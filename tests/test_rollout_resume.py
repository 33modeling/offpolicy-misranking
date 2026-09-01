"""rollout shard 중간 재개(salvage_partial + collect_rollouts .partial 경로) 검증.

배경: v4-27b에서 간헐 ULF가 fresh rollout을 46/128 지점에서 죽였고, 기존
전체-.tmp 쓰기는 재시도마다 프롬프트 0부터 다시 돌아 간헐 크래시 노드에서
shard가 영원히 완주하지 못했다. 이 테스트는 ① 부분 파일 구제 규칙과
② 크래시→재실행 시 완료 프롬프트를 건너뛰고 정확한 최종 산출물을 발행하는
경로를 GPU 없이 검증한다.
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import rollout
from rollout import collect_rollouts, salvage_partial

PASS = 0
FAIL = 0


def ok(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def row_line(p, j, tail="\n"):
    return json.dumps({"prompt_idx": p, "rollout_idx": j, "input_ids": [1, 2, 3],
                       "resp_start": 2, "resp_end": 3, "reward": 0.0}) + tail


def test_salvage():
    with tempfile.TemporaryDirectory() as d:
        part = Path(d) / "r.shard0.partial"

        ok("없는 파일 → 빈 집합", salvage_partial(part, 2) == set())

        # 완주 2개 + 미완 1개 → 완주만 남는다
        part.write_text(row_line(0, 0) + row_line(0, 1)
                        + row_line(1, 0) + row_line(1, 1) + row_line(2, 0))
        got = salvage_partial(part, 2)
        kept = [json.loads(l)["prompt_idx"] for l in part.read_text().splitlines()]
        ok("완주 프롬프트만 재개 집합", got == {0, 1}, got)
        ok("미완 프롬프트 행 제거", kept == [0, 0, 1, 1], kept)

        # 찢긴 마지막 줄 — 그 지점부터 버림
        part.write_text(row_line(0, 0) + row_line(0, 1) + '{"prompt_idx": 1, "rol')
        ok("찢긴 꼬리 이후 폐기", salvage_partial(part, 2) == {0})

        # 마지막 줄에 개행 없음(정상 JSON) — 보존되고 개행 보충
        part.write_text(row_line(3, 0) + row_line(3, 1, tail=""))
        got = salvage_partial(part, 2)
        ok("개행 없는 완결 행 보존", got == {3}, got)
        ok("재작성 후 전 행 개행 종결", part.read_text().endswith("\n"))

        part.write_text(row_line(0, 0) + row_line(0, 0))
        ok("중복 rollout_idx 그룹 폐기", salvage_partial(part, 2) == set())


class FakeTok:
    eos_token_id = 9

    def apply_chat_template(self, msgs, add_generation_prompt=True,
                            tokenize=False, **kw):
        return "Q"

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        return SimpleNamespace(input_ids=torch.tensor([[5, 6]]))

    def decode(self, ids, skip_special_tokens=True):
        return "답"


class FakeModel:
    """crash_at 번째 generate 호출에서 ULF를 흉내낸다 (None이면 정상)."""

    device = "cpu"

    def __init__(self, crash_at=None):
        self.crash_at = crash_at
        self.calls = 0

    def generate(self, batch_ids, attention_mask=None, **gkw):
        self.calls += 1
        if self.crash_at is not None and self.calls >= self.crash_at:
            raise RuntimeError("CUDA error: unspecified launch failure")
        # 프롬프트(2토큰) + RNG-dependent 응답 [7|8, eos]
        n = batch_ids.shape[0]
        response = torch.cat(
            [torch.randint(7, 9, (n, 1)), torch.full((n, 1), 9)], dim=1
        )
        return torch.cat([batch_ids, response], dim=1)


def test_crash_resume(monkeypatched_reward=True):
    rollout.reward = lambda text, answer: 1.0
    prompts = [{"question": f"q{i}", "answer": "1"} for i in range(4)]
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "rollouts_fresh_train.shard0.jsonl"

        # 1차: 3번째 generate 호출(프롬프트 2)에서 크래시 → 프롬프트 0·1만 내구
        try:
            collect_rollouts(FakeModel(crash_at=3), FakeTok(), prompts, k=2,
                             max_new_tokens=4, temperature=1.0, out_path=out,
                             sampling_seed_base=123)
            ok("크래시 발생", False)
        except RuntimeError as e:
            ok("크래시 발생", "unspecified launch failure" in str(e))
        part = out.with_suffix(".partial")
        ok("크래시 후 .partial 보존", part.exists())
        ok("최종 파일 미발행", not out.exists())
        done = {json.loads(l)["prompt_idx"] for l in part.open()}
        ok("완료 프롬프트 0·1 내구", done == {0, 1}, done)

        # 2차: 정상 모델로 재실행 → 0·1 스킵, 2·3만 생성, 발행
        m2 = FakeModel()
        collect_rollouts(m2, FakeTok(), prompts, k=2,
                         max_new_tokens=4, temperature=1.0, out_path=out,
                         sampling_seed_base=123)
        ok("재실행 시 미완분만 생성(2프롬프트×1배치)", m2.calls == 2, m2.calls)
        ok("최종 파일 발행", out.exists() and not part.exists())
        rows = [json.loads(l) for l in out.open()]
        ok("행 수 = 4×2", len(rows) == 8, len(rows))
        seen = {(r["prompt_idx"], r["rollout_idx"]) for r in rows}
        ok("중복 없음", len(seen) == 8)
        ok("전 프롬프트 커버", {r["prompt_idx"] for r in rows} == {0, 1, 2, 3})
        man = json.loads((out.parent / (out.stem + ".manifest.json")).read_text())
        ok("manifest sha 발행", "artifact_sha256" in man)
        ok("manifest RNG seed 발행", man.get("sampling_seed_base") == 123)

        clean_dir = Path(d) / "clean"
        clean_dir.mkdir()
        clean = clean_dir / out.name
        collect_rollouts(
            FakeModel(), FakeTok(), prompts, k=2,
            max_new_tokens=4, temperature=1.0, out_path=clean,
            sampling_seed_base=123,
        )
        ok("중단 재개와 무중단 생성이 동일", out.read_bytes() == clean.read_bytes())

        # 3차: 구버전 .tmp 승계 — 완주 1프롬프트가 든 legacy tmp에서 재개
        out2 = Path(d) / "rollouts_fresh_train.shard1.jsonl"
        legacy = out2.with_suffix(".tmp")
        legacy.write_text(row_line(0, 0) + row_line(0, 1))
        m3 = FakeModel()
        collect_rollouts(m3, FakeTok(), prompts, k=2,
                         max_new_tokens=4, temperature=1.0, out_path=out2,
                         sampling_seed_base=123)
        ok("legacy .tmp 승계 후 미완분만 생성", m3.calls == 3, m3.calls)
        ok("legacy 경로 발행·행수 정확",
           out2.exists() and sum(1 for _ in out2.open()) == 8)

        # partial의 generation 계약이 달라지면 섞어 쓰지 않고 전부 재생성한다.
        out3 = Path(d) / "contract-change.jsonl"
        try:
            collect_rollouts(FakeModel(crash_at=3), FakeTok(), prompts, k=2,
                             max_new_tokens=4, temperature=1.0, out_path=out3,
                             sampling_seed_base=1)
        except RuntimeError:
            pass
        m4 = FakeModel()
        collect_rollouts(m4, FakeTok(), prompts, k=2,
                         max_new_tokens=4, temperature=1.0, out_path=out3,
                         sampling_seed_base=2)
        ok("partial 계약 변경 시 전 프롬프트 재생성", m4.calls == 4, m4.calls)
        ok("불일치 partial 격리", (Path(d) / ".restart-quarantine").is_dir())


def test_publication_recovery():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "rollouts.jsonl"
        out.write_text(
            row_line(0, 0) + row_line(0, 1) + row_line(1, 0) + row_line(1, 1)
        )
        in_progress = Path(d) / "rollouts.manifest.json.tmp"
        in_progress.write_text(json.dumps({"k": 2, "n_prompts": 2, "idx_offset": 0}))
        ok("JSONL 뒤 manifest 발행 중단 복구", rollout.rollout_artifact_ready(out))
        manifest = json.loads((Path(d) / "rollouts.manifest.json").read_text())
        ok("복구 manifest가 artifact hash 결합", bool(manifest.get("artifact_sha256")))
        ok("복구 후 in-progress manifest 제거", not in_progress.exists())


test_salvage()
test_crash_resume()
test_publication_recovery()
print(f"\nPASS {PASS} / FAIL {FAIL}")


def test_rollout_resume_regressions() -> None:
    assert FAIL == 0


if __name__ == "__main__":
    sys.exit(1 if FAIL else 0)
