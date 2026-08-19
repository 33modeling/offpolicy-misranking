"""rollout_contract (P0-1·P0-2 수정) 검증 — pytest 수집·스크립트 직접 실행 겸용."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rollout_contract import (eos_ids_of, gen_kwargs, resolved_manifest,
                              resp_end_index, trim_row)

EOS = 151645          # Qwen2.5 <|im_end|>
EOS2 = 151643         # Qwen2.5 <|endoftext|> (generation_config eos 리스트의 둘째)


def test_resp_end_basic():
    # 프롬프트 3토큰 + 응답 [7, 8, EOS] + padding [EOS, EOS]
    ids = [1, 2, 3, 7, 8, EOS, EOS, EOS]
    assert resp_end_index(ids, 3, EOS) == 6          # 첫 EOS 포함
    assert resp_end_index(ids, 3, {EOS, EOS2}) == 6
    # EOS 없이 max length 도달
    assert resp_end_index([1, 2, 3, 7, 8], 3, EOS) == 5
    # 프롬프트 안의 EOS(챗 템플릿 im_end)는 응답 절단에 쓰지 않는다
    assert resp_end_index([1, EOS, 3, 7, EOS], 2, EOS) == 5
    # 둘째 EOS id(endoftext)로 끝난 뒤 pad(im_end)가 붙은 경우
    ids2 = [1, 2, 3, 7, EOS2, EOS, EOS]
    assert resp_end_index(ids2, 3, {EOS, EOS2}) == 5
    # 텐서 입력
    try:
        import torch
        assert resp_end_index(torch.tensor(ids), 3, EOS) == 6
    except ImportError:
        pass


def test_trim_row():
    r = {"input_ids": [1, 2, 3, 7, 8, EOS, EOS], "resp_start": 3}
    trim_row(r, {EOS})
    assert r["resp_end"] == 6 and r["input_ids"] == [1, 2, 3, 7, 8, EOS]
    # 멱등: 저장된 resp_end가 있으면 eos_ids 없이도 같은 결과
    r2 = {"input_ids": [1, 2, 3, 7, 8, EOS], "resp_start": 3, "resp_end": 6}
    trim_row(r2)
    assert r2["input_ids"] == [1, 2, 3, 7, 8, EOS]
    # 구버전 + eos 미지정 → 무변경
    r3 = {"input_ids": [1, 2, 3, 7, 8, EOS, EOS], "resp_start": 3}
    trim_row(r3)
    assert "resp_end" not in r3 and len(r3["input_ids"]) == 7


def test_gen_kwargs_blocks_config_defaults():
    """Qwen2.5-Instruct 실 배포 generation_config에 gen_kwargs를 병합했을 때
    top_k·repetition_penalty가 실제로 차단되는지 — P0-1의 핵심."""
    from transformers import GenerationConfig
    import glob
    gc = None
    try:
        gc = GenerationConfig.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:
        # HF_HOME이 달라도 알려진 캐시 위치에서 스냅샷을 직접 찾는다
        pats = [os.path.expanduser("~/.cache/huggingface/hub"),
                str(Path(__file__).resolve().parents[1] / ".work/cache/huggingface/hub")]
        for root in pats:
            hits = glob.glob(f"{root}/models--Qwen--Qwen2.5-*-Instruct/"
                             "snapshots/*/generation_config.json")
            if hits:
                gc = GenerationConfig.from_pretrained(str(Path(hits[0]).parent))
                break
    if gc is None:  # 캐시 자체가 없는 환경 — 환경 문제로만 스킵
        print("  (스킵: Qwen2.5-Instruct generation_config 캐시 없음)")
        return
    # 전제: 배포 기본값에 위험 인자가 실재해야 이 테스트가 의미 있다
    # (top_k=20 전 크기 공통, repetition_penalty는 0.5B=1.1 / 7B·14B=1.05)
    assert gc.top_k == 20, f"전제 깨짐: top_k={gc.top_k}"
    assert gc.repetition_penalty > 1.0, \
        f"전제 깨짐: repetition_penalty={gc.repetition_penalty}"
    assert isinstance(gc.eos_token_id, list), "전제 깨짐: eos가 리스트가 아님"
    kw = gen_kwargs(temperature=1.0, top_p=1.0, max_new_tokens=8,
                    pad_token_id=EOS)
    merged = gc.to_dict()
    merged.update({k: v for k, v in kw.items() if k != "max_new_tokens"})
    resolved = GenerationConfig(**merged)
    assert resolved.top_k == 0
    assert resolved.repetition_penalty == 1.0
    assert resolved.top_p == 1.0
    assert resolved.temperature == 1.0
    assert resolved.do_sample is True


def test_manifest_contents():
    class _GC:
        do_sample, temperature, top_p, top_k = True, 0.7, 0.8, 20
        repetition_penalty, eos_token_id = 1.05, [EOS, EOS2]
    class _Cfg:
        _name_or_path, eos_token_id = "dummy/model", EOS
    class _M:
        generation_config, config = _GC(), _Cfg()
    class _Tok:
        eos_token_id = EOS
    kw = gen_kwargs(1.0, 1.0, 512, EOS)
    m = resolved_manifest(_M(), _Tok(), kw)
    assert m["explicit_kwargs"]["top_k"] == 0
    assert m["explicit_kwargs"]["repetition_penalty"] == 1.0
    assert m["model_generation_config"]["top_k"] == 20  # 원본 위험값이 기록됨
    assert set(m["eos_token_ids"]) == {EOS, EOS2}


def test_read_rollouts_trim(tmp_path=None):
    import tempfile
    import torch
    from experiment import read_rollouts
    d = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    p = d / "rollouts.jsonl"
    rows = [
        {"prompt_idx": 0, "rollout_idx": 0, "resp_start": 3,
         "input_ids": [1, 2, 3, 7, 8, EOS, EOS, EOS], "reward": 1.0},       # 구버전
        {"prompt_idx": 0, "rollout_idx": 1, "resp_start": 3, "resp_end": 5,
         "input_ids": [1, 2, 3, 9, EOS], "reward": 0.0},                    # 신형
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    # 구버전 + OM_EOS_IDS → 재유도 절단
    os.environ["OM_EOS_IDS"] = f"{EOS},{EOS2}"
    try:
        out = read_rollouts(p)
        assert out[0][0]["resp_end"] == 6 and out[0][0]["input_ids"].numel() == 6
        assert out[0][1]["resp_end"] == 5 and out[0][1]["input_ids"].numel() == 5
        # 미지정 → 구버전 행은 무변경(레거시), 신형 행은 resp_end로 절단
        del os.environ["OM_EOS_IDS"]
        out2 = read_rollouts(p)
        assert out2[0][0]["input_ids"].numel() == 8
        assert out2[0][1]["input_ids"].numel() == 5
    finally:
        os.environ.pop("OM_EOS_IDS", None)
    assert isinstance(out[0][0]["input_ids"], torch.Tensor)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL PASS ({len(fns)} tests)")
