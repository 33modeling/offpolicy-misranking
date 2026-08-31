#!/usr/bin/env python3
"""Rollout collection and policy loading.

Policy training lives in train_policy_grpo.py. Keeping supervised fine-tuning
out of this module prevents it from being substituted for the RLVR objective.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from artifact_contract import sha256_file
from data import build_user_msg, reward
from rollout_contract import eos_ids_of, gen_kwargs, resolved_manifest, resp_end_index

# torch 2.7 + Hopper: cuDNN SDPA가 산발적 'unspecified launch failure'를 낸다
# (비동기라 보고 지점은 attention이 아닐 수도 있음 — modeling_qwen2.py:47 사례).
# load_model 안에만 두면 drift 학습(rollout.py 직접 로드)·downstream이 빠지므로
# import 시점 전역 비활성 — flash/efficient/math 커널로 폴백된다.
try:
    torch.backends.cuda.enable_cudnn_sdp(False)
except AttributeError:
    pass

# 샘플링 분포 — 감사 blocker A: top-p 절단 분포에서 표본을 뽑고 raw-softmax로
# ratio를 계산하면 정의한 IS estimand가 아니다. 기본을 full softmax(top_p=1.0)로
# 통일한다. 예전 조건 재현 시에만 OM_TOP_P=0.95 지정.
SAMPLING = {"top_p": float(os.environ.get("OM_TOP_P", "1.0"))}


def _eta(done: int, total: int, t_start: float) -> str:
    import time as _t
    if done == 0:
        return "?"
    rem = (total - done) * (_t.time() - t_start) / done
    h, m = int(rem // 3600), int(rem % 3600 // 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _attn_kwargs() -> dict:
    """OM_ATTN=eager|sdpa|flash_attention_2 — CUDA 커널 문제 시 코드 수정 없이 우회."""
    import os
    attn = os.environ.get("OM_ATTN")
    return {"attn_implementation": attn} if attn else {}


def load_model(name_or_path: str, device: str | None = None, dtype: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or auto_device()
    is_cuda = str(device).startswith("cuda")
    if dtype is None:
        dtype = "bfloat16" if is_cuda else "float32"
    tok = AutoTokenizer.from_pretrained(name_or_path)
    # device_map 대신 CPU 로드 → .to(cuda) 2단계: 신아키텍처(Qwen3.8 등)가
    # meta-init을 못 타면 device_map 경로가 GPU에 스켈레톤+체크포인트 이중
    # 상주(27B에서 ~52+49GB)로 OOM — CPU 경유는 GPU에 정확히 한 벌만 올린다.
    # (GPU 선택은 CUDA_VISIBLE_DEVICES가 담당하므로 기능 동일)
    want = getattr(torch, dtype)
    kw = dict(low_cpu_mem_usage=True, **_attn_kwargs())

    def _load(cls):
        try:
            return cls.from_pretrained(name_or_path, dtype=want, **kw)
        except TypeError:  # 구버전 transformers: dtype 인자 미지원 → 옛 이름
            return cls.from_pretrained(name_or_path, torch_dtype=want, **kw)

    try:
        model = _load(AutoModelForCausalLM)
    except (ValueError, KeyError):
        # Qwen3.6/3.8 계열은 멀티모달 automap이라 CausalLM 매핑에 없다 —
        # MM 클래스로 폴백 (텍스트 전용 사용, vision 경로는 안 탄다)
        from transformers import AutoModelForMultimodalLM
        model = _load(AutoModelForMultimodalLM)
    # dtype 무조건 통일 — 첫 파라미터만 bf16이고 일부 모듈(vision·MTP 등)이
    # fp32로 남는 혼합 로드까지 방어 (이미 맞는 텐서는 no-op라 비용 없음)
    model = model.to(want)
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) \
        + sum(b.numel() * b.element_size() for b in model.buffers())
    print(f"[load_model] 파라미터+버퍼 총 {n_bytes / 1e9:.1f}GB ({want})", flush=True)
    if is_cuda:
        free, total = torch.cuda.mem_get_info(device)
        print(f"[load_model] GPU free {free / 1e9:.1f}/{total / 1e9:.1f}GB, "
              f"필요 {n_bytes / 1e9:.1f}GB", flush=True)
        model.to(device)
    model.eval()
    gpu = torch.cuda.get_device_name(device) if is_cuda else "CPU"
    print(f"model loaded: {name_or_path} → {device} ({gpu}, {dtype})", flush=True)
    return model, tok


def _lora_targets() -> list | str:
    """OM_LORA_TARGETS: 콤마 목록 또는 'all-linear' — DeltaNet류 신아키텍처는
    q/v_proj가 일부 층에만 있어 all-linear가 안전하다 (27B G블록용)."""
    v = os.environ.get("OM_LORA_TARGETS", "")
    if not v:
        return ["q_proj", "v_proj"]
    return v if v == "all-linear" else [s.strip() for s in v.split(",") if s.strip()]


def _gen_batch_size(total: int) -> int:
    """OM_GEN_BATCH: generate 배치 상한 (0/미설정 = 전체 한 배치 — 기존 동작 그대로).
    27B처럼 가중치가 GPU를 거의 채우는 모델은 8 정도로 제한해 KV 캐시/프리필
    활성값 OOM을 막는다 — 긴 프롬프트에서 한참 돌다 터지는 그 경로."""
    try:
        v = int(os.environ.get("OM_GEN_BATCH", "0"))
    except ValueError:
        v = 0
    return total if v <= 0 else max(1, min(total, v))


def chat_ids(tok, question: str) -> torch.Tensor:
    msgs = [{"role": "user", "content": build_user_msg(question)}]
    # transformers 4/5 양쪽에서 안전: 템플릿은 텍스트로 뽑고 별도로 토크나이즈.
    # thinking 계열(Qwen3+)은 기본 OFF — 짧은 rollout이라는 추정량 정의를 보존.
    # (해당 인자를 모르는 템플릿에서는 TypeError 폴백)
    kw = {} if os.environ.get("OM_THINKING") == "on" else {"enable_thinking": False}
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
    except TypeError:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]


def salvage_partial(part_path: Path, k: int) -> set[int]:
    """중단된 shard의 .partial에서 K개를 전부 갖춘 프롬프트만 남기고 재개 집합을 반환.

    ULF류 간헐 크래시 노드에서 수 시간짜리 shard를 프롬프트 0부터 다시 돌지 않기
    위한 장치 — 강제 종료로 찢긴 마지막 줄부터는 버리고, K개 미달 프롬프트는
    통째로 제거해 중복 행·부분 행이 최종 산출물에 들어갈 수 없게 한다."""
    if not part_path.exists():
        return set()
    rows_by_prompt: dict[int, list[str]] = {}
    with part_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                break  # 찢긴 꼬리 줄 — 이 지점부터는 신뢰하지 않는다
            rows_by_prompt.setdefault(int(row["prompt_idx"]), []).append(
                line if line.endswith("\n") else line + "\n"
            )
    complete = {p: rows for p, rows in rows_by_prompt.items() if len(rows) == k}
    tmp = part_path.with_suffix(part_path.suffix + ".rewrite")
    with tmp.open("w") as f:
        for p in sorted(complete):
            f.writelines(complete[p])
    tmp.replace(part_path)
    return set(complete)


@torch.no_grad()
def collect_rollouts(
    model,
    tok,
    prompts: list[dict],
    k: int,
    max_new_tokens: int,
    temperature: float,
    out_path: Path,
    batch_prompts: int = 8,
    idx_offset: int = 0,
) -> None:
    """프롬프트별 K개 응답 생성 → jsonl (token id·reward 저장, logp는 나중에 재계산)."""
    import time

    from grads import ts

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{ts()}] rollout 시작: {len(prompts)} prompts × K={k}, "
          f"max_new={max_new_tokens}, temp={temperature} → {out_path.name}", flush=True)
    t_start = time.time()
    # P0-1: 샘플링 인자 전체 명시(generation_config 병합 차단) + manifest 기록
    gkw = gen_kwargs(temperature, SAMPLING["top_p"], max_new_tokens,
                     tok.eos_token_id)
    eos_set = eos_ids_of(model, tok, pad_id=tok.eos_token_id)
    manifest_path = out_path.parent / (out_path.stem + ".manifest.json")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest = resolved_manifest(model, tok, gkw)
    manifest.update({"k": k, "n_prompts": len(prompts), "idx_offset": idx_offset})
    # Keep an in-progress record without replacing a previously valid sidecar.
    # The final manifest is published only after the JSONL and its hash exist.
    manifest_tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    # 프롬프트 단위 내구 저장 + 중간 재개 — 간헐 크래시(ULF) 노드에서 shard 전체를
    # 처음부터 다시 돌지 않는다. 구버전이 남긴 .tmp가 있으면 진행분을 승계한다.
    part_path = out_path.with_suffix(".partial")
    legacy_tmp = out_path.with_suffix(".tmp")
    if not part_path.exists() and legacy_tmp.exists():
        legacy_tmp.rename(part_path)
    done = salvage_partial(part_path, k)
    if done:
        print(f"[{ts()}] rollout 재개: 완료 프롬프트 {len(done)}개 스킵 "
              f"({part_path.name})", flush=True)
    todo_total = sum(1 for i in range(len(prompts)) if idx_offset + i not in done)
    session_done = 0
    with part_path.open("a") as f:
        for i, item in enumerate(prompts):
            if idx_offset + i in done:
                continue
            t0 = time.time()
            ids = chat_ids(tok, item["question"]).to(model.device)
            resp_start = ids.numel()
            n_correct = 0
            bs = _gen_batch_size(k)
            j = 0
            for s in range(0, k, bs):
                nb = min(bs, k - s)
                batch_ids = ids.unsqueeze(0).expand(nb, -1)
                gen = model.generate(
                    batch_ids,
                    attention_mask=torch.ones_like(batch_ids),
                    **gkw,
                )
                for seq in gen:
                    # P0-2: 첫 EOS(포함)에서 절단 — padding을 저장하지 않는다
                    end = resp_end_index(seq, resp_start, eos_set)
                    seq = seq[:end]
                    text = tok.decode(seq[resp_start:], skip_special_tokens=True)
                    r = reward(text, item["answer"])
                    n_correct += r > 0.5
                    f.write(
                        json.dumps(
                            {
                                "prompt_idx": idx_offset + i,
                                "rollout_idx": j,
                                "input_ids": seq.tolist(),
                                "resp_start": resp_start,
                                "resp_end": end,
                                "reward": r,
                            }
                        )
                        + "\n"
                    )
                    j += 1
            f.flush()  # 프롬프트 단위 내구 지점 — 크래시 시 여기까지는 재개 가능
            session_done += 1
            n_have = len(done) + session_done
            print(f"[{ts()}]  rollout {n_have}/{len(prompts)} "
                  f"({100 * n_have // len(prompts)}%, {time.time() - t0:.0f}s/개, "
                  f"정답 {n_correct}/{k}, ETA {_eta(session_done, todo_total, t_start)})",
                  flush=True)
    # fail-closed 발행: 행 수가 정확히 n_prompts×K일 때만 최종 이름을 얻는다
    n_rows = sum(1 for _ in part_path.open())
    if n_rows != len(prompts) * k:
        raise RuntimeError(
            f"rollout 산출물 행 수 불일치: {n_rows} != {len(prompts)}×{k} — "
            f"{part_path} 보존, 발행 중단")
    part_path.rename(out_path)  # 원자적 완료 표시 — 중단된 부분 파일은 .partial로 남는다
    manifest.update({
        "artifact_file": out_path.name,
        "artifact_sha256": sha256_file(out_path),
    })
    manifest_tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    manifest_tmp.replace(manifest_path)


def load_policy(base: str, adapter: Path | None, device: str | None = None):
    """π 로드 — adapter가 있으면 base+LoRA 병합, 없으면 base 그대로 (=β)."""
    model, tok = load_model(base, device=device)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        model = model.merge_and_unload()
        model.eval()
    return model, tok
