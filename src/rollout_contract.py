#!/usr/bin/env python3
"""생성·소비 계약 단일화 — P0-1·P0-2 수정 (docs/PAPER_REVIEW_2026-08-19.md §14.3-A).

P0-1: HF generate()는 미지정 인자를 모델 generation_config에서 병합한다.
Qwen2.5-Instruct 배포 기본값(top_k=20, repetition_penalty=1.05)이 끼어들면
샘플링 분포가 teacher-forcing raw-softmax와 달라져 IS ratio가 논문의
estimand와 어긋난다(음수 KL의 원인 후보 1). 샘플링에 영향을 주는 인자
전부를 gen_kwargs()에서 명시해 차단하고, 해석된 설정을 manifest로 남긴다.

P0-2: 배치 생성은 이른 종료 행을 pad(=eos)로 채운다. 응답 구간은
[resp_start, resp_end) — resp_end는 첫 EOS를 포함한 직후 인덱스다. 생성
시점에 잘라 저장하고, 구버전 산출물(resp_end 미저장)은 EOS id 집합으로
재유도한다.
"""

from __future__ import annotations

ROLLOUT_STREAM_IDS = {
    "rollouts_behavior_train": 101,
    "rollouts_fresh_train": 211,
    "rollouts_fresh_val": 307,
}
ROLLOUT_SEED_SCHEME = "independent-source-per-prompt-v1"


def rollout_seed_base(experiment_seed: int, drift: int, source: str) -> int:
    """Return a stable, source-disjoint generation seed domain."""
    if source not in ROLLOUT_STREAM_IDS:
        raise ValueError(f"unknown rollout seed source: {source}")
    source_id = ROLLOUT_STREAM_IDS[source]
    source_drift = 0 if source == "rollouts_behavior_train" else int(drift)
    return (
        int(experiment_seed) * 1_000_003
        + source_drift * 65_537
        + source_id * 104_729
        + 17
    ) & 0x7FFFFFFF


def rollout_prompt_seed(seed_base: int, prompt_idx: int) -> int:
    """Derive a reshard- and resume-stable seed for one prompt."""
    return (int(seed_base) + int(prompt_idx) * 15_485_863) & 0x7FFFFFFF


def gen_kwargs(temperature: float, top_p: float, max_new_tokens: int,
               pad_token_id: int) -> dict:
    """model.generate()에 그대로 풀어 넣는 샘플링 인자 전부.

    top_k=0·repetition_penalty=1.0·no_repeat_ngram_size=0은 '기본값이라 생략'이
    아니라 generation_config 병합 차단용 명시다 — 지우면 P0-1이 재발한다.
    """
    return {
        "do_sample": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": 0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": int(pad_token_id),
    }


def eos_ids_of(model=None, tok=None, pad_id: int | None = None) -> set[int]:
    """종료로 취급할 토큰 id 집합 — tokenizer eos + generation_config/config의
    eos(단일 또는 리스트, Qwen2.5는 [im_end, endoftext]) + pad."""
    s: set[int] = set()
    if tok is not None and getattr(tok, "eos_token_id", None) is not None:
        s.add(int(tok.eos_token_id))
    for src in (getattr(model, "generation_config", None),
                getattr(model, "config", None)):
        v = getattr(src, "eos_token_id", None) if src is not None else None
        if v is None:
            continue
        s.update(int(x) for x in (v if isinstance(v, (list, tuple)) else [v]))
    if pad_id is not None:
        s.add(int(pad_id))
    return s


def resp_end_index(ids, resp_start: int, eos_ids) -> int:
    """응답 끝(exclusive) — resp_start 이후 첫 EOS를 포함한 위치+1, 없으면 len."""
    seq = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    eos = {int(eos_ids)} if isinstance(eos_ids, int) else {int(x) for x in eos_ids}
    for t in range(int(resp_start), len(seq)):
        if seq[t] in eos:
            return t + 1
    return len(seq)


def trim_row(r: dict, eos_ids=None) -> dict:
    """rollout 행 dict의 input_ids를 응답 끝에서 자르고 resp_end를 보장한다.

    - 신형 산출물: resp_end 저장돼 있음 → 그 값으로 자른다(멱등).
    - 구버전: eos_ids가 주어지면 재유도, 없으면 그대로 둔다(호출자가 판단).
    """
    ids = r["input_ids"]
    end = r.get("resp_end")
    if end is None and eos_ids:
        end = resp_end_index(ids, r["resp_start"], eos_ids)
    if end is not None:
        end = int(end)
        r["resp_end"] = end
        r["input_ids"] = ids[:end]
    return r


def resolved_manifest(
    model, tok, kwargs: dict, *, prompt_format: str | None = None
) -> dict:
    """실제 적용될 생성 설정 스냅샷 — 명시 인자 + 모델 generation_config 원본."""
    gc = getattr(model, "generation_config", None)
    keys = ("do_sample", "temperature", "top_p", "top_k", "min_p",
            "repetition_penalty", "no_repeat_ngram_size", "typical_p",
            "penalty_alpha", "num_beams", "eos_token_id")
    cfg = {k: getattr(gc, k, None) for k in keys} if gc is not None else {}
    for k, v in list(cfg.items()):
        if isinstance(v, (list, tuple)):
            cfg[k] = list(v)
    return {
        "explicit_kwargs": dict(kwargs),
        "model_generation_config": cfg,
        "eos_token_ids": sorted(eos_ids_of(model, tok)),
        "model_name_or_path": getattr(getattr(model, "config", None),
                                      "_name_or_path", None),
        "policy_adapter": getattr(model, "_om_policy_adapter", None),
        "prompt_format": prompt_format,
        "contract": ("sampling = raw softmax + temperature/top_p만 적용; "
                     "explicit_kwargs가 generation_config를 덮는다. "
                     "응답 구간 = [resp_start, resp_end), 첫 EOS 포함."),
    }
