"""2×2 hybrid rollout — occupancy/continuation을 직접 바꾸는 유일한 처치축 (concept 10절).

cell 표기: (prefix 정책, continuation 정책)
  bb = β rollout 그대로              bp = β-prefix + π-continuation
  pb = π-prefix + β-continuation     pp = π fresh 그대로
prefix 절단점 f ∈ {0.25, 0.5, 0.75}에서 응답을 자르고 반대 정책으로 이어 쓴다.

각 cell의 on-policy식 LOO group gradient(비율 교정 없음)로 프롬프트 점수를 내고
oracle(pp) 대비 top-k precision을 비교한다 — g10/g01의 실패가 각각 continuation/
occupancy 축을 바꿀 때 줄어드는지가 게이트 통과 조건의 처치 검증이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from data import reward
from grads import ProjectionSpec, cosine, loo_advantages, prompt_gradient


@torch.no_grad()
def continue_rollouts_batch(model, tok, prefixes: list[torch.Tensor],
                            max_new_tokens: int, temperature: float) -> list[torch.Tensor]:
    """프리픽스 묶음을 left-padding 배치로 이어 생성 — 배치1 대비 GPU 활용 대폭 개선.

    반환: 각 항목의 (원 프리픽스 + 생성 토큰) 시퀀스 (pad 제거).
    """
    max_len = max(p.numel() for p in prefixes)
    pad_id = tok.pad_token_id or tok.eos_token_id
    batch = torch.full((len(prefixes), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros_like(batch)
    for b, p in enumerate(prefixes):  # left padding — 생성 시작 위치를 정렬
        batch[b, max_len - p.numel():] = p
        mask[b, max_len - p.numel():] = 1
    batch, mask = batch.to(model.device), mask.to(model.device)
    out = model.generate(
        batch, attention_mask=mask, do_sample=True, temperature=temperature,
        top_p=0.95, max_new_tokens=max_new_tokens, pad_token_id=pad_id,
    )
    seqs = []
    for b, p in enumerate(prefixes):
        gen_part = out[b, max_len:].cpu()
        seqs.append(torch.cat([p, gen_part]))
    return seqs


def make_hybrid_cells(
    beta, pi, tok,
    behavior_rollouts: dict[int, list[dict]],
    fresh_rollouts: dict[int, list[dict]],
    prompts: list[dict],
    cut_frac: float,
    max_new_tokens: int,
    temperature: float,
    n_prompts: int,
    out_path: Path,
) -> None:
    """bp·pb cell 생성 (bb·pp는 기존 rollout 재사용). jsonl 저장."""
    with out_path.open("w") as f:
        for pi_idx in sorted(behavior_rollouts)[:n_prompts]:
            gold = prompts[pi_idx]["answer"]
            for cell, src_rows, cont_model in (
                ("bp", behavior_rollouts[pi_idx], pi),
                ("pb", fresh_rollouts[pi_idx], beta),
            ):
                rows = src_rows[:8]  # cell당 소표본 (concept: '소표본으로 만든다')
                prefixes = []
                for r in rows:
                    resp_len = r["input_ids"].numel() - r["resp_start"]
                    cut = r["resp_start"] + max(1, int(resp_len * cut_frac))
                    prefixes.append(r["input_ids"][:cut])
                seqs = continue_rollouts_batch(cont_model, tok, prefixes,
                                               max_new_tokens, temperature)
                for r, seq in zip(rows, seqs):
                    text = tok.decode(seq[r["resp_start"]:], skip_special_tokens=True)
                    f.write(json.dumps({
                        "cell": cell, "prompt_idx": pi_idx,
                        "input_ids": seq.tolist(), "resp_start": r["resp_start"],
                        "reward": reward(text, gold), "cut_frac": cut_frac,
                    }) + "\n")
            from grads import ts; print(f"[{ts()}]  hybrid {pi_idx} (cut={cut_frac})", flush=True)


def score_cells(
    grad_model, params, cell_rows: dict[str, dict[int, list[dict]]],
    val_grad: torch.Tensor, spec: ProjectionSpec,
) -> dict:
    """cell별 프롬프트 점수 (gradient는 항상 π에서 계산 — 처치는 데이터 분포뿐)."""
    out: dict[str, dict[int, float]] = {}
    for cell, by_prompt in cell_rows.items():
        out[cell] = {}
        for pi_idx, rows in sorted(by_prompt.items()):
            advs = loo_advantages(torch.tensor([r["reward"] for r in rows]))
            weights = [
                torch.full((r["input_ids"].numel() - r["resp_start"],), float(a))
                for r, a in zip(rows, advs)
            ]
            g = prompt_gradient(grad_model, params, rows, weights, spec)
            out[cell][pi_idx] = cosine(g, val_grad)
    return out
