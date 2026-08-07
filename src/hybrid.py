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
def continue_rollout(model, tok, input_ids: torch.Tensor, cut: int, max_new_tokens: int,
                     temperature: float) -> torch.Tensor:
    """input_ids[:cut]를 프리픽스로 반대 정책이 이어서 생성."""
    prefix = input_ids[:cut].unsqueeze(0).to(model.device)
    out = model.generate(
        prefix,
        attention_mask=torch.ones_like(prefix),
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
    )
    return out[0].cpu()


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
                for r in src_rows[:8]:  # cell당 소표본 (concept: '소표본으로 만든다')
                    resp_len = r["input_ids"].numel() - r["resp_start"]
                    cut = r["resp_start"] + max(1, int(resp_len * cut_frac))
                    seq = continue_rollout(cont_model, tok, r["input_ids"], cut,
                                           max_new_tokens, temperature)
                    text = tok.decode(seq[r["resp_start"]:], skip_special_tokens=True)
                    f.write(json.dumps({
                        "cell": cell, "prompt_idx": pi_idx,
                        "input_ids": seq.tolist(), "resp_start": r["resp_start"],
                        "reward": reward(text, gold), "cut_frac": cut_frac,
                    }) + "\n")
            print(f"  hybrid {pi_idx} (cut={cut_frac})", flush=True)


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
