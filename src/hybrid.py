"""2×2 hybrid(splice) intervention — occupancy/continuation 축을 데이터로 직접 처치.

감사(blocker C) 반영 재설계:
  · 네 cell 전부 **동일 K(기본 8)**, 전부 **oracle fresh와 독립** 생성
      bb = β rollout(기존 behavior에서 K개 — oracle과 무관)
      pp = π가 프롬프트부터 새로 생성 (oracle fresh 재사용 금지)
      bp = β-prefix 절단 + π-continuation
      pb = **새 pp의 π-prefix** 절단 + β-continuation (oracle 표본 미사용)
  · 프롬프트 subset은 호출측에서 seeded random(live 한정)으로 선정해 전달
  · 명칭은 taxonomy cell 직접 구현이 아니라 splice intervention임을 유지
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

from data import reward
from grads import ProjectionSpec, cosine, loo_advantages, prompt_gradient
from rollout import SAMPLING, _gen_batch_size, chat_ids
from rollout_contract import eos_ids_of, gen_kwargs, resp_end_index


@torch.no_grad()
def continue_rollouts_batch(model, tok, prefixes: list[torch.Tensor],
                            max_new_tokens: int, temperature: float,
                            resp_starts: list[int] | None = None) -> list[torch.Tensor]:
    """프리픽스 묶음을 left-padding 배치로 이어 생성 (샘플링 분포는 SAMPLING 공유).
    보존한 response prefix는 원래 horizon에서 차감한다. OM_GEN_BATCH 설정 시 그
    크기로 쪼개 생성한다 (27B KV 캐시 OOM 방어)."""
    if resp_starts is not None and len(resp_starts) != len(prefixes):
        raise ValueError("resp_starts and prefixes must have equal length")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    bs = _gen_batch_size(len(prefixes))
    budgets: dict[int, list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for idx, prefix in enumerate(prefixes):
        consumed = 0 if resp_starts is None else max(0, prefix.numel() - resp_starts[idx])
        remaining = max(0, max_new_tokens - consumed)
        budgets[remaining].append((idx, prefix))

    seqs: list[torch.Tensor | None] = [None] * len(prefixes)
    for remaining, indexed in budgets.items():
        if remaining == 0:
            for idx, prefix in indexed:
                seqs[idx] = prefix.cpu()
            continue
        for s in range(0, len(indexed), bs):
            chunk = indexed[s:s + bs]
            max_len = max(prefix.numel() for _, prefix in chunk)
            batch = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
            mask = torch.zeros_like(batch)
            for b, (_, prefix) in enumerate(chunk):  # left padding — 생성 시작 위치 정렬
                batch[b, max_len - prefix.numel():] = prefix
                mask[b, max_len - prefix.numel():] = 1
            batch, mask = batch.to(model.device), mask.to(model.device)
            generated = model.generate(
                batch, attention_mask=mask,
                **gen_kwargs(temperature, SAMPLING["top_p"], remaining, pad_id),
            )
            for b, (idx, prefix) in enumerate(chunk):
                gen_part = generated[b, max_len:].cpu()
                seqs[idx] = torch.cat([prefix.cpu(), gen_part])
    if any(seq is None for seq in seqs):
        raise RuntimeError("hybrid generation did not produce every requested sequence")
    return [seq for seq in seqs if seq is not None]


def _cut_prefixes(rows: list[dict], cut_frac: float) -> list[torch.Tensor]:
    out = []
    for r in rows:
        resp_len = r["input_ids"].numel() - r["resp_start"]
        cut = r["resp_start"] + max(1, int(resp_len * cut_frac))
        out.append(r["input_ids"][:cut])
    return out


def make_hybrid_cells(
    beta, pi, tok,
    behavior_rollouts: dict[int, list[dict]],
    prompts: list[dict],
    cut_frac: float,
    max_new_tokens: int,
    temperature: float,
    subset: list[int],
    out_path: Path,
    k_cell: int = 8,
) -> None:
    """subset 프롬프트마다 bb/pp/bp/pb 네 cell(각 K=k_cell)을 jsonl로 저장."""
    from grads import ts

    if not 0 < cut_frac < 1:
        raise ValueError(f"cut_frac must be in (0, 1), got {cut_frac}")
    if k_cell < 2:
        raise ValueError(f"k_cell must be at least 2 for LOO advantages, got {k_cell}")
    eos_set = eos_ids_of(beta, tok) | eos_ids_of(pi, tok)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        def emit(cell: str, pi_idx: int, seq: torch.Tensor, resp_start: int, gold: str):
            # P0-2: 첫 EOS(포함)에서 절단해 저장
            end = resp_end_index(seq, resp_start, eos_set)
            seq = seq[:end]
            text = tok.decode(seq[resp_start:], skip_special_tokens=True)
            f.write(json.dumps({
                "cell": cell, "prompt_idx": pi_idx,
                "input_ids": seq.tolist(), "resp_start": resp_start,
                "resp_end": end,
                "reward": reward(text, gold), "cut_frac": cut_frac,
            }) + "\n")

        for pi_idx in subset:
            gold = prompts[pi_idx]["answer"]
            b_rows = behavior_rollouts[pi_idx][:k_cell]
            if len(b_rows) != k_cell:
                raise ValueError(
                    f"prompt {pi_idx}: behavior cell needs K={k_cell}, got {len(b_rows)}"
                )

            # bb — 기존 β rollout 그대로 (oracle과 독립)
            for r in b_rows:
                emit("bb", pi_idx, r["input_ids"], r["resp_start"], gold)

            # pp — π가 프롬프트부터 새로 생성 (oracle fresh 미재사용, K 동일)
            pids = chat_ids(tok, prompts[pi_idx]["question"])
            pp_seqs = continue_rollouts_batch(pi, tok, [pids] * k_cell,
                                              max_new_tokens, temperature)
            pp_rows = [{"input_ids": s, "resp_start": pids.numel()} for s in pp_seqs]
            for r in pp_rows:
                emit("pp", pi_idx, r["input_ids"], r["resp_start"], gold)

            # bp — β-prefix 절단 + π-continuation
            b_prefixes = _cut_prefixes(b_rows, cut_frac)
            for r, seq in zip(b_rows, continue_rollouts_batch(
                    pi, tok, b_prefixes, max_new_tokens, temperature,
                    [r["resp_start"] for r in b_rows]), strict=True):
                emit("bp", pi_idx, seq, r["resp_start"], gold)

            # pb — 새 π-prefix 절단 + β-continuation
            p_prefixes = _cut_prefixes(pp_rows, cut_frac)
            for r, seq in zip(pp_rows, continue_rollouts_batch(
                    beta, tok, p_prefixes, max_new_tokens, temperature,
                    [r["resp_start"] for r in pp_rows]), strict=True):
                emit("pb", pi_idx, seq, r["resp_start"], gold)

            print(f"[{ts()}]  hybrid {pi_idx} (cut={cut_frac}, 4cells×K={k_cell})", flush=True)
    tmp_path.replace(out_path)


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
                for r, a in zip(rows, advs, strict=True)
            ]
            g = prompt_gradient(grad_model, params, rows, weights, spec)
            out[cell][pi_idx] = cosine(g, val_grad)
    return out
