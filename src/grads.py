"""2×2 추정량과 projected per-prompt gradient.

concept #68의 population 정의를 그대로 구현한다:
  g00 = Σ_t E_β[ r_t      R z_t ]   (CROPI token-ratio, raw outcome)
  g10 = Σ_t E_β[ P_t r_t  R z_t ]   (prefix만 교정)
  g01 = Σ_t E_β[ r_t S_t  R z_t ]   (suffix만 교정)
  g11 = Σ_t E_β[ P_t r_t S_t R z_t ] = g_π  (full trajectory IS)
  oracle/val = on-policy fresh rollout의 LOO group gradient (unbiased for E[Rz])

R 자리는 전부 leave-one-out 비정규화 advantage A_j = R_j - mean_{l≠j} R_l 를 쓴다
(baseline은 score expectation에서 소거 — verify_theory.py의 LOO unbiasedness 참조).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


ESTIMATORS = ("g00", "g10", "g01", "g11")


def loo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """rewards: (K,) → LOO 비정규화 advantage (K,). K=1이면 0."""
    k = rewards.numel()
    if k < 2:
        return torch.zeros_like(rewards)
    total = rewards.sum()
    return rewards - (total - rewards) / (k - 1)


def token_weights(
    logp_pi: torch.Tensor,
    logp_beta: torch.Tensor,
    advantage: float,
    estimator: str,
    clip_cap: float = 10.0,
) -> torch.Tensor:
    """한 rollout의 토큰별 가중치 w_t (grad에 곱해질 계수, detach 상태로 반환).

    logp_pi/logp_beta: (T,) 응답 토큰의 로그확률 (teacher forcing).
    clip_cap: 곱 비율의 상한 (하한은 1/cap). 안정화용 — population 정의엔 없음.
    """
    log_r = (logp_pi - logp_beta).detach()  # (T,)
    if estimator == "g00":
        log_w = log_r
    elif estimator == "g10":
        # P_t * r_t = exp( Σ_{u<=t} log r_u )  (누적 prefix, 현재 토큰 포함)
        log_w = torch.cumsum(log_r, dim=0)
    elif estimator == "g01":
        # r_t * S_t = exp( Σ_{u>=t} log r_u )  (현재 토큰 포함 미래 곱)
        total = log_r.sum()
        log_w = total - torch.cumsum(log_r, dim=0) + log_r
    elif estimator == "g11":
        log_w = log_r.sum().expand_as(log_r)
    else:
        raise ValueError(f"unknown estimator {estimator}")
    cap = math.log(clip_cap)
    return torch.exp(log_w.clamp(-cap, cap)) * advantage


@dataclass
class ProjectionSpec:
    dim: int = 4096
    seed: int = 20260807
    chunk: int = 8_000_000  # flatten grad를 이 크기로 잘라 투영 (메모리 상한)


@torch.no_grad()
def project_grads(params: list[torch.Tensor], spec: ProjectionSpec) -> torch.Tensor:
    """현재 .grad 를 flatten → 고정 시드 CountSketch(sparse JL) 투영 ((dim,) 반환).

    out[h(i)] += σ(i)·g_i — 좌표당 해시 인덱스·부호만 생성하므로 추가 메모리가
    청크당 수 MB다 (밀집 JL은 chunk×dim 행렬이 수십 GB라 OOM). 내적/cosine을
    기대값에서 보존하며(CROPI sparse projection 계열), 시드 소비 순서가 고정이라
    호출 간 일관된다. 청크 크기와 무관하게 param별 원소 순서로 결정된다.
    """
    device = params[0].device
    out = torch.zeros(spec.dim, dtype=torch.float32, device=device)
    offset = 0
    for p in params:
        n = p.numel()
        flat = None if p.grad is None else p.grad.detach().float().reshape(-1)
        if flat is not None:
            for start in range(0, n, spec.chunk):
                m = min(spec.chunk, n - start)
                # 전역 원소 위치의 정수 해시(splitmix형) — RNG 스트림과 무관해
                # 청크 크기·호출 순서가 결과를 바꾸지 않는다.
                pos = torch.arange(offset + start, offset + start + m,
                                   device=device, dtype=torch.int64)
                x = (pos + spec.seed) * 6364136223846793005 + 1442695040888963407
                x = x ^ (x >> 33)
                x = x * -7046029254386353131  # 0x9E3779B97F4A7C15 (int64 wrap)
                x = x ^ (x >> 29)
                idx = (x & 0x7FFFFFFF) % spec.dim
                sign = ((x >> 31) & 1).to(torch.float32) * 2 - 1
                out.scatter_add_(0, idx, flat[start : start + m].to(device) * sign)
                del pos, x, idx, sign
        offset += n
    return out.cpu()


def grad_params(model, last_n_layers: int) -> list[torch.Tensor]:
    """gradient 계산·투영 대상 파라미터 — 마지막 n개 decoder block (+ 최종 norm).

    lm_head/embedding은 크기 대비 신호가 중복이라 기본 제외 (concept '같은 gradient
    layer 범위' — 모든 estimator·oracle이 같은 목록을 쓰는 것이 유일한 요구).
    """
    layers = model.model.layers
    chosen: list[torch.Tensor] = []
    for layer in layers[len(layers) - last_n_layers :]:
        chosen += [p for p in layer.parameters()]
    chosen += list(model.model.norm.parameters())
    # LoRA merge_and_unload 이후엔 전체가 requires_grad=False일 수 있다 —
    # 대상만 켜고 나머지는 동결(backward 메모리 절약 겸용).
    model.requires_grad_(False)
    for p in chosen:
        p.requires_grad_(True)
    return chosen


def prompt_gradient(
    model,
    params: list[torch.Tensor],
    sequences: list[dict],
    weights: list[torch.Tensor],
    spec: ProjectionSpec,
    micro_batch: int = 2,
) -> torch.Tensor:
    """ĝ = (1/K) Σ_j Σ_t w_{j,t} ∇logπ(a_t|h) 를 한 프롬프트에 대해 계산해 투영.

    sequences[j]: {"input_ids": (L,), "resp_start": int} — 전체 시퀀스와 응답 시작.
    weights[j]: (T_j,) 토큰 가중치 (detach).
    """
    model.zero_grad(set_to_none=True)
    k = len(sequences)
    for start in range(0, k, micro_batch):
        batch = sequences[start : start + micro_batch]
        ws = weights[start : start + micro_batch]
        loss = 0.0
        for seq, w in zip(batch, ws):
            ids = seq["input_ids"].unsqueeze(0).to(model.device)
            logits = model(ids).logits[0, :-1].float()
            logp = torch.log_softmax(logits, dim=-1)
            tgt = ids[0, 1:]
            tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            resp = tok_logp[seq["resp_start"] - 1 :]
            t = min(resp.numel(), w.numel())
            loss = loss + (w[:t].to(resp.device) * resp[:t]).sum() / k
        loss.backward()
    return project_grads(params, spec)


@torch.no_grad()
def sequence_logprobs(model, input_ids: torch.Tensor, resp_start: int) -> torch.Tensor:
    """주어진 시퀀스의 응답 구간 토큰 로그확률 (T,) — teacher forcing, no grad."""
    ids = input_ids.unsqueeze(0).to(model.device)
    logits = model(ids).logits[0, :-1].float()
    logp = torch.log_softmax(logits, dim=-1)
    tgt = ids[0, 1:]
    tok = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return tok[resp_start - 1 :].cpu()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))


def ts() -> str:
    import time as _t
    return _t.strftime("%H:%M:%S")
