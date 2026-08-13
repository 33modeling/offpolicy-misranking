"""게이트 실험 orchestrator — stage 단위 실행, 산출물은 outputs/<run>/ 아래 저장.

stages:
  rollout-behavior  β(base)로 train+val 프롬프트 rollout 수집
  drift             정답 rollout LoRA RFT로 π 체크포인트 생성 (steps 스윕)
  score             β rollout에 4개 추정량(g00/g10/g01/g11) 적용 → 프롬프트 점수
  oracle            π fresh rollout 수집 → oracle 점수·split-half noise floor·micro-group grads
  report            top-k precision/Jaccard 표 + margin + CertaGrad vs uniform
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from certagrad import certagrad, uniform_baseline
from data import load_prompts
from grads import (
    ESTIMATORS,
    ProjectionSpec,
    cosine,
    grad_params,
    loo_advantages,
    prompt_gradient,
    sequence_logprobs,
    token_weights,
    weight_stats,
)
from rollout import collect_rollouts, load_policy, train_drift_lora


def _atomic_text(path: Path, text: str) -> None:
    """쓰다 죽어도 완성본만 남는다 — 부분 파일이 exists() 재개 검사를 통과하는 오염 방지."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def read_rollouts(path: Path) -> dict[int, list[dict]]:
    by_prompt: dict[int, list[dict]] = defaultdict(list)
    for line in path.open():
        r = json.loads(line)
        r["input_ids"] = torch.tensor(r["input_ids"])
        by_prompt[r["prompt_idx"]].append(r)
    return by_prompt


def stage_score(args, run: Path, pi=None, beta=None, shard: tuple[int, int] | None = None) -> None:
    """β rollout에 대해 π/β 로그확률 → 4개 추정량 → projected gradient → 점수.

    shard=(i,n)이면 프롬프트를 인터리브(i::n)로 나눠 scores_offpolicy.shard{i}.json에 쓴다."""
    spec = ProjectionSpec(dim=args.proj_dim)
    rollouts = read_rollouts(run / "rollouts_behavior_train.jsonl")
    if shard is not None:
        keys = sorted(rollouts)[shard[0]::shard[1]]
        rollouts = {k: rollouts[k] for k in keys}

    # 대형 모델 대응 2-pass: 두 모델을 동시에 올리지 않는다 (π+β 동시 로드가
    # 14B에서 attention OOM의 원인). β 로그확률을 먼저 전부 계산해 두고 β를
    # 내린 뒤 π를 올려 gradient까지 계산한다. 모델이 주입된 경우(analyze)는
    # 기존 동시 경로 유지.
    two_pass = pi is None and beta is None
    beta_logps: dict[int, list] = {}
    if two_pass:
        beta, _ = load_policy(args.model, None)
        for pi_idx, rows in sorted(rollouts.items()):
            beta_logps[pi_idx] = [
                sequence_logprobs(beta, r["input_ids"], r["resp_start"]) for r in rows
            ]
        del beta
        beta = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"score 2-pass: β 로그확률 {len(beta_logps)} prompts 완료 — β 언로드 후 π 로드")
    if pi is None:
        pi, _ = load_policy(args.model, Path(args.adapter) if args.adapter else None)
    if beta is None and not two_pass:
        beta, _ = load_policy(args.model, None)
    params = grad_params(pi, args.grad_layers)

    # validation 방향은 π fresh가 정본이지만, score stage에서는 우선 저장된 oracle
    # stage 산출물을 쓴다 (oracle을 먼저 돌릴 것).
    val = torch.load(run / "val_gradient.pt", weights_only=True)

    import time as _time

    from rollout import _eta
    t_start = _time.time()
    n_done = 0
    out = {est: {} for est in ESTIMATORS}
    div_rows: list[dict] = []
    for pi_idx, rows in sorted(rollouts.items()):
        n_done += 1
        rewards = torch.tensor([r["reward"] for r in rows])
        advs = loo_advantages(rewards)
        logps_pi = [sequence_logprobs(pi, r["input_ids"], r["resp_start"]) for r in rows]
        logps_beta = beta_logps[pi_idx] if two_pass else [
            sequence_logprobs(beta, r["input_ids"], r["resp_start"]) for r in rows
        ]
        for lp, lb in zip(logps_pi, logps_beta):
            div_rows.append(weight_stats(lp, lb, clip_cap=args.clip_cap))
        for est in ESTIMATORS:
            weights = [
                token_weights(lp, lb, float(a), est, clip_cap=args.clip_cap)
                for lp, lb, a in zip(logps_pi, logps_beta, advs)
            ]
            g = prompt_gradient(pi, params, rows, weights, spec, micro_batch=args.micro_batch)
            out[est][pi_idx] = {"score": cosine(g, val), "norm": float(g.norm())}
        if n_done % 5 == 0:
            from grads import ts
            print(f"[{ts()}]  score {n_done}/{len(rollouts)} "
                  f"({100 * n_done // len(rollouts)}%, ETA {_eta(n_done, len(rollouts), t_start)})", flush=True)
    out_name = ("scores_offpolicy.json" if shard is None
                else f"scores_offpolicy.shard{shard[0]}.json")
    _atomic_text(run / out_name, json.dumps(out, indent=1))
    # 진단 통계 (감사 §5·§6·§16): 토큰 KL̂(β‖π)·궤적 ESS·추정량별 clip 비율
    if div_rows:
        n_tok = sum(d["tokens"] for d in div_rows)
        lr = torch.tensor([d["traj_logratio"] for d in div_rows], dtype=torch.float64)
        w = torch.exp(lr - lr.max())
        ess = float(w.sum() ** 2 / (w ** 2).sum() / len(w))
        stats = {"token_kl_beta_pi": sum(d["kl_sum"] for d in div_rows) / max(1, n_tok),
                 "traj_ess_frac_g11": ess, "rollouts": len(div_rows)}
        for est in ESTIMATORS:
            stats[f"clipfrac_{est}"] = sum(d[f"clipfrac_{est}"] for d in div_rows) / len(div_rows)
        dname = ("divergence_stats.json" if shard is None
                 else f"divergence_stats.shard{shard[0]}.json")
        _atomic_text(run / dname, json.dumps(stats, indent=1))


def stage_oracle(args, run: Path, pi=None, tok=None, shard: tuple[int, int] | None = None) -> None:
    """π fresh rollout으로 oracle 점수·noise floor·CertaGrad용 micro-group 저장.

    shard=(i,n)이면 gradient 계산을 프롬프트 인터리브로 분담 — val 방향은 shard 0만."""
    spec = ProjectionSpec(dim=args.proj_dim)
    if pi is None:
        pi, tok = load_policy(args.model, Path(args.adapter) if args.adapter else None)
    params = grad_params(pi, args.grad_layers)
    prompts = json.loads((run / "prompts.json").read_text())

    fresh_path = run / "rollouts_fresh_train.jsonl"
    if not fresh_path.exists():
        collect_rollouts(
            pi, tok, prompts["train"], args.fresh_k, args.max_new_tokens,
            args.temperature, fresh_path,
        )
    val_path = run / "rollouts_fresh_val.jsonl"
    if not val_path.exists():
        collect_rollouts(
            pi, tok, prompts["val"], args.val_k, args.max_new_tokens,
            args.temperature, val_path,
        )

    # validation 방향 v — 비샤드 경로 전담 (샤드 모드는 val-grads 스테이지가 담당)
    if shard is None:
        # 산출물 2개(gradient+groups) — 둘 다 있어야 스킵 (부분 상태 오인 방지, B5류)
        if not ((run / "val_gradient.pt").exists() and (run / "val_groups.pt").exists()):
            v_sum, groups_v = None, []
            for pi_idx, rows in sorted(read_rollouts(val_path).items()):
                advs = loo_advantages(torch.tensor([r["reward"] for r in rows]))
                weights = [
                    torch.full((r["input_ids"].numel() - r["resp_start"],), float(a))
                    for r, a in zip(rows, advs)
                ]
                g = prompt_gradient(pi, params, rows, weights, spec, micro_batch=args.micro_batch)
                groups_v.append(g)
                v_sum = g if v_sum is None else v_sum + g
            val_grad = v_sum / len(groups_v)
            _atomic_save(torch.stack(groups_v), run / "val_groups.pt")
            _atomic_save(val_grad, run / "val_gradient.pt")

    # oracle 점수 + split-half + micro-group 저장
    import time as _time

    from rollout import _eta
    t_start = _time.time()
    n_done = 0
    oracle, halves, micro = {}, {}, {}
    fresh_by_prompt = read_rollouts(fresh_path)
    if shard is not None:
        keys = sorted(fresh_by_prompt)[shard[0]::shard[1]]
        fresh_by_prompt = {k: fresh_by_prompt[k] for k in keys}
    for pi_idx, rows in sorted(fresh_by_prompt.items()):
        n_done += 1
        gsize = args.micro_group
        group_grads = []
        for s in range(0, len(rows), gsize):
            chunk = rows[s : s + gsize]
            advs = loo_advantages(torch.tensor([r["reward"] for r in chunk]))
            weights = [
                torch.full((r["input_ids"].numel() - r["resp_start"],), float(a))
                for r, a in zip(chunk, advs)
            ]
            group_grads.append(prompt_gradient(pi, params, chunk, weights, spec, micro_batch=args.micro_batch))
        stack = torch.stack(group_grads)
        micro[pi_idx] = stack
        if shard is None:
            mu = stack.mean(dim=0)
            oracle[pi_idx] = {"score": cosine(mu, val_grad), "norm": float(mu.norm())}
            h = stack.shape[0] // 2
            halves[pi_idx] = {
                "a": cosine(stack[:h].mean(dim=0), val_grad),
                "b": cosine(stack[h:].mean(dim=0), val_grad),
            }
        if n_done % 5 == 0:
            from grads import ts
            print(f"[{ts()}]  oracle {n_done}/{len(fresh_by_prompt)} "
                  f"({100 * n_done // len(fresh_by_prompt)}%, ETA {_eta(n_done, len(fresh_by_prompt), t_start)})", flush=True)
    if shard is None:
        # 산출물 3개 — 완료 마커 역할인 scores_oracle.json을 마지막에 쓴다
        _atomic_save(micro, run / "oracle_micro_groups.pt")
        _atomic_text(run / "scores_splithalf.json", json.dumps(halves, indent=1))
        _atomic_text(run / "scores_oracle.json", json.dumps(oracle, indent=1))
    else:
        _atomic_save(micro, run / f"oracle_micro_groups.shard{shard[0]}.pt")


def topk(scores: dict, frac: float, seed: int = 0) -> set:
    """동률은 seeded 난수로 무작위 절단 — index-순 절단의 임의성 제거 (감사 §11)."""
    import random as _r
    rng = _r.Random(seed ^ 0x5EED)
    jit = {i: rng.random() for i in scores}
    k = max(1, int(len(scores) * frac))
    return set(sorted(scores, key=lambda i: (-scores[i]["score"], jit[i]))[:k])


def stage_report(args, run: Path) -> None:
    oracle = {int(k): v for k, v in json.loads((run / "scores_oracle.json").read_text()).items()}
    off = json.loads((run / "scores_offpolicy.json").read_text())
    halves = {int(k): v for k, v in json.loads((run / "scores_splithalf.json").read_text()).items()}

    frac = args.topk_frac
    seed = getattr(args, "seed", 0)
    o_top = topk(oracle, frac, seed)
    k = len(o_top)
    # 동점 절단 jitter는 절반별로 독립이어야 한다 — 같은 seed면 동점이 많은
    # 체제에서 양쪽이 같은 임의 선택을 해 floor가 인위로 부풀려진다(2026-08-13 발견).
    ha = topk({i: {"score": h["a"]} for i, h in halves.items()}, frac, seed)
    hb = topk({i: {"score": h["b"]} for i, h in halves.items()}, frac, seed + 104729)
    noise_floor = len(ha & hb) / k

    def _ties_and_zeros(sc: dict) -> tuple[int, int]:
        vals = sorted((v["score"] for v in sc.values()), reverse=True)
        kth = vals[k - 1] if len(vals) >= k else vals[-1]
        ties = sum(1 for v in vals if abs(v - kth) < 1e-9)
        zeros = sum(1 for v in sc.values() if v.get("norm", 1.0) < 1e-6)
        return ties, zeros

    lines = [f"# report  (top-{int(frac*100)}%, k={k}, "
             f"split_half_reliability={noise_floor:.3f} — 참조치이며 상한 아님)", ""]
    lines.append("| estimator | top-k precision | Jaccard | Δ vs floor |")
    lines.append("|---|---|---|---|")
    results = {"noise_floor": noise_floor, "split_half_reliability": noise_floor,
               "k": k, "seed": seed, "boundary_ties": {}, "zero_grad": {}}
    ot, oz = _ties_and_zeros(oracle)
    results["boundary_ties"]["oracle"], results["zero_grad"]["oracle"] = ot, oz
    for est in ESTIMATORS:
        scores = {int(i): v for i, v in off[est].items()}
        t_, z_ = _ties_and_zeros(scores)
        results["boundary_ties"][est], results["zero_grad"][est] = t_, z_
        e_top = topk(scores, frac, seed)
        prec = len(e_top & o_top) / k
        jac = len(e_top & o_top) / len(e_top | o_top)
        results[est] = {"precision": prec, "jaccard": jac}
        lines.append(f"| {est} | {prec:.3f} | {jac:.3f} | {prec - noise_floor:+.3f} |")

    # CertaGrad vs uniform (fresh micro-group 시뮬레이션)
    micro = torch.load(run / "oracle_micro_groups.pt", weights_only=True)
    val_groups = torch.load(run / "val_groups.pt", weights_only=True)
    order = sorted(micro)
    pools = [micro[i].float() for i in order]
    cg = certagrad(pools, val_groups.float(), k, radius_mode=args.radius_mode)
    total_pool = sum(p.shape[0] for p in pools)
    uni = uniform_baseline(pools, val_groups.float(), k, groups_each=pools[0].shape[0])
    cg_sel = {order[i] for i in cg["selected"]}
    uni_sel = {order[i] for i in uni["selected"]}
    uniform_precision = len(uni_sel & o_top) / k
    results["certagrad"] = {
        "certified": cg["certified"],
        "fresh_groups": cg["fresh_groups"],
        "fresh_frac_of_uniform": cg["fresh_groups"] / uni["fresh_groups"],
        "precision_vs_oracle": len(cg_sel & o_top) / k,
        "uniform_precision_vs_oracle": uniform_precision,
    }
    lines += [
        "",
        f"CertaGrad: certified={cg['certified']} fresh={cg['fresh_groups']}/{uni['fresh_groups']} "
        f"({results['certagrad']['fresh_frac_of_uniform']:.2f}× of uniform), "
        f"precision={results['certagrad']['precision_vs_oracle']:.3f} "
        f"(uniform precision={uniform_precision:.3f})",
    ]
    _atomic_text(run / "report.md", "\n".join(lines))
    _atomic_text(run / "report.json", json.dumps(results, indent=1))
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["prep", "rollout-behavior", "drift", "score", "oracle",
                            "report", "hybrid", "analyze", "downstream",
                            "rollout-fresh", "oracle-grads", "score-shard",
                            "merge-grads", "val-grads", "val-deepen"])
    p.add_argument("--run", default="outputs/pilot")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=None, help="π LoRA adapter 경로 (없으면 β=π)")
    p.add_argument("--dataset", default="gsm8k",
                   choices=["gsm8k", "math500", "mbpp", "kk", "dapo-math", "apps"])
    p.add_argument("--n-train", type=int, default=256)
    p.add_argument("--n-val", type=int, default=50)
    p.add_argument("--behavior-k", type=int, default=8)
    p.add_argument("--fresh-k", type=int, default=32)
    p.add_argument("--val-k", type=int, default=8)
    p.add_argument("--micro-group", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--drift-steps", type=int, default=100)
    p.add_argument("--proj-dim", type=int, default=4096)
    p.add_argument("--grad-layers", type=int, default=4)
    p.add_argument("--clip-cap", type=float, default=10.0)
    p.add_argument("--topk-frac", type=float, default=0.10)
    p.add_argument("--radius-mode", default="gaussian", choices=["gaussian", "hoeffding"])
    p.add_argument("--cut-frac", type=float, default=0.5, help="hybrid prefix 절단점")
    p.add_argument("--hybrid-prompts", type=int, default=32)
    p.add_argument("--k-cell", type=int, default=8)
    p.add_argument("--downstream-source", default="oracle",
                   help="oracle|g00|g10|g01|g11|random")
    p.add_argument("--downstream-steps", type=int, default=200)
    p.add_argument("--micro-batch", type=int, default=2,
                   help="prompt_gradient 동시 시퀀스 수 (14B는 1 권장)")
    p.add_argument("--budget-rollouts", type=int, default=0,
                   help=">0이면 선택 비용 차감 후 남는 예산으로 학습 스텝 결정 (총연산 통일)")
    p.add_argument("--shard", default=None, help="rollout-behavior 샤딩 'i:n' (예: 0:4)")
    p.add_argument("--seed", type=int, default=0,
                   help="생성 샘플링·LoRA init·tie-break 시드 (프롬프트 분할은 고정)")
    args = p.parse_args()

    # 시드 관통 — 샤드마다 다른 스트림 (같은 시드면 샤드 간 표본 상관 방지)
    import random as _random
    _shard_i = int(args.shard.split(":")[0]) if args.shard else 0
    _base = (args.seed * 1_000_003 + _shard_i * 7919 + 17) & 0x7FFFFFFF
    torch.manual_seed(_base)
    _random.seed(_base)

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)

    if args.stage == "prep":
        prompts = load_prompts(args.dataset, args.n_train, args.n_val)
        (run / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=1))
        print(f"prep: train {len(prompts['train'])} / val {len(prompts['val'])}")
    elif args.stage == "rollout-behavior":
        merged = run / "rollouts_behavior_train.jsonl"
        if args.shard:
            i, n = map(int, args.shard.split(":"))
            out = run / f"rollouts_behavior_train.shard{i}.jsonl"
        else:
            i, n, out = 0, 1, merged
        # 병합본이 있으면 샤드도 스킵 — 다른 run에서 복사해 온 β 재사용(go_boost·
        # real_drift_check)과 병합 후 재시작 양쪽을 커버한다.
        if merged.exists():
            print("rollout-behavior: 병합본 존재 — 스킵 (재사용)")
        elif out.exists():
            print(f"rollout-behavior: {out.name} 이미 존재 — 스킵")
        else:
            beta, tok = load_policy(args.model, None)
            train = json.loads((run / "prompts.json").read_text())["train"]
            per = (len(train) + n - 1) // n
            lo, hi = i * per, min((i + 1) * per, len(train))
            collect_rollouts(beta, tok, train[lo:hi], args.behavior_k,
                             args.max_new_tokens, args.temperature, out,
                             idx_offset=lo)
    elif args.stage == "rollout-fresh":
        # π(adapter) fresh rollout을 샤딩 수집 — analyze는 완성 파일이 있으면 생성 스킵
        merged = run / "rollouts_fresh_train.jsonl"
        if args.shard:
            i, n = map(int, args.shard.split(":"))
            out = run / f"rollouts_fresh_train.shard{i}.jsonl"
        else:
            i, n, out = 0, 1, merged
        pi = tok = None
        # 병합본 존재 시 샤드 스킵 (병합 후 재시작 케이스 — 낡은 π 병합본은
        # run_14b의 adapter 시각 격리가 먼저 치우므로 여기 도달하면 현재 π 것)
        if merged.exists():
            print("rollout-fresh: 병합본 존재 — 스킵")
        elif out.exists():
            print(f"rollout-fresh: {out.name} 이미 존재 — 스킵")
        else:
            pi, tok = load_policy(args.model, Path(args.adapter) if args.adapter else None)
            train = json.loads((run / "prompts.json").read_text())["train"]
            per = (len(train) + n - 1) // n
            lo, hi = i * per, min((i + 1) * per, len(train))
            collect_rollouts(pi, tok, train[lo:hi], args.fresh_k,
                             args.max_new_tokens, args.temperature, out,
                             idx_offset=lo)
        # shard 0이 val fresh도 담당 — train 샤드 스킵 여부와 **무관하게** 검사.
        # (else 안에 있던 시절: 재시작하면 train과 함께 val 수집도 건너뛰어
        #  rollouts_fresh_val.jsonl 영구 누락 → val-grads가 line 37에서 사망)
        val_out = run / "rollouts_fresh_val.jsonl"
        if i == 0 and not val_out.exists():
            if pi is None:
                pi, tok = load_policy(args.model, Path(args.adapter) if args.adapter else None)
            prompts = json.loads((run / "prompts.json").read_text())
            collect_rollouts(pi, tok, prompts["val"], args.val_k,
                             args.max_new_tokens, args.temperature, val_out)
    elif args.stage == "drift":
        # 재개 시 재학습 금지 — LoRA 초기화가 랜덤이라 π가 바뀌면 이전에 계산된
        # 점수들과의 비교 일관성이 깨진다. 다시 학습하려면 adapter 폴더를 지울 것.
        adapter_dir = run / f"drift_{args.drift_steps}"
        if (adapter_dir / "adapter_config.json").exists():
            print(f"drift: {adapter_dir.name} 이미 존재 — 스킵 (재학습하려면 폴더 삭제)")
        else:
            train_drift_lora(args.model, run / "rollouts_behavior_train.jsonl",
                             adapter_dir, steps=args.drift_steps)
    elif args.stage == "score":
        stage_score(args, run)
    elif args.stage == "oracle":
        stage_oracle(args, run)
    elif args.stage == "val-deepen":
        # val fresh를 K만큼 추가 수집(append) 후 val gradient 재계산 — α_v 심화용
        pi, tok = load_policy(args.model, Path(args.adapter) if args.adapter else None)
        prompts = json.loads((run / "prompts.json").read_text())
        extra = run / "rollouts_fresh_val.extra.jsonl"
        collect_rollouts(pi, tok, prompts["val"], args.val_k,
                         args.max_new_tokens, args.temperature, extra)
        with (run / "rollouts_fresh_val.jsonl").open("a") as f:
            for line in extra.open():
                f.write(line)
        extra.unlink()
        for p in ("val_gradient.pt", "val_groups.pt"):
            (run / p).unlink(missing_ok=True)
        print(f"val-deepen: +K={args.val_k} 추가 완료 → val-grads 재계산 필요")
    elif args.stage == "val-grads":
        pi, _ = load_policy(args.model, Path(args.adapter) if args.adapter else None)
        params = grad_params(pi, args.grad_layers)
        spec = ProjectionSpec(dim=args.proj_dim)
        v_sum, groups_v = None, []
        for pi_idx, rows in sorted(read_rollouts(run / "rollouts_fresh_val.jsonl").items()):
            advs = loo_advantages(torch.tensor([r["reward"] for r in rows]))
            weights = [
                torch.full((r["input_ids"].numel() - r["resp_start"],), float(a))
                for r, a in zip(rows, advs)
            ]
            g = prompt_gradient(pi, params, rows, weights, spec, micro_batch=args.micro_batch)
            groups_v.append(g)
            v_sum = g if v_sum is None else v_sum + g
        _atomic_save(torch.stack(groups_v), run / "val_groups.pt")
        _atomic_save(v_sum / len(groups_v), run / "val_gradient.pt")
        print(f"val-grads: {len(groups_v)} prompts")
    elif args.stage == "oracle-grads":
        i, n = map(int, args.shard.split(":"))
        stage_oracle(args, run, shard=(i, n))
    elif args.stage == "score-shard":
        i, n = map(int, args.shard.split(":"))
        stage_score(args, run, shard=(i, n))
    elif args.stage == "merge-grads":
        # 샤드 산출물 병합 → 최종 파일
        for base, is_pt in (("scores_oracle", False), ("scores_splithalf", False),
                            ("oracle_micro_groups", True), ("scores_offpolicy", False)):
            shards = sorted(run.glob(f"{base}.shard*.{'pt' if is_pt else 'json'}"))
            if not shards:
                continue
            if is_pt:
                merged: dict = {}
                for p in shards:
                    merged.update(torch.load(p, weights_only=True))
                _atomic_save(merged, run / f"{base}.pt")
            elif base == "scores_offpolicy":
                merged = {}
                for p in shards:
                    part = json.loads(p.read_text())
                    for est, sc in part.items():
                        merged.setdefault(est, {}).update(sc)
                _atomic_text(run / f"{base}.json", json.dumps(merged, indent=1))
            else:
                merged = {}
                for p in shards:
                    merged.update(json.loads(p.read_text()))
                _atomic_text(run / f"{base}.json", json.dumps(merged, indent=1))
            print(f"merge: {base} ← {len(shards)} shards")
        # 병합된 micro + val 방향에서 oracle 점수·split-half 도출 (비샤드 경로와 동일 수식)
        micro_p = run / "oracle_micro_groups.pt"
        val_p = run / "val_gradient.pt"
        # 산출물 2개(oracle+splithalf) — 둘 다 있어야 스킵 (부분 상태 오인 방지)
        if micro_p.exists() and val_p.exists() and not (
                (run / "scores_oracle.json").exists()
                and (run / "scores_splithalf.json").exists()):
            micro = torch.load(micro_p, weights_only=True)
            val_grad = torch.load(val_p, weights_only=True)
            oracle, halves = {}, {}
            for pi_idx, stack in sorted(micro.items()):
                mu = stack.mean(dim=0)
                oracle[pi_idx] = {"score": cosine(mu, val_grad), "norm": float(mu.norm())}
                h = stack.shape[0] // 2
                halves[pi_idx] = {
                    "a": cosine(stack[:h].mean(dim=0), val_grad),
                    "b": cosine(stack[h:].mean(dim=0), val_grad),
                }
            _atomic_text(run / "scores_splithalf.json", json.dumps(halves, indent=1))
            _atomic_text(run / "scores_oracle.json", json.dumps(oracle, indent=1))
            print(f"merge: oracle 점수 도출 {len(oracle)} prompts")
    elif args.stage == "report":
        stage_report(args, run)
    elif args.stage == "hybrid":
        run_hybrid(args, run, None, None, None, args.cut_frac)
    elif args.stage == "analyze":
        # oracle→score→report→hybrid×3 을 한 프로세스로 — 7B 재로드 제거 (GPU 유휴 최소화)
        pi, tok = load_policy(args.model, Path(args.adapter) if args.adapter else None)
        beta, _ = load_policy(args.model, None)
        # oracle 산출물 3개 전부 있어야 스킵 (부분 상태 오인 방지)
        if not all((run / f).exists() for f in
                   ("scores_oracle.json", "scores_splithalf.json", "oracle_micro_groups.pt")):
            stage_oracle(args, run, pi=pi, tok=tok)
        else:
            print("analyze: oracle 산출물 존재 — 스킵")
        if not (run / "scores_offpolicy.json").exists():
            stage_score(args, run, pi=pi, beta=beta)
        else:
            print("analyze: score 산출물 존재 — 스킵")
        stage_report(args, run)
        for cut in (0.25, 0.5, 0.75):
            run_hybrid(args, run, pi, beta, tok, cut)
    elif args.stage == "downstream":
        from train_downstream import run_downstream

        run_downstream(run, args.model, args.downstream_source, args.downstream_steps,
                       args.behavior_k, args.max_new_tokens, args.temperature,
                       args.topk_frac, budget_rollouts=args.budget_rollouts)


def run_hybrid(args, run: Path, pi, beta, tok, cut_frac: float) -> None:
    from hybrid import make_hybrid_cells, score_cells

    spec = ProjectionSpec(dim=args.proj_dim)
    if pi is None:
        pi, tok = load_policy(args.model, Path(args.adapter) if args.adapter else None)
        beta, _ = load_policy(args.model, None)
    if (run / f"scores_hybrid_{cut_frac}.json").exists():
        print(f"hybrid cut={cut_frac}: 이미 존재 — 스킵")
        return
    if True:
        prompts = json.loads((run / "prompts.json").read_text())["train"]
        behavior = read_rollouts(run / "rollouts_behavior_train.jsonl")
        hy_path = run / f"rollouts_hybrid_{cut_frac}.jsonl"
        if not hy_path.exists():
            # subset: live(β 보상 혼합) 프롬프트에서 seeded random — cut 간 동일 (감사 §7)
            import random as _r
            live = [i for i in behavior
                    if len({r["reward"] > 0.5 for r in behavior[i]}) > 1]
            rng = _r.Random(getattr(args, "seed", 0) * 104_729 + 11)
            pool = live if len(live) >= args.hybrid_prompts else sorted(behavior)
            subset = sorted(rng.sample(pool, min(args.hybrid_prompts, len(pool))))
            make_hybrid_cells(beta, pi, tok, behavior, prompts, cut_frac,
                              args.max_new_tokens, args.temperature,
                              subset, hy_path, k_cell=args.k_cell)
        hy = read_rollouts(hy_path)  # prompt_idx 기준 — cell 분리 다시
        cells: dict[str, dict[int, list[dict]]] = {"bb": {}, "bp": {}, "pb": {}, "pp": {}}
        for idx, rows in hy.items():
            for r in rows:
                cells[r["cell"]].setdefault(idx, []).append(r)
        params = grad_params(pi, args.grad_layers)
        val = torch.load(run / "val_gradient.pt", weights_only=True)
        scores = score_cells(pi, params, cells, val, spec)
        (run / f"scores_hybrid_{cut_frac}.json").write_text(json.dumps(scores, indent=1))
        print(f"hybrid cut={cut_frac} 저장: {sorted(scores)}")


if __name__ == "__main__":
    main()
