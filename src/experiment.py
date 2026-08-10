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
)
from rollout import collect_rollouts, load_policy, train_drift_lora


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
    for pi_idx, rows in sorted(rollouts.items()):
        n_done += 1
        rewards = torch.tensor([r["reward"] for r in rows])
        advs = loo_advantages(rewards)
        logps_pi = [sequence_logprobs(pi, r["input_ids"], r["resp_start"]) for r in rows]
        logps_beta = beta_logps[pi_idx] if two_pass else [
            sequence_logprobs(beta, r["input_ids"], r["resp_start"]) for r in rows
        ]
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
    (run / out_name).write_text(json.dumps(out, indent=1))


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
        if not (run / "val_gradient.pt").exists():
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
            torch.save(val_grad, run / "val_gradient.pt")
            torch.save(torch.stack(groups_v), run / "val_groups.pt")

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
        (run / "scores_oracle.json").write_text(json.dumps(oracle, indent=1))
        (run / "scores_splithalf.json").write_text(json.dumps(halves, indent=1))
        torch.save(micro, run / "oracle_micro_groups.pt")
    else:
        torch.save(micro, run / f"oracle_micro_groups.shard{shard[0]}.pt")


def topk(scores: dict, frac: float) -> set:
    k = max(1, int(len(scores) * frac))
    return set(sorted(scores, key=lambda i: -scores[i]["score"])[:k])


def stage_report(args, run: Path) -> None:
    oracle = {int(k): v for k, v in json.loads((run / "scores_oracle.json").read_text()).items()}
    off = json.loads((run / "scores_offpolicy.json").read_text())
    halves = {int(k): v for k, v in json.loads((run / "scores_splithalf.json").read_text()).items()}

    frac = args.topk_frac
    o_top = topk(oracle, frac)
    k = len(o_top)
    ha = topk({i: {"score": h["a"]} for i, h in halves.items()}, frac)
    hb = topk({i: {"score": h["b"]} for i, h in halves.items()}, frac)
    noise_floor = len(ha & hb) / k

    lines = [f"# report  (top-{int(frac*100)}%, k={k}, noise_floor(split-half precision)={noise_floor:.3f})", ""]
    lines.append("| estimator | top-k precision | Jaccard | Δ vs floor |")
    lines.append("|---|---|---|---|")
    results = {"noise_floor": noise_floor, "k": k}
    for est in ESTIMATORS:
        scores = {int(i): v for i, v in off[est].items()}
        e_top = topk(scores, frac)
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
    (run / "report.md").write_text("\n".join(lines))
    (run / "report.json").write_text(json.dumps(results, indent=1))
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
                   choices=["gsm8k", "math500", "mbpp", "kk"])
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
    p.add_argument("--downstream-source", default="oracle",
                   help="oracle|g00|g10|g01|g11|random")
    p.add_argument("--downstream-steps", type=int, default=200)
    p.add_argument("--micro-batch", type=int, default=2,
                   help="prompt_gradient 동시 시퀀스 수 (14B는 1 권장)")
    p.add_argument("--budget-rollouts", type=int, default=0,
                   help=">0이면 선택 비용 차감 후 남는 예산으로 학습 스텝 결정 (총연산 통일)")
    p.add_argument("--shard", default=None, help="rollout-behavior 샤딩 'i:n' (예: 0:4)")
    args = p.parse_args()

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)

    if args.stage == "prep":
        prompts = load_prompts(args.dataset, args.n_train, args.n_val)
        (run / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=1))
        print(f"prep: train {len(prompts['train'])} / val {len(prompts['val'])}")
    elif args.stage == "rollout-behavior":
        if args.shard:
            i, n = map(int, args.shard.split(":"))
            out = run / f"rollouts_behavior_train.shard{i}.jsonl"
        else:
            i, n, out = 0, 1, run / "rollouts_behavior_train.jsonl"
        if out.exists():
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
        if args.shard:
            i, n = map(int, args.shard.split(":"))
            out = run / f"rollouts_fresh_train.shard{i}.jsonl"
        else:
            i, n, out = 0, 1, run / "rollouts_fresh_train.jsonl"
        pi = tok = None
        if out.exists():
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
        train_drift_lora(args.model, run / "rollouts_behavior_train.jsonl",
                         run / f"drift_{args.drift_steps}", steps=args.drift_steps)
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
        torch.save(v_sum / len(groups_v), run / "val_gradient.pt")
        torch.save(torch.stack(groups_v), run / "val_groups.pt")
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
                torch.save(merged, run / f"{base}.pt")
            elif base == "scores_offpolicy":
                merged = {}
                for p in shards:
                    part = json.loads(p.read_text())
                    for est, sc in part.items():
                        merged.setdefault(est, {}).update(sc)
                (run / f"{base}.json").write_text(json.dumps(merged, indent=1))
            else:
                merged = {}
                for p in shards:
                    merged.update(json.loads(p.read_text()))
                (run / f"{base}.json").write_text(json.dumps(merged, indent=1))
            print(f"merge: {base} ← {len(shards)} shards")
        # 병합된 micro + val 방향에서 oracle 점수·split-half 도출 (비샤드 경로와 동일 수식)
        micro_p = run / "oracle_micro_groups.pt"
        val_p = run / "val_gradient.pt"
        if micro_p.exists() and val_p.exists() and not (run / "scores_oracle.json").exists():
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
            (run / "scores_oracle.json").write_text(json.dumps(oracle, indent=1))
            (run / "scores_splithalf.json").write_text(json.dumps(halves, indent=1))
            print(f"merge: oracle 점수 도출 {len(oracle)} prompts")
    elif args.stage == "report":
        stage_report(args, run)
    elif args.stage == "hybrid":
        run_hybrid(args, run, None, None, None, args.cut_frac)
    elif args.stage == "analyze":
        # oracle→score→report→hybrid×3 을 한 프로세스로 — 7B 재로드 제거 (GPU 유휴 최소화)
        pi, tok = load_policy(args.model, Path(args.adapter) if args.adapter else None)
        beta, _ = load_policy(args.model, None)
        if not (run / "scores_oracle.json").exists():
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
        fresh = read_rollouts(run / "rollouts_fresh_train.jsonl")
        hy_path = run / f"rollouts_hybrid_{cut_frac}.jsonl"
        if not hy_path.exists():
            make_hybrid_cells(beta, pi, tok, behavior, fresh, prompts, cut_frac,
                              args.max_new_tokens, args.temperature,
                              args.hybrid_prompts, hy_path)
        hy = read_rollouts(hy_path)  # prompt_idx 기준 — cell 분리 다시
        cells: dict[str, dict[int, list[dict]]] = {"bp": {}, "pb": {}}
        for idx, rows in hy.items():
            for r in rows:
                cells[r["cell"]].setdefault(idx, []).append(r)
        sub = set(cells["bp"])
        cells["bb"] = {i: behavior[i] for i in sub}
        cells["pp"] = {i: fresh[i] for i in sub}
        params = grad_params(pi, args.grad_layers)
        val = torch.load(run / "val_gradient.pt", weights_only=True)
        scores = score_cells(pi, params, cells, val, spec)
        (run / f"scores_hybrid_{cut_frac}.json").write_text(json.dumps(scores, indent=1))
        print(f"hybrid cut={cut_frac} 저장: {sorted(scores)}")


if __name__ == "__main__":
    main()
