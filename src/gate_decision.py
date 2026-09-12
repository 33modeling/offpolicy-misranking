"""Bounded reliability diagnostic (random-fallback gate) on a completed point.

For a fixed selector at a fixed checkpoint, the pilot is a uniform sample of
``pilot_size`` candidate prompts. Every pilot prompt carries two measurements
of the selector's actual ranking score (half A and half B). The gate retains
the selector when the lower confidence bound of the half-score correlation
reaches ``r_min`` and the remaining scoring cost fits the budget; otherwise
the continuation trains on a uniform subset. The rule is a frozen JSON file
written before any test reward is examined.

Signals and the origin of their halves::

    difficulty  -|p - 1/2| on the two halves of the stored behavior responses
                (rollouts_behavior_train.jsonl); the passrate_beta selector's
                score, obtainable during training at no extra generation
    passrate    raw pass-rate halves (diagnostic only, not a selector score)
    fresh       scores_splithalf.json a/b: two independent current-policy
                scorings against the validation direction (fresh_r selector)
    g00..g11    scores_stale_splithalf.json (src/stale_splithalf.py): the
                reuse estimators recomputed on the two response halves

Usage::

    python src/gate_decision.py decide --run POINT --rule RULE.json [--e5 SEED_DIR] [--out FILE]
    python src/gate_decision.py table [--r-min 0.25] [--confidence 0.9]
    python src/gate_decision.py write-rule --out RULE.json [--pilot-size 40] [--r-min 0.25]

``decide`` writes gate_decision.json and .csv (next to --e5 when given) and,
when the E5 seed directory has downstream_results.csv, maps the decision to
the fixed arms' test rewards: the reward of the retained selector, of the
random arm, and of the branch the gate actually chose (forgone reward
= reward of the selector minus reward of the chosen branch). CPU only.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RULE_SCHEMA = "offpolicy-gate-rule/v1"
DECISION_SCHEMA = "offpolicy-gate-decision/v1"
SIGNAL_SELECTOR = {"difficulty": "passrate_beta", "passrate": None, "fresh": "fresh_r",
                   "g00": "g00", "g10": "g10", "g01": "g01", "g11": "g11"}
DEFAULT_SIGNALS = ("difficulty", "fresh", "g11")
# Signals whose measurement is a by-product of a uniformly sampled training
# block: no candidate scoring remains after the pilot.
BYPRODUCT = ("difficulty", "passrate")
PROMPTS_PER_STEP = 4  # four ranks, one prompt each, in the matched trainer


def default_rule(pilot_size: int = 40, r_min: float = 0.25, confidence: float = 0.90,
                 seed: int = 20260913, budget_seconds: float | None = None) -> dict:
    return {"schema": RULE_SCHEMA, "pilot_size": int(pilot_size), "r_min": float(r_min),
            "confidence": float(confidence), "seed": int(seed),
            "scoring_budget_seconds": budget_seconds, "cost_per_prompt_seconds": {},
            "interval": "Fisher z, two-sided at `confidence`; the lower endpoint is the one-sided bound used for retention",
            "threshold_origin": "r_min = f^2 retains a score whose noisy top-k recovers at least the fraction f of the oracle score gain under the Gaussian model (Proposition gain); f = 0.5 gives 0.25",
            "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def validate_rule(rule: dict) -> dict:
    if not isinstance(rule, dict) or rule.get("schema") != RULE_SCHEMA:
        raise ValueError("unsupported gate rule")
    n, r_min, conf = rule.get("pilot_size"), rule.get("r_min"), rule.get("confidence")
    if not isinstance(n, int) or n < 4:
        raise ValueError("pilot_size must be an integer of at least four")
    if not isinstance(r_min, (int, float)) or not 0 < r_min < 1:
        raise ValueError("r_min must lie in (0, 1)")
    if not isinstance(conf, (int, float)) or not 0.5 <= conf < 1:
        raise ValueError("confidence must lie in [0.5, 1)")
    budget = rule.get("scoring_budget_seconds")
    if budget is not None and (not isinstance(budget, (int, float)) or budget < 0):
        raise ValueError("scoring_budget_seconds must be nonnegative or null")
    costs = rule.get("cost_per_prompt_seconds", {})
    if not isinstance(costs, dict) or any(not isinstance(v, (int, float)) or v < 0 for v in costs.values()):
        raise ValueError("cost_per_prompt_seconds must map signals to nonnegative seconds")
    return rule


# ----------------------------------------------------------------- statistics
def z_score(confidence: float) -> float:
    """Two-sided normal quantile for the given coverage."""
    return statistics.NormalDist().inv_cdf(0.5 + confidence / 2)


def fisher_bounds(r: float, n: int, confidence: float) -> tuple[float, float]:
    """Fisher-z confidence interval for a correlation from n independent pairs."""
    if n <= 3 or not math.isfinite(r):
        return float("nan"), float("nan")
    r = max(-0.999999, min(0.999999, r))
    half = z_score(confidence) / math.sqrt(n - 3)
    return math.tanh(math.atanh(r) - half), math.tanh(math.atanh(r) + half)


def required_pilot_size(rho: float, r_min: float, confidence: float) -> float:
    """Pairs needed for the Fisher-z bound to separate a true correlation rho
    from r_min: n = 3 + (z / (atanh rho - atanh r_min))^2. Infinite at rho == r_min."""
    if not (-1 < rho < 1) or not (-1 < r_min < 1):
        raise ValueError("correlations must lie in (-1, 1)")
    gap = abs(math.atanh(rho) - math.atanh(r_min))
    if gap == 0:
        return float("inf")
    return 3 + (z_score(confidence) / gap) ** 2


def pilot_size_table(r_min: float, confidence: float,
                     rhos=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> list[dict]:
    rows = []
    for rho in rhos:
        n = required_pilot_size(rho, r_min, confidence)
        rows.append({"rho": rho, "r_min": r_min, "confidence": confidence,
                     "outcome": "retain" if rho > r_min else ("reject" if rho < r_min else "undecidable"),
                     "pairs": n, "steps": n / PROMPTS_PER_STEP if math.isfinite(n) else float("inf")})
    return rows


def pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 3:
        return float("nan")
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    xa, xb = [v - ma for v in a], [v - mb for v in b]
    denom = math.sqrt(sum(v * v for v in xa) * sum(v * v for v in xb))
    if denom == 0:
        return float("nan")
    return max(-1.0, min(1.0, sum(u * v for u, v in zip(xa, xb)) / denom))


# ------------------------------------------------------------------- signals
def behavior_halves(run: Path) -> dict[int, tuple[float, float]]:
    """Half pass rates from the stored behavior responses: first half of the
    response indices against the second half."""
    rewards: dict[int, dict[int, float]] = defaultdict(dict)
    path = run / "rollouts_behavior_train.jsonl"
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            idx, j, reward = int(row["prompt_idx"]), int(row["rollout_idx"]), float(row["reward"])
            if j in rewards[idx]:
                raise ValueError(f"duplicate behavior response {idx}/{j} in {path}")
            rewards[idx][j] = reward
    if not rewards:
        raise ValueError(f"no behavior responses in {path}")
    sizes = {len(v) for v in rewards.values()}
    if len(sizes) != 1 or min(sizes) < 2:
        raise ValueError("behavior groups must have equal size of at least two responses")
    halves = {}
    for idx, group in rewards.items():
        ordered = [group[j] for j in sorted(group)]
        cut = len(ordered) // 2
        halves[idx] = (statistics.fmean(ordered[:cut]), statistics.fmean(ordered[cut:]))
    return halves


def signal_halves(run: Path, signal: str) -> dict[int, tuple[float, float]]:
    if signal in ("difficulty", "passrate"):
        halves = behavior_halves(run)
        if signal == "passrate":
            return halves
        return {i: (-abs(pa - 0.5), -abs(pb - 0.5)) for i, (pa, pb) in halves.items()}
    if signal == "fresh":
        from score_artifacts import load_complete_score_artifacts
        artifacts = load_complete_score_artifacts(run)
        return {i: (float(h["a"]), float(h["b"])) for i, h in artifacts.splithalf.items()}
    if signal in ("g00", "g10", "g01", "g11"):
        path = run / "scores_stale_splithalf.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing: run src/stale_splithalf.py on this point first")
        stale = json.loads(path.read_text())
        if signal not in stale:
            raise ValueError(f"{path} has no {signal} halves")
        return {int(i): (float(h["a"]), float(h["b"])) for i, h in stale[signal].items()}
    raise ValueError(f"unknown signal: {signal}")


def topk_overlap(halves: dict[int, tuple[float, float]], frac: float = 0.1) -> dict:
    """Set-level repeatability of a nonlinear score: fraction of the top-k of
    half A that is also in the top-k of half B over the whole pool, with the
    chance level k/n (ties broken by index)."""
    ids = sorted(halves)
    n = len(ids)
    k = max(1, int(n * frac))
    top_a = set(sorted(ids, key=lambda i: (-halves[i][0], i))[:k])
    top_b = set(sorted(ids, key=lambda i: (-halves[i][1], i))[:k])
    return {"k": k, "n": n, "overlap": len(top_a & top_b) / k, "chance": k / n}


def pilot_indices(ids, n: int, seed: int) -> list[int]:
    ids = sorted(ids)
    if n >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, n))


# ------------------------------------------------------------------- decision
def decide_signal(halves: dict[int, tuple[float, float]], signal: str, rule: dict,
                  pool_size: int | None = None) -> dict:
    rule = validate_rule(rule)
    pool_size = pool_size or len(halves)
    pilot = pilot_indices(halves, rule["pilot_size"], rule["seed"])
    a = [halves[i][0] for i in pilot]
    b = [halves[i][1] for i in pilot]
    n = len(pilot)
    r = pearson(a, b)
    valid = n > 3 and math.isfinite(r)
    lower, upper = fisher_bounds(r, n, rule["confidence"]) if valid else (float("nan"), float("nan"))
    cost_per_prompt = rule.get("cost_per_prompt_seconds", {}).get(signal)
    if signal in BYPRODUCT:
        pilot_cost, remaining = 0.0, 0.0
    elif cost_per_prompt is None:
        pilot_cost, remaining = None, None
    else:
        pilot_cost, remaining = cost_per_prompt * n, cost_per_prompt * max(0, pool_size - n)
    budget = rule.get("scoring_budget_seconds")
    r_min = rule["r_min"]
    if not valid:
        decision, reason = "random", "invalid"  # zero variance or too few pilot pairs
    elif budget is not None and remaining is None:
        decision, reason = "random", "unknown_cost"
    elif budget is not None and remaining > budget:
        decision, reason = "random", "over_budget"
    elif lower >= r_min:
        decision, reason = "retain", "reliable"
    elif upper < r_min:
        decision, reason = "random", "weak"
    else:
        decision, reason = "random", "unresolved"
    record = {"signal": signal, "selector": SIGNAL_SELECTOR[signal], "pilot_pairs": n,
              "pool_size": pool_size, "r_half": r if valid else None, "lower": lower if valid else None,
              "upper": upper if valid else None, "valid": valid, "r_min": r_min,
              "confidence": rule["confidence"], "decision": decision, "reason": reason,
              "required_pairs_at_r": (required_pilot_size(r, r_min, rule["confidence"]) if valid and abs(r) < 1 else None),
              "pilot_cost_seconds": pilot_cost, "remaining_scoring_seconds": remaining,
              "budget_seconds": budget,
              "decision_steps": n / PROMPTS_PER_STEP if signal in BYPRODUCT else None,
              "pilot_indices_sha": None}
    record["pilot_indices_sha"] = _sha_of(pilot)
    # descriptive set-level repeatability over the whole pool (not part of the rule)
    overlap = topk_overlap(halves)
    record.update(pool_topk_overlap=overlap["overlap"], pool_topk_chance=overlap["chance"], pool_topk_k=overlap["k"])
    return record


def _sha_of(values) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(list(values)).encode()).hexdigest()[:16]


def e5_rewards(seed_dir: Path) -> dict[str, dict]:
    path = seed_dir / "downstream_results.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["selector"]: row for row in csv.DictReader(handle)}


def _float(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def map_to_e5(record: dict, rewards: dict[str, dict]) -> dict:
    """Attach the fixed arms' test rewards to a decision: the selector's own
    arm, the random arm, and the branch the gate chose."""
    out = {"reward_selector": None, "reward_random": None, "reward_decision": None,
           "forgone_reward": None, "selector_vs_random": None,
           "selector_vs_random_lower": None, "selector_vs_random_upper": None}
    selector = record["selector"]
    if not rewards or selector is None:
        return out
    random_row, selector_row = rewards.get("random"), rewards.get(selector)
    if random_row:
        out["reward_random"] = _float(random_row["reward_after"])
    if selector_row:
        out["reward_selector"] = _float(selector_row["reward_after"])
        out["selector_vs_random"] = _float(selector_row.get("difference_vs_random"))
        out["selector_vs_random_lower"] = _float(selector_row.get("random_lower"))
        out["selector_vs_random_upper"] = _float(selector_row.get("random_upper"))
    chosen = out["reward_selector"] if record["decision"] == "retain" else out["reward_random"]
    out["reward_decision"] = chosen
    if out["reward_selector"] is not None and chosen is not None:
        out["forgone_reward"] = out["reward_selector"] - chosen
    return out


def decide(run: Path, rule: dict, signals=DEFAULT_SIGNALS, e5: Path | None = None) -> dict:
    rule = validate_rule(rule)
    pool = len(json.loads((run / "prompts.json").read_text())["train"])
    rewards = e5_rewards(e5) if e5 else {}
    rows, skipped = [], []
    for signal in signals:
        try:
            halves = signal_halves(run, signal)
        except FileNotFoundError as exc:
            skipped.append({"signal": signal, "reason": str(exc)})
            continue
        record = decide_signal(halves, signal, rule, pool_size=pool)
        record.update(map_to_e5(record, rewards))
        rows.append(record)
    return {"schema": DECISION_SCHEMA, "run": str(run), "e5": str(e5) if e5 else None,
            "rule": rule, "rewards_available": bool(rewards), "rows": rows, "skipped": skipped,
            "scope": "offline application of the frozen rule to stored half scores; decision "
                     "time for by-product signals counts pilot prompts at four per GRPO step; "
                     "rewards are those of the fixed arms, so this maps the decision, it does not "
                     "train a gated continuation"}


def write_outputs(report: dict, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    rows = report["rows"]
    if rows:
        with target.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return target


def render(report: dict) -> str:
    lines = [f"gate decision: run={report['run']}",
             f"  rule: pilot={report['rule']['pilot_size']} r_min={report['rule']['r_min']} "
             f"confidence={report['rule']['confidence']} seed={report['rule']['seed']}"]
    if not report["rewards_available"]:
        lines.append("  (no downstream_results.csv: decisions only, no reward mapping)")
    f = lambda v, w=6: ("-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:+.3f}").rjust(w)  # noqa: E731
    lines.append("  signal      n    r_half   lower   upper  decision   reason      need_n  steps   J_sel  J_rand   J_dec  forgone  pool top-k overlap (chance)")
    for r in report["rows"]:
        need = r["required_pairs_at_r"]
        need_s = "-" if need is None else ("inf" if not math.isfinite(need) else f"{need:.0f}")
        steps = "-" if r["decision_steps"] is None else f"{r['decision_steps']:.0f}"
        lines.append(f"  {r['signal']:10s} {r['pilot_pairs']:4d} {f(r['r_half'], 8)} {f(r['lower'], 7)} {f(r['upper'], 7)}  "
                     f"{r['decision']:9s} {r['reason']:11s} {need_s:>6s} {steps:>5s} "
                     f"{f(r['reward_selector'], 7)} {f(r['reward_random'], 7)} {f(r['reward_decision'], 7)} {f(r['forgone_reward'], 8)}  "
                     f"{r['pool_topk_overlap']:.3f} ({r['pool_topk_chance']:.3f})")
    for s in report["skipped"]:
        lines.append(f"  {s['signal']:10s} skipped: {s['reason']}")
    return "\n".join(lines)


# ------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("decide")
    p.add_argument("--run", type=Path, required=True, help="completed matrix point")
    p.add_argument("--rule", type=Path, required=True, help="frozen rule JSON")
    p.add_argument("--e5", type=Path, help="E5 seed directory with downstream_results.csv")
    p.add_argument("--out", type=Path, help="output JSON (default: <e5>/gate_decision.json or <run>/gate_decision.json)")
    p.add_argument("--signals", nargs="+", default=list(DEFAULT_SIGNALS), choices=sorted(SIGNAL_SELECTOR))
    p = sub.add_parser("table")
    p.add_argument("--r-min", type=float, default=0.25)
    p.add_argument("--confidence", type=float, default=0.90)
    p = sub.add_parser("write-rule")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pilot-size", type=int, default=40)
    p.add_argument("--r-min", type=float, default=0.25)
    p.add_argument("--confidence", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--budget-seconds", type=float)
    args = parser.parse_args(argv)
    try:
        if args.command == "table":
            print(f"pairs needed for the Fisher-z bound to separate rho from r_min={args.r_min} "
                  f"(two-sided coverage {args.confidence}; steps at {PROMPTS_PER_STEP} prompts per step)")
            for row in pilot_size_table(args.r_min, args.confidence):
                n = row["pairs"]
                print(f"  rho={row['rho']:.2f} {row['outcome']:11s} pairs={'inf' if not math.isfinite(n) else f'{n:7.0f}'} "
                      f"steps={'inf' if not math.isfinite(n) else f'{row['steps']:6.0f}'}")
            return 0
        if args.command == "write-rule":
            if args.out.exists():
                print(f"[rule] exists, left unchanged: {args.out}")
                return 0
            rule = default_rule(args.pilot_size, args.r_min, args.confidence, args.seed, args.budget_seconds)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(validate_rule(rule), indent=1) + "\n")
            print(f"[rule] frozen at {args.out}")
            return 0
        rule = validate_rule(json.loads(args.rule.read_text()))
        report = decide(args.run.resolve(), rule, args.signals, args.e5.resolve() if args.e5 else None)
        target = args.out or ((args.e5 or args.run) / "gate_decision.json")
        write_outputs(report, target)
        print(render(report))
        print(f"[gate] written: {target} (and .csv)")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
