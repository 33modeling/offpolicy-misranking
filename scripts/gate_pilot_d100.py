"""Executed correlation-gate pilot at MATH d=100 (added 2026-09-24).

Appendix E / Table 31 ran the pilot (gate_passrate: ten uniform updates, the
frozen Fisher rule, then 90 updates on cached SR or random) at d=0 and d=400.
This adds d=100 for seeds 0-2 in its own root. Only the gate arm is trained
here; it is compared with the existing, unchanged E5 d=100 uniform and cached-SR
arms (read only), exactly the controls Table 31 used at d=0/400. The pairing is
valid only when both roots froze the same source point, subsets and evaluation
questions; check() refuses otherwise.

    python scripts/gate_pilot_d100.py check|status|results --root NEW_ROOT
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evidence_downstream as ed  # noqa: E402

DRIFT = 100
SEEDS = (0, 1, 2)
ARM = "gate_passrate"
CONTROLS = ("random", "passrate_beta")
DEFAULT_OUT = Path.home() / "gate-pilot-d100-results.txt"


def points(root: Path, work: Path):
    for seed in SEEDS:
        yield seed, root / f"math500-d{DRIFT}" / f"s{seed}", work / f"runs/e5-reduced/math500-d{DRIFT}/s{seed}"


def done(out: Path, arm: str) -> bool:
    return all((out / arm / "evaluation" / f"shard-{s}.done.json").is_file() for s in range(4))


def same_inputs(new: Path, e5: Path) -> None:
    """Refuse a cross-root comparison unless the frozen inputs are identical."""
    a, b = ed.read(new / "experiment.json"), ed.read(e5 / "experiment.json")
    for key in ("source_run", "source_hashes", "evaluation_payload_sha256", "eval_seed", "eval_k",
                "eval_prompts", "steps", "seed", "drift"):
        if a.get(key) != b.get(key):
            raise ValueError(f"{new} and {e5} differ in {key}; the E5 controls are not a paired comparison")
    mine, theirs = ed.read(new / "subsets_hashes.json"), ed.read(e5 / "subsets_hashes.json")
    for arm in CONTROLS:
        if mine.get(arm) != theirs.get(arm):
            raise ValueError(f"{arm} subset differs between {new} and {e5}")


def check(root: Path, work: Path) -> list[str]:
    notes = []
    for seed, new, e5 in points(root, work):
        if not (e5 / "experiment.json").is_file():
            raise ValueError(f"E5 d={DRIFT} seed {seed} control root missing: {e5}")
        missing = [arm for arm in CONTROLS if not done(e5, arm)]
        if missing:
            raise ValueError(f"E5 d={DRIFT} seed {seed} controls not fully evaluated: {missing}")
        if (new / "experiment.json").is_file():
            same_inputs(new, e5)
            notes.append(f"s{seed}: inputs identical to E5 d={DRIFT}; controls complete")
        else:
            notes.append(f"s{seed}: not prepared yet; E5 controls complete")
    return notes


def status(root: Path, work: Path) -> str:
    lines = [f"GATE PILOT d={DRIFT} ({root})"]
    for seed, new, _ in points(root, work):
        if not (new / "experiment.json").is_file():
            lines.append(f"  s{seed}: not prepared")
            continue
        decision = new / ARM / "decision.json"
        decided = ed.read(decision)["decision"] if decision.is_file() else "undecided"
        lines.append(f"  s{seed}: before={ed.arm_state(new, 'before')} | {ARM}={ed.arm_state(new, ARM)} | decision={decided}")
    return "\n".join(lines)


def results(root: Path, work: Path) -> tuple[list[dict], list[str]]:
    rows, pending = [], []
    for seed, new, e5 in points(root, work):
        if not (new / "experiment.json").is_file() or not done(new, ARM):
            pending.append(f"s{seed}")
            continue
        same_inputs(new, e5)
        d = ed.read(new / ARM / "decision.json")
        gate = ed.evaluation_means(new, ARM)
        uniform, cached = (ed.evaluation_means(e5, arm) for arm in CONTROLS)
        vs_random = gate - uniform
        forgone = cached - gate
        rlo, rhi = ed.paired_interval(vs_random, seed + 7)      # same seeds as summarize()
        flo, fhi = ed.paired_interval(forgone, seed + 11)
        rows.append({"drift": DRIFT, "seed": seed, "r_half": d["r_half"], "decision": d["decision"],
                     "chosen_subset": d["chosen_subset"], "gate_reward": 100 * float(gate.mean()),
                     "vs_random": 100 * float(vs_random.mean()), "vs_random_ci": (100 * rlo, 100 * rhi),
                     # Table 31 reports gate minus cached, i.e. the negated forgone reward.
                     "vs_cached": -100 * float(forgone.mean()), "vs_cached_ci": (-100 * fhi, -100 * flo)})
    return rows, pending


def text(rows, pending) -> str:
    out = [f"GATE PILOT d={DRIFT} — MATH-300 reward differences (pp), paired 95% question bootstrap",
           "controls: unchanged E5 d=100 uniform and cached-SR arms (same source, subsets and questions)",
           "d\tseed\tr_half\tdecision\tvs_random\tvs_random_95\tvs_cached\tvs_cached_95"]
    for r in rows:
        out.append(f"{r['drift']}\t{r['seed']}\t{r['r_half']:.2f}\t{r['chosen_subset']}\t{r['vs_random']:+.2f}\t"
                   f"[{r['vs_random_ci'][0]:+.2f}, {r['vs_random_ci'][1]:+.2f}]\t{r['vs_cached']:+.2f}\t"
                   f"[{r['vs_cached_ci'][0]:+.2f}, {r['vs_cached_ci'][1]:+.2f}]")
    out.append("pending: " + (", ".join(pending) if pending else "none"))
    return "\n".join(out) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("check", "status", "results"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--work", type=Path, default=Path(os.environ.get("OM_WORK", "/group-volume/minsoo3.kim/offpolicy-misranking")))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    root = args.root.resolve()
    if args.command == "check":
        print("\n".join(check(root, args.work)))
    elif args.command == "status":
        print(status(root, args.work))
    else:
        body = text(*results(root, args.work))
        args.out.write_text(body)
        print(body, end="")
        print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"[gate-pilot-d100-error] {exc}", file=sys.stderr)
        raise SystemExit(1)
