"""RLOO objective control at MATH d=100 in its own root (added 2026-09-24).

The original RLOO control (scripts/run_rloo.sh, root rloo-selector-v2) covers
d=0 and d=400. This adds the missing middle checkpoint for the same three seeds
and the same three selectors (9 continuations + 3 parent evaluations). It never
edits src/: the frozen per-point contract, training, evaluation, queue and
report functions are reused unchanged, only their point list is set to d=100.

    python scripts/rloo_d100.py prepare|check|status|queue|results --root NEW_ROOT
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import rloo_experiment as experiment  # noqa: E402

DRIFT = 100
SEEDS = (0, 1, 2)
POINTS = tuple((DRIFT, seed) for seed in SEEDS)
ORIGINAL_ROOT_NAME = "rloo-selector-v2"
DEFAULT_OUT = Path.home() / "rloo-d100-results.txt"
FIELDS = ("model", "dataset", "max_new_tokens", "temperature", "prompt_format",
          "attn", "gen_batch", "lora_targets", "thinking", "top_p",
          "grpo_gradient_checkpointing")


def use_d100(module) -> None:
    module.POINTS = POINTS


def guard_root(root: Path) -> Path:
    root = root.resolve()
    if root.name == ORIGINAL_ROOT_NAME or (root / "math500-d0").exists() or (root / "math500-d400").exists():
        raise ValueError(f"refusing {root}: the original d0/d400 RLOO root is never reused")
    return root


def outs(root: Path) -> list[Path]:
    return [root / f"math500-d{DRIFT}" / f"s{seed}" for seed in SEEDS]


def prepare(work: Path, root: Path, source_root: Path | None, eval_prompts: Path | None) -> list[Path]:
    """d=100 copy of rloo_experiment.prepare_matrix (whose d0/d400 checks are literal)."""
    use_d100(experiment)
    ed = experiment.ed
    points = outs(root)
    existing = [ed.read(out / "experiment.json") if (out / "experiment.json").exists() else None for out in points]
    evaluations = [eval_prompts or (Path(c["evaluation_input"]) if c else
                   work / f"inputs/e5-reduced/test-math500-d{DRIFT}.json") for c in existing]
    if len({e.resolve() for e in evaluations}) != 1:
        raise ValueError("prepared seeds use different evaluation inputs")
    runs = [Path(c["source"]["source_run"]) if c and source_root is None else
            experiment.source_paths(work, source_root, seed, DRIFT)
            for seed, c in zip(SEEDS, existing, strict=True)]
    experiment.disjoint(root, [*runs, *(e.parent for e in evaluations)])
    configs = [ed.read(run / "run_config.json") for run in runs]
    fields = (*FIELDS, *ed.TRAIN_FLAGS.values())
    if any(any(config.get(key) != configs[0].get(key) for key in fields) for config in configs[1:]):
        raise ValueError("source model/runtime/training configuration differs across seeds")
    for seed, config, run, out, evaluation in zip(SEEDS, configs, runs, points, evaluations, strict=True):
        if config["seed"] != seed or config["drift"] != DRIFT:
            raise ValueError(f"source {run} is not seed {seed} at d={DRIFT}")
        experiment.prepare(run, out, evaluation, dry=True)
    with experiment.lock(root / ".matrix-prepare.lock", blocking=True):
        for run, out, evaluation in zip(runs, points, evaluations, strict=True):
            experiment.prepare(run, out, evaluation)
    print(f"Prepared {len(points) * len(experiment.ARMS)} RLOO training arms at d={DRIFT}; "
          "original d0/d400 root untouched.")
    return points


def matching_grpo_subsets(work: Path, root: Path) -> list[str]:
    """The control reuses the GRPO study's subsets; compare with the E5 d=100 copies when present."""
    notes = []
    for seed, out in zip(SEEDS, outs(root), strict=True):
        grpo = work / f"runs/e5-reduced/math500-d{DRIFT}/s{seed}/subsets"
        for arm in experiment.ARMS:
            mine, theirs = out / "subsets" / f"subset-{arm}.json", grpo / f"subset-{arm}.json"
            if not theirs.is_file():
                notes.append(f"s{seed}/{arm}: GRPO subset not found at {theirs}; not compared")
                continue
            a, b = experiment.ed.read(mine), experiment.ed.read(theirs)
            if a["selected_idx"] != b["selected_idx"] or a["train"] != b["train"]:
                raise ValueError(f"s{seed}/{arm}: rebuilt subset differs from the GRPO d={DRIFT} subset {theirs}")
            notes.append(f"s{seed}/{arm}: identical to GRPO d={DRIFT} subset")
    return notes


def run_experiment_command(command: str, root: Path, work: Path) -> None:
    use_d100(experiment)
    sys.argv = ["rloo_experiment.py", command, "--root", str(root), "--work", str(work)]
    experiment.main()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "prepare", "check", "status", "queue", "results"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--work", type=Path, default=Path(os.environ.get("OM_WORK", "/group-volume/minsoo3.kim/offpolicy-misranking")))
    p.add_argument("--source-root", type=Path)
    p.add_argument("--eval-prompts", type=Path)
    p.add_argument("--max-phase-seconds", type=float, default=86400.0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    root = guard_root(args.root)
    if args.command == "plan":
        print(f"RLOO d={DRIFT}: seeds {list(SEEDS)} x arms {list(experiment.ARMS)} = "
              f"{len(SEEDS) * len(experiment.ARMS)} continuations + {len(SEEDS)} parent evaluations; root {root}")
        return 0
    if args.command == "prepare":
        prepare(args.work, root, args.source_root, args.eval_prompts)
        for note in matching_grpo_subsets(args.work, root):
            print(note)
        return 0
    if args.command in ("check", "status"):
        run_experiment_command(args.command, root, args.work)
        if args.command == "check":
            for note in matching_grpo_subsets(args.work, root):
                print(note)
        return 0
    if args.command == "queue":
        import queue_rloo
        use_d100(queue_rloo.experiment)
        sys.argv = ["queue_rloo.py", "--root", str(root), "--max-phase-seconds", str(args.max_phase_seconds)]
        return queue_rloo.main()
    import rloo_report
    use_d100(rloo_report.experiment)
    sys.argv = ["rloo_report.py", "--root", str(root), "--out", str(args.out)]
    rloo_report.main()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"[rloo-d100-error] {exc}", file=sys.stderr)
        raise SystemExit(1)
