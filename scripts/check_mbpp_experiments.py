#!/usr/bin/env python3
"""Read-only MBPP switch-suite preflight. No model loading or GPU allocation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _status_summary import MBPP_SUITE_LABELS, gate_label

import evidence_downstream as ed
from selection_switch_gpu import mbpp_items, resolve_sources


def check(args):
    provenance = ed.read(args.manifest)
    if (provenance.get("source_repository") != "google-research-datasets/mbpp"
            or not provenance.get("source_revision")
            or provenance.get("sha256") != ed.digest(args.pool)):
        raise ValueError("MBPP pool manifest/revision/hash mismatch; check fetch_datasets.sh mbpp")
    pool = mbpp_items([json.loads(line) for line in args.pool.read_text().splitlines() if line.strip()])
    pool_answers = dict(zip(ed.questions(pool), (row["answer"] for row in pool)))
    pool_keys = set(pool_answers)
    if not all(answer.lstrip().startswith("assert") for answer in pool_answers.values()):
        raise ValueError("MBPP pool must use assertion-execution rewards")
    used = set()
    runs = resolve_sources(args.matrix, range(5), 0, "mbpp")
    for seed, run in enumerate(runs):
        config = ed.read(run / "run_config.json")
        if not (run / "DONE").is_file():
            raise ValueError(f"MBPP source not complete: {run}")
        if config.get("dataset") != "mbpp" or config.get("seed") != seed or config.get("drift") != 0:
            raise ValueError(f"MBPP source identity mismatch: {run}")
        source = ed.read(run / "prompts.json")
        train, val = ed.questions(source["train"]), ed.questions(source["val"])
        if set(train) & set(val) or not (set(train) | set(val)) <= pool_keys:
            raise ValueError(f"source train/validation overlaps or differs from the MBPP pool: {run}")
        if any(row["answer"] != pool_answers[key]
               for key, row in zip(train + val, source["train"] + source["val"])):
            raise ValueError(f"source assertion rewards differ from the MBPP pool: {run}")
        used.update(train)
        used.update(val)
        print(f"[source] seed={seed} candidates={len(train)} ranking-validation={len(val)} {run}")
    eligible = pool_keys - used
    if len(eligible) < 4:
        raise ValueError(f"only {len(eligible)} MBPP questions remain after all five source pools; at least four required")
    print(f"[evaluation] {len(eligible)} disjoint questions available; new roots use min(300, available)")
    print("[evaluation] internal held-out split of the full MBPP pool, NOT the official MBPP test protocol")

    reference = resolve_sources(args.matrix, [0], 100, "mbpp")[0] / "policy_step_100/grpo_stats.jsonl"
    timings = [float(json.loads(line)["step_seconds"]) for line in reference.read_text().splitlines() if line.strip()]
    if not timings or not all(math.isfinite(value) and value > 0 for value in timings):
        raise ValueError(f"invalid MBPP seed-0 d100 update timings: {reference}")

    specs = {"fresh": (args.fresh_root, "fresh_r", "budget", "final"),
             "quality": (args.quality_root, "fresh_r", "matched", "convergence"),
             "difficulty": (args.difficulty_root, "difficulty", "budget", "convergence"),
             "long": (getattr(args, "long_root", None) or args.fresh_root.with_name("selection-switch-mbpp-long-v1"),
                      "fresh_r", "budget", "final")}
    selected = ["quality"] if args.suite == "all" else [args.suite]
    needs_prefixes = any(name != "fresh" for name in selected)
    # Variant-only launches still depend on the fresh root's certified prefixes.
    inspect = list(dict.fromkeys(["fresh", *selected]))
    fresh_evaluation = None
    for name in inspect:
        root, selector, accounting, gate = specs[name]
        label = MBPP_SUITE_LABELS[name]
        print(f"[experiment] {label}; gate={gate_label(gate)}; root={root}")
        path = root / "switch.json"
        if not path.exists():
            if name == "fresh" and needs_prefixes:
                raise ValueError("On-policy · 선택비용 포함: shared MBPP prefixes/evaluation are not prepared")
            continue
        p = ed.read(path)
        expected = {"dataset": "mbpp", "selector": selector, "accounting": accounting, "gate": gate}
        actual = {key: p.get(key, "budget" if key == "accounting" else "final" if key == "gate" else None)
                  for key in expected}
        if actual != expected:
            raise ValueError(f"existing {label} root has a different protocol: {actual}; use a new root")
        if name == "long" and (isinstance(p.get("budget_gpu_seconds"), bool) or p.get("budget_gpu_seconds") != 87120):
            raise ValueError(f"existing {label} has a different budget; expected MATH long cap 87120 GPU-seconds; saved settings were not changed")
        evaluation = p["evaluation"]
        if (evaluation.get("provenance", {}).get("dataset") != provenance["source_repository"]
                or evaluation["provenance"].get("revision") != provenance["source_revision"]
                or len(evaluation.get("test", [])) < 4):
            raise ValueError(f"{label} evaluation is not a held-out set from this MBPP revision")
        if any(key not in eligible or row["answer"] != pool_answers[key]
               for key, row in zip(ed.questions(evaluation["test"]), evaluation["test"])):
            raise ValueError(f"{label} evaluation differs from the disjoint MBPP pool")
        for run in runs:
            ed.independent_test(ed.read(run / "prompts.json"), evaluation)
        if name == "fresh":
            fresh_evaluation = p["evaluation"]
        elif (p.get("prefix_source", {}).get("root") != str(args.fresh_root.resolve())
              or p["evaluation"] != fresh_evaluation):
            raise ValueError(f"{label} must reuse the MBPP on-policy prefixes and identical evaluation")
    if needs_prefixes:
        for seed in range(5):
            for step in (25, 50, 100):
                path = args.fresh_root / "prefixes" / f"seed-{seed}" / f"prefix-{step}.json"
                if not path.is_file():
                    raise ValueError(f"shared on-policy prefix not ready: {path}; requires On-policy · 선택비용 포함")
    print(f"[check] {len(selected)} condition(s), {48 * len(selected)} continuation branches; existing frozen allocations are unchanged")
    print("[check] inputs ready; full artifact/code/GRPO contracts are checked by the original launcher before GPU work")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("matrix", "pool", "manifest", "fresh-root", "quality-root", "difficulty-root"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--long-root", type=Path)
    parser.add_argument("--suite", choices=("all", "fresh", "quality", "difficulty", "long"), default="all")
    args = parser.parse_args()
    try:
        check(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"[mbpp preflight failed] {exc}\n")


if __name__ == "__main__":
    main()
