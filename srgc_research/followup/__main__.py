"""Explicit N12/N13 offline analysis, checkpoint audit and torchrun measurements."""

import argparse
from collections import defaultdict
from pathlib import Path

from . import n12, n13
from .common import csv_file, fingerprint, new_output, read_bound, write


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="CPU-only .pt inventory; no base model or GPU is loaded")
    audit.add_argument("roots", nargs="+", type=Path)
    for name in ("audit", "n12-report", "n12-panel", "n12-panels", "collect"):
        sub = audit if name == "audit" else commands.add_parser(name)
        sub.add_argument("--bundle", type=Path, required=True, help="original complete seed input/cache JSON")
        if name in {"audit", "collect"}:
            sub.add_argument("--plan", type=Path, required=True, help="original source plan")
        if name != "n12-report":
            sub.add_argument("--seed", type=int, required=True)
        if name == "n12-report":
            sub.add_argument("records", nargs="+", type=Path)
            sub.add_argument("--model")
            sub.add_argument("--dataset", choices=("math", "mbpp"))
        if name == "n12-panels":
            sub.add_argument("records", nargs="+", type=Path)
        if name == "collect":
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--experiment", choices=("n12", "n13"), required=True)
            sub.add_argument("--output-root", type=Path, required=True, help="separate diagnostic root; resume by rerunning the same command")
            sub.add_argument("--a-only", action="store_true", help="N13: measure scoring A only (default also independent B)")
        else:
            sub.add_argument("--output-dir", type=Path, required=True, help="new directory, never overwrite existing reports")
    sub = commands.add_parser("n13-report", help="compare raw vector acquisitions; scalar-only inputs stay unmeasured")
    sub.add_argument("records", nargs="+", type=Path)
    sub.add_argument("--output-dir", type=Path, required=True)
    sub.add_argument("--reward-records", nargs="+", type=Path, default=[], help="optional original results/curve exports; held-out associations only")
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            if args.experiment == "n12" and args.a_only:
                raise ValueError("--a-only is an N13 option")
            from .measure import collect
            collect(args)
            return
        sources = [*getattr(args, "records", []), *getattr(args, "roots", []), *getattr(args, "reward_records", [])]
        sources += [p for p in (getattr(args, "bundle", None), getattr(args, "plan", None)) if p is not None]
        if args.output_dir.exists():
            raise ValueError("output directory already exists")
        if args.command == "audit":
            from .checkpoints import inventory
            report = inventory(args.roots, args.bundle, args.plan, args.seed)
        elif args.command == "n12-report":
            report = n12.report(args.bundle, args.records, model=args.model, dataset=args.dataset)
        elif args.command == "n12-panel":
            document, sha = read_bound(args.bundle)
            report = n12.panel(document, sha, args.seed)
        elif args.command == "n12-panels":
            report = n12.panel_report(args.bundle, args.records, args.seed)
        else:
            rows, seen, units, historical = [], set(), set(), {}
            duplicates = 0
            for path in args.records:
                archive, payload = read_bound(path)
                if payload in seen:
                    raise ValueError("duplicate vector/scalar input payload")
                seen.add(payload)
                for document in n13.records(archive):
                    row = n13.compare(document)
                    if row["status"] == "measured":
                        identity = row["identity"]
                        unit = fingerprint({"identity": {k: identity.get(k) for k in
                            ("input_sha256", "plan_sha256", "policy_sha256", "measurement_sha256")},
                            "stage": row["stage"], "seed": row["seed"], "candidate_ids": row["candidate_ids"],
                            "sr_comparison_ids": row["sr_comparison_ids"]})
                        if unit in units:
                            raise ValueError("duplicate policy/checkpoint scoring acquisition; do not count overlapping exports twice")
                        units.add(unit)
                    elif row.get("recorded_historical_d") is not None:
                        identity = row.get("identity", {})
                        if all(identity.get(k) for k in ("input_sha256", "plan_sha256", "implementation_sha256")):
                            unit = fingerprint({"identity": identity, "seed": row["seed"], "arm": row["arm"], "stage": row["stage"]})
                            if unit in historical:
                                if historical[unit] != row["recorded_historical_d"]:
                                    raise ValueError("conflicting overlapping historical SR-GC checks")
                                duplicates += 1
                                continue
                            historical[unit] = row["recorded_historical_d"]
                    row.update(source=str(path), source_sha256=payload)
                    rows.append(row)
            cohorts = defaultdict(list)
            from srgc_research.stage_rollout_analysis import Analysis
            rewards = Analysis()
            for path in args.reward_records:
                document, sha = read_bound(path)
                rewards.document(document, {"path": str(path), "sha256": sha})
            for row in rows:
                row.update(n13.nearby_evaluation(row, rewards))
                identity = row.get("identity", {})
                cohorts[fingerprint({k: identity.get(k) for k in ("input_sha256", "plan_sha256", "measurement_sha256")})].append(row)
            report = {"schema": "srgc-n13-report-v1", "rows": rows, "coverage": n13.aggregate(rows),
                      "duplicate_historical_checks_removed": duplicates,
                      "cohort_coverage": {key: n13.aggregate(values) for key, values in cohorts.items()},
                      "actual_optimizer_utility_measured": False, "changes_live_switch_rule": False}
        output = new_output(args.output_dir, sources)
        write(output / "report.json", report)
        for key in ("rows", "windows_25", "training_rewards", "held_out_rewards", "requested_stage_coverage"):
            if key in report:
                csv_file(output / f"{key}.csv", report[key])
        (output / "README.md").write_text(
            f"# {report['schema']}\n\nSource-bound results: `report.json` and separate CSV files.\n\n"
            "Missing values remain null. Initial-cache strata are finite estimates, not true difficulty.\n"
            "Panel rewards are training-pool diagnostics; training and held-out rewards stay separate.\n"
            "N13 projected contrasts do not measure actual AdamW updates or change the Switch rule.\n")
        print(f"Saved: {output / 'report.json'}")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
