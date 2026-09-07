"""Rewrite execution-only runtime fields of an unfinished point's run_config.json.

`gen_batch` (generation batch) and `gradient_micro_batch` are memory/throughput
knobs: they are not part of the registered matrix contract
(`regime_contract.RUN_CONFIG_FIELDS`), the rollout partial manifest does not
record them, and the pipeline already lowers the generation batch mid-stage on
OOM recovery. The pinned `run_point.sh` nevertheless refuses to re-enter a point
whose recorded values differ from the launch environment (`[config-abort]`), so a
supervisor that wants a different batch for an unfinished point has to update the
record first. This tool does exactly that, and nothing else:

- only points without a non-empty `DONE` are touched, unless `--include-done`
  (run_matrix.sh passes it for the one finished point it is about to re-enter
  because run_complete rejected it, 2026-09-08);
- only `gen_batch` / `gradient_micro_batch` are changed;
- the digest is recomputed with the same serialization `run_point.sh` uses
  (`json.dumps(config_without_digest, sort_keys=True, separators=(",", ":"))`);
- `manifest.json` (a copy of the config plus software versions) is kept in sync;
- every change is printed as `[repair] <run>: <field> <old> -> <new>`.

Usage:
    python src/repair_run_config.py --family-root DIR --gen-batch 32 --gradient-micro-batch 4 --apply
    python src/repair_run_config.py --run RUN_DIR --gen-batch 16              # dry run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

RUNTIME_FIELDS = ("gen_batch", "gradient_micro_batch")


def config_digest(config: dict) -> str:
    """Digest exactly as scripts/run_point.sh computes it (digest key excluded)."""
    body = {key: value for key, value in config.items() if key != "digest"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def planned_changes(config: dict, wanted: dict[str, int | None]) -> dict[str, tuple[object, object]]:
    """Return {field: (old, new)} for fields whose recorded value differs.

    `gen_batch` is stored as the string run_point reads from OM_GEN_BATCH;
    `gradient_micro_batch` as an int (env_int). Types are preserved so the digest
    run_point recomputes at re-entry matches the rewritten record.
    """
    changes: dict[str, tuple[object, object]] = {}
    gen_batch = wanted.get("gen_batch")
    if gen_batch is not None:
        new = str(int(gen_batch))
        old = config.get("gen_batch")
        if old is None or str(old) != new:
            changes["gen_batch"] = (old, new)
    micro = wanted.get("gradient_micro_batch")
    if micro is not None:
        new_int = int(micro)
        old = config.get("gradient_micro_batch")
        try:
            same = int(old) == new_int
        except (TypeError, ValueError):
            same = False
        if not same:
            changes["gradient_micro_batch"] = (old, new_int)
    return changes


def repair_run(
    run: Path, wanted: dict[str, int | None], *, apply: bool, include_done: bool = False
) -> list[str]:
    """Repair one point directory. Returns the human-readable change lines."""
    config_path = run / "run_config.json"
    if not config_path.is_file():
        return []
    done = run / "DONE"
    if not include_done and done.is_file() and done.stat().st_size > 0:
        return []
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"[repair-skip] {run}: unreadable run_config.json ({exc})"]
    if not isinstance(config, dict):
        return [f"[repair-skip] {run}: run_config.json is not an object"]
    changes = planned_changes(config, wanted)
    if not changes:
        return []
    lines = []
    for field, (old, new) in changes.items():
        config[field] = new
        lines.append(f"[repair] {run.name}: {field} {old!r} -> {new!r}")
    config["digest"] = config_digest(config)
    if apply:
        _atomic_write(config_path, json.dumps(config, indent=1))
        manifest_path = run / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = None
            if isinstance(manifest, dict):
                for field, (_, new) in changes.items():
                    manifest[field] = new
                if "digest" in manifest:
                    manifest["digest"] = config["digest"]
                _atomic_write(manifest_path, json.dumps(manifest, indent=1))
    else:
        lines = [line + " (dry run)" for line in lines]
    return lines


def iter_runs(family_root: Path):
    for entry in sorted(family_root.iterdir()):
        if entry.is_dir() and (entry / "run_config.json").is_file():
            yield entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", type=Path, help="one point directory")
    target.add_argument("--family-root", type=Path, help="every point directory below this family")
    parser.add_argument("--gen-batch", type=int, default=None)
    parser.add_argument("--gradient-micro-batch", type=int, default=None)
    parser.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    parser.add_argument(
        "--include-done",
        action="store_true",
        help="also rewrite a finished point (its DONE marker is non-empty); used right before re-entry",
    )
    args = parser.parse_args(argv)
    if args.gen_batch is None and args.gradient_micro_batch is None:
        parser.error("nothing to repair: pass --gen-batch and/or --gradient-micro-batch")
    for name, value in (("--gen-batch", args.gen_batch), ("--gradient-micro-batch", args.gradient_micro_batch)):
        if value is not None and value < 1:
            parser.error(f"{name} must be >= 1")
    wanted = {"gen_batch": args.gen_batch, "gradient_micro_batch": args.gradient_micro_batch}
    runs = [args.run] if args.run else list(iter_runs(args.family_root)) if args.family_root.is_dir() else []
    lines: list[str] = []
    for run in runs:
        lines.extend(repair_run(run, wanted, apply=args.apply, include_done=args.include_done))
    for line in lines:
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
