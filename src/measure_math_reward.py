"""How much would the corrected math verifier change the stored math500 rewards?

The registered launcher scores math500 with `OM_MATH_VERIFIER=math_verify`, so a
response that is not an exact or numeric match is decided by
`data._math_reward`, which calls `math_verify.parse(prediction)` on bare text.
Bare text makes the numeric extractor read "2x" as 2 and leaves roots and
symbolic expressions unextracted. Parsing the same string as mathematics
(`parse("$" + expression + "$", extraction_config=[LatexExtractionConfig()])`)
is what the verifier was meant to do.

This measures the difference on rollouts that already exist. It reads nothing but
finished artifacts, needs no GPU and no regeneration, and never writes into a run
directory. The point is to decide, with a number, whether the finished matrix has
to be rescored:

    reward flips 0 -> 1 and 1 -> 0, per rollout file
    prompts whose K rewards are all equal (a group with no gradient) before/after

A self-check runs first: the old verifier is recomputed from the stored response
and compared with the stored `reward` field. If those disagree the replication is
wrong and the measurement is not reported.

    python src/measure_math_reward.py --root <run root> [--limit-rows N]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

PINNED_DEFAULT = "0e4cd412"
ROLLOUT_FILES = (
    "rollouts_behavior_train.jsonl",
    "rollouts_fresh_train.jsonl",
    "rollouts_val.jsonl",
)


def load_pinned_data_module(repo: Path, commit: str):
    """Import the generation commit's data.py, not the working tree's.

    The working tree may already carry a corrected copy; the matrix was scored
    with the pinned one, so the "old" side must come from the pinned one.
    """
    source = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:src/data.py"],
        capture_output=True, text=True,
    )
    if source.returncode != 0:
        raise SystemExit(f"[abort] cannot read {commit}:src/data.py ({source.stderr.strip()})")
    directory = Path(tempfile.mkdtemp(prefix="pinned-data-"))
    path = directory / "data.py"
    path.write_text(source.stdout, encoding="utf-8")
    # Load it under a name of its own. importlib.import_module("data") would hand
    # back a "data" module that something else imported first - in this repo that
    # is the WORKING TREE copy, which may already carry the corrected verifier.
    # The old side would then be silently identical to the new one and the tool
    # would report "no change" for a bug that is really there.
    if str(repo / "src") not in sys.path:
        sys.path.insert(0, str(repo / "src"))      # data.py's own imports
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"pinned_data_{commit}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DEFAULT_TIMEOUT = 5          # math-verify's own default for parse and verify
GENEROUS_TIMEOUT = int(os.environ.get("OM_MATH_VERIFY_TIMEOUT", "60"))


def verifier_pair(timeout_seconds: int = GENEROUS_TIMEOUT):
    """(old, new) math verifiers; both raise if math-verify is missing.

    math-verify gives parse and verify 5 seconds each. The matrix was scored on
    H100 nodes; on a slower CPU node the same comparison can run out of time and
    score 0, so the recomputed "old" value disagrees with the stored one for a
    reason that has nothing to do with the answer. Both verifiers get a generous
    budget here; the caller uses DEFAULT_TIMEOUT to tell a timeout-sensitive row
    from a real mismatch.
    """
    from math_verify import LatexExtractionConfig, parse, verify

    def old(prediction: str, gold: str) -> float:
        try:
            parsed_gold = parse(gold, parsing_timeout=timeout_seconds)
            parsed_prediction = parse(prediction, parsing_timeout=timeout_seconds)
            return 1.0 if parsed_gold and parsed_prediction and verify(
                parsed_gold, parsed_prediction, timeout_seconds=timeout_seconds
            ) else 0.0
        except Exception:
            return 0.0

    def new(prediction: str, gold: str) -> float:
        def as_math(expression: str):
            return parse("$" + expression + "$", extraction_config=[LatexExtractionConfig()],
                         parsing_timeout=timeout_seconds)

        try:
            parsed_gold = as_math(gold)
            parsed_prediction = as_math(prediction)
            return 1.0 if parsed_gold and parsed_prediction and verify(
                parsed_gold, parsed_prediction, timeout_seconds=timeout_seconds
            ) else 0.0
        except Exception:
            return 0.0

    return old, new


def reproduces_stored(text: str, gold: str, data, stored: float, old_verifier) -> str:
    """'exact' when the pinned verifier reproduces the stored reward, 'timeout'
    when only the time budget explains the difference, 'mismatch' otherwise."""
    if abs(score(text, gold, data, old_verifier) - stored) <= 1e-9:
        return "exact"
    old_default, _ = verifier_pair(DEFAULT_TIMEOUT)
    if abs(score(text, gold, data, old_default) - stored) <= 1e-9:
        return "timeout"
    return "mismatch"


def score(text: str, gold: str, data, verifier) -> float:
    """data.reward()'s math path, with the verifier swapped in.

    The steps before the verifier (answer extraction, exact match, numeric match)
    are identical in both variants, so only the verifier can move a reward.
    """
    prediction = data.extract_answer(text)
    if prediction is None:
        return 0.0
    gold = gold.strip().rstrip(".").replace(",", "").replace("$", "")
    if prediction == gold:
        return 1.0
    try:
        if abs(float(prediction) - float(gold)) < 1e-6:
            return 1.0
    except ValueError:
        pass
    return verifier(prediction, gold)


def group_key(row: dict) -> int:
    return int(row.get("prompt_idx", -1))


def measure_run(run: Path, data, old_verifier, new_verifier, limit_rows: int | None) -> dict:
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    prompts = json.loads((run / "prompts.json").read_text(encoding="utf-8"))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model_resolved"], local_files_only=True)
    report = {"run": run.name, "dataset": config.get("dataset"), "files": []}
    for name in ROLLOUT_FILES:
        path = run / name
        if not path.is_file() or path.stat().st_size == 0:
            continue
        split = prompts["val"] if "val" in name else prompts["train"]
        stats = Counter()
        old_by_prompt: dict[int, list[float]] = {}
        new_by_prompt: dict[int, list[float]] = {}
        mismatch_examples: list[str] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if limit_rows and stats["rows"] >= limit_rows:
                    break
                try:
                    row = json.loads(line)
                    ids = row["input_ids"][int(row["resp_start"]):]
                    stored = float(row["reward"])
                    index = group_key(row)
                    gold = split[index]["answer"]
                except (ValueError, KeyError, IndexError, TypeError):
                    stats["unreadable"] += 1
                    continue
                text = tokenizer.decode(ids, skip_special_tokens=True)
                recomputed_old = score(text, gold, data, old_verifier)
                recomputed_new = score(text, gold, data, new_verifier)
                stats["rows"] += 1
                if abs(recomputed_old - stored) > 1e-9:
                    kind = reproduces_stored(text, gold, data, stored, old_verifier)
                    if kind == "timeout":
                        stats["timeout_sensitive"] += 1
                        recomputed_old = stored      # the pinned value is the stored one
                    else:
                        stats["replication_mismatch"] += 1
                        if len(mismatch_examples) < 3:
                            mismatch_examples.append(
                                f"line {line_number}: stored={stored} recomputed_old={recomputed_old} gold={gold!r}"
                            )
                if recomputed_new > recomputed_old:
                    stats["flip_0_to_1"] += 1
                elif recomputed_new < recomputed_old:
                    stats["flip_1_to_0"] += 1
                old_by_prompt.setdefault(index, []).append(recomputed_old)
                new_by_prompt.setdefault(index, []).append(recomputed_new)
        flat_old = sum(1 for values in old_by_prompt.values() if len(set(values)) == 1)
        flat_new = sum(1 for values in new_by_prompt.values() if len(set(values)) == 1)
        report["files"].append(
            {
                "file": name,
                "rows": stats["rows"],
                "unreadable": stats["unreadable"],
                "replication_mismatch": stats["replication_mismatch"],
                "timeout_sensitive": stats["timeout_sensitive"],
                "mismatch_examples": mismatch_examples,
                "flip_0_to_1": stats["flip_0_to_1"],
                "flip_1_to_0": stats["flip_1_to_0"],
                "prompts": len(old_by_prompt),
                "flat_groups_old": flat_old,
                "flat_groups_new": flat_new,
            }
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="experiment root (holds family-* directories)")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pinned", default=os.environ.get("OM_GENERATION_GIT", PINNED_DEFAULT))
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--limit-rows", type=int, default=int(os.environ.get("OM_MATH_CHECK_ROWS", "0")) or None)
    parser.add_argument("--max-points", type=int, default=int(os.environ.get("OM_MATH_CHECK_POINTS", "0")) or None)
    args = parser.parse_args(argv)

    data = load_pinned_data_module(args.repo, args.pinned[:12])
    try:
        old_verifier, new_verifier = verifier_pair()
    except ImportError:
        print("[abort] math-verify is not importable; run scripts/setup_env.sh first")
        return 2

    points = sorted(
        run
        for family in sorted(args.root.glob(f"family-{args.dataset}-s*"))
        for run in sorted(family.iterdir())
        if run.is_dir()
        and (run / "DONE").is_file()
        and (run / "DONE").stat().st_size
        and (run / "run_config.json").is_file()
        and (run / "prompts.json").is_file()
    )
    if args.max_points:
        points = points[: args.max_points]
    if not points:
        print(f"[abort] no finished {args.dataset} point under {args.root}")
        return 1

    print(f"[math-reward-check] pinned={args.pinned[:8]} dataset={args.dataset} finished points={len(points)}")
    totals = Counter()
    for run in points:
        try:
            report = measure_run(run, data, old_verifier, new_verifier, args.limit_rows)
        except Exception as exc:  # one unreadable point must not lose the rest
            print(f"  {run.name}: skipped ({type(exc).__name__}: {exc})")
            continue
        for entry in report["files"]:
            totals["rows"] += entry["rows"]
            totals["flip_0_to_1"] += entry["flip_0_to_1"]
            totals["flip_1_to_0"] += entry["flip_1_to_0"]
            totals["replication_mismatch"] += entry["replication_mismatch"]
            totals["prompts"] += entry["prompts"]
            totals["flat_old"] += entry["flat_groups_old"]
            totals["flat_new"] += entry["flat_groups_new"]
            share = 100.0 * entry["flip_0_to_1"] / entry["rows"] if entry["rows"] else 0.0
            print(
                f"  {report['run']}  {entry['file']}: rows={entry['rows']} "
                f"0->1={entry['flip_0_to_1']} ({share:.2f}%) 1->0={entry['flip_1_to_0']} "
                f"no-gradient prompts {entry['flat_groups_old']}/{entry['prompts']} -> "
                f"{entry['flat_groups_new']}/{entry['prompts']}"
                + (f"  REPLICATION MISMATCH={entry['replication_mismatch']}" if entry["replication_mismatch"] else "")
            )
            for example in entry["mismatch_examples"]:
                print(f"      {example}")

    print("")
    if totals["replication_mismatch"]:
        print(
            f"[verdict] UNRELIABLE: the old verifier could not be reproduced on "
            f"{totals['replication_mismatch']}/{totals['rows']} stored rows. The numbers above are not trustworthy; "
            "the replication of data.reward() has to be corrected before any conclusion."
        )
        return 3
    share = 100.0 * totals["flip_0_to_1"] / totals["rows"] if totals["rows"] else 0.0
    freed = totals["flat_old"] - totals["flat_new"]
    print(
        f"[totals] rows={totals['rows']} 0->1={totals['flip_0_to_1']} ({share:.2f}%) "
        f"1->0={totals['flip_1_to_0']} no-gradient prompts {totals['flat_old']}/{totals['prompts']} -> "
        f"{totals['flat_new']}/{totals['prompts']} ({freed:+d})"
    )
    if totals["flip_0_to_1"] == 0 and totals["flip_1_to_0"] == 0:
        print("[verdict] NO CHANGE: the corrected verifier scores every stored response identically. Nothing to rescore.")
    elif share < 0.5 and freed <= 0.01 * max(totals["prompts"], 1):
        print(
            f"[verdict] SMALL: {share:.2f}% of responses change and {freed} prompt groups gain a gradient. "
            "Finish the matrix as it is and record the verifier limitation."
        )
    else:
        print(
            f"[verdict] LARGE: {share:.2f}% of responses change and {freed} prompt groups gain a gradient. "
            "The stored rollouts have to be rescored with the corrected verifier before the analysis is read. "
            "Rescoring needs no regeneration: the responses are already on disk."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
