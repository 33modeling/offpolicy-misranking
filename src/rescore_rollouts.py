"""Rescore the stored MATH-500 rollouts of a finished family with the corrected verifier.

Registered in the paper plan (§9.1, 2026-09-08): the matrix finishes under the
pinned verifier, then every point is rescored from its stored responses. This
module does the rescoring without regenerating or retraining anything:

1. every rollout row gets `reward` recomputed with the corrected verifier and
   keeps the pinned value in `reward_pinned` (a second run recomputes from
   `reward_pinned`, so the operation is idempotent);
2. every rollout manifest is resealed with the new `artifact_sha256`; the pinned
   hashes and the verifier identity go to `<prefix>.rescore.json` beside it;
3. the artifacts derived from rewards (gradients, scores, protocols, report,
   DONE, the deep-validation marker) are moved, not deleted, into
   `pinned-scoring/<stamp>/`; the family completion stamp is removed.

A normal worker then re-enters the family: generation and GRPO are skipped
(their artifacts are complete and hash-bound), only gradients, scores and the
report are recomputed. The pinned scoring stays readable under pinned-scoring/.

Before any file is touched the pinned verifier is recomputed from the stored
text and compared with the stored reward on every row; one mismatch aborts the
point, because it would mean the responses or the gold answers were misread.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import measure_math_reward as mmr

VERIFIER_ID = "math_verify-latex-2026-09-08"
ROLLOUT_PREFIXES = ("rollouts_behavior_train", "rollouts_fresh_train", "rollouts_fresh_val")
DERIVED_ARTIFACTS = (
    "val_gradient.pt", "val_groups.pt", "oracle_micro_groups.pt",
    "scores_oracle.json", "scores_offpolicy.json", "scores_splithalf.json",
    "score_protocol.json", "oracle_protocol.json", "divergence_stats.json",
    "report.json", "DONE", ".regime_validated.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rollout_files(run: Path, prefix: str) -> list[tuple[Path, Path | None]]:
    """(jsonl, manifest-or-None) for the merged file and every shard of a prefix."""
    out = []
    for path in sorted(run.glob(f"{prefix}*.jsonl")):
        manifest = path.with_name(path.name[: -len(".jsonl")] + ".manifest.json")
        out.append((path, manifest if manifest.is_file() else None))
    return out


def rewrite_rows(path: Path, split: list[dict], tokenizer, data, old_verifier, new_verifier) -> Counter:
    """Rewrite one rollout file in place. Returns the change statistics."""
    stats = Counter()
    temporary = path.with_name(path.name + f".rescore.{os.getpid()}")
    with path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as sink:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            ids = row["input_ids"][int(row["resp_start"]):]
            gold = split[int(row["prompt_idx"])]["answer"]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            pinned = float(row.get("reward_pinned", row["reward"]))
            kind = mmr.reproduces_stored(text, gold, data, pinned, old_verifier)
            if kind == "mismatch":
                temporary.unlink(missing_ok=True)
                raise ValueError(
                    f"{path.name}:{line_number}: the pinned verifier does not reproduce the stored reward "
                    f"({pinned}); responses or gold answers are misread, nothing was changed"
                )
            stats["timeout_sensitive"] += kind == "timeout"
            corrected = mmr.score(text, gold, data, new_verifier)
            if "reward_pinned" not in row:
                row["reward_pinned"] = pinned
            row["reward"] = corrected
            stats["rows"] += 1
            if corrected > pinned:
                stats["flip_0_to_1"] += 1
            elif corrected < pinned:
                stats["flip_1_to_0"] += 1
            sink.write(json.dumps(row) + "\n")
    temporary.replace(path)
    return stats


def scan_rows(path: Path, split: list[dict], tokenizer, data, old_verifier, new_verifier) -> Counter:
    """Read-only pass: prove the pinned verifier reproduces every stored reward and
    count what the corrected one would change. Raises on the first row it cannot
    reproduce, before anything has been written."""
    stats = Counter()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            text = tokenizer.decode(row["input_ids"][int(row["resp_start"]):], skip_special_tokens=True)
            gold = split[int(row["prompt_idx"])]["answer"]
            pinned = float(row.get("reward_pinned", row["reward"]))
            kind = mmr.reproduces_stored(text, gold, data, pinned, old_verifier)
            if kind == "mismatch":
                raise ValueError(
                    f"{path.name}:{line_number}: the pinned verifier does not reproduce the stored reward "
                    f"({pinned}); responses or gold answers are misread, nothing was changed"
                )
            stats["timeout_sensitive"] += kind == "timeout"
            corrected = mmr.score(text, gold, data, new_verifier)
            stats["rows"] += 1
            stats["flip_0_to_1"] += corrected > pinned
            stats["flip_1_to_0"] += corrected < pinned
    return stats


def reseal(manifest_path: Path, artifact: Path, sidecar: dict) -> None:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    sidecar.setdefault("pinned_artifact_sha256", {})[artifact.name] = document.get("artifact_sha256")
    document["artifact_sha256"] = sha256_file(artifact)
    temporary = manifest_path.with_name(manifest_path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    temporary.replace(manifest_path)


def retire_derived(run: Path, stamp: str) -> list[str]:
    """Move reward-derived artifacts aside; never delete."""
    parking = run / "pinned-scoring" / stamp
    moved = []
    for name in DERIVED_ARTIFACTS:
        source = run / name
        if source.exists():
            parking.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(parking / name))
            moved.append(name)
    return moved


def rescore_point(run: Path, data, old_verifier, new_verifier, *, apply: bool, stamp: str) -> dict:
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    prompts = json.loads((run / "prompts.json").read_text(encoding="utf-8"))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model_resolved"], local_files_only=True)
    report = {"run": run.name, "files": [], "retired": []}
    for prefix in ROLLOUT_PREFIXES:
        files = rollout_files(run, prefix)
        if not files:
            continue
        split = prompts["val"] if prefix.endswith("_val") else prompts["train"]
        sidecar_path = run / f"{prefix}.rescore.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.is_file() else {}
        sidecar.update({"verifier": VERIFIER_ID, "rewritten_at_utc": stamp})
        # Verify every file of this prefix before rewriting any of them: a
        # mismatch on the second file must not leave the first one rewritten.
        for path, _ in files:
            scan_rows(path, split, tokenizer, data, old_verifier, new_verifier)
        for path, manifest in files:
            if not apply:
                stats = scan_rows(path, split, tokenizer, data, old_verifier, new_verifier)
            else:
                stats = rewrite_rows(path, split, tokenizer, data, old_verifier, new_verifier)
                if manifest is not None:
                    reseal(manifest, path, sidecar)
            report["files"].append({
                "file": path.name,
                "rows": int(stats["rows"]),
                "flip_0_to_1": int(stats["flip_0_to_1"]),
                "flip_1_to_0": int(stats["flip_1_to_0"]),
                "timeout_sensitive": int(stats["timeout_sensitive"]),
            })
        if apply:
            sidecar_path.write_text(json.dumps(sidecar, indent=1, ensure_ascii=False), encoding="utf-8")
    if apply:
        report["retired"] = retire_derived(run, stamp)
    return report


def owner_is_fresh(queue: Path, family: str, seconds: int = 1800) -> bool:
    marker = queue / f"{family}.owner.json"
    try:
        return time.time() - marker.stat().st_mtime < seconds
    except OSError:
        return False


def rescore_family(root: Path, dataset: str, seed: int, drifts: list[int], tag: str, *,
                   data, old_verifier, new_verifier, apply: bool) -> dict:
    family = f"{dataset}-s{seed}"
    family_root = root / f"family-{family}"
    queue = root / ".families"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if owner_is_fresh(queue, family):
        raise SystemExit(f"[abort] a worker claimed {dataset}/s{seed} in the last 30 minutes; stop it first")
    lock_path = queue / f"{family}.lock"
    queue.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"[abort] {dataset}/s{seed} is locked by a running worker")
        points = [family_root / f"{tag}-s{seed}-{dataset}-d{d}" for d in drifts]
        missing = [p.name for p in points if not (p / "run_config.json").is_file()]
        if missing:
            raise SystemExit(f"[abort] {dataset}/s{seed} is not a finished family; missing {missing}")
        if apply:
            # The whole family is verified read-only first; only then is any
            # point rewritten, so a family is never left half rescored.
            for p in points:
                rescore_point(p, data, old_verifier, new_verifier, apply=False, stamp=stamp)
        reports = [rescore_point(p, data, old_verifier, new_verifier, apply=apply, stamp=stamp) for p in points]
        if apply:
            (family_root / ".family-complete").unlink(missing_ok=True)
    return {"family": f"{dataset}/s{seed}", "stamp": stamp, "points": reports}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="model tag prefix of point directories")
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--seed", type=int, action="append", help="family seed (repeatable); default: every complete family")
    parser.add_argument("--drifts", default="0 25 100 400")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pinned", default=os.environ.get("OM_GENERATION_GIT", mmr.PINNED_DEFAULT))
    parser.add_argument("--apply", action="store_true", help="rewrite (default: dry run, counts only)")
    args = parser.parse_args(argv)
    drifts = [int(d) for d in args.drifts.split()]

    data = mmr.load_pinned_data_module(args.repo, args.pinned[:12])
    try:
        old_verifier, new_verifier = mmr.verifier_pair()
    except ImportError:
        print("[abort] math-verify is not importable")
        return 2

    seeds = args.seed
    if not seeds:
        seeds = []
        for family_root in sorted(args.root.glob(f"family-{args.dataset}-s*")):
            seed = int(family_root.name.rsplit("-s", 1)[1])
            points = [family_root / f"{args.tag}-s{seed}-{args.dataset}-d{d}" for d in drifts]
            done = all((p / "DONE").is_file() for p in points)
            # A family already rescored once has no DONE and no stamp any more;
            # it is still ours to rescore again (idempotent), so a re-run after a
            # crash or a second correction needs no seed argument.
            rescored = all(any(p.glob("*.rescore.json")) for p in points)
            if done or rescored or (family_root / ".family-complete").is_file():
                seeds.append(seed)
    if not seeds:
        print(f"[rescore] no complete {args.dataset} family under {args.root}")
        return 1

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[rescore] {mode} verifier={VERIFIER_ID} pinned={args.pinned[:8]} families={[f'{args.dataset}/s{s}' for s in seeds]}")
    totals = Counter()
    for seed in seeds:
        result = rescore_family(args.root, args.dataset, seed, drifts, args.tag,
                                data=data, old_verifier=old_verifier, new_verifier=new_verifier, apply=args.apply)
        for point in result["points"]:
            for entry in point["files"]:
                totals["rows"] += entry["rows"]
                totals["flip_0_to_1"] += entry["flip_0_to_1"]
                totals["flip_1_to_0"] += entry["flip_1_to_0"]
                print(f"  {point['run']}  {entry['file']}: rows={entry['rows']} 0->1={entry['flip_0_to_1']} 1->0={entry['flip_1_to_0']}"
                      + (f" timeout-sensitive={entry['timeout_sensitive']}" if entry["timeout_sensitive"] else ""))
            if point["retired"]:
                print(f"  {point['run']}  moved to pinned-scoring/{result['stamp']}/: {', '.join(point['retired'])}")
    print(f"[rescore] rows={totals['rows']} 0->1={totals['flip_0_to_1']} 1->0={totals['flip_1_to_0']}")
    if args.apply:
        print("[rescore] the families are now incomplete on purpose; a worker recomputes gradients, scores and reports:")
        print("          bash scripts/run_olmo3_rlzero.sh run h100")
    return 0


if __name__ == "__main__":
    sys.exit(main())
