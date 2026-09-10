#!/usr/bin/env bash
# Download the MATH training split (EleutherAI/hendrycks_math, 7 subjects) as the
# candidate pool for the independent E5 test set. Run this ONCE in a shell that
# can reach the Hub (the login node); compute nodes read the frozen copy.
#
#   bash scripts/fetch_math_train.sh
#
# Writes $DATASETS_DIR/math_train/math_train.jsonl (question, answer from the
# last \boxed{...} of the reference solution, level, type) and a manifest with
# the resolved dataset revision and file hash. Existing copies are kept.
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=1
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
DIR="$DATASETS_DIR/math_train"; FILE="$DIR/math_train.jsonl"; MANIFEST="$DIR/dataset_manifest.json"
if [ -s "$FILE" ] && [ -s "$MANIFEST" ]; then
  echo "[fetch] math_train already present: $FILE ($(wc -l < "$FILE") rows); delete it to refetch"
  exit 0
fi
mkdir -p "$DIR"
fetch() {
  HF_HUB_ETAG_TIMEOUT=15 timeout 1800 "$PY" - "$DIR" <<'PYEOF'
import hashlib, json, sys, datetime
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi

REPO = "EleutherAI/hendrycks_math"
CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
           "number_theory", "prealgebra", "precalculus"]
out = Path(sys.argv[1]); target = out / "math_train.jsonl"


def boxed(solution: str):
    i = solution.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 1, i + len("\\boxed{")
    while j < len(solution) and depth:
        depth += {"{": 1, "}": -1}.get(solution[j], 0)
        j += 1
    return solution[i + len("\\boxed{"):j - 1].strip() if depth == 0 else None


try:
    revision = HfApi().dataset_info(REPO).sha
except Exception as exc:  # the manifest records what could be resolved
    revision = f"unresolved({type(exc).__name__})"
rows, skipped = [], 0
for config in CONFIGS:
    ds = load_dataset(REPO, config, split="train")
    for row in ds:
        answer = boxed(row["solution"])
        if not answer:
            skipped += 1
            continue
        rows.append({"question": row["problem"], "answer": answer, "level": row.get("level"),
                     "type": row.get("type"), "subject": config})
tmp = target.with_suffix(".tmp")
with tmp.open("w") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
tmp.replace(target)
manifest = {"schema_version": 1, "dataset": "math_train", "source_repository": REPO,
            "source_revision": revision, "split": "train", "configs": CONFIGS,
            "rows": len(rows), "skipped_without_boxed_answer": skipped,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "fetched_at_utc": datetime.datetime.now(datetime.UTC).isoformat()}
(out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"[manifest] math_train: rows={len(rows)} skipped={skipped} revision={revision[:12]} sha256={manifest['sha256'][:12]}")
PYEOF
}
if ! fetch; then
  echo "[fetch] hub failed; retrying through hf-mirror.com"
  HF_ENDPOINT=https://hf-mirror.com fetch || { echo "[fetch] math_train could not be downloaded" >&2; exit 1; }
fi
echo "[fetch] done: $FILE"
