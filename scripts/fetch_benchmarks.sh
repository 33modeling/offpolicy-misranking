#!/usr/bin/env bash
# Download the public evaluation sets for the trained E5 policies. Run ONCE in
# a shell that can reach the Hub (the login node); compute nodes read the
# frozen copies.
#
#   bash scripts/fetch_benchmarks.sh
#
# Writes $DATASETS_DIR/benchmarks/<set>.jsonl (question, answer, source_id)
# and <set>.manifest.json (repository, revision, split, rows, sha256) for
#   aime24     HuggingFaceH4/aime_2024 (30)        aime25  math-ai/aime25 (30)
#   amc23      math-ai/amc23 (40)                 gsm8k   openai/gsm8k main/test (1319)
#   math_rest  EleutherAI/hendrycks_math test minus the MATH-500 problems
#              (HuggingFaceH4/MATH-500), boxed answers
# Existing files are kept; delete a set to refetch it. Subsampling (gsm8k,
# math_rest) happens later, per E5 seed, with a frozen seed.
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=1
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
DIR="$DATASETS_DIR/benchmarks"
mkdir -p "$DIR"
fetch() {
  HF_HUB_ETAG_TIMEOUT=15 timeout 3600 "$PY" src/benchmark_eval.py fetch --datasets-dir "$DIR"
}
if ! fetch; then
  echo "[fetch] hub failed; retrying through hf-mirror.com"
  HF_ENDPOINT=https://hf-mirror.com fetch || { echo "[fetch] benchmark sets could not be downloaded" >&2; exit 1; }
fi
echo "[fetch] done: $DIR"
ls -la "$DIR"
