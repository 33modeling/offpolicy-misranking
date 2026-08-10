#!/usr/bin/env bash
# 공용 데이터셋을 $DATASETS_DIR(기본 /group-volume/datasets)에 jsonl로 받아둔다.
# 오프라인 컴퓨트 노드가 그대로 읽는 배치본 — 온라인/미러 되는 셸에서 실행.
#   bash scripts/fetch_datasets.sh              # 전부 (mbpp math500 gsm8k)
#   bash scripts/fetch_datasets.sh mbpp         # 골라서
# 이미 있으면 스킵. HF 실패 시 hf-mirror.com 폴백.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || PY=python3
TARGETS="${*:-mbpp math500 gsm8k}"
mkdir -p "$DATASETS_DIR"
echo "[fetch] DATASETS_DIR=$DATASETS_DIR → $TARGETS"

_fetch() {  # _fetch <name> <python-snippet>
  local name="$1" snip="$2"
  if HF_HUB_ETAG_TIMEOUT=15 timeout 900 "$PY" -c "$snip" "$DATASETS_DIR"; then return 0; fi
  echo "[fetch] $name 실패 — 미러 재시도(hf-mirror.com)"
  HF_HUB_ETAG_TIMEOUT=15 HF_ENDPOINT=https://hf-mirror.com timeout 900 "$PY" -c "$snip" "$DATASETS_DIR" \
    || { echo "[fetch] $name 확보 실패" >&2; return 1; }
}

fail=0
for t in $TARGETS; do
  case "$t" in
    mbpp)
      if [ -e "$DATASETS_DIR/mbpp/mbpp.jsonl" ]; then echo "[fetch] mbpp 있음, 스킵"; continue; fi
      _fetch mbpp '
import json, sys
from pathlib import Path
from datasets import load_dataset
out = Path(sys.argv[1]) / "mbpp"; out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("google-research-datasets/mbpp", "full")
n = 0
with open(out / "mbpp.jsonl", "w") as f:
    for split in ds:
        for r in ds[split]:
            f.write(json.dumps({"task_id": r["task_id"], "text": r["text"],
                                "code": r["code"], "test_list": r["test_list"]}) + "\n")
            n += 1
print("mbpp.jsonl:", n, "rows")' || fail=1 ;;
    math500)
      if [ -e "$DATASETS_DIR/math500/math500_test.jsonl" ]; then echo "[fetch] math500 있음, 스킵"; continue; fi
      _fetch math500 '
import json, sys
from pathlib import Path
from datasets import load_dataset
out = Path(sys.argv[1]) / "math500"; out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
with open(out / "math500_test.jsonl", "w") as f:
    for r in ds:
        f.write(json.dumps({"problem": r["problem"], "answer": str(r["answer"])}) + "\n")
print("math500_test.jsonl:", len(ds), "rows")' || fail=1 ;;
    gsm8k)
      if [ -e "$DATASETS_DIR/gsm8k/gsm8k_train.jsonl" ]; then echo "[fetch] gsm8k 있음, 스킵"; continue; fi
      _fetch gsm8k '
import json, sys
from pathlib import Path
from datasets import load_dataset
out = Path(sys.argv[1]) / "gsm8k"; out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("openai/gsm8k", "main", split="train")
with open(out / "gsm8k_train.jsonl", "w") as f:
    for r in ds:
        f.write(json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n")
print("gsm8k_train.jsonl:", len(ds), "rows")' || fail=1 ;;
    *) echo "[fetch] 모르는 데이터셋: $t (mbpp|math500|gsm8k)"; fail=1 ;;
  esac
done
exit "$fail"
