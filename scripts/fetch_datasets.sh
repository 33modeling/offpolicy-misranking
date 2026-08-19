#!/usr/bin/env bash
# 공용 데이터셋을 $DATASETS_DIR(기본 /group-volume/datasets)에 jsonl로 받아둔다.
# 오프라인 컴퓨트 노드가 그대로 읽는 배치본 — 온라인/미러 되는 셸에서 실행.
#   bash scripts/fetch_datasets.sh              # 전부 (mbpp math500 gsm8k)
#   bash scripts/fetch_datasets.sh mbpp         # 골라서
# 이미 있으면 스킵. HF 실패 시 hf-mirror.com 폴백.
set -uo pipefail
cd "$(dirname "$0")/.."
# 다운로드 전용 스크립트 — setup_env의 기본 오프라인(HF_HUB_OFFLINE=1)을 그대로
# 물려받으면 모든 fetch가 OfflineModeIsEnabled로 즉사한다. 항상 온라인으로 탄다.
export OM_ONLINE=1
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || PY=python3
TARGETS="${*:-mbpp math500 gsm8k kk}"   # apps·dapo-math는 이름 명시 시에만
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
import json, os, sys
from pathlib import Path
from datasets import load_dataset
out = Path(sys.argv[1]) / "mbpp"; out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("google-research-datasets/mbpp", "full")
n = 0
tmp = out / ("mbpp.jsonl.tmp." + str(os.getpid()))
with open(tmp, "w") as f:
    for split in ds:
        for r in ds[split]:
            f.write(json.dumps({"task_id": r["task_id"], "text": r["text"],
                                "code": r["code"], "test_list": r["test_list"]}) + "\n")
            n += 1
tmp.replace(out / "mbpp.jsonl")
print("mbpp.jsonl:", n, "rows")' || fail=1 ;;
    math500)
      if [ -e "$DATASETS_DIR/math500/math500_test.jsonl" ]; then echo "[fetch] math500 있음, 스킵"; continue; fi
      _fetch math500 '
import json, os, sys
from pathlib import Path
from datasets import load_dataset
out = Path(sys.argv[1]) / "math500"; out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
tmp = out / ("math500_test.jsonl.tmp." + str(os.getpid()))
with open(tmp, "w") as f:
    for r in ds:
        f.write(json.dumps({"problem": r["problem"], "answer": str(r["answer"])}) + "\n")
tmp.replace(out / "math500_test.jsonl")
print("math500_test.jsonl:", len(ds), "rows")' || fail=1 ;;
    gsm8k)
      if [ -e "$DATASETS_DIR/gsm8k/gsm8k_train.jsonl" ]; then echo "[fetch] gsm8k 있음, 스킵"; continue; fi
      _fetch gsm8k '
import json, os, sys
from pathlib import Path
from datasets import load_dataset
out = Path(sys.argv[1]) / "gsm8k"; out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("openai/gsm8k", "main", split="train")
tmp = out / ("gsm8k_train.jsonl.tmp." + str(os.getpid()))
with open(tmp, "w") as f:
    for r in ds:
        f.write(json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n")
tmp.replace(out / "gsm8k_train.jsonl")
print("gsm8k_train.jsonl:", len(ds), "rows")' || fail=1 ;;
    kk)
      if [ -e "$DATASETS_DIR/kk/kk.jsonl" ]; then echo "[fetch] kk 있음, 스킵"; continue; fi
      _fetch kk '
import json, os, sys
from pathlib import Path
from datasets import get_dataset_config_names, load_dataset
out = Path(sys.argv[1]) / "kk"; out.mkdir(parents=True, exist_ok=True)
repo = "K-and-K/knights-and-knaves"
try:
    configs = get_dataset_config_names(repo)
except Exception:
    configs = [None]
n = 0
tmp = out / ("kk.jsonl.tmp." + str(os.getpid()))
with open(tmp, "w") as f:
    for cfg in configs:
        ds = load_dataset(repo, cfg) if cfg else load_dataset(repo)
        for split in ds:
            for r in ds[split]:
                r = dict(r); r["_config"], r["_split"] = cfg, split
                f.write(json.dumps(r) + "\n"); n += 1
tmp.replace(out / "kk.jsonl")
print("kk.jsonl:", n, "rows")' || fail=1 ;;
    apps)
      if [ -e "$DATASETS_DIR/apps/apps.jsonl" ]; then echo "[fetch] apps 있음, 스킵"; continue; fi
      # 스크립트형 데이터셋이라 최신 datasets 라이브러리로는 load_dataset 불가 —
      # parquet 변환본(refs/convert/parquet)을 hub에서 직접 받아 파싱한다.
      _fetch apps '
import json, os, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
out = Path(sys.argv[1]) / "apps"; out.mkdir(parents=True, exist_ok=True)
files = [f for f in HfApi().list_repo_files("codeparrot/apps", repo_type="dataset",
                                            revision="refs/convert/parquet")
         if f.endswith(".parquet")]
assert files, "parquet 변환본 목록이 비었음"
n = 0
tmp = out / ("apps.jsonl.tmp." + str(os.getpid()))
with open(tmp, "w") as f:
    for rf in sorted(files):
        local = hf_hub_download("codeparrot/apps", rf, repo_type="dataset",
                                revision="refs/convert/parquet")
        for r in pq.read_table(local).to_pylist():
            f.write(json.dumps({"question": r.get("question"),
                                "input_output": r.get("input_output"),
                                "difficulty": r.get("difficulty"),
                                "_split": ("test" if "test" in rf else "train")}) + "\n")
            n += 1
tmp.replace(out / "apps.jsonl")
print("apps.jsonl:", n, "rows")' || fail=1 ;;
    dapo-math)
      if [ -e "$DATASETS_DIR/dapo-math/dapo_math.jsonl" ]; then echo "[fetch] dapo-math 있음, 스킵"; continue; fi
      _fetch dapo-math '
import json, os, sys
from pathlib import Path
from datasets import load_dataset
out = Path(sys.argv[1]) / "dapo-math"; out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
tmp = out / ("dapo_math.jsonl.tmp." + str(os.getpid()))
with open(tmp, "w") as f:
    for r in ds:
        f.write(json.dumps(dict(r)) + "\n")
tmp.replace(out / "dapo_math.jsonl")
print("dapo_math.jsonl:", len(ds), "rows")' || fail=1 ;;
    *) echo "[fetch] 모르는 데이터셋: $t (mbpp|math500|gsm8k|kk|apps|dapo-math)"; fail=1 ;;
  esac
done
exit "$fail"
