#!/usr/bin/env bash
# 공용 데이터셋을 $DATASETS_DIR(기본 /group-volume/datasets)에 jsonl로 받아둔다.
# 오프라인 컴퓨트 노드가 그대로 읽는 배치본 — 온라인/미러 되는 셸에서 실행.
#   bash scripts/fetch_datasets.sh              # 기본 (registered non-APPS sets)
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
TARGETS="${*:-mbpp math500 gsm8k kk arc-challenge}" # apps·dapo-math는 명시 시에만
mkdir -p "$DATASETS_DIR"
echo "[fetch] DATASETS_DIR=$DATASETS_DIR → $TARGETS"

_fetch() {  # _fetch <name> <python-snippet>
  local name="$1" snip="$2"
  if HF_HUB_ETAG_TIMEOUT=15 timeout 900 "$PY" -c "$snip" "$DATASETS_DIR"; then return 0; fi
  echo "[fetch] $name 실패 — 미러 재시도(hf-mirror.com)"
  HF_HUB_ETAG_TIMEOUT=15 HF_ENDPOINT=https://hf-mirror.com timeout 900 "$PY" -c "$snip" "$DATASETS_DIR" \
    || { echo "[fetch] $name 확보 실패" >&2; return 1; }
}

_snapshot_ok() {  # _snapshot_ok <manifest> <revision> <jsonl>
  "$PY" - "$1" "$2" "$3" <<'PYEOF' >/dev/null 2>&1
import hashlib, json, sys
from pathlib import Path
manifest, revision, data = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
if not manifest.is_file() or not data.is_file():
    raise SystemExit(1)
m = json.loads(manifest.read_text())
digest = hashlib.sha256(data.read_bytes()).hexdigest()
raise SystemExit(0 if m.get("source_revision") == revision and m.get("sha256") == digest else 1)
PYEOF
}

_manifest() {  # _manifest <name> <repo> <revision> <jsonl> <selection>
  "$PY" - "$1" "$2" "$3" "$4" "$5" <<'PYEOF'
import datetime, hashlib, json, os, sys
from pathlib import Path
name, repo, revision, filename, selection = sys.argv[1:]
data = Path(filename)
rows = sum(1 for line in data.open() if line.strip())
manifest = {
    "schema_version": 1,
    "dataset": name,
    "source_repository": repo,
    "source_revision": revision,
    "selection": selection,
    "artifact": data.name,
    "rows": rows,
    "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
}
target = data.parent / "dataset_manifest.json"
tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
tmp.replace(target)
print(f"[manifest] {name}: rows={rows} sha256={manifest['sha256'][:12]} revision={revision[:12]}")
PYEOF
}

fail=0
for t in $TARGETS; do
  case "$t" in
    mbpp)
      rev=4bb6404fdc6cacfda99d4ac4205087b89d32030c
      file="$DATASETS_DIR/mbpp/mbpp.jsonl"
      manifest="$DATASETS_DIR/mbpp/dataset_manifest.json"
      if _snapshot_ok "$manifest" "$rev" "$file"; then
        echo "[fetch] mbpp 고정 스냅샷 있음, 스킵"
      elif ! _fetch mbpp '
import json, os, sys
from pathlib import Path
from datasets import load_dataset
out = Path(sys.argv[1]) / "mbpp"; out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("google-research-datasets/mbpp", "full",
                  revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c")
n = 0
tmp = out / ("mbpp.jsonl.tmp." + str(os.getpid()))
with open(tmp, "w") as f:
    for split in ds:
        for r in ds[split]:
            f.write(json.dumps({"task_id": r["task_id"], "text": r["text"],
                                "code": r["code"], "test_list": r["test_list"]}) + "\n")
            n += 1
tmp.replace(out / "mbpp.jsonl")
print("mbpp.jsonl:", n, "rows")'; then fail=1; continue; fi
      _manifest mbpp google-research-datasets/mbpp "$rev" "$file" "all published splits, full config" ;;
    math500)
      rev=6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be
      file="$DATASETS_DIR/math500/math500_test.jsonl"
      manifest="$DATASETS_DIR/math500/dataset_manifest.json"
      if _snapshot_ok "$manifest" "$rev" "$file"; then
        echo "[fetch] math500 고정 스냅샷 있음, 스킵"
      elif ! _fetch math500 '
import json, os, sys
from pathlib import Path
from datasets import load_dataset
out = Path(sys.argv[1]) / "math500"; out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("HuggingFaceH4/MATH-500", split="test",
                  revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be")
tmp = out / ("math500_test.jsonl.tmp." + str(os.getpid()))
with open(tmp, "w") as f:
    for r in ds:
        f.write(json.dumps({"problem": r["problem"], "answer": str(r["answer"])}) + "\n")
tmp.replace(out / "math500_test.jsonl")
print("math500_test.jsonl:", len(ds), "rows")'; then fail=1; continue; fi
      _manifest math500 HuggingFaceH4/MATH-500 "$rev" "$file" "test split" ;;
    gsm8k)
      rev=740312add88f781978c0658806c59bc2815b9866
      file="$DATASETS_DIR/gsm8k/gsm8k_train.jsonl"
      manifest="$DATASETS_DIR/gsm8k/dataset_manifest.json"
      if _snapshot_ok "$manifest" "$rev" "$file"; then
        echo "[fetch] gsm8k 고정 스냅샷 있음, 스킵"
      elif ! _fetch gsm8k '
import json, os, sys
from pathlib import Path
from datasets import load_dataset
out = Path(sys.argv[1]) / "gsm8k"; out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("openai/gsm8k", "main", split="train",
                  revision="740312add88f781978c0658806c59bc2815b9866")
tmp = out / ("gsm8k_train.jsonl.tmp." + str(os.getpid()))
with open(tmp, "w") as f:
    for r in ds:
        f.write(json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n")
tmp.replace(out / "gsm8k_train.jsonl")
print("gsm8k_train.jsonl:", len(ds), "rows")'; then fail=1; continue; fi
      _manifest gsm8k openai/gsm8k "$rev" "$file" "main train split only" ;;
    kk)
      rev=2f68547989981b1af37cb3dde5fdefa847aa8619
      file="$DATASETS_DIR/kk/kk.jsonl"
      manifest="$DATASETS_DIR/kk/dataset_manifest.json"
      if _snapshot_ok "$manifest" "$rev" "$file"; then
        echo "[fetch] kk 고정 스냅샷 있음, 스킵"
      elif ! _fetch kk '
import json, os, sys
from pathlib import Path
from datasets import get_dataset_config_names, load_dataset
out = Path(sys.argv[1]) / "kk"; out.mkdir(parents=True, exist_ok=True)
repo = "K-and-K/knights-and-knaves"
revision = "2f68547989981b1af37cb3dde5fdefa847aa8619"
try:
    configs = get_dataset_config_names(repo, revision=revision)
except Exception:
    configs = [None]
n = 0
tmp = out / ("kk.jsonl.tmp." + str(os.getpid()))
with open(tmp, "w") as f:
    for cfg in configs:
        ds = load_dataset(repo, cfg, revision=revision) if cfg else load_dataset(repo, revision=revision)
        for split in ds:
            for r in ds[split]:
                r = dict(r); r["_config"], r["_split"] = cfg, split
                f.write(json.dumps(r) + "\n"); n += 1
tmp.replace(out / "kk.jsonl")
print("kk.jsonl:", n, "rows")'; then fail=1; continue; fi
      _manifest kk K-and-K/knights-and-knaves "$rev" "$file" "all published configs and splits" ;;
    arc-challenge)
      rev=210d026faf9955653af8916fad021475a3f00453
      file="$DATASETS_DIR/arc-challenge/arc_challenge.jsonl"
      manifest="$DATASETS_DIR/arc-challenge/dataset_manifest.json"
      if _snapshot_ok "$manifest" "$rev" "$file"; then
        echo "[fetch] arc-challenge 고정 스냅샷 있음, 스킵"
      elif ! _fetch arc-challenge '
import json, os, sys
from pathlib import Path
from datasets import load_dataset
out = Path(sys.argv[1]) / "arc-challenge"; out.mkdir(parents=True, exist_ok=True)
revision = "210d026faf9955653af8916fad021475a3f00453"
ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", revision=revision)
n = 0
tmp = out / ("arc_challenge.jsonl.tmp." + str(os.getpid()))
with open(tmp, "w") as f:
    for split in ("train", "validation"):
        for r in ds[split]:
            f.write(json.dumps({"id": r["id"], "question": r["question"],
                                "choices": r["choices"], "answerKey": r["answerKey"],
                                "_split": split}) + "\n")
            n += 1
tmp.replace(out / "arc_challenge.jsonl")
print("arc_challenge.jsonl:", n, "rows")'; then fail=1; continue; fi
      _manifest arc-challenge allenai/ai2_arc "$rev" "$file" "ARC-Challenge train+validation (labeled only)" ;;
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
    *) echo "[fetch] 모르는 데이터셋: $t (mbpp|math500|gsm8k|kk|arc-challenge|apps|dapo-math)"; fail=1 ;;
  esac
done
exit "$fail"
