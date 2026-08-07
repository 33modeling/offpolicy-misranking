#!/usr/bin/env bash
# 클러스터 프로비저닝 — failure-atlas/influence-staleness 패턴. 멱등.
#   source scripts/setup_env.sh && bash scripts/provision.sh
#
# torch 2.7.1+cu126 constraints 고정, venv는 group-volume($VENV_DIR) 하나,
# 모델은 고정 revision 스냅샷 + hf-mirror.com 폴백, GSM8K는 로컬 jsonl로 저장
# (컴퓨트 노드 오프라인 대비). 끝에 로직 테스트까지 돌린다.
set -Eeuo pipefail
step() { echo "[provision $(date '+%T')] $*"; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
[[ -n "${STORAGE_ROOT:-}" ]] || { echo "setup_env.sh 를 먼저 source 하라" >&2; exit 1; }
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true

PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.12}"
command -v "$PYTHON_BOOTSTRAP" >/dev/null || PYTHON_BOOTSTRAP=python3

# 1) venv — get-pip 폴백, 인트라넷 미러는 PIP_INDEX_URL로
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[provision] venv 생성: $VENV_DIR"
  "$PYTHON_BOOTSTRAP" -m venv "$VENV_DIR" 2>/dev/null || {
    "$PYTHON_BOOTSTRAP" -m venv --without-pip "$VENV_DIR"
    curl -sS --connect-timeout 15 --max-time 180 https://bootstrap.pypa.io/get-pip.py | "$VENV_DIR/bin/python" - \
      || { echo "[provision] get-pip 실패 — PIP_INDEX_URL=<인트라넷 미러> 지정 후 재실행" >&2; exit 1; }
  }
fi
step "pip 설치 시작 (진행 표시 — 멈춰 보이면 pypi 접근 불가, PIP_INDEX_URL=<인트라넷 미러> 지정)"
"$VENV_DIR/bin/pip" install --timeout 60 --retries 2 -c constraints/h100-cu126.txt -r requirements.txt \
  ${PIP_INDEX_URL:+-i "$PIP_INDEX_URL"} \
  || { echo "[provision] pip 설치 실패 — PIP_INDEX_URL 지정 또는 로컬 wheel 필요" >&2; exit 1; }
"$VENV_DIR/bin/python" -c "import torch, transformers, peft, datasets; \
print('[provision] torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
'gpus', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# 2) 모델 — failure-atlas와 동일한 고정 revision, 미러 폴백
fetch_model() {
  local repo="$1" revision="$2" dest="$3"
  if [[ -f "$dest/config.json" ]] && ls "$dest"/*.safetensors >/dev/null 2>&1; then
    echo "[provision] 모델 확인, 스킵: $dest"; return 0
  fi
  mkdir -p "$dest"
  echo "[provision] 모델 다운로드: $repo@$revision -> $dest"
  _dl() {
    HF_HUB_ETAG_TIMEOUT=15 timeout "${OM_FETCH_TIMEOUT:-3600}" \
      "$VENV_DIR/bin/python" - "$repo" "$revision" "$dest" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3],
                  ignore_patterns=["*.bin", "*.pth", "*.gguf"])
EOF
  }
  _dl || { echo "[provision] 일반 다운로드 실패 — 미러 재시도(hf-mirror.com)";
           HF_ENDPOINT=https://hf-mirror.com _dl; } \
      || { echo "[provision] 모델 확보 실패: $repo" >&2; return 1; }
}
fetch_model "Qwen/Qwen2.5-0.5B-Instruct" "main" "$MODEL_QWEN25_05B"
fetch_model "Qwen/Qwen2.5-7B-Instruct" "a09a35458c702b33eeacc393d103063234e8bc28" "$MODEL_QWEN25_7B"

# 3) 데이터 — GSM8K 로컬 jsonl (오프라인 노드에서 datasets 없이 읽게)
mkdir -p "$OM_DATA"
if [[ -f "$OM_DATA/gsm8k_train.jsonl" ]]; then
  echo "[provision] 데이터 확인, 스킵: $OM_DATA/gsm8k_train.jsonl"
else
  _fetch_data() {
    HF_HUB_ETAG_TIMEOUT=15 timeout 900 "$VENV_DIR/bin/python" - "$OM_DATA" <<'EOF'
import json, sys
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="train")
with open(sys.argv[1] + "/gsm8k_train.jsonl", "w") as f:
    for row in ds:
        f.write(json.dumps({"question": row["question"], "answer": row["answer"]}) + "\n")
print("gsm8k_train.jsonl:", len(ds), "rows")
EOF
  }
  _fetch_data || { echo "[provision] 데이터 실패 — 미러 재시도"; HF_ENDPOINT=https://hf-mirror.com _fetch_data; } \
    || { echo "[provision] GSM8K 확보 실패" >&2; exit 1; }
fi

# 4) 로직 테스트 (모델 불필요)
"$VENV_DIR/bin/python" tests/test_core.py

echo "[provision] done — 다음: bash scripts/run_h100_all.sh"
