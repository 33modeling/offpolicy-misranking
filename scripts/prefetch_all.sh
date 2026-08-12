#!/usr/bin/env bash
# 데이터셋·모델 사전 다운로드 원샷 — 검증된 방식(HF HTTP API 파일 목록 + aria2c
# 병렬, 미러 폴백) 그대로. 다운로드는 CPU/네트워크 작업이라 GPU 사용률이 0으로
# 떨어지는데, 이 시스템은 사용률이 낮으면 잡을 죽인다 — 그래서 다운로드 동안
# gpu_keepalive를 함께 돌린다 (사용자 요청 반영).
#
#   bash scripts/prefetch_all.sh                       # 온라인 머신에서
#   MODELS_PREFETCH="Org/Name ..." bash scripts/...    # 모델 목록 교체
#
# 중단돼도 같은 명령으로 이어받기(-c). 있으면 전부 스킵(멱등).
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh 2>/dev/null || true
command -v aria2c >/dev/null || echo "[info] aria2c 없음 — curl 8병렬 폴백으로 진행 (설치 불필요)"

# 다운로드 동안 GPU 사용률 유지 (GPU 없는 머신이면 자동 생략)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
PYV="${VENV_DIR:-}/bin/python"; [ -x "$PYV" ] || PYV=python3
if [ "${NGPU:-0}" -gt 0 ]; then
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU - 1))) "$PYV" scripts/gpu_keepalive.py > /dev/null 2>&1 &
  KA=$!
  trap 'kill $KA 2>/dev/null' EXIT
  echo "[prefetch] GPU ${NGPU}장 — 다운로드 동안 keepalive 가동"
fi

fetch_model() {  # fetch_model <HF repo> — 목록은 HTTP API, 파일은 aria2c 병렬
  local REPO="$1" DEST TREE LIST EP n
  DEST="$MODELS_DIR/$(basename "$REPO")"
  if [ -f "$DEST/config.json" ] && ls "$DEST"/*.safetensors >/dev/null 2>&1; then
    echo "[모델] 있음, 스킵: $(basename "$REPO")"; return 0
  fi
  mkdir -p "$DEST"
  TREE="$DEST/.tree.json"; LIST="$DEST/.aria2-list"
  EP="https://huggingface.co"
  if ! curl -fsSL -m 60 "$EP/api/models/$REPO/tree/main?recursive=true" -o "$TREE"; then
    EP="https://hf-mirror.com"
    curl -fsSL -m 60 "$EP/api/models/$REPO/tree/main?recursive=true" -o "$TREE" \
      || { echo "[모델] ✘ 목록 실패(본진·미러): $REPO — 레포명 확인"; return 1; }
  fi
  python3 - "$TREE" "$EP" "$REPO" > "$LIST" <<'PYEOF'
import json
import sys

tree, ep, repo = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
for f in tree:
    if f.get("type") == "file":
        print(f"{ep}/{repo}/resolve/main/{f['path']}")
        print(f"  out={f['path']}")
PYEOF
  n=$(grep -c "out=" "$LIST" || true)
  [ "${n:-0}" -gt 0 ] || { echo "[모델] ✘ 파일 목록 비었음: $REPO"; return 1; }
  if command -v aria2c >/dev/null; then
    echo "[모델] $REPO — ${n}개 파일 (aria2c -x16 -s16 -j4, 이어받기 -c)"
    aria2c -x16 -s16 -j4 -c --file-allocation=none --console-log-level=warn \
      --summary-interval=60 -d "$DEST" -i "$LIST" \
      || { echo "[모델] ✘ $REPO 중단 — 같은 명령으로 이어받기"; return 1; }
  else
    echo "[모델] $REPO — ${n}개 파일 (curl 8병렬 폴백, 이어받기 -C -)"
    python3 - "$LIST" "$DEST" <<'PYFALLBACK'
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

lines = [l for l in open(sys.argv[1]).read().splitlines() if l.strip()]
dest = sys.argv[2]
pairs = []
for i in range(0, len(lines), 2):
    url, out = lines[i], lines[i + 1].strip().split("out=", 1)[1]
    pairs.append((url, out))

def dl(p):
    url, out = p
    r = subprocess.run(
        ["curl", "-fL", "-C", "-", "--retry", "3", "--retry-delay", "5",
         "--create-dirs", "-o", f"{dest}/{out}", url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(("  OK  " if r.returncode == 0 else "  FAIL"), out, flush=True)
    return r.returncode

with ThreadPoolExecutor(8) as ex:
    codes = list(ex.map(dl, pairs))
sys.exit(1 if any(codes) else 0)
PYFALLBACK
    [ $? -eq 0 ] || { echo "[모델] ✘ $REPO 일부 실패 — 같은 명령 재실행(이어받기)"; return 1; }
  fi
  rm -f "$LIST" "$TREE"
  echo "[모델] ✔ $(basename "$REPO") ($(du -sh "$DEST" | cut -f1))"
}

echo "== 1) 데이터셋 전부 (있으면 스킵)"
bash scripts/fetch_datasets.sh mbpp math500 gsm8k kk dapo-math apps \
  || echo "[warn] 일부 데이터셋 실패 — 위 로그 확인 후 개별 재시도"

echo "== 2) 모델 스냅샷 (없는 것만)"
DEFAULT_MODELS="Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct \
Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.6-27B"
fail=0
for m in ${MODELS_PREFETCH:-$DEFAULT_MODELS}; do
  fetch_model "$m" || fail=1
done

echo "== 3) 최종 점검 (preflight)"
bash scripts/preflight.sh || true

if [ "$fail" = 0 ]; then
  echo "== prefetch 전부 완료"
else
  echo "== 일부 실패 — 레포명이 다르면 MODELS_PREFETCH로 지정, 아니면 같은 명령 재실행(이어받기)"
  exit 1
fi
