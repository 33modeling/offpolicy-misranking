#!/usr/bin/env bash
# Qwen3.6-27B 스냅샷 — HF HTTP API(파일 목록) + aria2c(병렬 다운로드) 조합.
# 온라인 머신에서:
#   bash scripts/fetch_27b.sh
# 다른 레포/FP8판: REPO27B=Qwen/Qwen3.6-27B-FP8 bash scripts/fetch_27b.sh
# 실패·중단 시 같은 명령으로 이어받기(-c). 본진 실패 시 hf-mirror 자동 폴백.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh 2>/dev/null || true

REPO="${REPO27B:-Qwen/Qwen3.6-27B}"
DEST="${MODELS_DIR:?MODELS_DIR 필요 — setup_env.sh 확인}/$(basename "$REPO")"
if [ -f "$DEST/config.json" ] && ls "$DEST"/*.safetensors >/dev/null 2>&1; then
  echo "[fetch] 이미 있음: $DEST"; exit 0
fi
command -v aria2c >/dev/null || { echo "[abort] aria2c 없음 — 설치 후 재실행 (apt install aria2)"; exit 1; }
mkdir -p "$DEST"
LIST="$DEST/.aria2-list"; TREE="$DEST/.tree.json"

fetch_tree() { curl -fsSL -m 60 "$1/api/models/$REPO/tree/main?recursive=true" -o "$TREE"; }
EP="https://huggingface.co"
if ! fetch_tree "$EP"; then
  echo "[fetch] 본진 API 실패 — hf-mirror로 폴백"
  EP="https://hf-mirror.com"
  fetch_tree "$EP" || { echo "[abort] 파일 목록 API 실패 (본진·미러 모두)"; exit 1; }
fi

python3 - "$TREE" "$EP" "$REPO" > "$LIST" <<'PYEOF'
import json
import sys

tree, ep, repo = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
for f in tree:
    if f.get("type") != "file":
        continue
    p = f["path"]
    print(f"{ep}/{repo}/resolve/main/{p}")
    print(f"  out={p}")
PYEOF

n=$(grep -c "out=" "$LIST" || true)
[ "${n:-0}" -gt 0 ] || { echo "[abort] 파일 목록이 비었음 — $TREE 확인"; exit 1; }
echo "[fetch] $n개 파일 → $DEST (aria2c -x16 -s16 -j4, 이어받기 -c)"
aria2c -x16 -s16 -j4 -c --file-allocation=none --console-log-level=warn \
  --summary-interval=30 -d "$DEST" -i "$LIST" \
  || { echo "[abort] aria2c 실패 — 같은 명령으로 이어받기 재시도"; exit 1; }
rm -f "$LIST" "$TREE"
echo "[fetch] 완료:"; du -sh "$DEST"
