#!/usr/bin/env bash
# P4-0 — K-scaling floor curve (GPU 0, several minutes).
#   bash scripts/kcurve.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
source scripts/_report_io.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
make_report_dir kcurve || { echo "[kcurve-abort] 출력 폴더 생성 실패" >&2; exit 1; }
publish_report "$REPORT_DIR/KCURVE.md" "0 3 4" yes \
  "$PY" src/kcurve_floor.py "$OM_WORK/runs" || exit 1
echo
echo "== 저장: $REPORT_DIR/KCURVE.md"
