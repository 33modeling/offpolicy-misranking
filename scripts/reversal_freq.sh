#!/usr/bin/env bash
# Sign-reversal frequency report from existing corrected artifacts.
#   bash scripts/reversal_freq.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
source scripts/_report_io.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
make_report_dir reversal || { echo "[reversal-abort] 출력 폴더 생성 실패" >&2; exit 1; }
publish_report "$REPORT_DIR/REVERSAL.md" "0" yes \
  "$PY" src/reversal_freq.py "$OM_WORK/runs" || exit 1
echo
echo "== 저장 완료: $REPORT_DIR/REVERSAL.md"
