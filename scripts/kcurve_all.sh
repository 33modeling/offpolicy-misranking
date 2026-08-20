#!/usr/bin/env bash
# B1 — K-scaling floor curves for preregistered and extended conditions.
#   bash scripts/kcurve_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
source scripts/_report_io.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
make_report_dir kcurve-all || { echo "[kcurve-all-abort] 출력 폴더 생성 실패" >&2; exit 1; }
publish_report "$REPORT_DIR/KCURVE_ALL.md" "0" yes \
  env PYTHONPATH=src "$PY" src/kcurve_all.py "$OM_WORK/runs" || exit 1
echo
echo "== 저장: $REPORT_DIR/KCURVE_ALL.md"
