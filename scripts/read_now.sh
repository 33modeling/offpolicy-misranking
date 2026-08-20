#!/usr/bin/env bash
# 즉시 판독 — READOUT과 reversal을 검증 후 하나의 고유 폴더에 publish한다.
#   bash scripts/read_now.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
source scripts/_report_io.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
make_report_dir readout || { echo "[readout-abort] 출력 폴더 생성 실패" >&2; exit 1; }

failures=0
publish_report "$REPORT_DIR/READOUT.md" "0" yes \
  "$PY" src/readout_summary.py "$OM_WORK/runs" || failures=$((failures + 1))
publish_report "$REPORT_DIR/REVERSAL.md" "0" no \
  "$PY" src/reversal_freq.py "$OM_WORK/runs" || failures=$((failures + 1))

shopt -s nullglob
for result_dir in "$OM_WORK"/results/*/; do
  tag=$(basename "$result_dir")
  [ -s "$result_dir/TABLES.md" ] \
    && cp "$result_dir/TABLES.md" "$REPORT_DIR/TABLES-$tag.md"
  [ -s "$result_dir/FRONTIER.md" ] \
    && cp "$result_dir/FRONTIER.md" "$REPORT_DIR/FRONTIER-$tag.md"
done

echo
if [ "$failures" -gt 0 ]; then
  echo "[readout-abort] 실패 $failures건: $REPORT_DIR" >&2
  ls "$REPORT_DIR"
  exit 1
fi
echo "== 저장 완료: $REPORT_DIR"
ls "$REPORT_DIR"
