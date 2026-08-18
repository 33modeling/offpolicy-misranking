#!/usr/bin/env bash
# 수확 원스톱 — 한 번 실행하고 마지막 줄에 찍히는 폴더 하나만 전달하면 끝:
#   KCURVE(GPU 0, 수 분) + READOUT + REVERSAL(닻 포함) + 표 사본을 같은 폴더에.
#   bash scripts/harvest.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
STAMP_DIR="$OM_WORK/readouts/$(date '+%Y-%m-%d_%H%M')-harvest"
mkdir -p "$STAMP_DIR"

"$PY" src/kcurve_floor.py "$OM_WORK/runs" | tee "$STAMP_DIR/KCURVE.md" || true
"$PY" src/readout_summary.py "$OM_WORK/runs" | tee "$STAMP_DIR/READOUT.md"
"$PY" src/reversal_freq.py "$OM_WORK/runs" > "$STAMP_DIR/REVERSAL.md" 2>/dev/null || true
# 원고 A8a — run별 정확 p·부트스트랩 CI (v1 계열 포함, CPU)
{ for d in "$OM_WORK"/runs/*/; do
    [ -f "$d/scores_oracle.json" ] || continue
    case "$(basename "$d")" in *smoke*) continue;; esac
    echo "## $(basename "$d")"; "$PY" src/stats_extra.py "$d" 2>/dev/null || true; echo
  done; } > "$STAMP_DIR/STATS.md" || true
# 결과 폴더는 세대별(v2·qwen3.8-27b 등)로 분리될 수 있다 — 전부 태그 붙여 동봉
for rd in "$OM_WORK"/results/*/; do
  rtag=$(basename "$rd")
  cp "$rd/TABLES.md"   "$STAMP_DIR/TABLES-$rtag.md"   2>/dev/null || true
  cp "$rd/FRONTIER.md" "$STAMP_DIR/FRONTIER-$rtag.md" 2>/dev/null || true
done
cp results/TABLES.md "$STAMP_DIR/" 2>/dev/null || true
echo
echo "== 전달할 폴더 하나: $STAMP_DIR"
ls "$STAMP_DIR"
