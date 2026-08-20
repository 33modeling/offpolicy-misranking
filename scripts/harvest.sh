#!/usr/bin/env bash
# 수확 원스톱 — 분석 산출물을 검증한 뒤 하나의 고유 폴더에 원자적으로 publish한다.
#   KCURVE(GPU 0, 수 분) + READOUT + REVERSAL(닻 포함) + STATS + 표 사본을 같은 폴더에.
#   bash scripts/harvest.sh
#
# 0820 수확 교훈: 실패를 숨기면(2>/dev/null, || true) 빈 파일이 성공처럼 전달된다
# (READOUT.md 0바이트). stderr와 partial stdout을 보존하고, 검증을 통과한 파일만
# 최종 Markdown 경로로 publish한다.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
READOUTS_ROOT="$OM_WORK/readouts"
mkdir -p "$READOUTS_ROOT" || {
  echo "[harvest-abort] readouts 경로 생성 실패: $READOUTS_ROOT" >&2
  exit 1
}
STAMP_DIR=$(mktemp -d "$READOUTS_ROOT/$(date '+%Y-%m-%d_%H%M%S')-harvest.XXXXXX") || {
  echo "[harvest-abort] 고유 수확 폴더 생성 실패" >&2
  exit 1
}
failures=()

publish_markdown() {
  local label=$1 output=$2 allowed=$3 display=$4
  shift 4
  local tmp="$output.tmp"
  local stem="${output%.md}"
  local err="$stem.err"
  local partial="$stem.partial.md"
  local rc reason

  rm -f "$tmp" "$partial" "$err"
  "$@" > "$tmp" 2> "$err"
  rc=$?
  if [[ " $allowed " != *" $rc "* ]]; then
    reason="exit=$rc"
  elif [ ! -s "$tmp" ]; then
    reason="empty-output"
  else
    if ! mv "$tmp" "$output"; then
      reason="publish-failed"
    else
      [ -s "$err" ] || rm -f "$err"
      [ "$display" = "yes" ] && cat "$output"
      return 0
    fi
  fi

  [ -e "$tmp" ] && mv "$tmp" "$partial"
  failures+=("$label:$reason")
  echo "[harvest] $label 실패 ($reason); stderr=$err" >&2
  return 1
}

# 3/4 are scientific k-curve verdicts (unreachable/no eligible run), not crashes.
publish_markdown kcurve "$STAMP_DIR/KCURVE.md" "0 3 4" yes \
  "$PY" src/kcurve_floor.py "$OM_WORK/runs" || :
publish_markdown readout "$STAMP_DIR/READOUT.md" "0" yes \
  "$PY" src/readout_summary.py "$OM_WORK/runs" || :
publish_markdown reversal "$STAMP_DIR/REVERSAL.md" "0" no \
  "$PY" src/reversal_freq.py "$OM_WORK/runs" || :

# 원고 A8a — corrected protocol이 있는 run별 정확 p·bootstrap CI.
stats_tmp="$STAMP_DIR/STATS.md.tmp"
stats_partial="$STAMP_DIR/STATS.partial.md"
stats_err="$STAMP_DIR/STATS.err"
: > "$stats_tmp"
: > "$stats_err"
stats_runs=0
stats_failed=0
shopt -s nullglob
for d in "$OM_WORK"/runs/*/; do
  [ -f "$d/scores_oracle.json" ] || continue
  [ -f "$d/score_protocol.json" ] || continue
  [ -f "$d/oracle_protocol.json" ] || continue
  case "$(basename "$d")" in *smoke*) continue;; esac
  stats_runs=$((stats_runs + 1))
  run_name=$(basename "$d")
  run_out="$STAMP_DIR/.stats-$run_name.out"
  run_err="$STAMP_DIR/.stats-$run_name.err"
  "$PY" src/stats_extra.py "$d" > "$run_out" 2> "$run_err"
  rc=$?
  if [ "$rc" -ne 0 ] || [ ! -s "$run_out" ]; then
    reason="exit=$rc"
    [ "$rc" -eq 0 ] && reason="empty-output"
    failures+=("stats:$run_name:$reason")
    stats_failed=1
    mv "$run_out" "$STAMP_DIR/STATS-$run_name.partial.md"
  else
    printf '## %s\n' "$run_name" >> "$stats_tmp"
    cat "$run_out" >> "$stats_tmp"
    printf '\n' >> "$stats_tmp"
    rm -f "$run_out"
  fi
  if [ -s "$run_err" ]; then
    printf '## %s\n' "$run_name" >> "$stats_err"
    cat "$run_err" >> "$stats_err"
    printf '\n' >> "$stats_err"
  fi
  rm -f "$run_err"
done
if [ "$stats_runs" -eq 0 ]; then
  failures+=("stats:no-corrected-runs")
  stats_failed=1
fi
if [ "$stats_failed" -eq 0 ] && [ -s "$stats_tmp" ]; then
  mv "$stats_tmp" "$STAMP_DIR/STATS.md"
else
  mv "$stats_tmp" "$stats_partial"
fi
[ -s "$stats_err" ] || rm -f "$stats_err"

# 결과 폴더는 세대별(v2·v3·qwen3.8-27b 등)로 분리될 수 있다.
for rd in "$OM_WORK"/results/*/; do
  [ -d "$rd" ] || continue
  rtag=$(basename "$rd")
  [ -s "$rd/TABLES.md" ] \
    && cp "$rd/TABLES.md" "$STAMP_DIR/TABLES-$rtag.md"
  [ -s "$rd/FRONTIER.md" ] \
    && cp "$rd/FRONTIER.md" "$STAMP_DIR/FRONTIER-$rtag.md"
done
[ -s results/TABLES.md ] && cp results/TABLES.md "$STAMP_DIR/"

if [ "${#failures[@]}" -gt 0 ]; then
  {
    echo "# Harvest failures"
    echo
    printf -- '- `%s`\n' "${failures[@]}"
  } > "$STAMP_DIR/HARVEST_FAILURES.md"
  echo "[harvest-abort] 실패 폴더: $STAMP_DIR" >&2
  printf '[harvest-abort] %s\n' "${failures[*]}" >&2
  ls "$STAMP_DIR"
  exit 1
fi

{
  echo "# Harvest status"
  echo
  echo "- status: complete"
  echo "- source_commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "- generated_at: $(date --iso-8601=seconds)"
} > "$STAMP_DIR/HARVEST_STATUS.md"

echo
echo "== 전달할 폴더 하나: $STAMP_DIR"
ls "$STAMP_DIR"
