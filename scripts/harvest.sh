#!/usr/bin/env bash
# 수확 원스톱 — 분석 산출물을 검증한 뒤 하나의 고유 폴더에 원자적으로 publish한다.
#   KCURVE(사전등록) + KCURVE_ALL(전 세대) + READOUT + REVERSAL(닻 포함)
#   + STATS + 표 사본 + regime 정본을 같은 폴더에.
#   bash scripts/harvest.sh
#
# 0820 수확 교훈: 실패를 숨기면(2>/dev/null, || true) 빈 파일이 성공처럼 전달된다
# (READOUT.md 0바이트). stderr와 partial stdout을 보존하고, 검증을 통과한 파일만
# 최종 Markdown 경로로 publish한다.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
source scripts/_report_cache.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
READOUTS_ROOT="$OM_WORK/readouts"
mkdir -p "$READOUTS_ROOT" || {
  echo "[harvest-abort] readouts 경로 생성 실패: $READOUTS_ROOT" >&2
  exit 1
}

# Harvest is a global publication step. Multiple workers may arrive here at
# once; only the first computes, and later workers reuse the exact validated
# bundle when neither analysis code nor input artifacts changed.
command -v flock >/dev/null 2>&1 || {
  echo "[harvest-abort] flock command missing" >&2
  exit 1
}
mkdir -p "$OM_WORK/locks" "$OM_WORK/results" || exit 1
exec 8>"$OM_WORK/locks/harvest.lock"
flock 8 || exit 1

harvest_code=(
  scripts/harvest.sh scripts/_report_cache.sh
  src/kcurve_floor.py src/kcurve_all.py src/readout_summary.py
  src/reversal_freq.py src/stats_extra.py src/run_select.py
  src/gate_rules.py src/score_artifacts.py src/select_rules.py
)
harvest_inputs=()
if [ -d "$OM_WORK/runs" ]; then
  while IFS= read -r -d '' artifact; do
    harvest_inputs+=("$artifact")
  done < <(find "$OM_WORK/runs" -type f \
    \( -name DONE -o -name '*.json' -o -name '*.jsonl' -o -name '*.pt' \) \
    -print0 | sort -z)
fi
while IFS= read -r -d '' artifact; do
  harvest_inputs+=("$artifact")
done < <(find "$OM_WORK/results" -mindepth 2 -maxdepth 2 -type f \
  \( -name 'TABLES.md' -o -name 'FRONTIER.md' -o -name 'frontier.json' \
     -o -name 'REGIME.json' -o -name 'REGIME.csv' \
     -o -name 'REGIME_SUMMARY.csv' -o -name 'FINAL_REPORT.md' \
     -o -name '.regime_collection.json' \) -print0 | sort -z)
harvest_key=$(report_cache_key "${harvest_code[@]}" -- "${harvest_inputs[@]}") \
  || exit 1
harvest_key=$(report_cache_key_values "$harvest_key" "harvest-schema=2") || exit 1
HARVEST_CURRENT="$OM_WORK/results/.harvest-current"
if [ -s "$HARVEST_CURRENT" ]; then
  cached_key=$(sed -n '1p' "$HARVEST_CURRENT")
  cached_dir=$(sed -n '2p' "$HARVEST_CURRENT")
  if [ "$cached_key" = "$harvest_key" ] \
     && [[ "$cached_dir" == "$READOUTS_ROOT/"* ]] \
     && [ -s "$cached_dir/HARVEST_STATUS.md" ] \
     && [ -s "$cached_dir/HARVEST_MANIFEST.sha256" ] \
     && (cd "$cached_dir" && sha256sum -c HARVEST_MANIFEST.sha256 >/dev/null 2>&1); then
    echo "== harvest 입력 변경 없음; 중복 계산 생략"
    echo "== 전달할 폴더 하나: $cached_dir"
    ls "$cached_dir"
    exit 0
  fi
fi

STAMP_DIR=$(mktemp -d "$READOUTS_ROOT/$(date '+%Y-%m-%d_%H%M%S')-harvest.XXXXXX") || {
  echo "[harvest-abort] 고유 수확 폴더 생성 실패" >&2
  exit 1
}
failures=()

validate_v4_matrix() {
  local seen=0 missing=0 model seed suffix run artifact
  local err="$STAMP_DIR/V4_MATRIX.err"
  local required=(DONE run_config.json manifest.json score_protocol.json oracle_protocol.json report.json)

  for run in "$OM_WORK"/runs/v4-27b-s* "$OM_WORK"/runs/v4-7b-s*; do
    [ -d "$run" ] || continue
    case "$(basename "$run")" in *smoke*) continue;; esac
    seen=1
    break
  done
  [ "$seen" -eq 1 ] || return 0

  : > "$err"
  for model in 27b 7b; do
    for seed in 0 1 2 3 4; do
      for suffix in "" -math500; do
        run="$OM_WORK/runs/v4-$model-s$seed$suffix"
        for artifact in "${required[@]}"; do
          if [ ! -s "$run/$artifact" ]; then
            printf '%s: %s missing or empty\n' "$(basename "$run")" "$artifact" >> "$err"
            missing=$((missing + 1))
          fi
        done
      done
    done
  done
  if [ "$missing" -gt 0 ]; then
    failures+=("v4-matrix:incomplete-$missing-artifacts")
    echo "[harvest] v4 matrix 불완전 ($missing개 필수 산출물 누락); stderr=$err" >&2
    return 1
  fi
  rm -f "$err"
}

validate_v4_matrix || :

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
publish_markdown kcurve-all "$STAMP_DIR/KCURVE_ALL.md" "0" no \
  "$PY" src/kcurve_all.py "$OM_WORK/runs" || :
publish_markdown readout "$STAMP_DIR/READOUT.md" "0" yes \
  "$PY" src/readout_summary.py "$OM_WORK/runs" || :
publish_markdown reversal "$STAMP_DIR/REVERSAL.md" "0" no \
  "$PY" src/reversal_freq.py "$OM_WORK/runs" || :

# 원고 A8a — corrected protocol이 있는 run별 정확 p·bootstrap CI.
stats_tmp="$STAMP_DIR/STATS.md.tmp"
stats_partial="$STAMP_DIR/STATS.partial.md"
stats_err="$STAMP_DIR/STATS.err"
stats_runs=0
stats_failed=0
stats_reused=0
stats_dirs=()
stats_names=()
stats_inputs=()
stats_code=(
  src/stats_extra.py src/gate_rules.py src/score_artifacts.py src/select_rules.py
)
shopt -s nullglob
for d in "$OM_WORK"/runs/*/; do
  [ -f "$d/scores_oracle.json" ] || continue
  [ -f "$d/score_protocol.json" ] || continue
  [ -f "$d/oracle_protocol.json" ] || continue
  case "$(basename "$d")" in *smoke*) continue;; esac
  stats_runs=$((stats_runs + 1))
  run_name=$(basename "$d")
  stats_dirs+=("$d")
  stats_names+=("$run_name")
  for artifact in run_config.json manifest.json score_protocol.json \
      oracle_protocol.json report.json scores_oracle.json scores_offpolicy.json \
      scores_splithalf.json; do
    stats_inputs+=("$d/$artifact")
  done
done
if [ "$stats_runs" -eq 0 ]; then
  failures+=("stats:no-corrected-runs")
  stats_failed=1
else
  stats_key=$(report_cache_key "${stats_code[@]}" -- "${stats_inputs[@]}") || exit 1
  stats_key=$(report_cache_key_values "$stats_key" "stats-schema=1" \
    "frac=0.10" "bootstrap=2000" "seed=0") || exit 1
  stats_cache_dir="$OM_WORK/results/.analysis-cache/harvest-stats"
  stats_cache_out="$stats_cache_dir/STATS.md"
  stats_cache_marker="$stats_cache_dir/STATS.key"
  mkdir -p "$stats_cache_dir" || exit 1

  if report_cache_hit "$stats_cache_marker" "$stats_key" "$stats_cache_out"; then
    cp -- "$stats_cache_out" "$STAMP_DIR/STATS.md" || exit 1
    stats_reused=1
    echo "[harvest] STATS 입력 변경 없음; 2,000-bootstrap 재사용"
  elif [ ! -e "$stats_cache_marker" ]; then
    # One-time migration for a successful STATS.md inside an older harvest that
    # failed later (for example, because REGIME files were still missing).
    while IFS= read -r legacy_stats; do
      [ -s "$legacy_stats" ] || continue
      legacy_is_fresh=1
      for artifact in "${stats_code[@]}" "${stats_inputs[@]}"; do
        [ -e "$artifact" ] || continue
        if [ "$artifact" -nt "$legacy_stats" ]; then
          legacy_is_fresh=0
          break
        fi
      done
      [ "$legacy_is_fresh" -eq 1 ] || continue
      [ "$(grep -c '^## ' "$legacy_stats" || true)" -eq "$stats_runs" ] || continue
      legacy_has_all=1
      for run_name in "${stats_names[@]}"; do
        grep -Fqx "## $run_name" "$legacy_stats" || {
          legacy_has_all=0
          break
        }
      done
      [ "$legacy_has_all" -eq 1 ] || continue

      cache_tmp="$stats_cache_out.tmp.$$"
      cp -- "$legacy_stats" "$cache_tmp" || exit 1
      mv -- "$cache_tmp" "$stats_cache_out" || exit 1
      report_cache_write "$stats_cache_marker" "$stats_key" \
        "$stats_cache_out" || exit 1
      cp -- "$stats_cache_out" "$STAMP_DIR/STATS.md" || exit 1
      stats_reused=1
      echo "[harvest] 이전 정상 STATS 검증 완료; 2,000-bootstrap 재계산 생략"
      break
    done < <(find "$READOUTS_ROOT" -mindepth 2 -maxdepth 2 -type f \
      -name STATS.md -printf '%T@ %p\n' | sort -nr | cut -d' ' -f2-)
  fi

  if [ "$stats_reused" -eq 0 ]; then
    : > "$stats_tmp"
    : > "$stats_err"
    for stats_index in "${!stats_dirs[@]}"; do
      d=${stats_dirs[$stats_index]}
      run_name=${stats_names[$stats_index]}
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
    if [ "$stats_failed" -eq 0 ] && [ -s "$stats_tmp" ]; then
      mv "$stats_tmp" "$STAMP_DIR/STATS.md"
      cache_tmp="$stats_cache_out.tmp.$$"
      cp -- "$STAMP_DIR/STATS.md" "$cache_tmp" || exit 1
      mv -- "$cache_tmp" "$stats_cache_out" || exit 1
      report_cache_write "$stats_cache_marker" "$stats_key" \
        "$stats_cache_out" || exit 1
    else
      mv "$stats_tmp" "$stats_partial"
    fi
    [ -s "$stats_err" ] || rm -f "$stats_err"
  fi
fi

# 결과 표는 선택 사항이지만, 파일이 존재하면 비어 있거나 복사에 실패한 상태를
# 정상 수확으로 숨기지 않는다.
copy_optional_report() {
  local label=$1 source=$2 destination=$3
  [ -e "$source" ] || return 0
  if [ ! -s "$source" ]; then
    failures+=("$label:empty-source")
    echo "[harvest] $label 실패 (empty-source): $source" >&2
    return 1
  fi
  if ! cp -- "$source" "$destination"; then
    failures+=("$label:copy-failed")
    echo "[harvest] $label 실패 (copy-failed): $source" >&2
    return 1
  fi
}

copy_required_report() {
  local label=$1 source=$2 destination=$3
  if [ ! -s "$source" ]; then
    failures+=("$label:missing-or-empty")
    echo "[harvest] $label 실패 (missing-or-empty): $source" >&2
    return 1
  fi
  if ! cp -- "$source" "$destination"; then
    failures+=("$label:copy-failed")
    echo "[harvest] $label 실패 (copy-failed): $source" >&2
    return 1
  fi
}

# 결과 폴더는 세대별(v2·v3·qwen3.8-27b 등)로 분리될 수 있다.
for rd in "$OM_WORK"/results/*/; do
  [ -d "$rd" ] || continue
  rtag=$(basename "$rd")
  copy_optional_report "tables:$rtag" "$rd/TABLES.md" \
    "$STAMP_DIR/TABLES-$rtag.md" || :
  copy_optional_report "frontier:$rtag" "$rd/FRONTIER.md" \
    "$STAMP_DIR/FRONTIER-$rtag.md" || :
  case "$rtag" in
    regime-*)
      copy_required_report "regime:$rtag:json" "$rd/REGIME.json" \
        "$STAMP_DIR/REGIME-$rtag.json" || :
      copy_required_report "regime:$rtag:csv" "$rd/REGIME.csv" \
        "$STAMP_DIR/REGIME-$rtag.csv" || :
      copy_required_report "regime:$rtag:summary" "$rd/REGIME_SUMMARY.csv" \
        "$STAMP_DIR/REGIME_SUMMARY-$rtag.csv" || :
      copy_required_report "regime:$rtag:report" "$rd/FINAL_REPORT.md" \
        "$STAMP_DIR/FINAL_REPORT-$rtag.md" || :
      copy_optional_report "regime:$rtag:collection" \
        "$rd/.regime_collection.json" \
        "$STAMP_DIR/REGIME_COLLECTION-$rtag.json" || :
      ;;
  esac
done
copy_optional_report "tables:legacy-repo" results/TABLES.md \
  "$STAMP_DIR/TABLES.md" || :

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

(
  cd "$STAMP_DIR" || exit 1
  find . -maxdepth 1 -type f ! -name HARVEST_MANIFEST.sha256 -print0 \
    | sort -z | xargs -0 sha256sum
) > "$STAMP_DIR/HARVEST_MANIFEST.sha256" || {
  echo "[harvest-abort] bundle manifest 생성 실패: $STAMP_DIR" >&2
  exit 1
}

current_tmp="$HARVEST_CURRENT.tmp.$$"
printf '%s\n%s\n' "$harvest_key" "$STAMP_DIR" > "$current_tmp" || exit 1
mv -- "$current_tmp" "$HARVEST_CURRENT" || exit 1

echo
echo "== 전달할 폴더 하나: $STAMP_DIR"
ls "$STAMP_DIR"
