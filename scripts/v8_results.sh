#!/bin/sh
# V8 결과만 파일 하나로: sh scripts/v8_results.sh
# worker가 collect·analyze를 이미 돌려 둔 analysis/report.md 를 코호트별로 모아 ~/v8-results.txt 에 쓴다.
# 계산은 하지 않는다(읽기 전용). 결과가 아직 없는 코호트는 그렇다고 적는다.
set -eu
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
V8_ROOT=${V8_ROOT:-$WORK/v8}
OUT=${V8_RESULTS_OUT:-$HOME/v8-results.txt}
TMP="$OUT.tmp.$$"
{
    echo "# V8 results  host=$(hostname)  at=$(date '+%Y-%m-%d %H:%M:%S %Z')  root=$V8_ROOT"
    for cohort in olmo_math qwen_mbpp olmo_mbpp qwen_math; do
        echo
        echo "################ $cohort ################"
        a="$V8_ROOT/$cohort/analysis"
        if [ ! -f "$a/report.md" ]; then
            echo "no result yet: $a/report.md missing"
            continue
        fi
        echo "## latest analysis ($(date -r "$a/report.md" '+%Y-%m-%d %H:%M'))"
        cat "$a/report.md"
        for d in "$a"/matched-update-*; do
            [ -f "$d/report.md" ] || continue
            echo
            echo "## primary result $(basename "$d") (frozen copy)"
            cat "$d/report.md"
        done
    done
} > "$TMP"
mv "$TMP" "$OUT"
cat "$OUT"
echo
echo "[v8] saved: $OUT"
