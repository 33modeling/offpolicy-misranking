#!/bin/sh
# V8 결과를 명령 하나로 파일 하나에: sh scripts/v8_export.sh
#   -> $WORK/v8-export/v8-export-<노드>-<시각>.tar.gz (+ .sha256)
# 안에 든 것:
#   v8-results.txt   worker가 만든 cohort별 analysis/report.md 모음 (scripts/v8_results.sh)
#   v8-recount.txt   현재 코드로 비용·예산 시점 재집계 (scripts/v8_recount.sh)
#   v8-explain.txt   위 재집계의 한글 풀이
#   recount-analysis/<cohort>/   재집계 analysis 폴더
#   v8-harvest-*.tar.gz          원자료 전체(체크포인트·rollout 캐시 제외, v8/experiments/harvest.sh)
#   steps.txt, logs/*.log        단계별 종료 코드와 출력
# 원래 결과(V8_ROOT)는 읽기만 한다. 한 단계가 실패해도 나머지를 끝까지 하고, 실패한 단계는 steps.txt와 끝 출력에 적는다.
set -u
export PYTHONDONTWRITEBYTECODE=1
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$HERE/.." && pwd)
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
V8_ROOT=${V8_ROOT:-$WORK/v8}
export V8_ROOT
HOST=$(hostname 2>/dev/null || echo node)
STAMP=$(date +%Y%m%d-%H%M%S)
NAME="v8-export-$HOST-$STAMP"
OUTDIR=${V8_EXPORT_OUT:-$WORK/v8-export}
mkdir -p "$OUTDIR" 2>/dev/null || OUTDIR=$HOME/v8-export
mkdir -p "$OUTDIR/$NAME/logs" || { echo "[v8-export] cannot create $OUTDIR/$NAME" >&2; exit 2; }
D="$OUTDIR/$NAME"
FAILED=
step() { # step <이름> <명령...>
    n=$1; shift
    echo "[v8-export] $n ..."
    "$@" > "$D/logs/$n.log" 2>&1
    rc=$?
    echo "$n exit=$rc" >> "$D/steps.txt"
    if [ "$rc" -ne 0 ]; then FAILED="$FAILED $n"; echo "[v8-export] $n FAILED (exit $rc), see logs/$n.log"; tail -5 "$D/logs/$n.log"; fi
}
{
    echo "host=$HOST at=$(date '+%Y-%m-%d %H:%M:%S %Z') v8_root=$V8_ROOT"
    echo "code=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
} > "$D/steps.txt"

step results env V8_RESULTS_OUT="$D/v8-results.txt" sh "$HERE/v8_results.sh"
step recount env V8_RECOUNT_OUT="$D/recount" V8_RECOUNT_TEXT="$D/v8-recount.txt" \
    V8_EXPLAIN_TEXT="$D/v8-explain.txt" sh "$HERE/v8_recount.sh"
# The recount mirror is symlinks into V8_ROOT; keep only its new analysis folders.
if [ -d "$D/recount" ]; then
    for a in "$D"/recount/*/analysis; do
        [ -d "$a" ] || continue
        c=$(basename "$(dirname "$a")")
        mkdir -p "$D/recount-analysis" && mv "$a" "$D/recount-analysis/$c"
    done
    rm -rf "$D/recount"
fi
step harvest sh "$REPO/v8/experiments/harvest.sh"
H=$(grep -m1 -E '/v8-harvest-[^ ]*\.tar\.gz$' "$D/logs/harvest.log" 2>/dev/null)
if [ -n "$H" ] && [ -f "$H" ]; then
    mv "$H" "$D/" && rm -f "$H.sha256" && (cd "$D" && sha256sum "$(basename "$H")" > "$(basename "$H").sha256")
fi

FINAL="$OUTDIR/$NAME.tar.gz"
if ! tar -czf "$FINAL" -C "$OUTDIR" "$NAME"; then
    echo "[v8-export] packing failed: $FINAL" >&2; exit 2
fi
(cd "$OUTDIR" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
echo
cat "$D/steps.txt"
echo "[v8-export] file: $FINAL ($(du -h "$FINAL" | cut -f1))"
if [ -n "$FAILED" ]; then echo "[v8-export] failed steps:$FAILED (the file still holds everything else)"; exit 1; fi
echo "[v8-export] all steps ok"
