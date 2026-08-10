# 공용: 산출물 루트 결정 — source 해서 쓴다 (OUT_ROOT 확정·export)
# 사용: source scripts/_find_root.sh [7b|14b|fast|<경로>]
#   인자가 있으면 그 대상 고정, 없으면 최근 산출물 자동 탐색(후보 목록 출력).
_arg="${1:-${OM_TARGET:-}}"
case "$_arg" in
  7b)   OUT_ROOT="$OM_WORK/runs/gate" ;;
  14b)  OUT_ROOT="$OM_WORK/runs/gate-14b" ;;
  fast) OUT_ROOT="$OM_WORK/runs/gate-fast" ;;
  "")   ;;
  *)    OUT_ROOT="$_arg" ;;
esac
if [ -z "${OUT_ROOT:-}" ]; then
  _om_candidates() {
    ls -dt "$OM_WORK"/runs/* 2>/dev/null
    ls -dt outputs/* 2>/dev/null
  }
  for c in $(_om_candidates); do
    [ -d "$c" ] || continue
    if [ -f "$c/logs/main.log" ] || ls "$c"/drift*/ >/dev/null 2>&1        || [ -f "$c/scores_oracle.json" ]; then
      OUT_ROOT="$c"; break
    fi
  done
  OUT_ROOT="${OUT_ROOT:-$OM_WORK/runs/gate}"
  echo "== 대상 자동 선택: $OUT_ROOT"
  echo "   (다른 대상: 7b/14b/fast 인자로 지정 — 예: bash scripts/result.sh 7b)"
else
  echo "== 대상: $OUT_ROOT"
fi
export OUT_ROOT
