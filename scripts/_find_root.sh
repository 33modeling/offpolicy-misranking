# 공용: 산출물 루트 자동 탐색 — source 해서 쓴다 (OUT_ROOT를 확정해 export)
# 우선순위: 명시된 OUT_ROOT → 로그/산출물이 있는 후보 중 가장 최근 것
_om_candidates() {
  [ -n "${OUT_ROOT:-}" ] && printf '%s\n' "$OUT_ROOT"
  ls -dt "$OM_WORK"/runs/* 2>/dev/null
  ls -dt outputs/* 2>/dev/null
}
_om_pick_root() {
  local c
  while IFS= read -r c; do
    [ -d "$c" ] || continue
    if [ -f "$c/logs/main.log" ] || ls "$c"/drift*/ >/dev/null 2>&1 \
       || [ -f "$c/scores_oracle.json" ]; then
      printf '%s\n' "$c"; return 0
    fi
  done < <(_om_candidates)
  return 1
}
if ROOT_PICKED="$(_om_pick_root)"; then
  OUT_ROOT="$ROOT_PICKED"
else
  OUT_ROOT="${OUT_ROOT:-$OM_WORK/runs/gate}"
fi
export OUT_ROOT
echo "== 산출물 위치: $OUT_ROOT"
