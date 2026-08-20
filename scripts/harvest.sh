#!/usr/bin/env bash
# 수확 원스톱 — 한 번 실행하고 마지막 줄에 찍히는 폴더 하나만 전달하면 끝:
#   KCURVE(GPU 0, 수 분) + READOUT + REVERSAL(닻 포함) + STATS + 표 사본을 같은 폴더에.
#   bash scripts/harvest.sh
#
# 0820 수확 교훈: 실패를 숨기면(2>/dev/null, || true) 빈 파일이 성공처럼 전달된다
# (READOUT.md 0바이트). 이제 stderr는 logs/ 에 남기고, 단계별 종료코드·산출물
# 크기·경고 수를 마지막에 표로 찍는다. 빈 산출물이 있으면 크게 경고한다.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
STAMP_DIR="$OM_WORK/readouts/$(date '+%Y-%m-%d_%H%M')-harvest"
mkdir -p "$STAMP_DIR/logs"

STATUS=()
run_step() {  # run_step <이름> <출력파일> <명령...>
  local name="$1" out="$2"; shift 2
  local log="$STAMP_DIR/logs/$name.log" rc=0 sz=0 errs=0
  echo "== [$name] 생성 중 → $(basename "$out")"
  "$@" > "$out" 2> "$log" || rc=$?
  sz=$(wc -c < "$out" 2>/dev/null || echo 0)
  errs=$(grep -ci "error\|traceback\|없음\|경고" "$log" 2>/dev/null | head -1)
  errs=${errs:-0}
  if [ "$rc" -ne 0 ] || [ "$sz" -lt 50 ]; then
    echo "   [실패] rc=$rc, ${sz}바이트 — 로그 마지막 5줄:"
    tail -5 "$log" 2>/dev/null | sed 's/^/     /'
  elif [ "$errs" -gt 0 ]; then
    echo "   [경고] ${sz}바이트, 로그에 경고 ${errs}건 (logs/$name.log)"
  fi
  STATUS+=("$name|$rc|$sz|$errs")
}

run_step KCURVE   "$STAMP_DIR/KCURVE.md"   "$PY" src/kcurve_floor.py    "$OM_WORK/runs"
run_step READOUT  "$STAMP_DIR/READOUT.md"  "$PY" src/readout_summary.py "$OM_WORK/runs"
run_step REVERSAL "$STAMP_DIR/REVERSAL.md" "$PY" src/reversal_freq.py   "$OM_WORK/runs"

# 원고 A8a — run별 정확 p·부트스트랩 CI. run 하나가 실패해도 파일에 기록하고 계속.
echo "== [STATS] 생성 중 → STATS.md"
{
  found=0
  for d in "$OM_WORK"/runs/*/; do
    [ -f "$d/scores_oracle.json" ] || continue
    case "$(basename "$d")" in *smoke*) continue;; esac
    found=$((found + 1))
    echo "## $(basename "$d")"
    "$PY" src/stats_extra.py "$d" 2>&1 || echo "  [실패] stats_extra 비정상 종료"
    echo
  done
  [ "$found" -gt 0 ] || echo "[실패] scores_oracle.json을 가진 run이 없다: $OM_WORK/runs"
} > "$STAMP_DIR/STATS.md" 2> "$STAMP_DIR/logs/STATS.log"
s_sz=$(wc -c < "$STAMP_DIR/STATS.md" 2>/dev/null || echo 0)
s_er=$(grep -c "실패\|경고" "$STAMP_DIR/STATS.md" 2>/dev/null | head -1)
s_er=${s_er:-0}
STATUS+=("STATS|0|$s_sz|$s_er")
[ "$s_er" -gt 0 ] && echo "   [경고] STATS.md에 실패·경고 ${s_er}건"

# 결과 폴더는 세대별(v2·v3·qwen3.8-27b 등)로 분리될 수 있다 — 전부 태그 붙여 동봉
copied=0
for rd in "$OM_WORK"/results/*/; do
  [ -d "$rd" ] || continue
  rtag=$(basename "$rd")
  for f in TABLES FRONTIER; do
    [ -f "$rd/$f.md" ] && { cp "$rd/$f.md" "$STAMP_DIR/$f-$rtag.md"; copied=$((copied + 1)); }
  done
done
[ "$copied" -gt 0 ] || echo "   [경고] results/*/ 에서 복사한 표가 없다 — tables.sh·frontier.sh를 먼저 돌렸는지 확인"
STATUS+=("표사본|0|$copied|0")

echo
echo "== 수확 상태 요약"
printf "%-10s %5s %10s %6s\n" 단계 종료 크기 경고
for row in "${STATUS[@]}"; do
  IFS='|' read -r n rc sz er <<< "$row"
  printf "%-10s %5s %10s %6s\n" "$n" "$rc" "$sz" "$er"
done
bad=$(printf '%s\n' "${STATUS[@]}" | awk -F'|' '$1!="표사본" && ($2!=0 || $3<50) {c++} END{print c+0}')
echo
[ "$bad" -gt 0 ] && echo "!! 실패·빈 산출물 ${bad}건 — 그대로 전달하지 말고 logs/ 를 먼저 볼 것"
echo "== 전달할 폴더 하나: $STAMP_DIR"
ls "$STAMP_DIR"
