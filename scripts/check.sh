#!/usr/bin/env bash
# 3줄 진단 — 돌고 있는지, 멈췄는지, 뭘 해야 하는지 결론까지 출력.
#   bash scripts/check.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1
source scripts/_find_root.sh

NPROC=$(pgrep -fc "python.*src/experiment.py" 2>/dev/null || true); NPROC=${NPROC:-0}
NBABY=$(pgrep -fc "bash.*scripts/babysit.sh" 2>/dev/null || true); NBABY=${NBABY:-0}

# 가장 최근 로그 파일과 나이(분)
NEWEST=$(ls -t "$OUT_ROOT"/logs/*.log 2>/dev/null | head -1)
if [ -n "${NEWEST:-}" ]; then
  AGE=$(( ($(date +%s) - $(stat -c %Y "$NEWEST")) / 60 ))
  LASTLINE=$(tail -n 1 "$NEWEST" | cut -c1-90)
else
  AGE=99999; LASTLINE="(로그 없음)"
fi
echo "상태: 프로세스=${NPROC}개, babysit=${NBABY}개, 최신로그=${AGE}분 전 [$(basename "${NEWEST:-없음}")]"
echo "마지막: $LASTLINE"

if [ -f "$OUT_ROOT/DONE" ] || ls "$OUT_ROOT"/drift*/report.json "$OUT_ROOT"/report.json >/dev/null 2>&1; then
  echo "결론: ✅ 완료 — bash scripts/result.sh 실행"
elif [ "$NPROC" -gt 0 ] && [ "$AGE" -le 15 ]; then
  echo "결론: 🟢 정상 진행 중 — 아무것도 하지 말 것"
elif [ "$NPROC" -gt 0 ]; then
  echo "결론: 🟡 프로세스는 있는데 로그가 ${AGE}분째 조용 — 15분 뒤 다시 check. 계속되면 멈춘 것"
elif [ "$NBABY" -gt 0 ]; then
  echo "결론: 🟡 babysit이 재시작 대기 중(최대 5분 주기) — 10분 뒤 다시 check. 그래도 프로세스 0이면:"
  grep -E "abort|실패|사망" "$OUT_ROOT"/logs/main.log 2>/dev/null | tail -3
else
  echo "결론: 🔴 완전히 죽음 — 마지막 사유:"
  tail -n 5 "$OUT_ROOT"/logs/main.log 2>/dev/null
  echo "재시작: nohup bash scripts/babysit.sh > babysit.log 2>&1 &"
fi
