#!/usr/bin/env bash
# 원샷 진단 리포트 — 원인 판정에 필요한 모든 것을 DIAGNOSIS.txt 하나로 수집.
# 실패가 반복되면 이것만 실행하고 파일(또는 화면) 전체를 사진으로 전달하면 된다.
#   bash scripts/diagnose.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh 2>/dev/null || true
OUT="DIAGNOSIS.txt"
{
  echo "===== 진단 리포트 $(date '+%F %T') ====="
  echo "-- 코드: $(git -c safe.directory='*' log --oneline -1 2>/dev/null || echo 'git 정보 없음')"
  echo "-- GPU 상태:"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  (nvidia-smi 불가)"
  echo "-- 디스크: $(df -h "${OM_WORK:-.}" 2>/dev/null | tail -1)"
  echo "-- 완주 현황:"
  for d in "${OM_WORK:-.}"/runs/v2-*; do
    [ -d "$d" ] && echo "  $(basename "$d"): $([ -f "$d/DONE" ] && echo DONE || echo 미완)"
  done
  echo "-- 에러 시그니처 집계 (전 로그, 빈도순):"
  grep -hiE "cublas|cuda error|out of memory|illegal|launch fail|xid|no space|killed|assert" \
    ./*.log "${OM_WORK:-.}"/runs/v2-*/logs/*.log 2>/dev/null \
    | sed 's/^[[:space:]]*//' | cut -c1-110 | sort | uniq -c | sort -rn | head -12
  echo "-- dmesg Xid (하드웨어 판정, 권한 없으면 생략):"
  dmesg 2>/dev/null | grep -i xid | tail -5 || echo "  (dmesg 접근 불가)"
  echo "-- 최근 로그 tail:"
  lf=$(ls -t go_full.console.log v2-*.log boost-*.log 2>/dev/null | head -1)
  if [ -n "${lf:-}" ]; then echo "  [$lf]"; tail -12 "$lf" | sed 's/^/  /'; fi
  echo "===== 끝 — 이 출력 전체를 사진으로 전달 ====="
} | tee "$OUT"
