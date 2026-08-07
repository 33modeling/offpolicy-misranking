#!/usr/bin/env bash
# 원샷 재시작: soft 리셋 → 게이트 백그라운드 실행 → 로그 tail.
#   git pull && bash scripts/go.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
OUT_ROOT="${OUT_ROOT:-$OM_WORK/runs/gate}"; export OUT_ROOT

bash scripts/reset_run.sh
echo
echo "== 게이트 시작 (백그라운드, nohup)"
mkdir -p "$OUT_ROOT/logs"
nohup bash scripts/run_h100_all.sh > "$OUT_ROOT/logs/nohup.out" 2>&1 &
echo "PID $! — 창을 닫아도 계속 돈다"
sleep 3
echo "== 라이브 로그 (Ctrl+C 해도 실행은 유지됨)"
tail -f "$OUT_ROOT/logs/main.log"
