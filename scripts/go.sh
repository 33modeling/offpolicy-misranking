#!/usr/bin/env bash
# 원샷 재시작: soft 리셋 → 게이트 백그라운드 실행 → 로그 tail.
#   git pull && bash scripts/go.sh
set -uo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" = "fast" ]; then
  # 빠른 모드 — 산출물 경로를 분리해 full과 절대 섞이지 않는다
  export DRIFTS="100" FRESH_K=16 HYBRID_PROMPTS=24 DOWNSTREAM_STEPS=100
  export OUT_ROOT_SUFFIX="-fast"
  echo "== 모드: FAST (경로: runs/gate-fast)"
else
  echo "== 모드: FULL — drift 50/100/200, FRESH_K=32, downstream 200 (경로: runs/gate)"
fi
source scripts/setup_env.sh
OUT_ROOT="${OUT_ROOT:-$OM_WORK/runs/gate${OUT_ROOT_SUFFIX:-}}"; export OUT_ROOT

bash scripts/reset_run.sh
echo
echo "== 게이트 시작 (백그라운드, nohup)"
mkdir -p "$OUT_ROOT/logs"
nohup bash scripts/run_h100_all.sh > "$OUT_ROOT/logs/nohup.out" 2>&1 &
echo "PID $! — 창을 닫아도 계속 돈다"
sleep 3
echo "== 라이브 로그 (Ctrl+C 해도 실행은 유지됨)"
tail -f "$OUT_ROOT/logs/main.log"
