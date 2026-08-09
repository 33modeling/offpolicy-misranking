#!/usr/bin/env bash
# 자가 회복 실행 — 게이트가 죽으면 자동으로 정리하고 이어서 재시작한다.
#
#   nohup bash scripts/babysit.sh fast > babysit.log 2>&1 &   # 이 한 줄이면 끝
#
# 완료($OUT_ROOT/DONE) 또는 최대 재시작 횟수(기본 12회)까지 5분 간격 감시.
set -uo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-}"
if [ "$MODE" = "fast" ]; then
  export DRIFTS="100" FRESH_K=16 HYBRID_PROMPTS=24 DOWNSTREAM_STEPS=100
  SUFFIX="-fast"
  echo "== 모드: FAST (경로: runs/gate-fast)"
else
  SUFFIX=""
  echo "== 모드: FULL — drift 50/100/200, FRESH_K=32, downstream 200 (경로: runs/gate)"
fi
source scripts/setup_env.sh
OUT_ROOT="${OUT_ROOT:-$OM_WORK/runs/gate$SUFFIX}"; export OUT_ROOT

# 싱글턴 — babysit이 여러 개 뜨면 서로의 실행을 죽였다고 오판해 재시작 폭풍이 난다
LOCK="$OM_WORK/.babysit.lock"
if [ -f "$LOCK/pid" ] && kill -0 "$(cat "$LOCK/pid" 2>/dev/null)" 2>/dev/null; then
  echo "이미 babysit 실행 중 (pid $(cat "$LOCK/pid")) — 추가 실행하지 않음. 상태는 scripts/check.sh"
  exit 0
fi
mkdir -p "$LOCK"; echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT
mkdir -p "$OUT_ROOT/logs"
BLOG="$OUT_ROOT/logs/babysit.log"
blog() { echo "[$(date '+%F %T')] $*" | tee -a "$BLOG"; }

MAX_RESTARTS="${MAX_RESTARTS:-12}"
restarts=0
blog "babysit 시작 (mode=${MODE:-full}, max_restarts=$MAX_RESTARTS)"

while true; do
  if [ -f "$OUT_ROOT/DONE" ]; then
    blog "완료 감지 — result.sh 실행 가능. babysit 종료"
    break
  fi
  if pgrep -f "scripts/run_h100_all.sh" >/dev/null || pgrep -f "src/experiment.py" >/dev/null; then
    LAST=$(tail -qn 1 "$OUT_ROOT"/logs/drift*.log "$OUT_ROOT"/logs/phase0-*.log 2>/dev/null | tail -1 | cut -c1-80)
    blog "생존 확인 — 진행: [$LAST]"
    sleep 300
    continue
  fi
  # 죽어 있음 — 재시작
  restarts=$((restarts + 1))
  if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
    blog "재시작 한도 초과($MAX_RESTARTS) — 포기. 마지막 로그 확인 필요"
    tail -n 15 "$OUT_ROOT/logs/main.log" 2>/dev/null | tee -a "$BLOG"
    exit 1
  fi
  blog "죽음 감지 → 재시작 #$restarts (미완성 산출물 정리 후 이어서)"
  bash scripts/reset_run.sh >> "$BLOG" 2>&1 || true
  nohup bash scripts/run_h100_all.sh > "$OUT_ROOT/logs/nohup-$restarts.out" 2>&1 &
  blog "재시작됨 (pid $!)"
  sleep 60
  if ! pgrep -f "scripts/run_h100_all.sh" >/dev/null; then
    blog "재시작 직후 사망 — 사유:"
    tail -n 6 "$OUT_ROOT/logs/nohup-$restarts.out" 2>/dev/null | tee -a "$BLOG"
    tail -n 4 "$OUT_ROOT/logs/main.log" 2>/dev/null | tee -a "$BLOG"
  fi
  sleep 240
done
