#!/usr/bin/env bash
# 14B MATH-500 게이트 시작/재개 원샷 (이 노드는 eager 필수):  bash scripts/go14m.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
LOGDIR="${OM_WORK:-.}/console-logs"; mkdir -p "$LOGDIR"
export OM_ATTN="${OM_ATTN:-eager}" DATASET=math500
nohup bash scripts/run_14b.sh > "$LOGDIR/14b-math.log" 2>&1 &
sleep 8
head -6 "$LOGDIR/14b-math.log"
echo "== 진행: tail -3 "$LOGDIR/14b-math.log" / 결과: bash scripts/result.sh 14bm"
