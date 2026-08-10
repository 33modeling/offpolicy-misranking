#!/usr/bin/env bash
# 7B MATH-500 게이트 시작/재개 원샷:  bash scripts/go7m.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
export MODEL_14B="$MODELS_DIR/Qwen2.5-7B-Instruct"
export DATASET=math500 FRESH_K="${FRESH_K:-32}" OUT_ROOT="$OM_WORK/runs/gate-7b"
nohup bash scripts/run_14b.sh > 7b-math.log 2>&1 &
sleep 8
head -6 7b-math.log
echo "== 진행: tail -3 7b-math.log / 결과: bash scripts/result.sh 7bm"
