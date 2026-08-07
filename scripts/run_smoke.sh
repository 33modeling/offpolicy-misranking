#!/usr/bin/env bash
# 스모크 — 0.5B, 프롬프트 8개, K=4, 짧은 생성으로 전 stage 완주만 확인 (GPU 권장, CPU 가능).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
RUN="outputs/smoke"
MODEL="${MODEL:-${MODEL_QWEN25_05B:-Qwen/Qwen2.5-0.5B-Instruct}}"
if [ -n "${MODEL_QWEN25_05B:-}" ] && [ ! -d "$MODEL" ]; then MODEL="Qwen/Qwen2.5-0.5B-Instruct"; fi
COMMON=(--run "$RUN" --model "$MODEL" --n-train 8 --n-val 4
        --behavior-k 4 --fresh-k 8 --val-k 4 --micro-group 2
        --max-new-tokens 384 --temperature 0.7
        --drift-steps 10 --proj-dim 512 --grad-layers 2
        --topk-frac 0.25)

python3 src/experiment.py --stage prep             "${COMMON[@]}"
python3 src/experiment.py --stage rollout-behavior "${COMMON[@]}"
python3 src/experiment.py --stage drift            "${COMMON[@]}"
python3 src/experiment.py --stage oracle           "${COMMON[@]}" --adapter "$RUN/drift_10"
python3 src/experiment.py --stage score            "${COMMON[@]}" --adapter "$RUN/drift_10"
python3 src/experiment.py --stage report           "${COMMON[@]}"
echo "SMOKE OK"
