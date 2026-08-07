#!/usr/bin/env bash
# 스모크 — 0.5B, 프롬프트 8개, K=4, 짧은 생성으로 전 stage 완주만 확인 (GPU 권장, CPU 가능).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh 2>/dev/null || true
PY="${VENV_DIR:+$VENV_DIR/bin/python}"; [ -x "${PY:-/none}" ] || PY=python3
export PYTHONPATH=src
RUN="outputs/smoke"
MODEL="${MODEL:-${MODEL_QWEN25_05B:-Qwen/Qwen2.5-0.5B-Instruct}}"
if [ -n "${MODEL_QWEN25_05B:-}" ] && [ ! -d "$MODEL" ]; then MODEL="Qwen/Qwen2.5-0.5B-Instruct"; fi
COMMON=(--run "$RUN" --model "$MODEL" --n-train 8 --n-val 4
        --behavior-k 4 --fresh-k 8 --val-k 4 --micro-group 2
        --max-new-tokens 384 --temperature 0.7
        --drift-steps 10 --proj-dim 512 --grad-layers 2
        --topk-frac 0.25)

"$PY" src/experiment.py --stage prep             "${COMMON[@]}"
"$PY" src/experiment.py --stage rollout-behavior "${COMMON[@]}"
"$PY" src/experiment.py --stage drift            "${COMMON[@]}"
"$PY" src/experiment.py --stage oracle           "${COMMON[@]}" --adapter "$RUN/drift_10"
"$PY" src/experiment.py --stage score            "${COMMON[@]}" --adapter "$RUN/drift_10"
"$PY" src/experiment.py --stage report           "${COMMON[@]}"
echo "SMOKE OK"
