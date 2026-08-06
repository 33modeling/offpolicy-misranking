#!/usr/bin/env bash
# H100 게이트 파일럿 — concept #68 10절 셋업의 축소판 (1 drift 수준).
# 사용: bash scripts/run_gate.sh [RUN_DIR] [DRIFT_STEPS]
# 순서: prep → β rollout → drift(LoRA RFT) → oracle(π fresh) → score(2×2) → report
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
RUN="${1:-outputs/h100-pilot}"
DRIFT="${2:-100}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

python3 src/experiment.py --stage prep             --run "$RUN" --model "$MODEL"
python3 src/experiment.py --stage rollout-behavior --run "$RUN" --model "$MODEL"
python3 src/experiment.py --stage drift            --run "$RUN" --model "$MODEL" --drift-steps "$DRIFT"
python3 src/experiment.py --stage oracle           --run "$RUN" --model "$MODEL" --adapter "$RUN/drift_$DRIFT"
python3 src/experiment.py --stage score            --run "$RUN" --model "$MODEL" --adapter "$RUN/drift_$DRIFT"
python3 src/experiment.py --stage report           --run "$RUN"
