#!/usr/bin/env bash
# Dr.GRPO and sequence-level RLOO: two model families x math/code x three seeds.
set -euo pipefail

cd "$(dirname "$0")/.."
export GENERALIZATION_CONFIG=configs/method_robustness.json
export GENERALIZATION_RUN_ID=method-dr-grpo-v1
bash scripts/run_generalization.sh

export GENERALIZATION_CONFIG=configs/method_rloo.json
export GENERALIZATION_RUN_ID=method-rloo-v1
exec bash scripts/run_generalization.sh
