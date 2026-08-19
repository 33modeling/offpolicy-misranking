#!/usr/bin/env bash
# v3 = P0-1·P0-2 계약 수정(c6ca013) 이후 7B 본실험 재생성 — 원커맨드.
#   bash scripts/go_v3.sh                        # gsm8k+dapo (s1,s2) → math500 → mbpp·kk(s0)
#   SEEDS_V3="0 1 2 3 4" bash scripts/go_v3.sh   # 시드 확장
#   OM_SKIP_EXTRA=1 bash scripts/go_v3.sh        # mbpp·kk 생략 (결정점 최소셋만)
# 산출물은 runs/v3·results/v3로 격리 — 수정 전 v2 완주분과 절대 섞이지 않는다.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh

export RUN_BASE="$OM_WORK/runs/v3" RESULTS_BASE="$OM_WORK/results/v3"
S="${SEEDS_V3:-1 2}"

echo "== go_v3: RUN_BASE=$RUN_BASE, SEEDS=[$S]"

# 1) 결정점 최소셋 — 본문 표를 떠받치는 풀부터
SEEDS="$S" DATASETS="gsm8k dapo-math" bash scripts/go_v2.sh || exit 1

# 2) math500
SEEDS="$S" DATASETS="math500" N_TRAIN=400 N_VAL=100 bash scripts/go_v2.sh || exit 1

# 3) 도메인 다각화 1-seed (B10)
if [ "${OM_SKIP_EXTRA:-0}" != "1" ]; then
  SEEDS="0" DATASETS="mbpp" bash scripts/go_v2.sh || exit 1
  SEEDS="0" DATASETS="kk" bash scripts/go_v2.sh || exit 1
fi

echo "== go_v3 완료 — git pull && bash scripts/harvest.sh 로 수확"
