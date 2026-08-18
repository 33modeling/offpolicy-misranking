#!/usr/bin/env bash
# 최신 세대 모델 1-seed 검증 (BACKLOG B11) — 메인 v2와 동일 프로토콜, 별도 폴더 격리.
#   bash scripts/go_new.sh                                    # 기본: Qwen3.8-27B
#   REPO27B=Qwen/Qwen3.6-27B bash scripts/go_new.sh           # 스모크 실패 시 폴백
# 모델 스냅샷이 없으면 fetch_27b.sh(미러 폴백·이어받기)로 먼저 받는다.
# 30분 스모크 게이트는 go_v2에 내장 — 신아키텍처 호환은 거기서 판정된다.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh

REPO="${REPO27B:-Qwen/Qwen3.8-27B-BF16}"
NAME="$(basename "$REPO")"
export MODEL_14B="$MODELS_DIR/$NAME"
if [ ! -f "$MODEL_14B/config.json" ]; then
  echo "== 스냅샷 없음 → fetch: $REPO"
  REPO27B="$REPO" bash scripts/fetch_27b.sh || exit 1
fi

TAG="$(echo "$NAME" | tr '[:upper:]' '[:lower:]')"
export RUN_BASE="$OM_WORK/runs/$TAG" RESULTS_BASE="$OM_WORK/results/$TAG"
export OM_LORA_TARGETS="${OM_LORA_TARGETS:-all-linear}"   # DeltaNet 층 대응
S="${SEEDS_NEW:-0}"

# 1) gsm8k — 자연 풀 그대로. 27B가 거의 다 맞추는 게 정상(포화 표본):
#    "능력이 오르면 체제가 포화로 이동한다"는 본문 서사의 최신 모델 실증용.
SEEDS="$S" DATASETS="gsm8k" bash scripts/go_v2.sh || exit 1

# 2) dapo-math — 자연 풀은 27B가 다 풀어 신호가 죽는다 → hard-slice 프리스크린
#    (모델로 선채점해 못 푸는 문제만 추려 풀 구성) 후 주입. 유신호 체제 담당.
POOL="$OM_WORK/pools/dapo-math-hard.jsonl"
if [ ! -s "$POOL" ]; then
  echo "== dapo-math hard-slice 프리스크린 (POOL_N=${POOL_N:-2000})"
  MODEL="$MODEL_14B" bash scripts/prescreen_pool.sh dapo-math "${POOL_N:-2000}" || exit 1
fi
OM_POOL_FILE="$POOL" RUN_BASE="$OM_WORK/runs/$TAG-hard" RESULTS_BASE="$OM_WORK/results/$TAG-hard" \
  SEEDS="$S" DATASETS="dapo-math" bash scripts/go_v2.sh || exit 1

# 3) math500 — 총 500문제뿐이라 hard-slice 불가, 자연 풀로 나오는 체제 그대로 기록.
SEEDS="$S" DATASETS="math500" N_TRAIN=400 N_VAL=100 bash scripts/go_v2.sh
echo "== 완료 — 수확은 평소대로: bash scripts/harvest.sh (한 폴더에 v2와 함께 동봉됨)"
