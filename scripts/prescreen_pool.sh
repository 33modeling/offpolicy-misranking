#!/usr/bin/env bash
# β pass-rate 프리스크린 → hard-slice 풀 생성 (27B G블록의 난이도 매칭 관문).
#   MODEL=<모델 경로> bash scripts/prescreen_pool.sh [dataset=dapo-math] [POOL_N=2000]
# 절차: prep(POOL_N개) → β rollout(K=8, GPU 샤딩) → make_hard_pool(0<rate<1)
# 산출: $OM_WORK/pools/<dataset>-hard.jsonl  → 본실행에서 OM_POOL_FILE로 주입.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || { echo "[abort] venv 없음"; exit 1; }

DS="${1:-dapo-math}"
POOL_N="${2:-${POOL_N:-2000}}"
MODEL="${MODEL:?MODEL=<모델 경로> 필요}"
TAG="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
RUN="$OM_WORK/runs/prescreen-$DS-$TAG"
OUT="${OUT:-$OM_WORK/pools/$DS-hard.jsonl}"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
mkdir -p "$RUN"

COMMON=(--run "$RUN" --model "$MODEL" --dataset "$DS" --n-train "$POOL_N" --n-val 8 --micro-batch 1)
echo "== prescreen: $DS × $POOL_N, model=$TAG, GPU ${NGPU}장"
"$PY" src/experiment.py --stage prep "${COMMON[@]}" || exit 1

pids=()
for i in $(seq 0 $((NGPU - 1))); do
  ( CUDA_VISIBLE_DEVICES=$i "$PY" src/experiment.py --stage rollout-behavior \
      "${COMMON[@]}" --shard "$i:$NGPU" > "$RUN/prescreen-shard$i.log" 2>&1 ) &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "[abort] rollout 샤드 실패 — $RUN/prescreen-shard*.log 확인 (재실행 시 완료 샤드 스킵)"; exit 1; }

"$PY" src/make_hard_pool.py "$RUN" "$OUT" 0.0 1.0
