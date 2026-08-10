#!/usr/bin/env bash
# 14B 규모 확인 실험 — 7B 게이트 통과 후의 scale confirmation.
#
# 설계 (풀 게이트 축소판):
#   · 모델: Qwen2.5-14B-Instruct — group-volume 로컬 스냅샷 전용 (다운로드 안 함)
#   · drift 100 단일 / fresh K=16 / hybrid 3절단점 / downstream 없음(C1·C1' 중심)
#   · 판정 대상: 7B 대비 g10/g01의 Δfloor가 커지는가 작아지는가 (스케일 축)
#   · GPU 배치: phase0 β rollout 4샤딩 → drift(1 GPU) → fresh rollout 4샤딩 → analyze
#
#   bash scripts/run_14b.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
MODEL_14B="${MODEL_14B:-$MODELS_DIR/Qwen2.5-14B-Instruct}"
[ -f "$MODEL_14B/config.json" ] || { echo "[abort] 14B 로컬 스냅샷 없음: $MODEL_14B — failure-atlas 규약대로 미러 경로를 MODEL_14B로 지정하거나 자산을 먼저 확보할 것"; exit 1; }
OUT_ROOT="${OUT_ROOT:-$OM_WORK/runs/gate-14b}"; export OUT_ROOT
LOGS="$OUT_ROOT/logs"; mkdir -p "$LOGS"
# GPU 수 자동 감지 — 인스턴스마다 다름 (4장 하드코딩이 invalid device ordinal의 원인)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
[ "$NGPU" -ge 1 ] || { echo "[abort] GPU 미감지"; exit 1; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/main.log"; }
# 다른 실행(7B babysit/run)과의 GPU 충돌 차단
if pgrep -f "bash.*scripts/babysit.sh" >/dev/null || pgrep -f "scripts/run_h100_all.sh" >/dev/null; then
  echo "[abort] 7B babysit/run_h100_all 이 아직 실행 중 — 14B와 GPU가 충돌한다."
  echo "        먼저:  pkill -f babysit.sh; pkill -f run_h100_all; pkill -f 'src/experiment.py'; pkill -f gpu_keepalive"
  exit 1
fi
BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 > 2000' | wc -l)
if [ "${BUSY:-0}" -gt 0 ] && [ "${OM_SKIP_GPU_CHECK:-0}" != "1" ]; then
  echo "[abort] GPU ${BUSY}개가 이미 점유 중:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
  exit 1
fi
run_stage() { local gpu="$1" lf="$2"; shift 2
  local t0=$SECONDS
  log "GPU$gpu ▶ $*"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/experiment.py "$@" >> "$lf" 2>&1; then
    log "GPU$gpu ✔ $1 $2 ($((SECONDS - t0))s)"
  else local rc=$?; log "GPU$gpu ✘ $* rc=$rc"; tail -8 "$lf" | tee -a "$LOGS/main.log"; return $rc; fi
}
DRIFT=100
COMMON=(--run "$OUT_ROOT" --model "$MODEL_14B" --fresh-k "${FRESH_K:-16}" --hybrid-prompts "${HYBRID_PROMPTS:-24}" --micro-batch 1)

"$PY" scripts/gpu_keepalive.py > "$LOGS/keepalive.log" 2>&1 &
KEEP=$!
# 종료(정상·에러 모두) 시 이 실행이 띄운 모든 자식 정리 — 고아 샤드가 GPU를 점유한 채
# 남아 "계속 실행되는 것처럼" 보이고 재시작 OOM을 일으키는 버그의 수정
cleanup() {
  kill "$KEEP" 2>/dev/null || true
  pkill -f -- "--run $OUT_ROOT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

log "=== 14B 시작: $MODEL_14B → $OUT_ROOT (GPU ${NGPU}장) ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv | tee -a "$LOGS/main.log" || true
run_stage 0 "$LOGS/prep.log" --stage prep "${COMMON[@]}"
# β rollout N샤딩
pids=(); for i in $(seq 0 $((NGPU - 1))); do
  ( run_stage "$i" "$LOGS/beta-shard$i.log" --stage rollout-behavior "${COMMON[@]}" --shard "$i:$NGPU" ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || exit 1; done
[ -f "$OUT_ROOT/rollouts_behavior_train.jsonl" ] || cat "$OUT_ROOT"/rollouts_behavior_train.shard*.jsonl > "$OUT_ROOT/rollouts_behavior_train.jsonl"
run_stage 0 "$LOGS/drift.log" --stage drift "${COMMON[@]}" --drift-steps "$DRIFT"
# π fresh N샤딩
pids=(); for i in $(seq 0 $((NGPU - 1))); do
  ( run_stage "$i" "$LOGS/fresh-shard$i.log" --stage rollout-fresh "${COMMON[@]}" --adapter "$OUT_ROOT/drift_$DRIFT" --shard "$i:$NGPU" ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || exit 1; done
[ -f "$OUT_ROOT/rollouts_fresh_train.jsonl" ] || cat "$OUT_ROOT"/rollouts_fresh_train.shard*.jsonl > "$OUT_ROOT/rollouts_fresh_train.jsonl"
# val 방향 ∥ oracle micro 샤딩 (GPU 여유가 있으면 마지막 GPU를 val 전용으로)
pids=()
if [ "$NGPU" -ge 2 ]; then
  NM=$((NGPU - 1))
  ( run_stage "$NM" "$LOGS/val-grads.log" --stage val-grads "${COMMON[@]}" --adapter "$OUT_ROOT/drift_$DRIFT" ) & pids+=($!)
else
  NM=1
  run_stage 0 "$LOGS/val-grads.log" --stage val-grads "${COMMON[@]}" --adapter "$OUT_ROOT/drift_$DRIFT" || exit 1
fi
for i in $(seq 0 $((NM - 1))); do
  ( run_stage "$i" "$LOGS/ograds-shard$i.log" --stage oracle-grads "${COMMON[@]}" --adapter "$OUT_ROOT/drift_$DRIFT" --shard "$i:$NM" ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || exit 1; done
# 2×2 score N샤딩
pids=(); for i in $(seq 0 $((NGPU - 1))); do
  ( run_stage "$i" "$LOGS/score-shard$i.log" --stage score-shard "${COMMON[@]}" --adapter "$OUT_ROOT/drift_$DRIFT" --shard "$i:$NGPU" ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || exit 1; done
run_stage 0 "$LOGS/merge.log" --stage merge-grads "${COMMON[@]}"
run_stage 0 "$LOGS/report.log" --stage report "${COMMON[@]}"
# hybrid 3절단점 — GPU 수만큼 병렬 (모자라면 라운드로빈 순차)
pids=(); gpu=0
for cut in 0.25 0.5 0.75; do
  ( run_stage "$((gpu % NGPU))" "$LOGS/hybrid-$cut.log" --stage hybrid "${COMMON[@]}" --adapter "$OUT_ROOT/drift_$DRIFT" --cut-frac "$cut" ) & pids+=($!)
  gpu=$((gpu + 1))
  [ $((gpu % NGPU)) -eq 0 ] && for p in "${pids[@]}"; do wait "$p" || exit 1; done && pids=()
done
for p in "${pids[@]}"; do wait "$p" || exit 1; done
log "=== 14B 완료 — bash scripts/result.sh 로 판정 (OUT_ROOT=$OUT_ROOT) ==="
touch "$OUT_ROOT/DONE"
