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
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/main.log"; }
run_stage() { local gpu="$1" lf="$2"; shift 2
  local t0=$SECONDS
  log "GPU$gpu ▶ $*"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/experiment.py "$@" >> "$lf" 2>&1; then
    log "GPU$gpu ✔ $1 $2 ($((SECONDS - t0))s)"
  else local rc=$?; log "GPU$gpu ✘ $* rc=$rc"; tail -8 "$lf" | tee -a "$LOGS/main.log"; return $rc; fi
}
DRIFT=100
COMMON=(--run "$OUT_ROOT" --model "$MODEL_14B" --fresh-k "${FRESH_K:-16}" --hybrid-prompts "${HYBRID_PROMPTS:-24}")

"$PY" scripts/gpu_keepalive.py > "$LOGS/keepalive.log" 2>&1 &
KEEP=$!; trap 'kill $KEEP 2>/dev/null' EXIT

log "=== 14B 시작: $MODEL_14B → $OUT_ROOT ==="
run_stage 0 "$LOGS/prep.log" --stage prep "${COMMON[@]}"
# β rollout 4샤딩
pids=(); for i in 0 1 2 3; do
  ( run_stage "$i" "$LOGS/beta-shard$i.log" --stage rollout-behavior "${COMMON[@]}" --shard "$i:4" ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || exit 1; done
[ -f "$OUT_ROOT/rollouts_behavior_train.jsonl" ] || cat "$OUT_ROOT"/rollouts_behavior_train.shard*.jsonl > "$OUT_ROOT/rollouts_behavior_train.jsonl"
run_stage 0 "$LOGS/drift.log" --stage drift "${COMMON[@]}" --drift-steps "$DRIFT"
# π fresh 4샤딩
pids=(); for i in 0 1 2 3; do
  ( run_stage "$i" "$LOGS/fresh-shard$i.log" --stage rollout-fresh "${COMMON[@]}" --adapter "$OUT_ROOT/drift_$DRIFT" --shard "$i:4" ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || exit 1; done
[ -f "$OUT_ROOT/rollouts_fresh_train.jsonl" ] || cat "$OUT_ROOT"/rollouts_fresh_train.shard*.jsonl > "$OUT_ROOT/rollouts_fresh_train.jsonl"
# analyze (생성은 스킵되고 gradient·report·hybrid만)
run_stage 0 "$LOGS/analyze.log" --stage analyze "${COMMON[@]}" --adapter "$OUT_ROOT/drift_$DRIFT"
log "=== 14B 완료 — bash scripts/result.sh 로 판정 (OUT_ROOT=$OUT_ROOT) ==="
touch "$OUT_ROOT/DONE"
