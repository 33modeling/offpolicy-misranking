#!/usr/bin/env bash
# 4×H100 병렬 게이트 파일럿 (기본 Qwen2.5-7B-Instruct).
#
# 배치:
#   phase 0  공유: prep + β rollout 1회 (GPU0) — drift와 무관하므로 재사용
#   phase 1  drift 50/100/200 파이프라인을 GPU 0/1/2에 병렬
#            (drift SFT → π fresh oracle → 2×2 score → report → hybrid 25/50/75%)
#   phase 2  downstream 4소스(oracle/g10/g01/random)를 GPU 0~3에 병렬 (drift100 기준)
#
# 사용 (클러스터 노드):
#   export HF_ENDPOINT=<HF 미러>                                  # 폐쇄망 필수
#   export OUT_ROOT=/group-volume/minsoo3.kim/offpolicy-misranking # 산출 경로
#   bash scripts/run_h100_all.sh
#
# 로그: $OUT_ROOT/logs/ 아래 스테이지별 파일 + main.log 타임라인.
# 실패 시: 해당 로그 tail을 main.log에 남기고 종료 코드 1.
set -uo pipefail
cd "$(dirname "$0")/.."
: "${HF_ENDPOINT:?HF_ENDPOINT 미러를 설정할 것}"
OUT_ROOT="${OUT_ROOT:-outputs/h100}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DRIFTS=(${DRIFTS:-50 100 200})
export PYTHONUNBUFFERED=1 PYTHONPATH=src
LOGS="$OUT_ROOT/logs"
mkdir -p "$LOGS"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/main.log"; }
run_stage() {  # run_stage <gpu> <logfile> <args...>
  local gpu="$1" lf="$2"; shift 2
  log "GPU$gpu ▶ experiment.py $* (log: $(basename "$lf"))"
  CUDA_VISIBLE_DEVICES="$gpu" python3 src/experiment.py "$@" >> "$lf" 2>&1
}

log "=== 시작: MODEL=$MODEL, DRIFTS=${DRIFTS[*]}, OUT_ROOT=$OUT_ROOT ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee -a "$LOGS/main.log" || true

# ---------- phase 0: 공유 (prep + β rollout) ----------
SHARED="$OUT_ROOT/shared"
if [ -f "$SHARED/rollouts_behavior_train.jsonl" ]; then
  log "phase0: 공유 산출물 존재 — 스킵"
else
  run_stage 0 "$LOGS/phase0-prep.log"    --stage prep             --run "$SHARED" --model "$MODEL" || { log "phase0 prep 실패"; tail -20 "$LOGS/phase0-prep.log" | tee -a "$LOGS/main.log"; exit 1; }
  run_stage 0 "$LOGS/phase0-rollout.log" --stage rollout-behavior --run "$SHARED" --model "$MODEL" || { log "phase0 rollout 실패"; tail -20 "$LOGS/phase0-rollout.log" | tee -a "$LOGS/main.log"; exit 1; }
  log "phase0 완료: $(wc -l < "$SHARED/rollouts_behavior_train.jsonl") rollouts"
fi

# ---------- phase 1: drift 병렬 (GPU 0/1/2) ----------
drift_pipeline() {  # drift_pipeline <gpu> <drift>
  local gpu="$1" drift="$2"
  local run="$OUT_ROOT/drift$drift" lf="$LOGS/drift$drift.log"
  mkdir -p "$run"
  ln -sf "$(realpath "$SHARED/prompts.json")" "$run/prompts.json"
  ln -sf "$(realpath "$SHARED/rollouts_behavior_train.jsonl")" "$run/rollouts_behavior_train.jsonl"
  run_stage "$gpu" "$lf" --stage drift  --run "$run" --model "$MODEL" --drift-steps "$drift" && \
  run_stage "$gpu" "$lf" --stage oracle --run "$run" --model "$MODEL" --adapter "$run/drift_$drift" && \
  run_stage "$gpu" "$lf" --stage score  --run "$run" --model "$MODEL" --adapter "$run/drift_$drift" && \
  run_stage "$gpu" "$lf" --stage report --run "$run" && \
  for cut in 0.25 0.5 0.75; do
    run_stage "$gpu" "$lf" --stage hybrid --run "$run" --model "$MODEL" \
      --adapter "$run/drift_$drift" --cut-frac "$cut" || return 1
  done
}
pids=(); names=()
gpu=0
for drift in "${DRIFTS[@]}"; do
  drift_pipeline "$gpu" "$drift" &
  pids+=($!); names+=("drift$drift(GPU$gpu)")
  gpu=$((gpu + 1))
done
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    log "phase1 ${names[$i]} 완료"
  else
    log "phase1 ${names[$i]} 실패 — tail:"
    tail -20 "$LOGS/${names[$i]%%(*}.log" | tee -a "$LOGS/main.log"
    fail=1
  fi
done
[ "$fail" -eq 0 ] || { log "phase1 실패 — 중단"; exit 1; }

# ---------- phase 2: downstream 병렬 (GPU 0~3, drift100 점수 기준) ----------
DS_RUN="$OUT_ROOT/drift100"
pids=(); srcs=()
gpu=0
for src in oracle g10 g01 random; do
  ( run_stage "$gpu" "$LOGS/downstream-$src.log" --stage downstream --run "$DS_RUN" \
      --model "$MODEL" --downstream-source "$src" ) &
  pids+=($!); srcs+=("$src")
  gpu=$((gpu + 1))
done
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then log "phase2 downstream-${srcs[$i]} 완료"; else
    log "phase2 downstream-${srcs[$i]} 실패 — tail:"
    tail -20 "$LOGS/downstream-${srcs[$i]}.log" | tee -a "$LOGS/main.log" || true
    fail=1
  fi
done

# ---------- 요약 ----------
log "=== 요약 ==="
for drift in "${DRIFTS[@]}"; do
  log "--- drift$drift report"
  cat "$OUT_ROOT/drift$drift/report.md" 2>/dev/null | tee -a "$LOGS/main.log" || log "(report 없음)"
done
for src in oracle g10 g01 random; do
  cat "$DS_RUN/downstream_$src.json" 2>/dev/null | tee -a "$LOGS/main.log" || true
done
log "=== 종료 (fail=$fail) ==="
exit "$fail"
