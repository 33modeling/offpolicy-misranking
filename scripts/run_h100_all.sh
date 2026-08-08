#!/usr/bin/env bash
# 4×H100 병렬 게이트 파일럿 (기본 Qwen2.5-7B-Instruct).
#
# 배치:
#   phase 0  공유: prep + β rollout 1회 (GPU0) — drift와 무관하므로 재사용
#   phase 1  drift 50/100/200 파이프라인을 GPU 0/1/2에 병렬
#            (drift SFT → analyze[oracle·score·report·hybrid — 모델 1회 로드])
#   phase 2  downstream 4소스(oracle/g10/g01/random)를 GPU 0~3에 병렬 (drift100 기준)
#
# 사용 (클러스터 노드):
#   source scripts/setup_env.sh && bash scripts/provision.sh   # 최초 1회 (온라인 머신)
#   bash scripts/run_h100_all.sh                                # 이후 이것만
#
# 로그: $OUT_ROOT/logs/ 아래 스테이지별 파일 + main.log 타임라인.
# 실패 시: 해당 로그 tail을 main.log에 남기고 종료 코드 1.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "venv 없음 — 'source scripts/setup_env.sh && bash scripts/provision.sh' 먼저"; exit 1; }
# 모델·데이터는 provision이 group-volume에 받아둔 로컬 스냅샷 사용 (미러 설정 불필요)
if [ -d "$MODEL_QWEN25_7B" ]; then MODEL="${MODEL:-$MODEL_QWEN25_7B}"; else MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"; fi
OUT_ROOT="${OUT_ROOT:-$OM_WORK/runs/gate}"
DRIFTS=(${DRIFTS:-50 100 200})
# 실행 시간 손잡이: FRESH_K=16 이면 oracle 수집 절반, DRIFTS="100" 이면 단일 파이프라인
EXTRA=(--fresh-k "${FRESH_K:-32}" --val-k "${VAL_K:-8}" --hybrid-prompts "${HYBRID_PROMPTS:-32}")
LOGS="$OUT_ROOT/logs"
mkdir -p "$LOGS"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/main.log"; }
run_stage() {  # run_stage <gpu> <logfile> <args...>
  local gpu="$1" lf="$2"; shift 2
  local t0=$SECONDS
  log "GPU$gpu ▶ $* (log: $(basename "$lf"))"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/experiment.py "$@" >> "$lf" 2>&1; then
    log "GPU$gpu ✔ $1 $2 완료 ($((SECONDS - t0))s)"
  else
    local rc=$?
    log "GPU$gpu ✘ $1 $2 실패 rc=$rc ($((SECONDS - t0))s) — tail:"
    tail -8 "$lf" | tee -a "$LOGS/main.log"
    return "$rc"
  fi
}

log "=== 시작: MODEL=$MODEL, DRIFTS=${DRIFTS[*]}, OUT_ROOT=$OUT_ROOT ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv | tee -a "$LOGS/main.log" || true
# 좀비 점유 검사 — 이전 실행이 GPU를 잡고 있으면 OOM 나므로 여기서 멈춘다
BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 > 2000' | wc -l)
if [ "${BUSY:-0}" -gt 0 ]; then
  log "[abort] GPU ${BUSY}개가 이미 2GB+ 점유 중 — 이전 프로세스를 먼저 종료할 것:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv | tee -a "$LOGS/main.log" || true
  log "        예: kill <PID>  후 재실행 (OM_SKIP_GPU_CHECK=1 로 무시 가능)"
  [ "${OM_SKIP_GPU_CHECK:-0}" = "1" ] || exit 1
fi

# ---------- phase 0: 공유 (prep + β rollout, 4-GPU 샤딩) ----------
SHARED="$OUT_ROOT/shared"
if [ -f "$SHARED/rollouts_behavior_train.jsonl" ]; then
  log "phase0: 공유 산출물 존재 — 스킵"
else
  run_stage 0 "$LOGS/phase0-prep.log" --stage prep --run "$SHARED" --model "$MODEL" || { log "phase0 prep 실패"; exit 1; }
  pids=()
  for i in 0 1 2 3; do
    ( run_stage "$i" "$LOGS/phase0-rollout-shard$i.log" --stage rollout-behavior         --run "$SHARED" --model "$MODEL" --shard "$i:4" ) &
    pids+=($!)
  done
  p0fail=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { log "phase0 shard$i 실패"; p0fail=1; }
  done
  [ "$p0fail" -eq 0 ] || exit 1
  cat "$SHARED"/rollouts_behavior_train.shard*.jsonl > "$SHARED/rollouts_behavior_train.jsonl"
  log "phase0 완료: $(wc -l < "$SHARED/rollouts_behavior_train.jsonl") rollouts (4샤드 병합)"
fi

# ---------- 진행 하트비트 (10분 간격, 종료 시 자동 정리) ----------
# GPU 유휴 킬 회피 — 모든 GPU에 5분마다 소형 연산 (클러스터 idle-kill 정책 대응)
"$PY" scripts/gpu_keepalive.py 15 >> "$LOGS/keepalive.log" 2>&1 &
KEEPALIVE_PID=$!

( while true; do
    sleep 600
    {
      echo "[$(date '+%F %T')] ---- heartbeat ----"
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  GPU /'
      for lf in "$LOGS"/drift*.log "$LOGS"/downstream-*.log; do
        [ -f "$lf" ] && echo "  $(basename "$lf" .log): $(tail -n 1 "$lf")"
      done
    } >> "$LOGS/main.log"
  done ) &
HEARTBEAT_PID=$!
trap 'kill $HEARTBEAT_PID $KEEPALIVE_PID 2>/dev/null' EXIT

# ---------- phase 1: drift 병렬 (GPU 0/1/2) ----------
drift_pipeline() {  # drift_pipeline <gpu> <drift>
  local gpu="$1" drift="$2"
  local run="$OUT_ROOT/drift$drift" lf="$LOGS/drift$drift.log"
  mkdir -p "$run"
  ln -sf "$(realpath "$SHARED/prompts.json")" "$run/prompts.json"
  ln -sf "$(realpath "$SHARED/rollouts_behavior_train.jsonl")" "$run/rollouts_behavior_train.jsonl"
  run_stage "$gpu" "$lf" --stage drift   --run "$run" --model "$MODEL" --drift-steps "$drift" && \
  run_stage "$gpu" "$lf" --stage analyze --run "$run" --model "$MODEL" --adapter "$run/drift_$drift" "${EXTRA[@]}"
}
pids=(); names=()
gpu=0
for drift in "${DRIFTS[@]}"; do
  drift_pipeline "$gpu" "$drift" &
  pids+=($!); names+=("drift$drift(GPU$gpu)")
  gpu=$((gpu + 1))
done
# GPU3: downstream-random은 점수가 필요 없어 phase1과 동시 실행 (prompts.json 대기)
( RUN_R="$OUT_ROOT/drift${DRIFTS[1]:-${DRIFTS[0]}}"
  for _ in $(seq 60); do [ -f "$RUN_R/prompts.json" ] && break; sleep 5; done
  run_stage 3 "$LOGS/downstream-random.log" --stage downstream --run "$RUN_R"     --model "$MODEL" --downstream-source random ) &
pids+=($!); names+=("downstream-random(GPU3)")
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
for src in oracle g10 g01; do
  [ -f "$DS_RUN/downstream_$src.json" ] && { log "phase2 $src 이미 완료 — 스킵"; continue; }
  ( run_stage "$gpu" "$LOGS/downstream-$src.log" --stage downstream --run "$DS_RUN" \
      --model "$MODEL" --downstream-source "$src" --downstream-steps "${DOWNSTREAM_STEPS:-200}" ) &
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
