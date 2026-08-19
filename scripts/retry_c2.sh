#!/usr/bin/env bash
# C2 재시도 원샷: drift를 더 세게(기본 400스텝) 학습한 run을 추가해 margin이
# 벌어지는지 본다. 점수 분산은 drift에 비례해 커지므로, 경계 동점(margin≈0)이
# "drift가 약해서"였다면 여기서 풀린다. val은 처음부터 K=24로 깊게 관측.
#   bash scripts/retry_c2.sh            # 7B gate 루트에 drift400 추가 (GPU 1장)
#   DRIFT=800 bash scripts/retry_c2.sh  # 더 세게
# 끝나면 frac 스캔 스윕(c2_sweep) + 진단(c2_diagnose)까지 자동.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
echo "[abort] retry_c2.sh는 config-locked run을 부분 재사용해 계약을 우회하므로 비활성화됨."
echo "        corrected 신규 OUT_ROOT에서 run_14b.sh를 실행할 것."
exit 2
source scripts/_find_root.sh 7b
PY="$VENV_DIR/bin/python"
MODEL="${MODEL:-$MODELS_DIR/Qwen2.5-7B-Instruct}"
DRIFT="${DRIFT:-400}"
RUN="$OUT_ROOT/drift$DRIFT"
LOGS="$OUT_ROOT/logs"; mkdir -p "$LOGS" "$RUN"
LF="$LOGS/retry-drift$DRIFT.log"
log() { echo "[$(date '+%F %T')] $*"; }

SHARED="$OUT_ROOT/shared"
[ -f "$SHARED/rollouts_behavior_train.jsonl" ] || { log "[abort] 공유 β rollout 없음: $SHARED — 7B gate 루트가 맞는지 확인"; exit 1; }

# 비어있는 GPU 자동 선택 (다른 실험과 충돌 방지 — 여유 30GB+ 필요)
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F', ' '$2 < 40000 {print $1; exit}')
[ -n "${GPU:-}" ] || { log "[abort] 여유 GPU 없음"; exit 1; }
log "== GPU $GPU 사용, drift=$DRIFT, run=$RUN (log: $LF)"

# GPU 유휴 3h 킬 회피 — 우리 GPU만 keepalive
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/gpu_keepalive.py 15 >> "$LOGS/keepalive-retry.log" 2>&1 &
KA_PID=$!
trap 'kill $KA_PID 2>/dev/null' EXIT

ln -sf "$(realpath "$SHARED/prompts.json")" "$RUN/prompts.json"
ln -sf "$(realpath "$SHARED/rollouts_behavior_train.jsonl")" "$RUN/rollouts_behavior_train.jsonl"

log "== [1/3] drift 학습 ($DRIFT steps) — 진행은 tail -f $LF"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" src/experiment.py --stage drift \
  --run "$RUN" --model "$MODEL" --drift-steps "$DRIFT" >> "$LF" 2>&1 \
  || { tail -8 "$LF"; exit 1; }

log "== [2/3] analyze (oracle·score·report·hybrid, val-k=24)"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" src/experiment.py --stage analyze \
  --run "$RUN" --model "$MODEL" --adapter "$RUN/drift_$DRIFT" \
  --fresh-k "${FRESH_K:-32}" --val-k "${VAL_K:-24}" \
  --hybrid-prompts "${HYBRID_PROMPTS:-32}" >> "$LF" 2>&1 \
  || { tail -8 "$LF"; exit 1; }

log "== [3/3] frac 스캔 스윕 + 진단"
"$PY" src/c2_sweep.py "$RUN"
"$PY" src/c2_diagnose.py "$RUN"
log "== 끝 — 'C2 PASS'면 게이트 재판정, margin이 여전히 ≈0이면 drift 무관 = 구조적 확정"
