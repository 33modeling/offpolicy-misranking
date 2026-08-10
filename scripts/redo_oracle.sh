#!/usr/bin/env bash
# oracle 심화 원샷 — fresh rollout을 더 깊게(K 기본 32) 다시 뽑아 floor를 끌어올린다.
# behavior·drift·val·2×2 점수는 재사용, oracle 쪽 산출물만 지우고 재실행.
#   OM_ATTN=eager bash scripts/redo_oracle.sh              # 기본: 14B math500, K=32
#   RUN=<경로> FRESH_K=48 bash scripts/redo_oracle.sh      # 대상·깊이 지정
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
RUN="${RUN:-$OM_WORK/runs/gate-14b-math500}"
[ -f "$RUN/prompts.json" ] || { echo "[abort] run 아님: $RUN (prompts.json 없음)"; exit 1; }
DATASET="${DATASET:-math500}"
export FRESH_K="${FRESH_K:-32}"
# 이 14B 노드는 fused SDPA 커널이 ULF를 내는 이력(C6)이 있어 eager가 기본.
# 빠른 커널을 쓰고 싶을 때만 OM_ATTN=sdpa 로 명시.
export OM_ATTN="${OM_ATTN:-eager}"

# 이전 실행 잔재 정리 (있으면)
pkill -f run_14b.sh 2>/dev/null || true
pkill -f gpu_keepalive 2>/dev/null || true
sleep 3

echo "== oracle 심화: $RUN (FRESH_K=$FRESH_K, DATASET=$DATASET, OM_ATTN=${OM_ATTN:-기본})"
echo "== 삭제 대상(oracle 계열만):"
ls -1 "$RUN"/rollouts_fresh_train*.jsonl "$RUN"/oracle_micro_groups*.pt \
      "$RUN"/scores_oracle.json "$RUN"/scores_splithalf.json \
      "$RUN"/report.md "$RUN"/report.json 2>/dev/null || true
rm -f "$RUN"/rollouts_fresh_train*.jsonl "$RUN"/oracle_micro_groups*.pt \
      "$RUN"/scores_oracle.json "$RUN"/scores_splithalf.json \
      "$RUN"/report.md "$RUN"/report.json

# run_14b는 OUT_ROOT에 DATASET 접미사를 붙이므로 접미사 없는 밑동을 넘긴다
BASE_ROOT="$RUN"
case "$RUN" in *-"$DATASET") BASE_ROOT="${RUN%-$DATASET}";; esac
LOG="14b-oracle-redo.log"
OUT_ROOT="$BASE_ROOT" DATASET="$DATASET" nohup bash scripts/run_14b.sh > "$LOG" 2>&1 &
sleep 8
echo "== 재실행 시작 (log: $LOG) — 완료 스테이지는 스킵, fresh부터 재생성"
head -5 "$LOG"
echo "== 끝나면: bash scripts/result.sh '$RUN' 에서 floor+precision 4개 확인"
