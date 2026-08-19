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
echo "[abort] redo_oracle.sh는 config-locked run의 일부만 바꿔 artifact를 혼합하므로 비활성화됨."
echo "        FRESH_K를 바꿀 때는 새 OUT_ROOT로 scripts/run_14b.sh를 실행할 것."
exit 2
