#!/usr/bin/env bash
# 준-개입 실험 — 같은 태스크(GSM8K)·같은 모델에서 풀 구성만 바꿔 floor(신호)를
# 키웠을 때 stale 붕괴가 따라오는지 확인. 역상관 주장에서 "태스크 차이" 교란을
# 제거하는 핵심 증거.  (tmux 포그라운드, go_v2류와 동시 실행 금지)
#   bash scripts/go_hard.sh          # prescreen(hard-slice) → 3-seed run
set -uo pipefail
cd "$(dirname "$0")/.."
if pgrep -f "scripts/go_v2.sh\|scripts/go_full.sh" >/dev/null; then
  echo "[abort] 다른 go_* 실행 중"; exit 1
fi
source scripts/setup_env.sh
M7="${MODEL_14B:-$MODELS_DIR/Qwen2.5-7B-Instruct}"
POOL="$OM_WORK/pools/gsm8k-hard.jsonl"
if [ ! -s "$POOL" ]; then
  echo "== [1] GSM8K hard-slice 프리스크린 (β pass-rate로 live만)"
  MODEL="$M7" OUT="$POOL" bash scripts/prescreen_pool.sh gsm8k "${POOL_N:-2000}" || exit 1
fi
[ -s "$POOL" ] || { echo "[abort] 풀 생성 실패"; exit 1; }
NP=$(wc -l < "$POOL")
NT=512; NV=100
[ "$NP" -lt 620 ] && { NT=256; NV=50; echo "[info] 풀 ${NP}개 — n=256+50으로 축소"; }
echo "== [2] hard-pool 3-seed (n=$NT+$NV)"
for s in 0 1 2; do
  dir="$OM_WORK/runs/v2-hard-s$s"
  [ -f "$dir/DONE" ] && { echo "  ✔ s$s 스킵"; continue; }
  if DATASET=gsm8k OM_POOL_FILE="$POOL" MODEL_14B="$M7" SEED="$s" \
     N_TRAIN=$NT N_VAL=$NV OUT_ROOT="$dir" bash scripts/run_14b.sh >> "hard-s$s.log" 2>&1; then
    echo "  ✔ s$s"
  else
    echo "  ✘ s$s — tail:"; tail -4 "hard-s$s.log" | sed 's/^/     /'
  fi
done
echo "== 끝 — 예측: hard 풀에서 floor↑ 하면 stale retention↓ (역상관 재현이면 준-개입 성립)"
