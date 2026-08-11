#!/usr/bin/env bash
# G블록 — Qwen3.6-27B (최신 오픈웨이트): 호환성 스모크 → hard-slice 프리스크린
# → 3-seed 본실행. tmux 포그라운드에서:
#   bash scripts/go_27b.sh
# 전제: 모델 스냅샷($MODELS_DIR/Qwen3.6-27B — fetch_27b.sh)과 dapo-math 배치본.
# thinking OFF 기본(OM_THINKING=on으로만 켜짐), LoRA는 all-linear(DeltaNet 층 포함).
# 코드 도메인: DATASETS_27B="dapo-math apps" 로 apps(하네스 구현됨)까지.
set -uo pipefail
cd "$(dirname "$0")/.."

if pgrep -f "scripts/go_v2.sh" >/dev/null || pgrep -f "scripts/go_full.sh" >/dev/null \
   || pgrep -f "scripts/go_boost.sh" >/dev/null; then
  echo "[abort] 다른 go_* 실행 중 — 완주 후 재실행"; exit 1
fi
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
M27="${MODEL27B:-$MODELS_DIR/Qwen3.6-27B}"
[ -f "$M27/config.json" ] || {
  echo "[abort] 27B 스냅샷 없음: $M27 — 온라인 머신에서 bash scripts/fetch_27b.sh"; exit 1; }
export OM_LORA_TARGETS="${OM_LORA_TARGETS:-all-linear}"
SEEDS_27B="${SEEDS_27B:-0 1 2}"
DATASETS_27B="${DATASETS_27B:-dapo-math}"

echo "== [0] 호환성 스모크 (8+4, 전 스테이지 — DeltaNet/checkpointing/LoRA 관문)"
SMK="$OM_WORK/runs/smoke-27b"
if [ -f "$SMK/report.json" ]; then
  echo "   스모크 산출물 존재 — 스킵"
else
  if ! DATASET=gsm8k OUT_ROOT="$SMK" N_TRAIN=8 N_VAL=4 FRESH_K=4 HYBRID_PROMPTS=4 \
       SEED=0 MODEL_14B="$M27" bash scripts/run_14b.sh > smoke-27b.log 2>&1; then
    echo "== [중단] 27B 스모크 실패 — 신아키텍처 호환성(transformers/PEFT/checkpointing) 의심. tail:"
    tail -12 smoke-27b.log | sed 's/^/   /'
    echo "   transformers 업그레이드가 필요하면 constraints 확인 후 provision 재실행."
    exit 1
  fi
  echo "   스모크 ✔"
fi

for DS in $DATASETS_27B; do
  echo "== [1/$DS] hard-slice 프리스크린 (β pass-rate, 전량 정답/오답 제외)"
  POOL="$OM_WORK/pools/$DS-hard.jsonl"
  if [ ! -s "$POOL" ]; then
    MODEL="$M27" bash scripts/prescreen_pool.sh "$DS" "${POOL_N:-2000}" || { echo "[abort] $DS 프리스크린 실패"; exit 1; }
  fi
  [ -s "$POOL" ] || { echo "[abort] 풀 없음: $POOL"; exit 1; }

  echo "== [2/$DS] 본실행 seeds($SEEDS_27B), n=512 (hard pool)"
  for s in $SEEDS_27B; do
    dir="$OM_WORK/runs/v2-27b-$DS-s$s"
    [ -f "$dir/DONE" ] && { echo "  ✔ $DS/s$s 완주 — 스킵"; continue; }
    if DATASET="$DS" OM_POOL_FILE="$POOL" MODEL_14B="$M27" SEED="$s" \
       N_TRAIN=512 N_VAL=100 OUT_ROOT="$dir" bash scripts/run_14b.sh >> "27b-$DS-s$s.log" 2>&1; then
      echo "  ✔ $DS/s$s"
    else
      echo "  ✘ $DS/s$s — tail:"; tail -4 "27b-$DS-s$s.log" | sed 's/^/     /'
    fi
  done
done

echo "== 종료 요약 =="
for DS in $DATASETS_27B; do for s in $SEEDS_27B; do
  d="$OM_WORK/runs/v2-27b-$DS-s$s"
  [ -d "$d" ] && echo "  v2-27b-$DS-s$s: $([ -f "$d/DONE" ] && echo ✔ || echo ✘)"
done; done
echo "== 끝 — 판정: hard pool에서 one-sided 열화+hybrid 회복 재현 시 본문 승격, 포화면 규칙(3) 사례로 수록"
