#!/usr/bin/env bash
# H블록 — Qwen3.5 소형 스케일 스윕 (0.8B/2B/4B × 3-seed, gsm8k n=512).
# 목적 2중: ① 능력 축 실측 — 같은 태스크에서 0.8→2→4→(7B)로 floor·live·
# misranking이 어떻게 움직이는가 ("capability frontier" 주장의 스윕 증거)
# ② 신세대 아키텍처 사전 검증 — 27B(G블록) 전 단계 게이트.
#   bash scripts/go_35.sh                                  # tmux 포그라운드
#   MODELS_35="/경로1 /경로2" bash scripts/go_35.sh        # 스냅샷 경로 지정
set -uo pipefail
cd "$(dirname "$0")/.."
if pgrep -f "scripts/go_v2.sh" >/dev/null || pgrep -f "scripts/go_full.sh" >/dev/null \
   || pgrep -f "scripts/go_boost.sh" >/dev/null || pgrep -f "scripts/go_27b.sh" >/dev/null; then
  echo "[abort] 다른 go_* 실행 중 — 완주 후 재실행"; exit 1
fi
source scripts/setup_env.sh
LOGDIR="${OM_WORK:-.}/console-logs"; mkdir -p "$LOGDIR"
export OM_LORA_TARGETS="${OM_LORA_TARGETS:-all-linear}"   # 신아키텍처 대비 (thinking은 기본 OFF)
SEEDS_35="${SEEDS_35:-0 1 2}"

if [ -z "${MODELS_35:-}" ]; then
  MODELS_35=""
  for name in Qwen3.5-0.8B-Instruct Qwen3.5-0.8B Qwen3.5-2B-Instruct Qwen3.5-2B \
              Qwen3.5-4B-Instruct Qwen3.5-4B; do
    [ -f "$MODELS_DIR/$name/config.json" ] && MODELS_35="$MODELS_35 $MODELS_DIR/$name"
  done
fi
[ -n "${MODELS_35// /}" ] || {
  echo "[abort] Qwen3.5 스냅샷을 못 찾음 ($MODELS_DIR) — MODELS_35=\"/경로 ...\"로 지정"; exit 1; }
echo "== 대상 모델:$MODELS_35"

for M in $MODELS_35; do
  TAG=$(basename "$M" | tr '[:upper:]' '[:lower:]' | sed 's/^qwen//; s/-instruct$//')
  echo
  echo "== [$TAG] 호환성 스모크 (8+4, 전 스테이지)"
  SMK="$OM_WORK/runs/smoke-$TAG"
  if [ -f "$SMK/report.json" ] && [ -f "$SMK/score_protocol.json" ] \
     && [ -f "$SMK/oracle_protocol.json" ]; then
    echo "   스모크 산출물 존재 — 스킵"
  elif ! DATASET=gsm8k OUT_ROOT="$SMK" N_TRAIN=8 N_VAL=4 FRESH_K=8 HYBRID_PROMPTS=4 \
        SEED=0 MODEL_14B="$M" bash scripts/run_14b.sh > "$LOGDIR/smoke-$TAG.log" 2>&1; then
    echo "== [$TAG] 스모크 실패 — 이 모델 스킵 (호환성 문제). tail:"
    tail -6 "$LOGDIR/smoke-$TAG.log" | sed 's/^/   /'
    continue
  else
    echo "   스모크 ✔"
  fi

  echo "== [$TAG] 본실행 seeds($SEEDS_35), gsm8k n=512"
  for s in $SEEDS_35; do
    dir="$OM_WORK/runs/v2-$TAG-s$s"
    [ -f "$dir/DONE" ] && [ -f "$dir/score_protocol.json" ] \
      && [ -f "$dir/oracle_protocol.json" ] \
      && { echo "  ✔ $TAG/s$s 완주 — 스킵"; continue; }
    if DATASET=gsm8k MODEL_14B="$M" SEED="$s" N_TRAIN=512 N_VAL=100 OUT_ROOT="$dir" \
       bash scripts/run_14b.sh >> "$LOGDIR/35-$TAG-s$s.log" 2>&1; then
      echo "  ✔ $TAG/s$s"
    else
      echo "  ✘ $TAG/s$s — tail:"; tail -4 "$LOGDIR/35-$TAG-s$s.log" | sed 's/^/     /'
    fi
  done
done

echo
echo "== 종료 요약 =="
for M in $MODELS_35; do
  TAG=$(basename "$M" | tr '[:upper:]' '[:lower:]' | sed 's/^qwen//; s/-instrict$//; s/-instruct$//')
  for s in $SEEDS_35; do
    d="$OM_WORK/runs/v2-$TAG-s$s"
    [ -d "$d" ] && echo "  v2-$TAG-s$s: $([ -f "$d/DONE" ] && echo ✔ || echo ✘)"
  done
done
echo "== 끝 — 능력 축(0.8→2→4→7B) 스윕. 표는 tables.sh (v2-* DONE 자동 포함)"
