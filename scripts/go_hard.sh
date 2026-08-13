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
LOGDIR="${OM_WORK:-.}/console-logs"; mkdir -p "$LOGDIR"
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3

echo "== [0] P3-0 사전 검력 게이트 — 기존 run의 live 부분집합 floor로 성패 선판정"
if "$PY" src/precheck_hard.py "$OM_WORK/runs" --gate; then
  echo "   GO — live 필터로 신호 체제 형성 근거 있음, 본실행 진행"
else
  rc=$?
  "$PY" src/precheck_hard.py "$OM_WORK/runs" | tail -15 | sed 's/^/   /'
  if [ "${FORCE_HARD:-0}" != "1" ]; then
    echo "[abort] 사전 검력 NO-GO/판정불가(rc=$rc) — GPU 태우지 않음."
    echo "        전체 보고서: bash scripts/precheck_hard.sh"
    echo "        그래도 강행: FORCE_HARD=1 bash scripts/go_hard.sh"
    exit 1
  fi
  echo "[warn] FORCE_HARD=1 — 게이트 무시하고 진행"
fi

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
     N_TRAIN=$NT N_VAL=$NV OUT_ROOT="$dir" bash scripts/run_14b.sh >> "$LOGDIR/hard-s$s.log" 2>&1; then
    echo "  ✔ s$s"
  else
    echo "  ✘ s$s — tail:"; tail -4 "$LOGDIR/hard-s$s.log" | sed 's/^/     /'
  fi
done
echo "== 끝 — 예측: hard 풀에서 floor↑ 하면 stale retention↓ (역상관 재현이면 준-개입 성립)"
