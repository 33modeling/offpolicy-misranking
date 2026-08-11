#!/usr/bin/env bash
# 메인 보강 3종 완전 자동 실행 (H100 4대, tmux 포그라운드):
#   bash scripts/go_all.sh
# 순서: GSM8K(n=512) → dapo-math(n=512) → mbpp(n=512), 각각 4-GPU 샤딩.
#  - run이 죽으면 잔재 정리 후 자동 재개(데이터셋당 최대 3회, 완료 스테이지는 스킵)
#  - preflight 실패한 데이터셋은 건너뛰고 다음으로 (전체가 죽지 않음)
#  - 전부 끝나면 tables.sh로 결과 표 자동 생성
# 진행 로그가 이 창에 흐름. Ctrl+C = 전체 정리 종료.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv python 없음: $PY"; exit 1; }
N=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "== GPU ${N}장 감지"

echo "== [0] GPU 건강검사"
sick=0
for i in $(seq 0 $((N - 1))); do
  if CUDA_VISIBLE_DEVICES="$i" timeout 120 "$PY" -c "
import torch
a = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
for _ in range(20):
    a = (a @ a).clamp(-1, 1)
torch.cuda.synchronize()
s = torch.randn(32, 2048, 2048, device='cuda', dtype=torch.bfloat16)
for _ in range(30):
    s.softmax(dim=-1).sum().item()
torch.cuda.synchronize()" 2>"$TMPDIR/hc$i.err"; then
    echo "  GPU$i OK"
  else
    echo "  GPU$i ✘ FAIL — $(tail -1 "$TMPDIR/hc$i.err" 2>/dev/null | cut -c1-100)"; sick=1
  fi
done
[ "$sick" -eq 0 ] || { echo "== [중단] 병든 GPU — 이 노드에서 돌리지 말 것"; exit 1; }

export N_TRAIN="${N_TRAIN:-512}" N_VAL="${N_VAL:-100}"
export FRESH_K="${FRESH_K:-32}" HYBRID_PROMPTS="${HYBRID_PROMPTS:-64}"
export MODEL_14B="${MODEL_14B:-$MODELS_DIR/Qwen2.5-7B-Instruct}"
BASE_ROOT="$OM_WORK/runs/big-7b"
DATASETS=(${DATASETS:-gsm8k dapo-math mbpp})
MAX_TRY=3

cleanup_strays() {  # 죽은 시도의 잔재 프로세스 정리 (이 실험 것만)
  pkill -f -- "--run $BASE_ROOT" 2>/dev/null || true
  pkill -f gpu_keepalive 2>/dev/null || true
  sleep 5
}
trap 'echo "== 중단 요청 — 전체 정리"; cleanup_strays; kill $W 2>/dev/null; exit 130' INT TERM

# 상세 진행 워처 — 현재 활동 중인 run의 최신 로그 한 줄을 15초마다
( prev=""
  while :; do
    sleep 15
    lf=$(ls -t "$BASE_ROOT"*/logs/*.log 2>/dev/null | head -1); [ -n "$lf" ] || continue
    line=$(tail -n 1 "$lf" 2>/dev/null | cut -c1-120)
    [ -n "$line" ] && [ "$line" != "$prev" ] && { echo "[detail·$(basename "$lf" .log)] $line"; prev="$line"; }
  done ) &
W=$!

declare -A RESULT
for DS in "${DATASETS[@]}"; do
  RUN_DIR="$BASE_ROOT"; [ "$DS" != "gsm8k" ] && RUN_DIR="$BASE_ROOT-$DS"
  LOG="big-$DS.log"
  echo
  echo "==== [$DS] 시작 → $RUN_DIR (n=$N_TRAIN, fresh=$FRESH_K, hybrid=$HYBRID_PROMPTS, log: $LOG)"
  ok=0
  for try in $(seq 1 "$MAX_TRY"); do
    echo "==== [$DS] 시도 $try/$MAX_TRY"
    cleanup_strays
    if DATASET="$DS" OUT_ROOT="$BASE_ROOT" bash scripts/run_14b.sh >> "$LOG" 2>&1; then
      ok=1; echo "==== [$DS] ✔ 완주"; break
    fi
    echo "==== [$DS] ✘ 시도 $try 실패 — 마지막 로그:"
    tail -4 "$LOG" | sed 's/^/     /'
    if grep -q "\[abort\].*데이터" "$LOG"; then
      echo "==== [$DS] 데이터 문제 — 재시도 무의미, 다음 데이터셋으로"
      break
    fi
    sleep 20   # 재개는 완료 스테이지를 스킵하므로 이어서 진행됨
  done
  RESULT[$DS]=$ok
done

cleanup_strays
kill "$W" 2>/dev/null
echo
echo "==== 전체 종료 요약 ===="
for DS in "${DATASETS[@]}"; do
  RUN_DIR="$BASE_ROOT"; [ "$DS" != "gsm8k" ] && RUN_DIR="$BASE_ROOT-$DS"
  if [ "${RESULT[$DS]:-0}" = "1" ]; then
    echo "  $DS: ✔ 완주 → $RUN_DIR"
  else
    echo "  $DS: ✘ 미완 (big-$DS.log 마지막 사인 확인)"
  fi
done
echo
echo "==== 결과 수집 (group-volume) ===="
RESULTS_DIR="$OM_WORK/results"
mkdir -p "$RESULTS_DIR"
DIRS=()
for DS in "${DATASETS[@]}"; do
  RUN_DIR="$BASE_ROOT"; [ "$DS" != "gsm8k" ] && RUN_DIR="$BASE_ROOT-$DS"
  [ -f "$RUN_DIR/report.json" ] || continue
  DIRS+=("$RUN_DIR")
  cp "$RUN_DIR/report.json" "$RESULTS_DIR/report-big-$DS.json" 2>/dev/null || true
  cp "$RUN_DIR/report.md"   "$RESULTS_DIR/report-big-$DS.md"   2>/dev/null || true
  "$PY" src/judge.py "$RUN_DIR" > "$RESULTS_DIR/judge-big-$DS.txt" 2>&1 || true
done
[ "${#DIRS[@]}" -gt 0 ] && OM_RESULTS="$RESULTS_DIR" "$PY" src/make_tables.py "${DIRS[@]}" | tail -3
echo "== 끝. 결과 일체: $RESULTS_DIR (TABLES.md·report-*·judge-*) — md 뽑아서 전달"
ls -la "$RESULTS_DIR" 2>/dev/null | tail -8
