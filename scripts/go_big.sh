#!/usr/bin/env bash
# 메인 실험 확장판 (n 512, hybrid 64, fresh 32) — tmux 포그라운드 원샷.
#   bash scripts/go_big.sh                # 7B GSM8K, n=512 → runs/big-7b
#   DATASET=dapo-math bash scripts/go_big.sh   # 수학 대용량 → runs/big-7b-dapo-math
#   N_TRAIN=1024 bash scripts/go_big.sh        # 더 크게
# 진행 로그(문항·ETA·loss)가 이 창에 흐르고, Ctrl+C면 정리 종료.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv python 없음: $PY"; exit 1; }
N=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$N" -ge 1 ] || { echo "[abort] GPU 미감지"; exit 1; }

echo "== [1/3] GPU 건강검사 (장당 ~15초)"
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
[ "$sick" -eq 0 ] || { echo "== [중단] 병든 GPU — 다른 인스턴스에서 재실행"; exit 1; }

export DATASET="${DATASET:-gsm8k}"
export N_TRAIN="${N_TRAIN:-512}" N_VAL="${N_VAL:-100}"
export FRESH_K="${FRESH_K:-32}" HYBRID_PROMPTS="${HYBRID_PROMPTS:-64}"
export MODEL_14B="${MODEL_14B:-$MODELS_DIR/Qwen2.5-7B-Instruct}"
export OUT_ROOT="${OUT_ROOT:-$OM_WORK/runs/big-7b}"
LOG="big-${DATASET}.log"
RUN_DIR="$OUT_ROOT"; [ "$DATASET" != "gsm8k" ] && RUN_DIR="$OUT_ROOT-$DATASET"
echo "== [2/3] 실행: 7B $DATASET n=$N_TRAIN(+val $N_VAL) fresh=$FRESH_K hybrid=$HYBRID_PROMPTS → $RUN_DIR"

bash scripts/run_14b.sh > "$LOG" 2>&1 &
P=$!
trap 'echo "== 중단 — 정리"; kill $P $T $W 2>/dev/null; exit 130' INT TERM

echo "== [3/3] 실시간 로그 — 완주까지 이 창 유지 (log: $LOG)"
( tail -n 2 -f "$LOG" | sed -u 's/^/[main ] /' ) &
T=$!
( prev=""
  while :; do
    sleep 15
    lf=$(ls -t "$RUN_DIR/logs"/*.log 2>/dev/null | head -1); [ -n "$lf" ] || continue
    line=$(tail -n 1 "$lf" 2>/dev/null | cut -c1-120)
    [ -n "$line" ] && [ "$line" != "$prev" ] && { echo "[detail·$(basename "$lf" .log)] $line"; prev="$line"; }
  done ) &
W=$!
R=0; wait "$P" || R=$?
kill "$T" "$W" 2>/dev/null
echo
echo "== 종료 rc=$R (0=완주)"
[ "$R" -eq 0 ] && echo "-- 판정:  bash scripts/result.sh '$RUN_DIR'   /   표: bash scripts/tables.sh '$RUN_DIR'"
[ "$R" -ne 0 ] && { echo "-- 사인:"; tail -6 "$LOG" | sed 's/^/   /'; }
