#!/usr/bin/env bash
# 7B+14B MATH-500 동시 실행 — tmux 포그라운드용:  bash scripts/go7_14.sh
#   ① GPU 건강검사 ② 반반 분할로 둘 다 실행 ③ 두 로그를 실시간 출력하며 완주까지 대기
# Ctrl+C 하면 둘 다 정리 종료. (백그라운드로 원하면 go7m.sh/go14m.sh 개별 사용)
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv python 없음: $PY — group-volume 마운트/provision 확인"; exit 1; }
N=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$N" -ge 2 ] || { echo "[abort] GPU 2장 이상 필요 (감지: $N)"; exit 1; }

echo "== [1/3] GPU 건강검사 (장당 ~10초)"
sick=0
for i in $(seq 0 $((N - 1))); do
  if CUDA_VISIBLE_DEVICES="$i" timeout 120 "$PY" -c "
import torch
a = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
for _ in range(20):            # matmul 부하
    a = (a @ a).clamp(-1, 1)
torch.cuda.synchronize()
s = torch.randn(32, 2048, 2048, device='cuda', dtype=torch.bfloat16)
for _ in range(30):            # 실제 사망 이력 커널(softmax) 부하
    s.softmax(dim=-1).sum().item()
torch.cuda.synchronize()" 2>"$TMPDIR/hc$i.err"; then
    echo "  GPU$i OK"
  else
    echo "  GPU$i ✘ FAIL — $(tail -1 "$TMPDIR/hc$i.err" 2>/dev/null | cut -c1-100)"; sick=1
  fi
done
[ "$sick" -eq 0 ] || { echo "== [중단] 병든 GPU 있음 — 이 노드 버리고 다른 인스턴스에서 재실행할 것"; exit 1; }

H=$((N / 2))
G7=$(seq -s, 0 $((H - 1)))
G14=$(seq -s, "$H" $((N - 1)))
echo "== [2/3] 실행: 7B=[GPU $G7]  14B=[GPU $G14]  (로그: 7b-math.log / 14b-math.log)"

OM_GPUS="$G7" MODEL_14B="$MODELS_DIR/Qwen2.5-7B-Instruct" DATASET=math500 \
  FRESH_K=32 OUT_ROOT="$OM_WORK/runs/gate-7b" \
  bash scripts/run_14b.sh > 7b-math.log 2>&1 &
P7=$!
OM_GPUS="$G14" DATASET=math500 FRESH_K=32 bash scripts/run_14b.sh > 14b-math.log 2>&1 &
P14=$!
trap 'echo "== 중단 요청 — 둘 다 정리"; kill $P7 $P14 2>/dev/null; exit 130' INT TERM

echo "== [3/3] 실시간 로그 — 스테이지 전환은 즉시, 상세 진행(문항·ETA·loss)은 15초마다"
( tail -n 2 -f 7b-math.log  | sed -u 's/^/[7b ] /' ) &
T1=$!
( tail -n 2 -f 14b-math.log | sed -u 's/^/[14b] /' ) &
T2=$!
# 스테이지 내부의 상세 진행은 runs/<run>/logs/*.log에 쌓인다 — 활동 중인 로그의
# 마지막 줄을 주기적으로 화면에 올려 "침묵 = 멈춤 오인"을 없앤다.
watch_detail() {  # watch_detail <태그> <logs디렉토리>
  local prev=""
  while :; do
    sleep 15
    local lf line
    lf=$(ls -t "$2"/*.log 2>/dev/null | head -1)
    [ -n "$lf" ] || continue
    line=$(tail -n 1 "$lf" 2>/dev/null | cut -c1-120)
    [ -n "$line" ] && [ "$line" != "$prev" ] && { echo "[$1·$(basename "$lf" .log)] $line"; prev="$line"; }
  done
}
watch_detail "7b " "$OM_WORK/runs/gate-7b-math500/logs" &
W1=$!
watch_detail "14b" "$OM_WORK/runs/gate-14b-math500/logs" &
W2=$!
trap 'echo "== 중단 요청 — 전부 정리"; kill $P7 $P14 $T1 $T2 $W1 $W2 2>/dev/null; exit 130' INT TERM
R7=0; R14=0
wait "$P7" || R7=$?
wait "$P14" || R14=$?
kill "$T1" "$T2" "$W1" "$W2" 2>/dev/null
echo
echo "== 종료: 7B rc=$R7 / 14B rc=$R14  (0=완주)"
[ "$R7" -eq 0 ] && echo "-- 7B 판정:  bash scripts/result.sh 7bm"
[ "$R7" -ne 0 ] && { echo "-- 7B 사인:"; tail -6 7b-math.log | sed 's/^/   /'; }
[ "$R14" -eq 0 ] && echo "-- 14B 판정: bash scripts/result.sh 14bm"
[ "$R14" -ne 0 ] && { echo "-- 14B 사인:"; tail -6 14b-math.log | sed 's/^/   /'; }
