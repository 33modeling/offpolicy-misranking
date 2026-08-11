#!/usr/bin/env bash
# 7B+14B MATH-500 동시 실행 (한 노드, GPU 반반) + 자가진단:  bash scripts/go7_14.sh
#   ① GPU 건강검사(matmul) ② 둘 다 실행 ③ 45초 후 생존 확인·사인 출력
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
N=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$N" -ge 2 ] || { echo "[abort] GPU 2장 이상 필요 (감지: $N)"; exit 1; }

echo "== [1/3] GPU 건강검사 (장당 ~10초)"
sick=0
for i in $(seq 0 $((N - 1))); do
  if CUDA_VISIBLE_DEVICES="$i" timeout 60 "$PY" -c "
import torch; a=torch.randn(2048,2048,device='cuda',dtype=torch.bfloat16)
(a@a).sum().item(); torch.cuda.synchronize()" 2>/dev/null; then
    echo "  GPU$i OK"
  else
    echo "  GPU$i ✘ FAIL — 이 GPU는 병듦"; sick=1
  fi
done
[ "$sick" -eq 0 ] || { echo "== [중단] 병든 GPU 있음 — 이 노드 버리고 다른 인스턴스에서 재실행할 것"; exit 1; }

H=$((N / 2))
G7=$(seq -s, 0 $((H - 1)))
G14=$(seq -s, "$H" $((N - 1)))
echo "== [2/3] 실행: 7B=[GPU $G7]  14B=[GPU $G14]"
OM_GPUS="$G7" bash scripts/go7m.sh
OM_GPUS="$G14" bash scripts/go14m.sh

echo "== [3/3] 45초 후 생존 확인..."
sleep 45
ok=1
if pgrep -f -- "--run .*gate-7b-math500" >/dev/null; then
  echo "  7B  ✔ 실행 중 — $(tail -1 7b-math.log | cut -c1-70)"
else
  ok=0; echo "  7B  ✘ 죽음 — 사인:"; tail -6 7b-math.log | sed 's/^/    /'
fi
if pgrep -f -- "--run .*gate-14b-math500" >/dev/null; then
  echo "  14B ✔ 실행 중 — $(tail -1 14b-math.log | cut -c1-70)"
else
  ok=0; echo "  14B ✘ 죽음 — 사인:"; tail -6 14b-math.log | sed 's/^/    /'
fi
if [ "$ok" -eq 1 ]; then
  echo "== 둘 다 정상 — 이제 접속 끊어도 됨. 결과: bash scripts/result.sh 7bm / 14bm"
else
  echo "== 위 '사인' 부분을 사진으로 찍어 전달할 것"
fi
