#!/usr/bin/env bash
# 7B+14B MATH-500 동시 실행 (한 노드, GPU 반반):  bash scripts/go7_14.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
N=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$N" -ge 2 ] || { echo "[abort] GPU 2장 이상 필요 (감지: $N)"; exit 1; }
H=$((N / 2))
G7=$(seq -s, 0 $((H - 1)))
G14=$(seq -s, "$H" $((N - 1)))
echo "== GPU 분할: 7B=[$G7]  14B=[$G14]"
OM_GPUS="$G7" bash scripts/go7m.sh
OM_GPUS="$G14" bash scripts/go14m.sh
echo "== 둘 다 시작됨. 진행:  tail -3 7b-math.log 14b-math.log"
echo "== 결과:  bash scripts/result.sh 7bm   /   bash scripts/result.sh 14bm"
