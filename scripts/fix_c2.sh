#!/usr/bin/env bash
# Legacy C2 replay depended on invalid in-place validation deepening.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[abort] legacy C2 replay mutates immutable val_k; use a new run with larger VAL_K"
exit 2
source scripts/setup_env.sh
source scripts/_find_root.sh 7b
PY="$VENV_DIR/bin/python"
MODEL="${MODEL:-$MODELS_DIR/Qwen2.5-7B-Instruct}"

# 비어있는 GPU 자동 선택 (14B 실행과 충돌 방지 — 여유 30GB+ 필요)
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F', ' '$2 < 40000 {print $1; exit}')
[ -n "${GPU:-}" ] || { echo "[abort] 여유 GPU 없음 (14B가 전부 점유 중이면 완료 후 재실행)"; exit 1; }
echo "== GPU $GPU 사용 (여유 확인됨)"

for D in "$OUT_ROOT"/drift*; do
  [ -d "$D" ] || continue
  case "$(basename "$D")" in drift_*) continue;; esac
  ADAPTER=$(ls -d "$D"/drift_* 2>/dev/null | head -1)
  [ -n "$ADAPTER" ] || continue
  echo "== [1/3] val 심화: $(basename "$D") (+K=24)"
  GPU="$GPU" CUDA_VISIBLE_DEVICES="$GPU" "$PY" src/experiment.py --stage val-deepen \
    --run "$D" --model "$MODEL" --adapter "$ADAPTER" --val-k 24 || exit 1
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" src/experiment.py --stage val-grads \
    --run "$D" --model "$MODEL" --adapter "$ADAPTER" || exit 1
done

echo "== [2/3] 스윕 재판정"
"$PY" src/c2_sweep.py "$OUT_ROOT"
echo "== [3/3] 진단"
"$PY" src/c2_diagnose.py "$OUT_ROOT"
echo "== 끝 — '← C2 PASS' 줄이 있으면 게이트 5/5, 없으면 하한(불가능성) 서사로 확정"
