#!/usr/bin/env bash
# GPU 자가진단 — 하드웨어/드라이버 문제와 attention 커널 문제를 분리 판정.
#   bash scripts/gpu_check.sh
# GPU마다: ① 순수 matmul(기본 연산) ② SDPA(fused attention 커널) 를 각각 실행.
#   matmul 실패 → 그 GPU/드라이버 자체가 병듦 (노드 교체·재부팅 대상)
#   matmul OK + SDPA 실패 → 커널 문제 (OM_ATTN=eager 로 우회 가능)
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "== GPU ${NGPU}장 진단 시작 (드라이버: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1))"
fail=0
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES="$i" timeout 120 "$PY" - <<'EOF' || fail=1
import sys
import torch
try:
    torch.backends.cuda.enable_cudnn_sdp(False)
except AttributeError:
    pass
i = torch.cuda.current_device()
name = torch.cuda.get_device_name(i)
try:
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    (a @ a).sum().item()
    torch.cuda.synchronize()
    print(f"  matmul OK   ({name})")
except Exception as e:
    print(f"  matmul FAIL ({name}): {type(e).__name__}: {e} → 이 GPU/드라이버 문제, 노드 교체 대상")
    sys.exit(1)
try:
    q = torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.bfloat16)
    torch.nn.functional.scaled_dot_product_attention(q, q, q).sum().item()
    torch.cuda.synchronize()
    print("  sdpa   OK")
except Exception as e:
    print(f"  sdpa   FAIL: {type(e).__name__}: {e} → OM_ATTN=eager 로 우회 가능")
    sys.exit(1)
EOF
  echo "-- GPU $i 완료"
done
if [ "$fail" -eq 0 ]; then
  echo "== 전 GPU 정상 — CUDA 에러가 계속되면 OM_ATTN=eager 를 붙여 재실행"
else
  echo "== 실패 GPU 있음 — 위 판정대로 조치 (matmul FAIL이면 하드웨어, sdpa FAIL이면 OM_ATTN=eager)"
fi
exit "$fail"
