#!/usr/bin/env bash
# 재실험 사다리 — 원인 미상의 반복 실패 시 단계적 재시도 (tmux 포그라운드):
#   bash scripts/go_retry.sh
#
# R1  같은 노드 + cublasLt 폴백(DISABLE_ADDMM_CUDA_LT=1)으로 "검증런"
#     — gsm8k seed0 하나만 (완주돼 있으면 즉시 통과). 몇 시간 안에 판정.
# R1 통과 → 같은 env로 A(gsm8k+dapo 5-seed)·B(math500 5-seed) 전체 재개.
# R1 실패 → 진단 리포트(DIAGNOSIS.txt) 자동 생성 후 중단 — 노드 교체 지시.
#
# 모든 단계 DONE 스킵 — 어떤 시점에 죽어도 같은 명령으로 재개.
set -uo pipefail
cd "$(dirname "$0")/.."
if pgrep -f "scripts/go_v2.sh" >/dev/null || pgrep -f "scripts/go_full.sh" >/dev/null; then
  echo "[abort] go_v2/go_full 실행 중 — 죽었다고 판단되면: pkill -f go_v2; pkill -f go_full 후 재실행"
  exit 1
fi
export DISABLE_ADDMM_CUDA_LT=1
echo "== [R1] cublasLt 폴백 검증런 — gsm8k seed0 단독 (완주분 스킵)"
SEEDS="0" DATASETS="gsm8k" bash scripts/go_v2.sh 2>&1 | tee retry-probe.log || true

if grep -q "gsm8k/s0 ✔" retry-probe.log; then
  echo
  echo "== [R1 통과] 같은 env(cublasLt 폴백)로 전체 재개"
  SEEDS="${SEEDS_ALL:-0 1 2 3 4}" DATASETS="gsm8k dapo-math" bash scripts/go_v2.sh 2>&1 | tee -a go_full.console.log
  SEEDS="${SEEDS_ALL:-0 1 2 3 4}" DATASETS="math500" N_TRAIN=400 N_VAL=100 bash scripts/go_v2.sh 2>&1 | tee -a go_full.console.log
  echo "== 재실험 완료 — bash scripts/backup_results.sh 실행 권장"
else
  echo
  echo "== [R1 실패] 이 노드에서는 폴백으로도 완주 불가 — 진단 리포트 생성"
  bash scripts/diagnose.sh || true
  echo
  echo "== 다음 행동 (RECOVERY.md 상황 1):"
  echo "   1) DIAGNOSIS.txt 를 사진으로 전달"
  echo "   2) 다른 인스턴스에서: git pull → bash scripts/provision.sh → bash scripts/preflight.sh"
  echo "      → bash scripts/go_full.sh   (완주분은 group-volume 덕에 전부 스킵)"
  exit 1
fi
