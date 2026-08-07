#!/usr/bin/env bash
# 실행 중단 + 산출물 정리 후 재시작 준비.
#
#   bash scripts/reset_run.sh          # soft: 프로세스 종료 + 투영 기반 산출물만 삭제
#                                      #       (rollout·drift adapter·prompts는 보존 → 재개 빠름)
#   bash scripts/reset_run.sh --hard   # run 디렉토리 전체 삭제 (처음부터)
#   bash scripts/reset_run.sh --dry-run  # 지울 것만 보여주고 안 지움
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
OUT_ROOT="${OUT_ROOT:-$OM_WORK/runs/gate}"

HARD=0; DRY=0
for a in "$@"; do case "$a" in
  --hard) HARD=1 ;;
  --dry-run) DRY=1 ;;
  *) echo "알 수 없는 옵션: $a"; exit 2 ;;
esac; done
RM="rm -rfv"; [ "$DRY" = 1 ] && RM="echo [dry-run] would remove:"

echo "== 1) 실행 프로세스 종료"
pkill -f "src/experiment.py" 2>/dev/null && echo "experiment.py 종료" || echo "실행 중인 experiment.py 없음"
pkill -f "run_h100_all.sh" 2>/dev/null || true
sleep 2

echo "== 2) GPU 점유 확인"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,memory.used --format=csv
  LEFT=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [ "$LEFT" -gt 0 ] && { echo "경고: 아직 GPU 프로세스 ${LEFT}개 남음:"; \
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; \
    echo "  → 'kill <PID>' 후 다시 실행"; }
fi

echo "== 3) 산출물 정리 ($OUT_ROOT)"
if [ ! -d "$OUT_ROOT" ]; then
  echo "run 디렉토리 없음 — 정리할 것 없음"
elif [ "$HARD" = 1 ]; then
  $RM "$OUT_ROOT"
else
  # 투영(gradient)이 들어간 산출물만 — 신·구 투영 혼입 방지.
  # 보존: prompts.json, rollouts_*.jsonl, drift_* adapter, logs/
  for run in "$OUT_ROOT"/drift*; do
    [ -d "$run" ] || continue
    $RM "$run"/val_gradient.pt "$run"/val_groups.pt "$run"/oracle_micro_groups.pt \
        "$run"/scores_*.json "$run"/report.md "$run"/report.json \
        "$run"/rollouts_hybrid_*.jsonl "$run"/downstream_* 2>/dev/null || true
  done
  echo "보존됨: shared/ (β rollout), drift*/rollouts_fresh_*.jsonl, drift*/drift_* (adapter)"
fi

echo "== 완료. 다음: git pull && bash scripts/run_h100_all.sh"
