#!/usr/bin/env bash
# H100 게이트 파일럿 원 스크립트 — drift 3수준(50/100/200) 순차 실행.
# 클러스터 규약: GitHub egress 없음 → HF_ENDPOINT 미러 필수, 산출물은 group-volume.
#
# 사용 (클러스터 노드에서):
#   export HF_ENDPOINT=<HF 미러 URL>            # 필수
#   export OUT_ROOT=<group-volume 산출 경로>     # 예: /group-volume/minsoo3.kim/offpolicy-misranking
#   bash scripts/run_h100_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_ENDPOINT:?HF_ENDPOINT 미러를 설정할 것 (클러스터는 GitHub/HF 직결 불가)}"
OUT_ROOT="${OUT_ROOT:-outputs}"
# 파일럿을 7B로 직행 (2026-08-07 사용자 결정). 1.5B로 낮추려면 MODEL 환경변수로.
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export MODEL

for DRIFT in 50 100 200; do
  RUN="$OUT_ROOT/h100-pilot-drift$DRIFT"
  echo "=== drift $DRIFT → $RUN ==="
  bash scripts/run_gate.sh "$RUN" "$DRIFT" 2>&1 | tee "$RUN.log" || {
    echo "drift $DRIFT 실패 — 로그: $RUN.log"; exit 1; }
done

echo "=== 요약 ==="
for DRIFT in 50 100 200; do
  echo "--- drift $DRIFT"
  cat "$OUT_ROOT/h100-pilot-drift$DRIFT/report.md" || true
done
