#!/usr/bin/env bash
# D1 — 실제-RL 체크포인트 재채점 (리뷰어 공격 'LoRA-RFT drift 대표성' 방어).
# 인공 RFT drift 대신 GRPO-lite로 실제 학습된 π에서 misranking이 재현되는지 확인.
#   bash scripts/real_drift_check.sh [SRC_RUN] [ADAPTER]
# 기본: SRC_RUN=$OM_WORK/runs/v2-s0, ADAPTER=$SRC_RUN/downstream_random
# 전제: go_full.sh C단계 완주(= downstream_random 어댑터 존재). β rollout·prompts는
# SRC에서 복사해 재사용(분할 결정적) → drift·β 수집 스킵, fresh/oracle/score/report
# 만 돈다 (GPU 1장, 반나절 이하). 판정: report의 g10/g01 열화 방향이 Table 1
# GSM8K 행과 일치하는가 + divergence 통계(KL̂)로 drift 축 위 위치 기록.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh

SRC="${1:-$OM_WORK/runs/v2-s0}"
ADAPTER="${2:-$SRC/downstream_random}"
DEST="$OM_WORK/runs/v2-realrl"

[ -f "$ADAPTER/adapter_config.json" ] || {
  echo "[abort] GRPO-lite 어댑터 없음: $ADAPTER — go_full.sh C단계(downstream) 먼저"; exit 1; }
[ -f "$SRC/prompts.json" ] || { echo "[abort] SRC prompts 없음: $SRC"; exit 1; }
[ -f "$SRC/rollouts_behavior_train.jsonl" ] || {
  echo "[abort] SRC β rollout 없음: $SRC/rollouts_behavior_train.jsonl"; exit 1; }

mkdir -p "$DEST"
cp -n "$SRC/prompts.json" "$DEST/" 2>/dev/null || true
cp -n "$SRC/rollouts_behavior_train.jsonl" "$DEST/" 2>/dev/null || true
# GRPO 어댑터를 drift_100 자리에 — run_14b가 drift 학습을 스킵하고 이걸 π로 쓴다
if [ ! -f "$DEST/drift_100/adapter_config.json" ]; then
  mkdir -p "$DEST/drift_100"
  cp -r "$ADAPTER/." "$DEST/drift_100/"
fi

echo "== real-RL 재채점 시작 — π = $(basename "$ADAPTER") (GRPO-lite), DEST=$DEST"
DATASET=gsm8k OUT_ROOT="$DEST" N_TRAIN=512 N_VAL=100 SEED=0 bash scripts/run_14b.sh || {
  echo "[abort] 파이프라인 실패 — $DEST/logs 확인"; exit 1; }

echo "== 판정 재료 =="
[ -f "$DEST/report.json" ] && "$VENV_DIR/bin/python" - "$DEST" <<'PYEOF'
import json, sys
from pathlib import Path
r = json.loads((Path(sys.argv[1]) / "report.json").read_text())
print("report keys:", list(r)[:8])
print("→ Table 1 'real-RL π' 행 재료. g10/g01 vs floor 방향을 GSM8K drift 행과 비교할 것.")
PYEOF
echo "== 끝 — 표 반영: bash scripts/tables.sh (v2-realrl 포함 여부는 수동 지정)"
