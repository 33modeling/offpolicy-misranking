#!/usr/bin/env bash
# v2 교정 파이프라인 본실행 (감사 P0 반영판) — tmux 포그라운드 원샷:
#   bash scripts/go_v2.sh
# 절차: GPU 건강검사 → 30분 스모크(전 스테이지 완주 확인) → 3-seed × {gsm8k, dapo-math}
#       n=512·val 100·fresh 32·hybrid 64 — 죽으면 자동 재개(2회), DONE 스킵
# 끝나면 결과 일체를 $OM_WORK/results/v2/ 로 수집.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv python 없음: $PY"; exit 1; }
N=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "== GPU ${N}장 감지"

echo "== [0] GPU 건강검사"
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
[ "$sick" -eq 0 ] || { echo "== [중단] 병든 GPU — 다른 인스턴스에서"; exit 1; }

export MODEL_14B="${MODEL_14B:-$MODELS_DIR/Qwen2.5-7B-Instruct}"
BASE="$OM_WORK/runs/v2"
DATASETS=(${DATASETS:-gsm8k dapo-math})
SEEDS=(${SEEDS:-0 1 2})

# 상세 진행 워처
( prev=""
  while :; do
    sleep 15
    lf=$(ls -t "$BASE"*/logs/*.log 2>/dev/null | head -1); [ -n "$lf" ] || continue
    line=$(tail -n 1 "$lf" 2>/dev/null | cut -c1-120)
    [ -n "$line" ] && [ "$line" != "$prev" ] && { echo "[detail·$(basename "$lf" .log)] $line"; prev="$line"; }
  done ) &
W=$!
cleanup_strays() { pkill -f -- "--run $BASE" 2>/dev/null || true; pkill -f gpu_keepalive 2>/dev/null || true; sleep 5; }
trap 'echo "== 중단 — 전체 정리"; cleanup_strays; kill $W 2>/dev/null; exit 130' INT TERM

echo
echo "== [1] 스모크 (~30분): 교정 파이프라인 전 스테이지가 실제로 완주하는지 먼저 확인"
SMOKE="$BASE-smoke"
if [ -f "$SMOKE/report.json" ] && ls "$SMOKE"/scores_hybrid_*.json >/dev/null 2>&1; then
  echo "   스모크 산출물 존재 — 스킵"
else
  cleanup_strays
  if ! DATASET=gsm8k OUT_ROOT="$SMOKE" N_TRAIN=32 N_VAL=16 FRESH_K=8 \
       HYBRID_PROMPTS=8 SEED=0 bash scripts/run_14b.sh > v2-smoke.log 2>&1; then
    echo "== [중단] 스모크 실패 — 본실행 진입 안 함. 사인:"
    tail -8 v2-smoke.log | sed 's/^/   /'
    cleanup_strays; kill $W 2>/dev/null; exit 1
  fi
  for want in report.json scores_hybrid_0.5.json divergence_stats.shard0.json manifest.json; do
    ls "$SMOKE"/$want >/dev/null 2>&1 || { echo "== [중단] 스모크 산출물 누락: $want"; kill $W 2>/dev/null; exit 1; }
  done
  echo "   스모크 ✔ (report·hybrid 4cell·divergence·manifest 확인)"
fi

export N_TRAIN="${N_TRAIN:-512}" N_VAL="${N_VAL:-100}"
export FRESH_K="${FRESH_K:-32}" HYBRID_PROMPTS="${HYBRID_PROMPTS:-64}"
declare -A RESULT
for SEED in "${SEEDS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    RUN_DIR="$BASE-s$SEED"; [ "$DS" != "gsm8k" ] && RUN_DIR="$RUN_DIR-$DS"
    KEY="$DS/s$SEED"; LOG="v2-$DS-s$SEED.log"
    echo
    echo "==== [$KEY] → $RUN_DIR (log: $LOG)"
    if [ -f "$RUN_DIR/DONE" ]; then echo "==== [$KEY] ✔ 완주(DONE) — 스킵"; RESULT[$KEY]=1; continue; fi
    ok=0
    for try in 1 2; do
      echo "==== [$KEY] 시도 $try/2"
      cleanup_strays
      if DATASET="$DS" OUT_ROOT="$RUN_DIR" SEED="$SEED" bash scripts/run_14b.sh >> "$LOG" 2>&1; then
        ok=1; echo "==== [$KEY] ✔ 완주"; break
      fi
      echo "==== [$KEY] ✘ 실패 — tail:"; tail -4 "$LOG" | sed 's/^/     /'
      grep -q "\[abort\].*데이터" "$LOG" && { echo "==== [$KEY] 데이터 문제 — 스킵"; break; }
      sleep 20
    done
    RESULT[$KEY]=$ok
  done
done

cleanup_strays; kill "$W" 2>/dev/null
# 사용률 기준 잡 킬 대응 — 이후 수집·표·frontier는 CPU 구간이라 GPU 사용률이
# 0으로 떨어지면 잡이 죽는다. 끝날 때까지 저강도 keepalive 유지.
ALLDEV=$(seq -s, 0 $((N - 1)))
CUDA_VISIBLE_DEVICES="$ALLDEV" "$PY" scripts/gpu_keepalive.py > /dev/null 2>&1 &
KA2=$!
trap 'kill $KA2 2>/dev/null' EXIT
echo
echo "==== 종료 요약 ===="
DIRS=()
for SEED in "${SEEDS[@]}"; do for DS in "${DATASETS[@]}"; do
  RUN_DIR="$BASE-s$SEED"; [ "$DS" != "gsm8k" ] && RUN_DIR="$RUN_DIR-$DS"
  KEY="$DS/s$SEED"
  if [ "${RESULT[$KEY]:-0}" = "1" ]; then echo "  $KEY ✔"; [ -f "$RUN_DIR/report.json" ] && DIRS+=("$RUN_DIR")
  else echo "  $KEY ✘ (v2-$DS-s$SEED.log 확인)"; fi
done; done

echo "==== 결과 수집: $OM_WORK/results/v2 ===="
RD="$OM_WORK/results/v2"; mkdir -p "$RD"
for d in "${DIRS[@]}"; do
  tag=$(basename "$d")
  cp "$d/report.json" "$RD/report-$tag.json" 2>/dev/null || true
  cp "$d/manifest.json" "$RD/manifest-$tag.json" 2>/dev/null || true
  cp "$d"/divergence_stats*.json "$RD/" 2>/dev/null || true
  "$PY" src/judge.py "$d" > "$RD/judge-$tag.txt" 2>&1 || true
done
[ "${#DIRS[@]}" -gt 0 ] && OM_RESULTS="$RD" "$PY" src/make_tables.py "${DIRS[@]}" | tail -3
echo "==== frontier 사후 분석 (비용–품질 Pareto·audit 정책·predictor baseline) ===="
[ "${#DIRS[@]}" -gt 0 ] && OM_RESULTS="$RD" "$PY" src/frontier.py "${DIRS[@]}" | tail -3
echo "== 끝 — $RD 의 TABLES.md·FRONTIER.md·report·judge·manifest 뽑아서 전달"
ls "$RD" 2>/dev/null | head