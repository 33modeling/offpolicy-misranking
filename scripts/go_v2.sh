#!/usr/bin/env bash
# v2 교정 파이프라인 본실행 (감사 P0 반영판) — tmux 포그라운드 원샷:
#   bash scripts/go_v2.sh
# 절차: GPU 건강검사 → 30분 스모크(전 스테이지 완주 확인) → 3-seed × {gsm8k, dapo-math}
#       n=512·val 100·fresh 32·hybrid 64 — 죽으면 자동 재개(2회), DONE 스킵
# 끝나면 결과 일체를 $OM_WORK/results/v2/ 로 수집.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
LOGDIR="${OM_WORK:-.}/console-logs"; mkdir -p "$LOGDIR"
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv python 없음: $PY"; exit 1; }
N=$(timeout 20 nvidia-smi -L 2>/dev/null | wc -l)
[ "${N:-0}" -ge 1 ] || { echo "[abort] nvidia-smi 무응답/GPU 0장 — 드라이버 wedge 의심, 노드 교체(RECOVERY 상황 1)"; exit 1; }
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
BASE="${RUN_BASE:-$OM_WORK/runs/v2}"   # 다른 모델 세대 런은 RUN_BASE로 폴더 분리 (7B 산출물 충돌 방지)
RUN_LABEL="${RUN_LABEL:-$(basename "$BASE")}"  # v4 등 호출 세대별 console log 격리
DATASETS=(${DATASETS:-gsm8k dapo-math})
SEEDS=(${SEEDS:-0 1 2})
command -v setsid >/dev/null 2>&1 || { echo "[abort] setsid 없음"; exit 1; }
ACTIVE_FILE="$TMPDIR/go-v2-$RUN_LABEL-$$.active"

run_pipeline() {  # run_pipeline <console-log> <command...>
  local console_log=$1 pid rc tmp
  shift
  setsid "$@" >> "$console_log" 2>&1 &
  pid=$!
  tmp="$ACTIVE_FILE.tmp"
  printf '%s\n%s\n' "$pid" "$console_log" > "$tmp"
  mv "$tmp" "$ACTIVE_FILE"
  wait "$pid"
  rc=$?
  rm -f "$ACTIVE_FILE"
  return "$rc"
}

# 상세 진행 워처 — stage 진입 전을 포함해 전체 pipeline 로그가 5분 멈추면
# process group을 종료한다. 산출물별 스킵과 retry loop가 마지막 완료 지점부터 재개한다.
( prev=""; still=0; stall_ticks=$(( ${OM_STALL_MINUTES:-5} * 4 ))
  while :; do
    sleep 15
    if [ ! -s "$ACTIVE_FILE" ]; then prev=""; still=0; continue; fi
    runner_pid=$(sed -n '1p' "$ACTIVE_FILE" 2>/dev/null)
    console_log=$(sed -n '2p' "$ACTIVE_FILE" 2>/dev/null)
    if [ -z "$runner_pid" ] || ! kill -0 "$runner_pid" 2>/dev/null; then
      prev=""; still=0; continue
    fi
    lf=$(ls -t "$console_log" "$BASE"*/logs/*.log 2>/dev/null | head -1)
    [ -n "$lf" ] || continue
    line=$(tail -n 1 "$lf" 2>/dev/null | cut -c1-120)
    sig=$(stat -c '%n:%Y:%s' "$lf" 2>/dev/null || true)
    if [ -n "$sig" ] && [ "$sig" != "$prev" ]; then
      [ -n "$line" ] && echo "[detail·$(basename "$lf" .log)] $line"
      prev="$sig"; still=0
    else
      still=$((still + 1))
      if [ "$still" -ge "$stall_ticks" ]; then
        echo "[워처] 로그 ${OM_STALL_MINUTES:-5}분 무변화 — smoke/run process group 종료 후 자동 재시작"
        kill -TERM -- "-$runner_pid" 2>/dev/null || true
        sleep 5
        kill -KILL -- "-$runner_pid" 2>/dev/null || true
        still=0
      fi
    fi
  done ) &
W=$!
cleanup_strays() {
  if [ -s "$ACTIVE_FILE" ]; then
    active_pid=$(sed -n '1p' "$ACTIVE_FILE" 2>/dev/null)
    [ -z "$active_pid" ] || kill -TERM -- "-$active_pid" 2>/dev/null || true
    rm -f "$ACTIVE_FILE"
  fi
  pkill -f -- "--run $BASE" 2>/dev/null || true; \
  find "${HF_HOME:-/nonexistent}" -name '*.lock' -mmin +30 -delete 2>/dev/null || true; sleep 5; }
trap 'echo "== 중단 — 전체 정리"; cleanup_strays; kill $W 2>/dev/null; exit 130' INT TERM

echo
echo "== [1] 스모크 (~30분): 교정 파이프라인 전 스테이지가 실제로 완주하는지 먼저 확인"
SMOKE="${RUN_BASE_SMOKE:-$BASE-smoke}"
smoke_ready=0
if [ -f "$SMOKE/report.json" ] && [ -f "$SMOKE/score_protocol.json" ] \
   && [ -f "$SMOKE/oracle_protocol.json" ]; then
  if [ "${OM_SKIP_HYBRID:-0}" = "1" ] \
     || ls "$SMOKE"/scores_hybrid_*.json >/dev/null 2>&1; then
    smoke_ready=1
  fi
fi
if [ "$smoke_ready" -eq 1 ]; then
  echo "   스모크 산출물 존재 — 스킵"
else
  cleanup_strays
  smoke_ok=0
  for smoke_try in $(seq 1 "${OM_MAX_RETRIES:-2}"); do
    if run_pipeline "$LOGDIR/$RUN_LABEL-smoke.log" env \
       DATASET=gsm8k OUT_ROOT="$SMOKE" N_TRAIN=32 N_VAL=16 FRESH_K=8 \
       HYBRID_PROMPTS=8 SEED=0 bash scripts/run_14b.sh; then
      smoke_ok=1
      break
    fi
    echo "   스모크 자동 재시작 $smoke_try/${OM_MAX_RETRIES:-2}"
    cleanup_strays
  done
  if [ "$smoke_ok" -ne 1 ]; then
    echo "== [중단] 스모크 실패 — 본실행 진입 안 함. 사인:"
    tail -8 "$LOGDIR/$RUN_LABEL-smoke.log" | sed 's/^/   /'
    cleanup_strays; kill $W 2>/dev/null; exit 1
  fi
  WANTS="report.json score_protocol.json oracle_protocol.json divergence_stats.shard0.json manifest.json"
  [ "${OM_SKIP_HYBRID:-0}" = "1" ] || WANTS="$WANTS scores_hybrid_0.5.json"
  for want in $WANTS; do
    ls "$SMOKE"/$want >/dev/null 2>&1 || { echo "== [중단] 스모크 산출물 누락: $want"; kill $W 2>/dev/null; exit 1; }
  done
  echo "   스모크 ✔ ($WANTS 확인)"
fi

# 본실행 전 좀비 정리 — 반드시 무조건 실행 (스모크 스킵 경로 포함).
# 죽은 런의 experiment.py가 모델 한 벌(27B≈52GB)을 문 채 남아 있으면
# drift 재로드가 "48.63GB 할당 실패/27.57GB 잔여" 꼴로 같은 자리 OOM 반복.
cleanup_strays

export N_TRAIN="${N_TRAIN:-512}" N_VAL="${N_VAL:-100}"
export FRESH_K="${FRESH_K:-32}" HYBRID_PROMPTS="${HYBRID_PROMPTS:-64}"
declare -A RESULT
for SEED in "${SEEDS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    RUN_DIR="$BASE-s$SEED"; [ "$DS" != "gsm8k" ] && RUN_DIR="$RUN_DIR-$DS"
    KEY="$DS/s$SEED"; LOG="$LOGDIR/$RUN_LABEL-$DS-s$SEED.log"
    echo
    echo "==== [$KEY] → $RUN_DIR (log: $LOG)"
    if [ -f "$RUN_DIR/DONE" ] && [ -f "$RUN_DIR/score_protocol.json" ] \
       && [ -f "$RUN_DIR/oracle_protocol.json" ]; then
      echo "==== [$KEY] ✔ 완주(DONE+protocols) — 스킵"; RESULT[$KEY]=1; continue
    fi
    ok=0
    for try in $(seq 1 "${OM_MAX_RETRIES:-2}"); do
      echo "==== [$KEY] 시도 $try/${OM_MAX_RETRIES:-2}"
      cleanup_strays
      if run_pipeline "$LOG" env DATASET="$DS" OUT_ROOT="$RUN_DIR" SEED="$SEED" \
         bash scripts/run_14b.sh; then
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
echo
echo "==== 종료 요약 ===="
DIRS=()
for SEED in "${SEEDS[@]}"; do for DS in "${DATASETS[@]}"; do
  RUN_DIR="$BASE-s$SEED"; [ "$DS" != "gsm8k" ] && RUN_DIR="$RUN_DIR-$DS"
  KEY="$DS/s$SEED"
  if [ "${RESULT[$KEY]:-0}" = "1" ]; then
    echo "  $KEY ✔"
    [ -f "$RUN_DIR/report.json" ] && [ -f "$RUN_DIR/score_protocol.json" ] \
      && [ -f "$RUN_DIR/oracle_protocol.json" ] && DIRS+=("$RUN_DIR")
  else echo "  $KEY ✘ ($LOGDIR/$RUN_LABEL-$DS-s$SEED.log 확인)"; fi
done; done

if [ "${OM_SKIP_POSTPROCESS:-0}" = "1" ]; then
  echo "==== 후처리 생략 (OM_SKIP_POSTPROCESS=1) — 병렬 worker 종료 후 한 번만 집계할 것"
  exit 0
fi

RD="${RESULTS_BASE:-$OM_WORK/results/v2}"; mkdir -p "$RD"
echo "==== 결과 수집: $RD ===="
for d in "${DIRS[@]}"; do
  tag=$(basename "$d")
  cp "$d/report.json" "$RD/report-$tag.json" || exit 1
  cp "$d/manifest.json" "$RD/manifest-$tag.json" || exit 1
  for f in "$d"/divergence_stats*.json; do
    [ -f "$f" ] || { echo "[abort] divergence stats 없음: $d"; exit 1; }
    base=$(basename "$f" .json)
    cp "$f" "$RD/$base-$tag.json" || exit 1
  done
  "$PY" src/judge.py "$d" > "$RD/judge-$tag.txt" 2>&1 || exit 1
done
post_fail=0
if [ "${#DIRS[@]}" -gt 0 ]; then
  OM_RESULTS="$RD" "$PY" src/make_tables.py "${DIRS[@]}" | tail -3 || post_fail=1
fi
echo "==== frontier 사후 분석 (비용–품질 Pareto·audit 정책·predictor baseline) ===="
if [ "${#DIRS[@]}" -gt 0 ]; then
  OM_RESULTS="$RD" "$PY" src/frontier.py "${DIRS[@]}" | tail -3 || post_fail=1
fi
[ "$post_fail" -eq 0 ] || { echo "[abort] 결과 표/frontier 생성 실패"; exit 1; }
echo "== 끝 — $RD 의 TABLES.md·FRONTIER.md·report·judge·manifest 뽑아서 전달"
ls "$RD" 2>/dev/null | head
