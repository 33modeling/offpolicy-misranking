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
PIPELINE_REPO="${OM_PIPELINE_REPO:-$PWD}"
PIPELINE_SCRIPT="${OM_PIPELINE_SCRIPT:-$PIPELINE_REPO/scripts/run_14b.sh}"
[ -f "$PIPELINE_SCRIPT" ] || { echo "[abort] pipeline script 없음: $PIPELINE_SCRIPT"; exit 1; }
ACTIVE_FILE="$TMPDIR/go-v2-$RUN_LABEL-$$.active"

run_complete() {
  local run=$1 artifact
  for artifact in DONE run_config.json manifest.json score_protocol.json \
      oracle_protocol.json report.json scores_oracle.json scores_offpolicy.json \
      scores_splithalf.json oracle_micro_groups.pt val_groups.pt; do
    [ -s "$run/$artifact" ] || return 1
  done
}

run_pipeline() {  # run_pipeline <console-log> <run-dir> <command...>
  local console_log=$1 active_run=$2 pid rc tmp
  shift 2
  setsid "$@" >> "$console_log" 2>&1 &
  pid=$!
  tmp="$ACTIVE_FILE.tmp"
  printf '%s\n%s\n%s\n' "$pid" "$console_log" "$active_run" > "$tmp"
  mv "$tmp" "$ACTIVE_FILE"
  wait "$pid"
  rc=$?
  rm -f "$ACTIVE_FILE"
  return "$rc"
}

group_cpu_seconds() {
  ps -eo pgid=,cputimes= 2>/dev/null \
    | awk -v pgid="$1" '$1 == pgid { total += $2 } END { print total + 0 }'
}

gpu_peak_util() {
  local peak=0 util sample
  for sample in 1 2 3; do
    util=$(timeout 10 nvidia-smi --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk '$1 > peak { peak=$1 } END { print peak + 0 }')
    [ "${util:-0}" -le "$peak" ] || peak=$util
    [ "$sample" -eq 3 ] || sleep 2
  done
  printf '%s\n' "$peak"
}

# 로그 정지만으로 정상 장시간 계산을 죽이지 않는다. 로그가 임계시간 동안
# 멈췄고 GPU와 process group CPU도 함께 정지했을 때만 마지막 체크포인트부터 재개한다.
( prev=""; still=0; watched_pid=""; cpu_mark=0
  stall_ticks=$(( ${OM_STALL_MINUTES:-5} * 4 ))
  while :; do
    sleep 15
    if [ ! -s "$ACTIVE_FILE" ]; then
      prev=""; still=0; watched_pid=""; cpu_mark=0; continue
    fi
    runner_pid=$(sed -n '1p' "$ACTIVE_FILE" 2>/dev/null)
    console_log=$(sed -n '2p' "$ACTIVE_FILE" 2>/dev/null)
    active_run=$(sed -n '3p' "$ACTIVE_FILE" 2>/dev/null)
    if [ -z "$runner_pid" ] || ! kill -0 "$runner_pid" 2>/dev/null; then
      prev=""; still=0; watched_pid=""; cpu_mark=0; continue
    fi
    if [ "$runner_pid" != "$watched_pid" ]; then
      prev=""; still=0; watched_pid=$runner_pid
      cpu_mark=$(group_cpu_seconds "$runner_pid")
    fi
    # 공유 볼륨의 다른 클러스터 로그를 진행 신호로 오인하지 않는다.
    lf=$(ls -t -- "$console_log" "$active_run"/logs/*.log 2>/dev/null | head -1)
    [ -n "$lf" ] || continue
    line=$(tail -n 1 "$lf" 2>/dev/null | cut -c1-120)
    sig=$(stat -c '%n:%Y:%s' "$lf" 2>/dev/null || true)
    if [ -n "$sig" ] && [ "$sig" != "$prev" ]; then
      [ -n "$line" ] && echo "[detail·$(basename "$lf" .log)] $line"
      prev="$sig"; still=0; cpu_mark=$(group_cpu_seconds "$runner_pid")
    else
      still=$((still + 1))
      if [ "$still" -ge "$stall_ticks" ]; then
        cpu_now=$(group_cpu_seconds "$runner_pid")
        cpu_delta=$((cpu_now > cpu_mark ? cpu_now - cpu_mark : 0))
        gpu_peak=$(gpu_peak_util)
        if [ "$gpu_peak" -gt 0 ] || [ "$cpu_delta" -gt 2 ]; then
          message="[워처] 로그 ${OM_STALL_MINUTES:-5}분 무변화지만 계산 활동 확인 (GPU ${gpu_peak}%, CPU +${cpu_delta}s) — 계속 실행"
          echo "$message"
          printf '%s\n' "$message" >> "$console_log"
          cpu_mark=$cpu_now
        else
          message="[워처] 로그·GPU·CPU 모두 ${OM_STALL_MINUTES:-5}분 정지 — process group 종료 후 자동 재시작"
          echo "$message"
          printf '%s\n' "$message" >> "$console_log"
          kill -TERM -- "-$runner_pid" 2>/dev/null || true
          sleep 5
          kill -KILL -- "-$runner_pid" 2>/dev/null || true
        fi
        still=0
      fi
    fi
  done ) &
W=$!
cleanup_strays() {
  active_pid=""
  if [ -s "$ACTIVE_FILE" ]; then
    active_pid=$(sed -n '1p' "$ACTIVE_FILE" 2>/dev/null)
    [ -z "$active_pid" ] || kill -TERM -- "-$active_pid" 2>/dev/null || true
  fi
  pkill -TERM -f -- "--run $BASE" 2>/dev/null || true
  for _ in $(seq 1 20); do
    group_alive=0
    if [ -n "$active_pid" ] && kill -0 -- "-$active_pid" 2>/dev/null; then
      group_alive=1
    fi
    if [ "$group_alive" -eq 0 ] && ! pgrep -f -- "--run $BASE" >/dev/null; then
      break
    fi
    sleep 0.5
  done
  [ -z "$active_pid" ] || kill -KILL -- "-$active_pid" 2>/dev/null || true
  pkill -KILL -f -- "--run $BASE" 2>/dev/null || true
  rm -f "$ACTIVE_FILE"
  find "${HF_HOME:-/nonexistent}" -name '*.lock' -mmin +30 -delete 2>/dev/null || true
  sleep 2
}
trap 'echo "== 중단 — 전체 정리"; cleanup_strays; kill $W 2>/dev/null; exit 130' INT TERM

echo
echo "== [1] 스모크 (~30분): 교정 파이프라인 전 스테이지가 실제로 완주하는지 먼저 확인"
SMOKE="${RUN_BASE_SMOKE:-$BASE-smoke}"
smoke_ready=0
if run_complete "$SMOKE"; then
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
  smoke_rc=0
  for smoke_try in $(seq 1 "${OM_MAX_RETRIES:-2}"); do
    if run_pipeline "$LOGDIR/$RUN_LABEL-smoke.log" "$SMOKE" env \
       DATASET=gsm8k OUT_ROOT="$SMOKE" N_TRAIN=32 N_VAL=16 FRESH_K=8 \
       HYBRID_PROMPTS=8 SEED=0 OM_RETRY_INDEX="$smoke_try" OM_REPO="$PIPELINE_REPO" \
       PYTHONPATH="$PIPELINE_REPO/src" bash "$PIPELINE_SCRIPT"; then
      smoke_ok=1
      break
    else
      smoke_rc=$?
    fi
    echo "   스모크 자동 재시작 $smoke_try/${OM_MAX_RETRIES:-2}"
    bash scripts/diagnose_run_failure.sh \
      "$SMOKE" "$LOGDIR/$RUN_LABEL-smoke.log" "$smoke_rc"
    cleanup_strays
  done
  if [ "$smoke_ok" -ne 1 ]; then
    echo "== [중단] 스모크 실패 — 본실행 진입 안 함"
    echo "   원인은 위 자동 진단 출력에 표시됨"
    cleanup_strays; kill $W 2>/dev/null; exit 1
  fi
  WANTS="DONE run_config.json report.json score_protocol.json oracle_protocol.json divergence_stats.shard0.json manifest.json"
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
    if run_complete "$RUN_DIR"; then
      echo "==== [$KEY] ✔ 완주(필수 artifact 6종) — 스킵"; RESULT[$KEY]=1; continue
    fi
    ok=0
    for try in $(seq 1 "${OM_MAX_RETRIES:-2}"); do
      echo "==== [$KEY] 시도 $try/${OM_MAX_RETRIES:-2}"
      cleanup_strays
      run_rc=0
      if run_pipeline "$LOG" "$RUN_DIR" env DATASET="$DS" OUT_ROOT="$RUN_DIR" SEED="$SEED" \
         OM_RETRY_INDEX="$try" \
         OM_REPO="$PIPELINE_REPO" PYTHONPATH="$PIPELINE_REPO/src" \
         bash "$PIPELINE_SCRIPT"; then
        if run_complete "$RUN_DIR"; then
          ok=1; echo "==== [$KEY] ✔ 완주"; break
        fi
        run_rc=3
        echo "==== [$KEY] ✘ child는 성공했지만 필수 artifact가 불완전"
      else
        run_rc=$?
      fi
      echo "==== [$KEY] ✘ 실패 — 진단:"
      bash scripts/diagnose_run_failure.sh "$RUN_DIR" "$LOG" "$run_rc"
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
FAILED_RUNS=0
for SEED in "${SEEDS[@]}"; do for DS in "${DATASETS[@]}"; do
  RUN_DIR="$BASE-s$SEED"; [ "$DS" != "gsm8k" ] && RUN_DIR="$RUN_DIR-$DS"
  KEY="$DS/s$SEED"
  if [ "${RESULT[$KEY]:-0}" = "1" ]; then
    echo "  $KEY ✔"
    [ -f "$RUN_DIR/report.json" ] && [ -f "$RUN_DIR/score_protocol.json" ] \
      && [ -f "$RUN_DIR/oracle_protocol.json" ] && DIRS+=("$RUN_DIR")
  else
    echo "  $KEY ✘ ($LOGDIR/$RUN_LABEL-$DS-s$SEED.log 확인)"
    FAILED_RUNS=$((FAILED_RUNS + 1))
  fi
done; done

if [ "$FAILED_RUNS" -gt 0 ]; then
  echo "==== [실패] ${FAILED_RUNS}개 run 미완료 — 성공으로 종료하지 않음"
  exit 1
fi

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
