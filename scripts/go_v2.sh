#!/usr/bin/env bash
# v2 교정 파이프라인 본실행 (감사 P0 반영판) — tmux 포그라운드 원샷:
#   bash scripts/go_v2.sh
# 절차: GPU 건강검사 → 30분 스모크(전 스테이지 완주 확인) → 3-seed × {gsm8k, dapo-math}
#       n=512·val 100·fresh 32·hybrid 64 — 죽으면 자동 재개(2회), DONE 스킵
# 끝나면 결과 일체를 $OM_WORK/results/v2/ 로 수집.
#
# 병렬: PAR=2 bash scripts/go_v2.sh  → GPU를 PAR개 그룹으로 나눠 seed×dataset job을
#       동시에 PAR개 돌린다(run_14b.sh의 OM_GPUS 분할). 기본 PAR=1은 종전과 동일한
#       완전 순차 — 6 run이 2~3일 걸리던 원인. 8장이면 PAR=2(4장씩)가 안전한 기본.
#       한 그룹은 최소 2장(val-grads ∥ oracle-grads 분리)을 권장.
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
DATASETS=(${DATASETS:-gsm8k dapo-math})
SEEDS=(${SEEDS:-0 1 2})

# 상세 진행 워처 — 5분 무변화마다 심장박동(무출력 스테이지 vs 진짜 hang 판별용)
( prev=""; still=0
  while :; do
    sleep 15
    lf=$(ls -t "$BASE"*/logs/*.log 2>/dev/null | head -1); [ -n "$lf" ] || continue
    line=$(tail -n 1 "$lf" 2>/dev/null | cut -c1-120)
    if [ -n "$line" ] && [ "$line" != "$prev" ]; then
      echo "[detail·$(basename "$lf" .log)] $line"; prev="$line"; still=0
    else
      still=$((still + 1))
      if [ $((still % 20)) -eq 0 ]; then
        util=$(timeout 10 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)
        echo "[워처] 로그 $((still * 15 / 60))분째 그대로 (GPU util ${util:-측정불가}%) — util>0이면 무출력 스테이지 진행 중(놔둘 것), 0%가 계속이면 hang → Ctrl+C 후 같은 명령 재실행(저장분 스킵)"
      fi
    fi
  done ) &
W=$!
# 좀비 정리는 **해당 run 디렉터리의 프로세스만** — "--run $BASE" 접두 매치는 같은
# BASE 접두를 쓰는 다른 인스턴스(별도 seed 병렬 실행)와 형제 run까지 죽인다.
cleanup_run() { pkill -f -- "--run $1( |\$)" 2>/dev/null || true; \
  find "${HF_HOME:-/nonexistent}" -name '*.lock' -mmin +30 -delete 2>/dev/null || true; sleep 5; }
cleanup_all_mine() { for d in "${MY_RUNS[@]:-}"; do [ -n "$d" ] && cleanup_run "$d"; done; }
MY_RUNS=()
trap 'echo "== 중단 — 이 인스턴스의 run 정리"; cleanup_all_mine; kill $W 2>/dev/null; exit 130' INT TERM

echo
echo "== [1] 스모크 (~30분): 교정 파이프라인 전 스테이지가 실제로 완주하는지 먼저 확인"
SMOKE="$BASE-smoke"
if [ -f "$SMOKE/report.json" ] && [ -f "$SMOKE/score_protocol.json" ] \
   && [ -f "$SMOKE/oracle_protocol.json" ] \
   && ls "$SMOKE"/scores_hybrid_*.json >/dev/null 2>&1; then
  echo "   스모크 산출물 존재 — 스킵"
else
  MY_RUNS+=("$SMOKE"); cleanup_run "$SMOKE"
  if ! DATASET=gsm8k OUT_ROOT="$SMOKE" N_TRAIN=32 N_VAL=16 FRESH_K=8 \
       HYBRID_PROMPTS=8 SEED=0 bash scripts/run_14b.sh > "$LOGDIR/v2-smoke.log" 2>&1; then
    echo "== [중단] 스모크 실패 — 본실행 진입 안 함. 사인:"
    tail -8 "$LOGDIR/v2-smoke.log" | sed 's/^/   /'
    cleanup_run "$SMOKE"; kill $W 2>/dev/null; exit 1
  fi
  WANTS="report.json score_protocol.json oracle_protocol.json divergence_stats.shard0.json manifest.json"
  [ "${OM_SKIP_HYBRID:-0}" = "1" ] || WANTS="$WANTS scores_hybrid_0.5.json"
  for want in $WANTS; do
    ls "$SMOKE"/$want >/dev/null 2>&1 || { echo "== [중단] 스모크 산출물 누락: $want"; kill $W 2>/dev/null; exit 1; }
  done
  echo "   스모크 ✔ ($WANTS 확인)"
fi

export N_TRAIN="${N_TRAIN:-512}" N_VAL="${N_VAL:-100}"
export FRESH_K="${FRESH_K:-32}" HYBRID_PROMPTS="${HYBRID_PROMPTS:-64}"
RESDIR="$LOGDIR/v2-results.$$"; mkdir -p "$RESDIR"
run_dir_of() { local d="$BASE-s$1"; [ "$2" != "gsm8k" ] && d="$d-$2"; echo "$d"; }
# 한 job(seed×dataset) — 죽으면 2회 재시도, 좀비 정리는 이 run 디렉터리 한정.
# 죽은 런의 experiment.py가 모델 한 벌(27B≈52GB)을 문 채 남아 있으면 drift 재로드가
# "48.63GB 할당 실패/27.57GB 잔여" 꼴로 같은 자리 OOM 반복 — 그래서 시도 전 정리.
run_job() {  # run_job <SEED> <DS> <OM_GPUS 또는 빈문자열>
  local SEED="$1" DS="$2" GSET="$3"
  local RUN_DIR KEY LOG ok try
  RUN_DIR=$(run_dir_of "$SEED" "$DS"); KEY="$DS/s$SEED"; LOG="$LOGDIR/v2-$DS-s$SEED.log"
  echo "==== [$KEY] → $RUN_DIR (log: $LOG${GSET:+, GPU $GSET})"
  if [ -f "$RUN_DIR/DONE" ] && [ -f "$RUN_DIR/score_protocol.json" ] \
     && [ -f "$RUN_DIR/oracle_protocol.json" ]; then
    echo "==== [$KEY] ✔ 완주(DONE+protocols) — 스킵"; echo 1 > "$RESDIR/$DS-s$SEED"; return 0
  fi
  ok=0
  for try in 1 2; do
    echo "==== [$KEY] 시도 $try/2"
    cleanup_run "$RUN_DIR"
    # 확장 결과로 생긴 VAR=값 단어는 bash가 환경 대입으로 안 본다 — export로 처리
    # (run_job은 워커 서브셸 안에서 돌아 밖으로 새지 않음; PAR=1이면 GSET이 비어 종전과 동일)
    [ -n "$GSET" ] && export OM_GPUS="$GSET"
    if DATASET="$DS" OUT_ROOT="$RUN_DIR" SEED="$SEED" \
         bash scripts/run_14b.sh >> "$LOG" 2>&1; then
      ok=1; echo "==== [$KEY] ✔ 완주"; break
    fi
    echo "==== [$KEY] ✘ 실패 — tail:"; tail -4 "$LOG" | sed 's/^/     /'
    grep -q "\[abort\].*데이터" "$LOG" && { echo "==== [$KEY] 데이터 문제 — 스킵"; break; }
    sleep 20
  done
  echo "$ok" > "$RESDIR/$DS-s$SEED"
}
JOBS=()
for SEED in "${SEEDS[@]}"; do for DS in "${DATASETS[@]}"; do
  JOBS+=("$SEED $DS"); MY_RUNS+=("$(run_dir_of "$SEED" "$DS")")
done; done

PAR="${PAR:-1}"
if [ "$PAR" -le 1 ]; then
  for job in "${JOBS[@]}"; do echo; run_job $job ""; done
else
  # GPU를 PAR개 그룹으로 라운드로빈 분할 → 워커 j는 자기 그룹으로 job j, j+PAR, ... 순차
  if [ -n "${OM_GPUS:-}" ]; then IFS=',' read -r -a ALLG <<< "$OM_GPUS"; else ALLG=($(seq 0 $((N - 1)))); fi
  [ "${#ALLG[@]}" -ge "$PAR" ] || { echo "[abort] GPU ${#ALLG[@]}장을 PAR=$PAR 그룹으로 못 나눔"; kill $W 2>/dev/null; exit 1; }
  declare -a GROUP
  for ((g = 0; g < ${#ALLG[@]}; g++)); do
    j=$((g % PAR)); GROUP[$j]="${GROUP[$j]:+${GROUP[$j]},}${ALLG[$g]}"
  done
  echo "== 병렬 PAR=$PAR — GPU 그룹: ${GROUP[*]}"
  wpids=()
  for ((j = 0; j < PAR; j++)); do
    (
      for ((i = j; i < ${#JOBS[@]}; i += PAR)); do
        echo; run_job ${JOBS[$i]} "${GROUP[$j]}"
      done
    ) & wpids+=($!)
  done
  for p in "${wpids[@]}"; do wait "$p"; done
fi
declare -A RESULT
for job in "${JOBS[@]}"; do
  set -- $job; RESULT["$2/s$1"]=$(cat "$RESDIR/$2-s$1" 2>/dev/null || echo 0)
done

cleanup_all_mine; kill "$W" 2>/dev/null
echo
echo "==== 종료 요약 ===="
DIRS=()
for SEED in "${SEEDS[@]}"; do for DS in "${DATASETS[@]}"; do
  RUN_DIR=$(run_dir_of "$SEED" "$DS")
  KEY="$DS/s$SEED"
  if [ "${RESULT[$KEY]:-0}" = "1" ]; then
    echo "  $KEY ✔"
    [ -f "$RUN_DIR/report.json" ] && [ -f "$RUN_DIR/score_protocol.json" ] \
      && [ -f "$RUN_DIR/oracle_protocol.json" ] && DIRS+=("$RUN_DIR")
  else echo "  $KEY ✘ ($LOGDIR/v2-$DS-s$SEED.log 확인)"; fi
done; done

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
