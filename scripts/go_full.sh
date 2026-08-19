#!/usr/bin/env bash
# 풀 패키지 원샷 — "할 거면 다": 5-seed 전체 + math500 + downstream 반복.
# tmux 포그라운드에서:
#   bash scripts/go_full.sh
# 전 단계가 DONE/산출물 스킵이라 끊겨도 같은 명령으로 재개된다.
#   A. gsm8k + dapo-math × seed 0~4 (n=512·val 100)
#   B. math500 × seed 0~4 (n=400+100 — 500문제 전량, k=40)
#   C. downstream 4소스(oracle/g10/g01/random) × gsm8k 완주 seed (drift 100)
# 축소 실행: SEEDS_ALL="0 1 2" bash scripts/go_full.sh
set -uo pipefail
cd "$(dirname "$0")/.."

if pgrep -f "scripts/go_v2.sh" >/dev/null; then
  echo "[abort] go_v2.sh 실행 중 — 완주 후 재실행 (pgrep -f go_v2.sh 로 확인)"
  exit 1
fi

source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || { echo "[abort] venv 없음 — provision.sh 먼저"; exit 1; }
MODEL="${MODEL_14B:-$MODELS_DIR/Qwen2.5-7B-Instruct}"
SEEDS_ALL="${SEEDS_ALL:-0 1 2 3 4}"

# 데이터 사전 점검 — 없으면 어느 run이 스킵될지 미리 알려준다
"$PY" -c "import sys; sys.path.insert(0,'src'); from data import load_prompts; \
load_prompts('dapo-math', 512, 100)" >/dev/null 2>&1 \
  || echo "[warn] dapo-math 배치본 없음 — A의 dapo run들이 스킵된다. 온라인 셸: bash scripts/fetch_datasets.sh dapo-math"

echo "== [A] gsm8k + dapo-math × seeds($SEEDS_ALL), n=512"
SEEDS="$SEEDS_ALL" DATASETS="gsm8k dapo-math" bash scripts/go_v2.sh \
  || { echo "[abort] A 실패 — 로그 확인 후 같은 명령으로 재개"; exit 1; }

echo "== [B] math500 × seeds($SEEDS_ALL), n=400+100"
if "$PY" -c "import sys; sys.path.insert(0,'src'); from data import load_prompts; \
r=load_prompts('math500',400,100); print('[preflight] math500 OK —', len(r['train']), '/', len(r['val']))"; then
  SEEDS="$SEEDS_ALL" DATASETS="math500" N_TRAIN=400 N_VAL=100 bash scripts/go_v2.sh \
    || { echo "[abort] B 실패 — 같은 명령으로 재개"; exit 1; }
else
  echo "[warn] math500 데이터 없음 — B 건너뜀. 온라인 셸: bash scripts/fetch_datasets.sh math500"
fi

echo "== [C] downstream 4소스 × {gsm8k, dapo} seeds — drift 100 기준"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "${NGPU:-0}" -ge 1 ] || NGPU=1
if [ -n "${OM_GPUS:-}" ]; then
  IFS=',' read -r -a GPUS <<< "$OM_GPUS"
else
  GPUS=($(seq 0 $((NGPU - 1))))
fi
NGPU=${#GPUS[@]}
# dapo 포함 — 신호 큰 체제에서 선택의 실제 가치(신-헤드라인의 downstream 증거)
C_DIRS=()
for s in $SEEDS_ALL; do
  for suf in "" "-dapo-math" "-dapo-math-dapo-math"; do
    dd="$OM_WORK/runs/v2-s$s$suf"
    # legacy 이중 접미사는 신형 완주가 있으면 제외 — 같은 seed의 downstream 중복 실행 방지
    newer="$OM_WORK/runs/v2-s$s-dapo-math"
    [ "$suf" = "-dapo-math-dapo-math" ] && [ -f "$newer/DONE" ] \
      && [ -f "$newer/score_protocol.json" ] && [ -f "$newer/oracle_protocol.json" ] \
      && continue
    [ -f "$dd/DONE" ] && [ -f "$dd/score_protocol.json" ] \
      && [ -f "$dd/oracle_protocol.json" ] && C_DIRS+=("$dd")
  done
done
for d in "${C_DIRS[@]}"; do
  mkdir -p "$d/logs"
  mapfile -t RUN_META < <("$PY" - "$d/run_config.json" <<'PYEOF'
import json, sys
config = json.load(open(sys.argv[1]))
for key in ("dataset", "n_train", "n_val"):
    print(config[key])
PYEOF
  ) || { echo "[abort] run metadata 읽기 실패: $d/run_config.json"; exit 1; }
  [ "${#RUN_META[@]}" -eq 3 ] || { echo "[abort] run metadata 불완전: $d"; exit 1; }
  RUN_DATASET="${RUN_META[0]}"
  RUN_N_TRAIN="${RUN_META[1]}"
  RUN_N_VAL="${RUN_META[2]}"
  pids=(); names=(); slot=0; failed=0
  wait_wave() {
    local j
    for j in "${!pids[@]}"; do
      if ! wait "${pids[$j]}"; then
        echo "  ✘ $(basename "$d")/${names[$j]} — downstream 실패"
        failed=1
      fi
    done
    pids=(); names=(); slot=0
  }
  for src in oracle g10 g01 random; do
    out="$d/downstream_${src}.json"
    [ -f "$out" ] && { echo "  [skip] $(basename "$d")/$src 완료분 존재"; continue; }
    gpu="${GPUS[$slot]}"; slot=$((slot + 1))
    ( if CUDA_VISIBLE_DEVICES=$gpu "$PY" src/experiment.py --stage downstream \
          --run "$d" --model "$MODEL" --dataset "$RUN_DATASET" \
          --n-train "$RUN_N_TRAIN" --n-val "$RUN_N_VAL" \
          --downstream-source "$src" > "$d/logs/downstream-$src.log" 2>&1; then
        echo "  ✔ $(basename "$d")/$src"
      else
        echo "  ✘ $(basename "$d")/$src — tail: $(tail -1 "$d/logs/downstream-$src.log" 2>/dev/null | cut -c1-80)"
        exit 1
      fi ) &
    pids+=($!)
    names+=("$src")
    [ "${#pids[@]}" -ge "$NGPU" ] && wait_wave
  done
  [ "${#pids[@]}" -gt 0 ] && wait_wave
  [ "$failed" -eq 0 ] || exit 1
done

echo "== 종료 요약 =="
for s in $SEEDS_ALL; do
  for tag in "" "-dapo-math" "-math500"; do
    d="$OM_WORK/runs/v2-s$s$tag"
    [ -d "$d" ] && echo "  v2-s$s$tag: $([ -f "$d/DONE" ] && echo ✔ || echo ✘)"
  done
done
echo "== 끝 — 표: bash scripts/tables.sh · frontier: bash scripts/frontier.sh (v2-s* 전부 자동 포함)"
