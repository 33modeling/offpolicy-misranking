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

echo "== [C] downstream 4소스 × gsm8k seeds — drift 100 기준"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
for s in $SEEDS_ALL; do
  d="$OM_WORK/runs/v2-s$s"
  [ -f "$d/DONE" ] || { echo "  [skip] v2-s$s 미완주 — downstream 생략"; continue; }
  mkdir -p "$d/logs"
  pids=(); i=0
  for src in oracle g10 g01 random; do
    out="$d/downstream_${src}.json"
    [ -f "$out" ] && { echo "  [skip] s$s/$src 완료분 존재"; continue; }
    gpu=$((i % NGPU)); i=$((i + 1))
    ( if CUDA_VISIBLE_DEVICES=$gpu "$PY" src/experiment.py --stage downstream \
          --run "$d" --model "$MODEL" --dataset gsm8k --n-train 512 --n-val 100 \
          --downstream-source "$src" > "$d/logs/downstream-$src.log" 2>&1; then
        echo "  ✔ s$s/$src"
      else
        echo "  ✘ s$s/$src — tail: $(tail -1 "$d/logs/downstream-$src.log" 2>/dev/null | cut -c1-80)"
      fi ) &
    pids+=($!)
  done
  [ "${#pids[@]}" -gt 0 ] && wait "${pids[@]}" 2>/dev/null
done

echo "== 종료 요약 =="
for s in $SEEDS_ALL; do
  for tag in "" "-dapo-math" "-math500"; do
    d="$OM_WORK/runs/v2-s$s$tag"
    [ -d "$d" ] && echo "  v2-s$s$tag: $([ -f "$d/DONE" ] && echo ✔ || echo ✘)"
  done
done
echo "== 끝 — 표: bash scripts/tables.sh · frontier: bash scripts/frontier.sh (v2-s* 전부 자동 포함)"
