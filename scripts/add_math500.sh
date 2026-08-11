#!/usr/bin/env bash
# math500 3-seed 추가 실행 — v2 본실행(gsm8k·dapo) **완주 후에만**, tmux 포그라운드:
#   bash scripts/add_math500.sh
# MATH-500은 500문제뿐이라 v2 기본 n=512+100이 안 들어간다 → 400+100(전량 사용).
# 기존 v2-s*(gsm8k)·v2-s*-dapo-math 완주분은 DONE 스킵으로 건드리지 않고
# v2-s*-math500 세 개만 돈다. k = top-10% of 400 = 40 (입도 0.025).
set -uo pipefail
cd "$(dirname "$0")/.."

# 동시 실행 방지 — go_v2가 돌고 있으면 cleanup_strays가 서로의 프로세스를 죽인다
if pgrep -f "scripts/go_v2.sh" >/dev/null; then
  echo "[abort] go_v2.sh 실행 중 — 완주 후에 다시. (pgrep -f go_v2.sh 로 확인)"
  exit 1
fi

source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3

# 데이터 사전 확인 — 오프라인 노드에서 스모크·본실행 헛돌기 방지
if ! "$PY" -c "
import sys; sys.path.insert(0, 'src')
from data import load_prompts
r = load_prompts('math500', 400, 100)
print('[preflight] math500 OK — train', len(r['train']), '/ val', len(r['val']))"; then
  echo "[abort] math500 데이터 확보 실패 — 온라인 셸에서: bash scripts/fetch_datasets.sh math500"
  exit 1
fi

DATASETS="math500" N_TRAIN=400 N_VAL=100 exec bash scripts/go_v2.sh
