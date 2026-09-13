#!/usr/bin/env bash
# Everything that remains, in priority order, on THIS node:
#
#   bash scripts/run_queue.sh          # run the queue (GPU node)
#   bash scripts/run_queue.sh status   # one-screen overview: one line per step (no GPU)
#
# Every step is leased (node lock, per-seed arms, per-point locks), so the
# same command on several nodes shares the work: a step that is finished or
# claimed elsewhere is skipped and the queue moves on. Rerunning resumes.
# Order: mixed-pool positive control (pool, point, arms, gate) -> reuse
# split-half scores (d400, d0) -> public benchmarks (d0, d400) -> d100
# continuation -> CPU analyses and the export bundle.
set -uo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-run}" = status ]; then
  # One screen: one line per queue step with a state word, seed detail below.
  export OM_ONLINE=0
  source scripts/setup_env.sh >/dev/null 2>&1
  PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
  PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" src/queue_status.py
  exit $?
fi
trap '' HUP
QUEUE=("run_mixed_pool.sh pool" "run_mixed_pool.sh point" "run_mixed_pool.sh e5" "run_mixed_pool.sh gate" \
       "run_stale_splithalf.sh" "run_stale_splithalf.sh d0" "run_e5_bench.sh d0" "run_e5_bench.sh" \
       "run_e5.sh d100")
for job in "${QUEUE[@]}"; do
  echo; echo "===== [$(date -u +%H:%M)] $job"
  bash scripts/$job 2>&1 | grep -v setup_env
  echo "===== [$(date -u +%H:%M)] $job finished (rc=${PIPESTATUS[0]})"
done
echo; echo "===== CPU analyses and export"
bash scripts/run_analyses.sh 2>&1 | grep -v setup_env | tail -n 6
echo "[queue] done on $(hostname); check:  bash scripts/run_queue.sh status"
