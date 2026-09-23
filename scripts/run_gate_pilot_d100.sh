#!/usr/bin/env bash
# New experiment (2026-09-24): executed correlation-gate pilot at MATH d=100.
#   Appendix E / Table 31 has d=0 and d=400; this adds d=100 for seeds 0-2.
#   Only the gate arm (gate_passrate: 10 uniform updates, frozen Fisher rule,
#   then 90 updates on cached SR or random) is trained, in its own root; it is
#   compared with the unchanged E5 d=100 uniform and cached-SR arms (read only).
#
#   bash scripts/run_gate_pilot_d100.sh           # prepare + check (CPU), then train/evaluate on this idle 4-GPU node
#   bash scripts/run_gate_pilot_d100.sh prepare   # prepare + check only, no GPU
#   bash scripts/run_gate_pilot_d100.sh status    # progress, no GPU
#   bash scripts/run_gate_pilot_d100.sh results   # one TXT: ~/gate-pilot-d100-results.txt
#
# Writes only under $OM_WORK/runs/gate-pilot-d100-v1. The E5 roots, the frozen
# test file and all code are read, never written. Several idle nodes may run
# the same command; the arm and baseline are leased and a busy node is left alone.
set -uo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
case "$MODE" in
  run|prepare|status|results) ;;
  *) echo "usage: bash scripts/run_gate_pilot_d100.sh [run|prepare|status|results]"; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo "[abort] no additional options; settings are fixed in this script"; exit 2; }
trap '' HUP
trap 'echo "[gate-pilot-d100] interrupted; nothing else will be started"; exit 130' INT TERM
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv missing: $PY"; exit 1; }
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
MATRIX=${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}
ROOT=${GATE_D100_ROOT:-$OM_WORK/runs/gate-pilot-d100-v1}
SEEDS=(0 1 2)
TEST="$OM_WORK/inputs/e5-reduced/test-math500-d100.json"
run_dir() { printf '%s/family-math500-s%s/%s-s%s-math500-d100\n' "$MATRIX" "$1" "$TAG" "$1"; }
out_dir() { printf '%s/math500-d100/s%s\n' "$ROOT" "$1"; }
PILOT=("$PY" scripts/gate_pilot_d100.py)

case "$MODE" in
  status) CUDA_VISIBLE_DEVICES="" exec "${PILOT[@]}" status --root "$ROOT" ;;
  results) CUDA_VISIBLE_DEVICES="" exec "${PILOT[@]}" results --root "$ROOT" ;;
esac
[ -s "$TEST" ] || { echo "[abort] frozen E5 d=100 test file missing: $TEST"; exit 1; }
for seed in "${SEEDS[@]}"; do
  [ -s "$(run_dir "$seed")/DONE" ] || { echo "[abort] source point is not complete: $(run_dir "$seed")"; exit 1; }
done
# The comparison needs the unchanged E5 controls; refuse before any preparation.
CUDA_VISIBLE_DEVICES="" "${PILOT[@]}" check --root "$ROOT" >/dev/null || exit 1
for seed in "${SEEDS[@]}"; do
  DOWNSTREAM_SELECTORS="gate_passrate" bash scripts/run_downstream_independent.sh "$(run_dir "$seed")" "$(out_dir "$seed")" \
    --eval-prompts "$TEST" --steps 100 --eval-k 8 --prepare-only >/dev/null || exit 1
done
CUDA_VISIBLE_DEVICES="" "${PILOT[@]}" check --root "$ROOT" || exit 1
[ "$MODE" = run ] || exit 0

export OUT_ROOT="$ROOT" E5_FORCE=0
source scripts/_e5_node.sh || exit 1
e5_acquire_node || exit "$?"
# Different nodes start at different seeds; the arm is leased, so nodes do not overlap.
offset=$(( $(hostname | cksum | cut -d' ' -f1) % ${#SEEDS[@]} ))
order=("${SEEDS[@]:offset}" "${SEEDS[@]:0:offset}")
rc_all=0
for seed in "${order[@]}"; do
  echo "== s$seed d100: $(out_dir "$seed")"
  OM_NODE_LOCK_HELD=1 OM_E5_CONTROLLER_PID="$$" DOWNSTREAM_SELECTORS="gate_passrate" \
    bash scripts/run_downstream_independent.sh "$(run_dir "$seed")" "$(out_dir "$seed")" \
    --eval-prompts "$TEST" --steps 100 --eval-k 8 7>&- 8>&-
  rc=$?
  if [ "$rc" -eq 75 ]; then echo "[abort] GPU admission failed; existing jobs untouched"; exit 75; fi
  if [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ]; then echo "[gate-pilot-d100] stopped by signal"; exit "$rc"; fi
  [ "$rc" -eq 0 ] || rc_all=1
done
echo "[gate-pilot-d100] pass complete; results: bash scripts/run_gate_pilot_d100.sh results"
exit "$rc_all"
