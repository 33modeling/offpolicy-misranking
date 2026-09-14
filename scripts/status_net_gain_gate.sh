#!/usr/bin/env bash
# Read-only status and ETA for an already prepared v3 net-gain suite.
set -euo pipefail
cd "$(dirname "$0")/.."

WORK=${OM_WORK:-$PWD/.work}
if [ -d "${GROUP_VOLUME:-/group-volume}" ]; then
  WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
fi
NET_ROOT=$(realpath -m "${NET_GATE_ROOT:-$WORK/runs/net-gain-gate-v3}")
PY=${NET_GATE_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
NODES=${NET_GATE_NODES:-4}

[[ "$NODES" =~ ^[1-9][0-9]*$ ]] || { echo '[abort] NET_GATE_NODES must be a positive integer'; exit 2; }
[ -f "$NET_ROOT/net_protocol.json" ] && [ -f "$NET_ROOT/suite.json" ] || {
  echo "[not prepared] $NET_ROOT"
  exit 0
}

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
"$PY" src/net_gain_gate_gpu.py status --root "$NET_ROOT"

# ETA is optional; missing reporting tools must not hide experiment status.
if ! command -v jq >/dev/null 2>&1; then
  printf '\n[ETA] unavailable without jq; experiment status is shown above\n'
  exit 0
fi

BUDGET=$(jq -er '.budget_gpu_seconds | numbers' "$NET_ROOT/suite.json")
TOTAL=0
DONE=0
RUNNING=0
FAILED=0
INVALID=0
QUEUED=0
REMAINING_GPU_SECONDS=0
EVAL_WALL_SECONDS=0
EVAL_SAMPLES=0
EVAL_PENDING=0
NOW=$(date +%s)

while IFS=$'\t' read -r point arm; do
  [ -n "$point" ] && [ -n "$arm" ] || continue
  TOTAL=$((TOTAL+1))
  ARM_DIR="$NET_ROOT/points/$point/$arm"
  DECISION="$ARM_DIR/decision.json"
  RESULT="$ARM_DIR/result.json"
  RESULT_HASH="$ARM_DIR/result.sha256.json"
  CAP=$BUDGET
  if [ -f "$DECISION" ]; then
    CAP=$(jq -er '.budget_gpu_seconds | numbers' "$DECISION")
  fi
  SPENT=0
  if [ -s "$ARM_DIR/cost.jsonl" ]; then
    SPENT=$(jq -s '[.[] | select(.state == "finished" and .ledger != "reporting") | .allocated_gpu_seconds] | add // 0' "$ARM_DIR/cost.jsonl")
  fi
  ACTIVE=0
  PROGRESS="$ARM_DIR/progress.json"
  if [ -f "$PROGRESS" ]; then
    STATE=$(jq -r '.state // ""' "$PROGRESS")
    UPDATED=$(jq -r '.updated // 0 | floor' "$PROGRESS")
    if [ "$STATE" = running ] && [ $((NOW-UPDATED)) -lt 90 ]; then
      ACTIVE=$(jq -r '(.seconds // 0) * (.gpus // 4)' "$PROGRESS")
      RUNNING=$((RUNNING+1))
    fi
  fi
  USED=$(awk -v a="$SPENT" -v b="$ACTIVE" 'BEGIN { printf "%.9f", a+b }')
  LEFT=$(awk -v cap="$CAP" -v used="$USED" 'BEGIN { x=cap-used; printf "%.9f", (x>0 ? x : 0) }')
  VALID_RESULT=0
  if [ -f "$RESULT" ] && [ -f "$RESULT_HASH" ]; then
    EXPECTED_HASH=$(jq -r '.sha256 // ""' "$RESULT_HASH")
    ACTUAL_HASH=$(sha256sum "$RESULT" | awk '{print $1}')
    if [ -n "$EXPECTED_HASH" ] && [ "$EXPECTED_HASH" = "$ACTUAL_HASH" ] \
        && jq -e '.complete == true and (.stop_reason == "budget_exhausted" or .stop_reason == "no_block_fits") and (.used_gpu_seconds <= .budget_gpu_seconds)' "$RESULT" >/dev/null; then
      VALID_RESULT=1
    fi
  fi
  if [ "$VALID_RESULT" -eq 1 ]; then
    DONE=$((DONE+1))
    REPORTING=$(jq -r '.cost.ledgers.reporting.wall_seconds // 0' "$RESULT")
    if awk -v value="$REPORTING" 'BEGIN { exit !(value > 0) }'; then
      EVAL_WALL_SECONDS=$(awk -v a="$EVAL_WALL_SECONDS" -v b="$REPORTING" 'BEGIN { printf "%.9f", a+b }')
      EVAL_SAMPLES=$((EVAL_SAMPLES+1))
    fi
    LEFT=0
  else
    REMAINING_GPU_SECONDS=$(awk -v a="$REMAINING_GPU_SECONDS" -v b="$LEFT" 'BEGIN { printf "%.9f", a+b }')
    SHARDS=0
    if [ -d "$ARM_DIR/evaluation" ]; then
      SHARDS=$(find "$ARM_DIR/evaluation" -maxdepth 1 -type f -name 'shard-*.done.json' | wc -l)
    fi
    [ "$SHARDS" -ge 4 ] || EVAL_PENDING=$((EVAL_PENDING+1))
    if [ -f "$RESULT" ]; then
      INVALID=$((INVALID+1))
    elif [ "$ACTIVE" != 0 ]; then
      :
    elif [ -f "$ARM_DIR/failure.json" ]; then
      FAILED=$((FAILED+1))
    else
      QUEUED=$((QUEUED+1))
    fi
  fi
done < <(jq -r --slurpfile p "$NET_ROOT/net_protocol.json" '.points[].name as $name | $p[0].arms[] | [$name, .] | @tsv' "$NET_ROOT/suite.json")

FLOOR_SECONDS=$(awk -v gpu="$REMAINING_GPU_SECONDS" -v nodes="$NODES" 'BEGIN { printf "%.0f", gpu/(4*nodes) }')
FLOOR_HOURS=$(awk -v seconds="$FLOOR_SECONDS" 'BEGIN { printf "%.2f", seconds/3600 }')
printf '\n[ETA] %d nodes x 4 GPUs | DONE %d/%d | RUNNING %d | QUEUED %d | FAILED %d | INVALID %d\n' \
  "$NODES" "$DONE" "$TOTAL" "$RUNNING" "$QUEUED" "$FAILED" "$INVALID"
printf '[ETA] remaining training/scoring allocation floor: %s h\n' "$FLOOR_HOURS"

TOTAL_SECONDS=$FLOOR_SECONDS
if [ "$EVAL_SAMPLES" -gt 0 ]; then
  EVAL_REMAINING=$(awk -v sum="$EVAL_WALL_SECONDS" -v samples="$EVAL_SAMPLES" -v pending="$EVAL_PENDING" -v nodes="$NODES" \
    'BEGIN { printf "%.0f", (sum/samples)*pending/nodes }')
  EVAL_HOURS=$(awk -v seconds="$EVAL_REMAINING" 'BEGIN { printf "%.2f", seconds/3600 }')
  TOTAL_SECONDS=$((FLOOR_SECONDS+EVAL_REMAINING))
  printf '[ETA] evaluation estimate from %d completed arms: %s h\n' "$EVAL_SAMPLES" "$EVAL_HOURS"
else
  printf '[ETA] evaluation: unknown until the first arm finishes evaluation\n'
fi

if [ "$FAILED" -gt 0 ] || [ "$INVALID" -gt 0 ]; then
  printf '[ETA] completion time withheld: failed/invalid arms require inspection; other arms continue\n'
else
  FINISH=$((NOW+TOTAL_SECONDS))
  printf '[ETA] estimated finish no earlier than: %s\n' "$(date -d "@$FINISH" '+%Y-%m-%d %H:%M:%S %Z')"
fi
printf '[ETA] estimate uses frozen allocation and observed reporting time; cluster queue/stragglers can extend it\n'
