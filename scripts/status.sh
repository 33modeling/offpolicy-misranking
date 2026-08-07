#!/usr/bin/env bash
# 현재 상태 한 방 요약: phase 위치·파이프라인별 마지막 로그·ETA·산출물 체크리스트.
#   bash scripts/status.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1
OUT_ROOT="${OUT_ROOT:-$OM_WORK/runs/gate}"
LOGS="$OUT_ROOT/logs"

echo "=== offpolicy-misranking 상태 ($(date '+%F %T')) — $OUT_ROOT"
echo
echo "-- GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi 없음)"
RUNNING=$(pgrep -fc "src/experiment.py" 2>/dev/null || true)
echo "  실행 중인 experiment.py: ${RUNNING:-0}개"
echo
echo "-- 파이프라인 마지막 로그"
for lf in "$LOGS"/phase0-*.log "$LOGS"/drift*.log "$LOGS"/downstream-*.log; do
  [ -f "$lf" ] || continue
  echo "  $(basename "$lf" .log): $(tail -n 1 "$lf" | cut -c1-120)"
done
echo
echo "-- ETA (rollout 진행 중인 로그에서 자동 계산)"
python3 - "$LOGS" <<'PY'
import glob, re, sys
for lf in sorted(glob.glob(sys.argv[1] + "/*.log")):
    lines = open(lf, errors="replace").readlines()[-200:]
    prog = [m for l in lines if (m := re.search(r"rollout (\d+)/(\d+) \((\d+)s", l))]
    if not prog:
        continue
    done, total, sec = int(prog[-1][1]), int(prog[-1][2]), int(prog[-1][3])
    if done < total:
        remain = (total - done) * sec
        print(f"  {lf.split('/')[-1]}: rollout {done}/{total}, ~{sec}s/개 → 남은 시간 ≈ {remain//3600}h {remain%3600//60}m")
PY
echo
echo "-- 산출물 체크리스트"
for run in "$OUT_ROOT"/drift*; do
  [ -d "$run" ] || continue
  d=$(basename "$run")
  ck() { [ -e "$run/$1" ] && echo "✔" || echo "·"; }
  echo "  $d: adapter $(ck "drift_${d#drift}") fresh $(ck rollouts_fresh_train.jsonl) oracle $(ck scores_oracle.json) 2×2 $(ck scores_offpolicy.json) report $(ck report.md) hybrid $(ck scores_hybrid_0.5.json)"
done
for f in "$OUT_ROOT"/drift*/downstream_*.json; do
  [ -f "$f" ] && echo "  downstream: $(basename "$f") ✔"
done
echo
echo "-- report 있으면 요약"
for r in "$OUT_ROOT"/drift*/report.md; do
  [ -f "$r" ] && { echo "  --- $r"; sed -n '1,12p' "$r" | sed 's/^/  /'; }
done
