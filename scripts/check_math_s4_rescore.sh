#!/usr/bin/env bash
# Read-only check (2026-09-24): was MATH seed 4 rescored like seeds 0-3?
#   Appendix C / Table 30 say seed-4 rescoring is unverified. The 2026-09-08
#   apply covered math500/s0-s3 only. This prints, per checkpoint d0/d25/d100/d400,
#   the evidence the rescoring tool leaves behind (rollouts_*.rescore.json with
#   verifier math_verify-latex-2026-09-08, reward_pinned in every row, a
#   pinned-scoring/ archive, and a DONE/report.json newer than the rescore).
#
#   bash scripts/check_math_s4_rescore.sh          # one TXT: ~/math-s4-rescore-check.txt
#
# Nothing is written under $OM_WORK: no locks, no setup_env.sh, no rescoring.
set -uo pipefail
OM_WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
FAM=${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}/family-math500-s4
OUT=${1:-$HOME/math-s4-rescore-check.txt}
VERIFIER=math_verify-latex-2026-09-08
{
  echo "MATH seed-4 rescoring check  $(date -u +%Y-%m-%dT%H:%M:%SZ)  family=$FAM"
  complete=0
  for d in 0 25 100 400; do
    P=$FAM/$TAG-s4-math500-d$d
    echo "== d$d  $P"
    [ -d "$P" ] || { echo "  missing point directory"; continue; }
    ok=1
    for f in "$P"/rollouts_*.jsonl; do
      [ -f "$f" ] || { echo "  no rollouts_*.jsonl"; ok=0; break; }
      rows=$(wc -l < "$f"); pinned=$(grep -c '"reward_pinned"' "$f")
      echo "  $(basename "$f") rows=$rows reward_pinned=$pinned"
      [ "$rows" -gt 0 ] && [ "$rows" -eq "$pinned" ] || ok=0
    done
    sidecars=0
    for s in "$P"/rollouts_*.rescore.json; do
      [ -f "$s" ] || continue
      sidecars=$((sidecars + 1))
      v=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("verifier"), d.get("rewritten_at_utc"))' "$s" 2>&1)
      echo "  sidecar $(basename "$s"): $v"
      [[ "$v" == "$VERIFIER "* ]] || ok=0
    done
    [ "$sidecars" -gt 0 ] || { echo "  no rescore sidecar"; ok=0; }
    ls -d "$P"/pinned-scoring/*/ >/dev/null 2>&1 && echo "  pinned-scoring archive present" || { echo "  no pinned-scoring archive"; ok=0; }
    for f in DONE report.json; do
      [ -s "$P/$f" ] && echo "  $f $(date -u -r "$P/$f" +%Y-%m-%dT%H:%M:%SZ)" || { echo "  $f missing (scores not recomputed after rescoring)"; ok=0; }
    done
    echo "  => $([ "$ok" = 1 ] && echo RESCORED || echo NOT-VERIFIED)"
    [ "$ok" = 1 ] && complete=$((complete + 1))
  done
  echo "summary: $complete/4 seed-4 checkpoints show complete rescoring evidence"
  grep -l "math500/s4" "$OM_WORK"/exports/rescore-math500-apply-*.txt 2>/dev/null | sed 's/^/apply log mentioning s4: /' || true
} 2>&1 | tee "$OUT"
echo "[saved] $OUT"
