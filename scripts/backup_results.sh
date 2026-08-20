#!/usr/bin/env bash
# 실험 산출물 오프사이트 백업 — 정본 소형 파일만 레포 안 results/backup/ 으로
# 수집해 로컬 커밋한다 (rollout jsonl·pt 등 대용량 제외). GitHub egress 없는
# 클러스터에서는 커밋까지만 되고, push는 온라인 셸에서 git push.
#   bash scripts/backup_results.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1
DEST="results/backup"
mkdir -p "$DEST"

# 집계 결과물(results/*) — TABLES/FRONTIER/report/judge 등
for d in "$OM_WORK"/results/*/; do
  [ -d "$d" ] || continue
  t="$DEST/$(basename "$d")"; mkdir -p "$t"
  find "$d" -maxdepth 1 -type f ! -name '*.pt' ! -name '*.jsonl' -size -20M \
    -exec cp {} "$t/" \; 2>/dev/null || true
done
# run별 정본(완주분만, 소형 json만)
for r in "$OM_WORK"/runs/v[0-9]*-*/; do
  [ -f "$r/DONE" ] && [ -f "$r/score_protocol.json" ] \
    && [ -f "$r/oracle_protocol.json" ] || continue
  t="$DEST/runs/$(basename "$r")"; mkdir -p "$t"
  for f in report.json manifest.json scores_oracle.json scores_offpolicy.json \
           scores_splithalf.json scores_hybrid_*.json divergence_stats*.json \
           score_protocol.json oracle_protocol.json hybrid_protocol_*.json \
           postprocess_manifest.json downstream_*.json; do
    cp "$r"/$f "$t/" 2>/dev/null || true
  done
done

echo "[backup] 수집 완료: $(du -sh "$DEST" | cut -f1)"
git -c safe.directory='*' add "$DEST" 2>/dev/null || true
if git -c safe.directory='*' -c user.name=33modeling -c user.email=33modeling@gmail.com \
     commit -m "results backup $(date +%F-%H%M)" >/dev/null 2>&1; then
  echo "[backup] 로컬 커밋 완료 — 온라인 셸에서 git push 하면 오프사이트 완성"
else
  echo "[backup] 새 변경 없음 (이미 백업됨)"
fi
