#!/usr/bin/env bash
# 실행 중인 v4 실험의 중간 결과물 스냅샷 — 돌고 있는 잡을 건드리지 않는다(전부 읽기 전용).
#   bash scripts/progress_snapshot.sh [클러스터라벨]
# 산출: $OM_WORK/progress/<MMDD-HHMM>-cluster<라벨>/ 폴더 하나
#   PROGRESS.md          진행률 표(런별 아티팩트·shard·재개분·에러 카운트·로그 tail)
#   contract.txt         check_contract.sh 판정(수정 후 산출물 여부)
#   <run>/               완료 run의 소형 아티팩트 사본(report·manifest·protocol·run_config)
#   judge-<run>.txt      report가 있는 run의 judge 판정
# 큰 jsonl(rollout)은 복사하지 않는다. 이 폴더를 통째로 홈에 전달하면 된다.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1 || source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3

LABEL="${1:-$(hostname -s 2>/dev/null || echo node)}"
OUT="$OM_WORK/progress/$(date +%m%d-%H%M)-cluster$LABEL"
mkdir -p "$OUT"
MD="$OUT/PROGRESS.md"

{
  echo "# v4 진행 스냅샷 — cluster $LABEL, $(date '+%F %T')"
  echo
  echo "- 코드: $(git rev-parse --short HEAD) ($(git log -1 --format=%s | cut -c1-60))"
  echo "- OM_WORK: $OM_WORK"
  echo
  echo "## GPU"
  echo '```'
  timeout 20 nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null || echo "nvidia-smi 실패"
  echo '```'
  echo
  echo "## 런별 진행"
} > "$MD"

for run in "$OM_WORK"/runs/v4-*/; do
  [ -d "$run" ] || continue
  name=$(basename "${run%/}")
  case "$name" in *smoke*) continue;; esac
  {
    echo
    echo "### $name"
    # 아티팩트 존재표
    arts=""
    for a in run_config.json manifest.json score_protocol.json oracle_protocol.json report.json DONE; do
      [ -s "$run$a" ] && arts="$arts $a✅" || arts="$arts $a—"
    done
    echo "-$arts"
    # provenance
    if [ -s "$run/run_config.json" ]; then
      "$PY" - "$run/run_config.json" <<'PYEOF' 2>/dev/null || true
import json, sys
c = json.load(open(sys.argv[1]))
print(f"- config: git={str(c.get('git'))[:12]} dataset={c.get('dataset')} "
      f"seed={c.get('seed')} n_train={c.get('n_train')} K(behavior)={c.get('behavior_k')}")
PYEOF
    fi
    # shard 진행률 (완성본 + .partial 재개분)
    for f in "$run"rollouts_*.jsonl "$run"rollouts_*.partial; do
      [ -f "$f" ] || continue
      lines=$(wc -l < "$f" 2>/dev/null || echo '?')
      sz=$(du -h "$f" 2>/dev/null | cut -f1)
      echo "- $(basename "$f"): ${lines}행 (${sz})"
    done
    # 로그: 에러 카운트 + 최신 로그 tail
    if [ -d "$run/logs" ]; then
      errs=$(grep -l "CUDA error\|Traceback" "$run"/logs/*.log 2>/dev/null | wc -l)
      echo "- 에러 있는 로그 파일: ${errs}개"
      lf=$(ls -t "$run"/logs/*.log 2>/dev/null | head -1)
      if [ -n "$lf" ]; then
        echo "- 최신 로그($(basename "$lf")) tail:"
        echo '```'
        tail -3 "$lf" 2>/dev/null
        echo '```'
      fi
    fi
  } >> "$MD"

  # 소형 아티팩트 사본 (읽기만, 큰 jsonl 제외)
  dst="$OUT/$name"; mkdir -p "$dst"
  for a in run_config.json manifest.json score_protocol.json oracle_protocol.json \
           report.json DONE; do
    [ -s "$run$a" ] && cp "$run$a" "$dst/" 2>/dev/null
  done
  cp "$run"rollouts_*.manifest.json "$dst/" 2>/dev/null || true

  # 완료 run이면 judge 판정 동봉
  if [ -s "$run/report.json" ] && [ -s "$run/score_protocol.json" ]; then
    timeout 180 "$PY" src/judge.py "${run%/}" > "$OUT/judge-$name.txt" 2>&1 || true
  fi
done

# 계약 판정 (수정 후 산출물인지)
bash scripts/check_contract.sh "$OM_WORK/runs" > "$OUT/contract.txt" 2>&1 || true

# 콘솔 로그 tail 모음
{
  echo
  echo "## 콘솔 로그 tail (v4 워커)"
  for cl in "$OM_WORK"/console-logs/v4-*.log; do
    [ -f "$cl" ] || continue
    echo; echo "### $(basename "$cl")"; echo '```'
    tail -5 "$cl" 2>/dev/null
    echo '```'
  done
} >> "$MD"

echo
echo "== 스냅샷 완료: $OUT"
echo "   이 폴더를 통째로 홈으로 전달하면 됩니다 (rollout 원본은 미포함, 전부 소형)."
du -sh "$OUT" 2>/dev/null | sed 's/^/   크기: /'
