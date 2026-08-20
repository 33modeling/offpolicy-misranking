#!/usr/bin/env bash
# run이 계약 수정(P0-1·P0-2) 이후 산출물인지 판정한다.
#   bash scripts/check_contract.sh              # runs/ 전체
#   bash scripts/check_contract.sh <run_dir>    # 하나만
#
# 판정 근거: 수정본 collect_rollouts는 rollout마다
#   ① `rollouts_*.manifest.json` (explicit_kwargs.top_k == 0, repetition_penalty == 1.0)
#   ② jsonl 각 행의 `resp_end` 필드
# 를 남긴다. 둘 다 없으면 수정 전 산출물이므로 논문 수치로 쓸 수 없다.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh 2>/dev/null || true
ROOT="${1:-${OM_WORK:-.}/runs}"
targets=()
if [ -f "$ROOT/prompts.json" ] || [ -f "$ROOT/DONE" ]; then targets=("$ROOT")
else for d in "$ROOT"/*/; do [ -d "$d" ] && targets+=("$d"); done; fi

printf "%-34s %-10s %-10s %s\n" run manifest resp_end 판정
post=0; pre=0
for d in "${targets[@]}"; do
  name=$(basename "${d%/}")
  case "$name" in *smoke*) continue;; esac
  man=no; rse=no
  ls "${d%/}"/rollouts_*.manifest.json >/dev/null 2>&1 && \
    grep -ql '"top_k": 0' "${d%/}"/rollouts_*.manifest.json 2>/dev/null && man=yes
  f=$(ls "${d%/}"/rollouts_behavior_train.jsonl 2>/dev/null | head -1)
  [ -n "$f" ] && head -1 "$f" 2>/dev/null | grep -q '"resp_end"' && rse=yes
  if [ "$man" = yes ] && [ "$rse" = yes ]; then verdict="✅ 수정 후"; post=$((post+1))
  elif [ "$man" = no ] && [ "$rse" = no ]; then verdict="❌ 수정 전 — 논문 수치 불가"; pre=$((pre+1))
  else verdict="⚠ 혼재 — 폴더 재사용 의심"; pre=$((pre+1)); fi
  printf "%-34s %-10s %-10s %s\n" "$name" "$man" "$rse" "$verdict"
done
echo
echo "요약: 수정 후 ${post}개 / 수정 전·혼재 ${pre}개"
[ "$pre" -gt 0 ] && echo "!! 수정 전 run이 있다 — 그 수치는 진단용으로만 쓰고 본문에 넣지 말 것"
exit 0
