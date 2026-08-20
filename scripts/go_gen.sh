#!/usr/bin/env bash
# 세대 태그를 받아 도는 범용 원커맨드 — go_v3.sh의 일반화판.
#   TAG=v4 bash scripts/go_gen.sh                     # 기본: gsm8k+dapo(s1,s2) → math500 → mbpp·kk(s0)
#   TAG=v4 SEEDS_GEN="0 1 2 3 4" bash scripts/go_gen.sh
#   TAG=v4 OM_SKIP_EXTRA=1 bash scripts/go_gen.sh     # mbpp·kk 생략(결정점 최소셋)
#
# 왜 새 태그가 필요한가 (2026-08-20):
#   `runs/v3-*`에는 **계약 수정(c6ca013, 8/19 22:57) 이전**에 만들어진 산출물이 남아 있다.
#   run_14b.sh는 `rollouts_*.jsonl`이 이미 있으면 생성을 건너뛰므로, 같은 폴더에 다시 돌리면
#   **수정 전 rollout을 그대로 재사용**해 수정이 무효가 된다. 그래서 폴더를 갈아탄다.
#
# 실행 전 자기점검: 이 스크립트는 대상 폴더에 기존 rollout이 있으면 멈춘다.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh

TAG="${TAG:-v4}"
export RUN_BASE="$OM_WORK/runs/$TAG" RESULTS_BASE="$OM_WORK/results/$TAG"
S="${SEEDS_GEN:-1 2}"

echo "== go_gen: TAG=$TAG  RUN_BASE=$RUN_BASE  SEEDS=[$S]"

# 오염 방지 게이트 — 기존 rollout이 있으면 계약 수정 전 산출물일 수 있다
existing=$(ls -d "$RUN_BASE"-* 2>/dev/null | wc -l)
if [ "$existing" -gt 0 ]; then
  echo "[중단] $RUN_BASE-* 폴더가 이미 ${existing}개 있다."
  echo "       계약 수정 전 산출물이 섞이면 rollout이 재사용돼 수정이 무효가 된다."
  echo "       다른 TAG를 쓰거나, 기존 폴더를 옮긴 뒤 다시 실행할 것:"
  echo "         mv $RUN_BASE $RUN_BASE.pre-fix   # (폴더별로)"
  exit 1
fi

# 1) 결정점 최소셋 — 본문 표를 떠받치는 풀
SEEDS="$S" DATASETS="gsm8k dapo-math" bash scripts/go_v2.sh || exit 1
# 2) math500
SEEDS="$S" DATASETS="math500" N_TRAIN=400 N_VAL=100 bash scripts/go_v2.sh || exit 1
# 3) 도메인 다각화 1-seed
if [ "${OM_SKIP_EXTRA:-0}" != "1" ]; then
  SEEDS="0" DATASETS="mbpp" bash scripts/go_v2.sh || exit 1
  SEEDS="0" DATASETS="kk" bash scripts/go_v2.sh || exit 1
fi

echo "== go_gen($TAG) 완료 — 계약 확인 후 수확:"
echo "   bash scripts/check_contract.sh   # manifest로 수정 후 실행인지 확인"
echo "   bash scripts/harvest.sh"
