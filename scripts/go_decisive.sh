#!/usr/bin/env bash
# 결정 실험 — 무누수 hybrid를 고검정력으로 재실행해 '회복 효과'의 유무를 확정한다.
# 배경(A0 코드 대조): v1의 21/21은 pp가 oracle 표본을 재사용한 누수 설계였고,
# v2 무누수판은 K=8 소표본이라 작은 진짜 효과는 미검출일 수 있다. 이 실험이 최종 판정.
#   bash scripts/go_decisive.sh        # 완주 run 전부 × cut=0.5, K=32, 프롬프트 96
# env: K_CELL(8 — equal-K 상한) HYB_N(96) CUTS("0.5") TARGETS("경로 ...")   — run당 수 시간(GPU 1장)
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
K_CELL="${K_CELL:-8}"; HYB_N="${HYB_N:-96}"   # K는 8 고정이 정설계 — behavior rollout이 프롬프트당 8개라 K>8이면 bb·bp만 부족해져 equal-K가 깨진다(v1 재발). 검정력은 프롬프트 수(96)가 지배.; CUTS="${CUTS:-0.5}"
MODEL="${MODEL_14B:-$MODELS_DIR/Qwen2.5-7B-Instruct}"

targets=()
if [ -n "${TARGETS:-}" ]; then
  targets=($TARGETS)
else
# 구버전 이중 접미사 run은 신형(단일 접미사) 완주가 있으면 제외 — 같은
# (dataset, seed)가 두 디렉터리로 이중 집계되는 것 방지
_legacy_dup() { case "$1" in
    *-dapo-math-dapo-math) echo "${1%-dapo-math}";;
    *-math500-math500)     echo "${1%-math500}";;
    *-mbpp-mbpp)           echo "${1%-mbpp}";;
    *-kk-kk)               echo "${1%-kk}";;
    *-apps-apps)           echo "${1%-apps}";;
    *) echo "";;
  esac; }
  for d in "$OM_WORK"/runs/v2-*; do
    case "$d" in *smoke*) continue;; esac
    dup=$(_legacy_dup "$d")
    [ -n "$dup" ] && [ -f "$dup/DONE" ] && { echo "  [skip] legacy 이중 접미사: $(basename "$d") (신형 완주 존재)"; continue; }
    [ -f "$d/DONE" ] && targets+=("$d")
  done
fi
echo "== 결정 실험: ${#targets[@]}개 run × cuts($CUTS), K=$K_CELL, 프롬프트 $HYB_N"

for d in "${targets[@]}"; do
  AD=$(ls -d "$d"/drift_* 2>/dev/null | head -1)
  [ -n "$AD" ] || { echo "  [skip] adapter 없음: $(basename "$d")"; continue; }
  DS=gsm8k; case "$d" in *dapo*) DS=dapo-math;; *math500*) DS=math500;; esac
  for cut in $CUTS; do
    # 기존 저검정력(K=8) 산출물은 .lowK 로 보존 — 재생성 유도
    for f in "$d/scores_hybrid_${cut}.json" "$d/rollouts_hybrid_${cut}.jsonl"; do
      [ -f "$f" ] && [ ! -f "$f.lowK" ] && mv "$f" "$f.lowK"
    done
    # FRESH=1 — 잘못된 설정(K=32 등)으로 만든 현재 파일 폐기 후 재생성 (.lowK 백업은 보존)
    if [ "${FRESH:-0}" = "1" ]; then
      rm -f "$d/scores_hybrid_${cut}.json" "$d/rollouts_hybrid_${cut}.jsonl"
    fi
    echo "== $(basename "$d") cut=$cut (K=$K_CELL, n=$HYB_N)"
    if ! CUDA_VISIBLE_DEVICES="${GPU:-0}" "$PY" src/experiment.py --stage hybrid \
        --run "$d" --model "$MODEL" --dataset "$DS" --adapter "$AD" \
        --cut-frac "$cut" --hybrid-prompts "$HYB_N" --k-cell "$K_CELL" \
        --micro-batch 1 --n-train 512 --n-val 100 --seed 0; then
      echo "  ✘ $(basename "$d") cut=$cut 실패 — 다음으로"
    fi
  done
done

echo "== 재판정 (READOUT.md 갱신)"
bash scripts/read_now.sh
