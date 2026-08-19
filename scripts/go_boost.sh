#!/usr/bin/env bash
# 메인 실험 보강 패키지 — go_full(A·B·C) 이후 남은 무방비 축 3개.
# tmux 포그라운드에서:
#   bash scripts/go_boost.sh              # D E F 전부
#   PHASES="D" bash scripts/go_boost.sh   # 골라서
#
#   D. drift 스윕 seed 반복 — 7B GSM8K drift {50,200,400} × seed {0,1,2}
#      (Table 1의 drift 행들에 오차대 공급. 같은 seed의 v2-s* 완주분에서
#       β rollout·prompts 재사용 → 수집 단계 생략)
#   E. 14B MATH-500 × seed {0,1,2} — n=400+100 (14B 행 오차대. β는 자체 수집)
#   F. mbpp × seed {0,1,2} — n=512 (regime 지도에 코드 도메인 — D3 방어)
#
# 전부 DONE 스킵 — 끊겨도 같은 명령으로 재개. go_v2/go_full과 동시 실행 금지.
# 폴더 규약: D→v2-d<drift>-s<seed>, E→v2-14bm-s<seed>, F→v2-s<seed>-mbpp
# (F만 v2-s* 글롭에 걸려 frontier/tables에 자동 포함된다 — 의도된 것)
set -uo pipefail
cd "$(dirname "$0")/.."

if pgrep -f "scripts/go_v2.sh" >/dev/null || pgrep -f "scripts/go_full.sh" >/dev/null; then
  echo "[abort] go_v2/go_full 실행 중 — 완주 후 재실행"; exit 1
fi
source scripts/setup_env.sh
LOGDIR="${OM_WORK:-.}/console-logs"; mkdir -p "$LOGDIR"
PHASES="${PHASES:-D E F}"

run_one() {  # run_one <OUT_ROOT> <ENV=V...>  — DONE 스킵 + 재시도 2회 + 데이터 문제 즉시 포기
  local dir="$1"; shift
  local tag; tag="$(basename "$dir")"
  if [ -f "$dir/DONE" ] && [ -f "$dir/score_protocol.json" ] \
     && [ -f "$dir/oracle_protocol.json" ]; then
    echo "  ✔ $tag 완주분 — 스킵"; return 0
  fi
  local lg="$LOGDIR/boost-$tag.log"
  for try in 1 2; do
    if env "$@" OUT_ROOT="$dir" bash scripts/run_14b.sh >> "$lg" 2>&1; then
      echo "  ✔ $tag"; return 0
    fi
    echo "  ✘ $tag 시도 $try/2 — tail:"; tail -3 "$lg" | sed 's/^/     /'
    grep -q "\[abort\].*데이터" "$lg" && { echo "  → 데이터 문제, $tag 포기"; return 1; }
    sleep 15
  done
  return 1
}

if [[ " $PHASES " == *" D "* ]]; then
  echo "== [D] drift 스윕 {50,200,400} × 3-seed (7B GSM8K, n=512)"
  for s in 0 1 2; do
    src="$OM_WORK/runs/v2-s$s"
    for d in 50 200 400; do
      dir="$OM_WORK/runs/v2-d$d-s$s"
      mkdir -p "$dir"
      if [ -f "$src/DONE" ] && [ -f "$src/score_protocol.json" ] \
         && [ -f "$src/oracle_protocol.json" ]; then
        cp -n "$src/prompts.json" "$dir/" 2>/dev/null || true
        cp -n "$src/rollouts_behavior_train.jsonl" "$dir/" 2>/dev/null || true
        for f in "$src"/rollouts_behavior_train*.manifest.json; do
          [ -f "$f" ] && cp -n "$f" "$dir/"
        done
      fi
      run_one "$dir" DATASET=gsm8k DRIFT="$d" SEED="$s" N_TRAIN=512 N_VAL=100 || true
    done
  done
fi

if [[ " $PHASES " == *" E "* ]]; then
  echo "== [E] 14B MATH-500 × 3-seed (n=400+100)"
  M14="${MODEL_14B:-$MODELS_DIR/Qwen2.5-14B-Instruct}"
  if [ -f "$M14/config.json" ]; then
    # 디렉터리에 데이터셋명 포함 — run_14b가 math500 접미사를 붙이므로 접미사
    # 없는 경로로 DONE을 검사하면 완주를 영구 미인식한다
    for s in 0 1 2; do
      run_one "$OM_WORK/runs/v2-14bm-s$s-math500" MODEL_14B="$M14" DATASET=math500 DRIFT=100 SEED="$s" N_TRAIN=400 N_VAL=100 || true
    done
  else
    echo "  [skip] 14B 스냅샷 없음: $M14 — provision 후 PHASES=E 재실행"
  fi
fi

if [[ " $PHASES " == *" F "* ]]; then
  echo "== [F] mbpp·kk × 3-seed — 코드·논리 도메인 (n=512, 풀 부족 시 256+50 폴백)"
  for DS in mbpp kk; do
    for s in 0 1 2; do
      dir="$OM_WORK/runs/v2-s$s-$DS"
      run_one "$dir" DATASET="$DS" DRIFT=100 SEED="$s" N_TRAIN=512 N_VAL=100 \
        || run_one "$dir" DATASET="$DS" DRIFT=100 SEED="$s" N_TRAIN=256 N_VAL=50 || true
    done
  done
fi

echo "== 종료 요약 =="
for pat in v2-d50-s v2-d200-s v2-d400-s; do
  for s in 0 1 2; do d="$OM_WORK/runs/$pat$s"
    [ -d "$d" ] && echo "  $(basename "$d"): $([ -f "$d/DONE" ] && echo ✔ || echo ✘)"
  done
done
for s in 0 1 2; do d="$OM_WORK/runs/v2-14bm-s$s-math500"
  [ -d "$d" ] && echo "  $(basename "$d"): $([ -f "$d/DONE" ] && echo ✔ || echo ✘)"
done
for DS in mbpp kk; do for s in 0 1 2; do d="$OM_WORK/runs/v2-s$s-$DS"
  [ -d "$d" ] && echo "  $(basename "$d"): $([ -f "$d/DONE" ] && echo ✔ || echo ✘)"
done; done
echo "== 끝 — 표는 tables.sh, frontier는 frontier.sh (mbpp만 자동 포함)"
