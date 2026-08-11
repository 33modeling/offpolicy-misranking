#!/usr/bin/env bash
# 14B 규모 확인 실험 — 7B 게이트 통과 후의 scale confirmation.
#
# 설계 (풀 게이트 축소판):
#   · 모델: Qwen2.5-14B-Instruct — group-volume 로컬 스냅샷 전용 (다운로드 안 함)
#   · drift 100 단일 / fresh K=16 / hybrid 3절단점 / downstream 없음(C1·C1' 중심)
#   · 판정 대상: 7B 대비 g10/g01의 Δfloor가 커지는가 작아지는가 (스케일 축)
#   · GPU 배치: phase0 β rollout 4샤딩 → drift(1 GPU) → fresh rollout 4샤딩 → analyze
#
#   bash scripts/run_14b.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
MODEL_14B="${MODEL_14B:-$MODELS_DIR/Qwen2.5-14B-Instruct}"
[ -f "$MODEL_14B/config.json" ] || { echo "[abort] 14B 로컬 스냅샷 없음: $MODEL_14B — failure-atlas 규약대로 미러 경로를 MODEL_14B로 지정하거나 자산을 먼저 확보할 것"; exit 1; }
OUT_ROOT="${OUT_ROOT:-$OM_WORK/runs/gate-14b}"; export OUT_ROOT
LOGS="$OUT_ROOT/logs"; mkdir -p "$LOGS"
# GPU 수 자동 감지 — 인스턴스마다 다름 (4장 하드코딩이 invalid device ordinal의 원인)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
[ "$NGPU" -ge 1 ] || { echo "[abort] GPU 미감지"; exit 1; }
# OM_GPUS="0,1" 지정 시 그 GPU들만 사용 — 한 노드에서 두 실험 동시 실행용
if [ -n "${OM_GPUS:-}" ]; then
  IFS=',' read -r -a GPUS <<< "$OM_GPUS"
  NGPU=${#GPUS[@]}
else
  GPUS=($(seq 0 $((NGPU - 1))))
fi
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/main.log"; }
# 다른 실행(7B babysit/run)과의 GPU 충돌 차단
if pgrep -f "bash.*scripts/babysit.sh" >/dev/null || pgrep -f "scripts/run_h100_all.sh" >/dev/null; then
  echo "[abort] 7B babysit/run_h100_all 이 아직 실행 중 — 14B와 GPU가 충돌한다."
  echo "        먼저:  pkill -f babysit.sh; pkill -f run_h100_all; pkill -f 'src/experiment.py'; pkill -f gpu_keepalive"
  exit 1
fi
# 점유 검사는 내가 쓸 GPU만 대상 (OM_GPUS 분할 실행 시 서로 간섭 금지)
BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' -v g="${OM_GPUS:-}" 'BEGIN{n=split(g,a,","); for(i=1;i<=n;i++) sel[a[i]]=1}
       { if ((n==0 || ($1 in sel)) && $2 > 2000) c++ } END{print c+0}')
if [ "${BUSY:-0}" -gt 0 ] && [ "${OM_SKIP_GPU_CHECK:-0}" != "1" ]; then
  echo "[abort] GPU ${BUSY}개가 이미 점유 중:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
  exit 1
fi
# 샤드 병합 + 커버리지 검증 — GPU 수가 바뀐 재시작이면 샤드 분할이 어긋나
# 조용한 누락/중복이 생기므로, prompt 전수·무중복을 확인하고 원자적으로 쓴다.
merge_rollouts() {  # merge_rollouts <base이름>
  local base="$1"
  [ -f "$OUT_ROOT/$base.jsonl" ] && return 0
  "$PY" - "$OUT_ROOT" "$base" <<'PYEOF'
import json, sys
from pathlib import Path
root, base = Path(sys.argv[1]), sys.argv[2]
shards = sorted(root.glob(base + ".shard*.jsonl"))
n_train = len(json.loads((root / "prompts.json").read_text())["train"])
seen = {}
for s in shards:
    for line in s.open():
        r = json.loads(line)
        seen.setdefault(r["prompt_idx"], []).append((r["rollout_idx"], line))
missing = [i for i in range(n_train) if i not in seen]
dup = [i for i, v in seen.items() if len({j for j, _ in v}) != len(v)]
if missing or dup:
    print(f"[merge-abort] {base}: 누락 {len(missing)}개(예 {missing[:5]}) 중복 {len(dup)}개 — "
          f"GPU 수 변경 등으로 샤드 분할이 어긋남. 정리 후 재실행:\n"
          f"  rm {root}/{base}.shard*.jsonl", flush=True)
    sys.exit(1)
tmp = root / (base + ".jsonl.tmp")
with tmp.open("w") as f:
    for i in range(n_train):
        for _, line in sorted(seen[i]):
            f.write(line)
tmp.rename(root / (base + ".jsonl"))
print(f"[merge] {base}: {n_train} prompts, {sum(len(v) for v in seen.values())} rollouts OK")
PYEOF
}
run_stage() { local dev="${GPUS[$1]:-$1}" lf="$2"; shift 2
  local t0=$SECONDS
  log "GPU$dev ▶ $*"
  if CUDA_VISIBLE_DEVICES="$dev" "$PY" src/experiment.py "$@" >> "$lf" 2>&1; then
    log "GPU$dev ✔ $1 $2 ($((SECONDS - t0))s)"
  else local rc=$?; log "GPU$dev ✘ $* rc=$rc"; tail -8 "$lf" | tee -a "$LOGS/main.log"; return $rc; fi
}
DRIFT=100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 클러스터 노드들의 fused SDPA ULF 이력(C5·C6) — 어떤 경로로 실행해도 eager 기본.
# 빠른 커널을 검증한 노드에서만 OM_ATTN=sdpa 로 명시 해제.
export OM_ATTN="${OM_ATTN:-eager}"
DATASET="${DATASET:-gsm8k}"
[ "$DATASET" != "gsm8k" ] && OUT_ROOT="${OUT_ROOT}-${DATASET}" && LOGS="$OUT_ROOT/logs" && mkdir -p "$LOGS"
# 데이터 사전 검사 — 오프라인 노드에서 허브 직행으로 죽는 것을 시작 전에 잡는다
# 데이터 사전 검사 — 로더 자신을 그대로 실행 (검사·실제 로드가 같은 코드 경로)
# 실패 시 로더가 '찾아본 위치' 목록을 출력하므로 원인 자가진단됨. GPU 잡기 전에 죽는다.
if ! "$PY" -c "import sys; sys.path.insert(0, 'src'); from data import load_prompts; \
r = load_prompts('$DATASET', 1, 1); print('[preflight] $DATASET 데이터 OK')"; then
  echo "[abort] $DATASET 데이터 로드 실패 — 위 '찾아본 위치'를 확인하거나 fetch_datasets.sh 실행"
  exit 1
fi
COMMON=(--run "$OUT_ROOT" --model "$MODEL_14B" --dataset "$DATASET" --fresh-k "${FRESH_K:-16}" --hybrid-prompts "${HYBRID_PROMPTS:-24}" --micro-batch 1)

KA_DEV=$(IFS=,; echo "${GPUS[*]}")
CUDA_VISIBLE_DEVICES="$KA_DEV" "$PY" scripts/gpu_keepalive.py > "$LOGS/keepalive.log" 2>&1 &
KEEP=$!
# 종료(정상·에러 모두) 시 이 실행이 띄운 모든 자식 정리 — 고아 샤드가 GPU를 점유한 채
# 남아 "계속 실행되는 것처럼" 보이고 재시작 OOM을 일으키는 버그의 수정
cleanup() {
  kill "$KEEP" 2>/dev/null || true
  pkill -f -- "--run $OUT_ROOT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

log "=== 14B 시작: $MODEL_14B → $OUT_ROOT (GPU ${NGPU}장) ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv | tee -a "$LOGS/main.log" || true
run_stage 0 "$LOGS/prep.log" --stage prep "${COMMON[@]}"

# ---- 정합성 사전 검사: 옛 실행 잔재가 이번 실행과 섞이는 것을 원천 차단 ----
# ① π(drift adapter)보다 오래된 π-의존 산출물 격리 — 재학습 후 옛 점수/rollout이
#    병합에 섞여 조용히 오염되는 경로 차단 (B6 확장)
AD=$(ls -t "$OUT_ROOT"/drift_*/adapter_config.json 2>/dev/null | head -1)
if [ -n "${AD:-}" ]; then
  SDIR="$OUT_ROOT/stale-$(date +%s)"; moved=0
  for f in "$OUT_ROOT"/rollouts_fresh_train*.jsonl "$OUT_ROOT"/rollouts_fresh_val.jsonl \
           "$OUT_ROOT"/oracle_micro_groups*.pt "$OUT_ROOT"/scores_oracle.json \
           "$OUT_ROOT"/scores_splithalf.json "$OUT_ROOT"/scores_offpolicy*.json \
           "$OUT_ROOT"/scores_hybrid_*.json "$OUT_ROOT"/rollouts_hybrid_*.jsonl \
           "$OUT_ROOT"/val_gradient.pt "$OUT_ROOT"/val_groups.pt \
           "$OUT_ROOT"/report.md "$OUT_ROOT"/report.json; do
    if [ -f "$f" ] && [ "$f" -ot "$AD" ]; then
      mkdir -p "$SDIR"; mv "$f" "$SDIR/"; moved=$((moved + 1))
    fi
  done
  [ "${moved:-0}" -gt 0 ] && log "[정합성] π 재학습 이전 산출물 ${moved}개 격리 → $SDIR"
fi
# ② 현재 GPU 분할(n)과 안 맞는 샤드 격리 — 다른 n으로 만든 샤드를 재사용하면
#    스킵→병합에서 누락/중복으로 죽거나(멈춤) 옛 값이 덮어써진다(오염)
"$PY" - "$OUT_ROOT" "$NGPU" <<'PYEOF' | tee -a "$LOGS/main.log"
import json, sys, time
from pathlib import Path
root, n = Path(sys.argv[1]), int(sys.argv[2])
nm = n - 1 if n >= 2 else 1
pj = root / "prompts.json"
if not pj.exists():
    sys.exit(0)
n_train = len(json.loads(pj.read_text())["train"])
per = (n_train + n - 1) // n
stale = []
for base in ("rollouts_behavior_train", "rollouts_fresh_train"):
    if (root / (base + ".jsonl")).exists():
        continue  # 병합 완료 — 샤드는 더 안 쓰임
    for p in root.glob(base + ".shard*.jsonl"):
        try:
            i = int(p.name.split("shard")[1].split(".")[0])
            lo, hi = i * per, min((i + 1) * per, n_train)
            idx = {json.loads(l)["prompt_idx"] for l in p.open()}
            ok = i < n and idx == set(range(lo, hi))
        except Exception:
            ok = False
        if not ok:
            stale.append(p)
for p in root.glob("scores_offpolicy.shard*.json"):
    try:
        ok = int(p.name.split("shard")[1].split(".")[0]) < n
    except Exception:
        ok = False
    if not ok:
        stale.append(p)
for p in root.glob("oracle_micro_groups.shard*.pt"):
    try:
        ok = int(p.name.split("shard")[1].split(".")[0]) < nm
    except Exception:
        ok = False
    if not ok:
        stale.append(p)
if stale:
    d = root / f"stale-shards-{int(time.time())}"
    d.mkdir(exist_ok=True)
    for p in stale:
        p.rename(d / p.name)
    print(f"[정합성] 현재 분할(n={n})과 안 맞는 샤드 {len(stale)}개 격리 → {d.name}")
PYEOF
# β rollout N샤딩
pids=(); for i in $(seq 0 $((NGPU - 1))); do
  ( run_stage "$i" "$LOGS/beta-shard$i.log" --stage rollout-behavior "${COMMON[@]}" --shard "$i:$NGPU" ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || exit 1; done
merge_rollouts rollouts_behavior_train || exit 1
run_stage 0 "$LOGS/drift.log" --stage drift "${COMMON[@]}" --drift-steps "$DRIFT"
# π fresh N샤딩
pids=(); for i in $(seq 0 $((NGPU - 1))); do
  ( run_stage "$i" "$LOGS/fresh-shard$i.log" --stage rollout-fresh "${COMMON[@]}" --adapter "$OUT_ROOT/drift_$DRIFT" --shard "$i:$NGPU" ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || exit 1; done
merge_rollouts rollouts_fresh_train || exit 1
# val 방향 ∥ oracle micro 샤딩 (GPU 여유가 있으면 마지막 GPU를 val 전용으로)
pids=()
if [ "$NGPU" -ge 2 ]; then
  NM=$((NGPU - 1))
  ( run_stage "$NM" "$LOGS/val-grads.log" --stage val-grads "${COMMON[@]}" --adapter "$OUT_ROOT/drift_$DRIFT" ) & pids+=($!)
else
  NM=1
  run_stage 0 "$LOGS/val-grads.log" --stage val-grads "${COMMON[@]}" --adapter "$OUT_ROOT/drift_$DRIFT" || exit 1
fi
for i in $(seq 0 $((NM - 1))); do
  ( run_stage "$i" "$LOGS/ograds-shard$i.log" --stage oracle-grads "${COMMON[@]}" --adapter "$OUT_ROOT/drift_$DRIFT" --shard "$i:$NM" ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || exit 1; done
# 2×2 score N샤딩
pids=(); for i in $(seq 0 $((NGPU - 1))); do
  ( run_stage "$i" "$LOGS/score-shard$i.log" --stage score-shard "${COMMON[@]}" --adapter "$OUT_ROOT/drift_$DRIFT" --shard "$i:$NGPU" ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || exit 1; done
run_stage 0 "$LOGS/merge.log" --stage merge-grads "${COMMON[@]}"
run_stage 0 "$LOGS/report.log" --stage report "${COMMON[@]}"
# hybrid 3절단점 — GPU 수만큼 병렬 (모자라면 라운드로빈 순차)
pids=(); gpu=0
for cut in 0.25 0.5 0.75; do
  ( run_stage "$((gpu % NGPU))" "$LOGS/hybrid-$cut.log" --stage hybrid "${COMMON[@]}" --adapter "$OUT_ROOT/drift_$DRIFT" --cut-frac "$cut" ) & pids+=($!)
  gpu=$((gpu + 1))
  [ $((gpu % NGPU)) -eq 0 ] && for p in "${pids[@]}"; do wait "$p" || exit 1; done && pids=()
done
for p in "${pids[@]}"; do wait "$p" || exit 1; done
log "=== 14B 완료 — bash scripts/result.sh 로 판정 (OUT_ROOT=$OUT_ROOT) ==="
touch "$OUT_ROOT/DONE"
