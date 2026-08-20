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
NGPU=$(timeout 20 nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
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
  echo "        해당 작업을 시작한 tmux 창에서 중단하거나 그 run의 PID만 종료할 것."
  exit 1
fi
# 점유 검사는 내가 쓸 GPU만 대상 (OM_GPUS 분할 실행 시 서로 간섭 금지)
BUSY=$(timeout 20 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' -v g="${OM_GPUS:-}" 'BEGIN{n=split(g,a,","); for(i=1;i<=n;i++) sel[a[i]]=1}
       { if ((n==0 || ($1 in sel)) && $2 > 2000) c++ } END{print c+0}')
if [ "${BUSY:-0}" -gt 0 ] && [ "${OM_SKIP_GPU_CHECK:-0}" != "1" ]; then
  echo "[abort] GPU ${BUSY}개가 이미 점유 중:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
  exit 1
fi
# 샤드 병합 + 커버리지 검증 — GPU 수가 바뀐 재시작이면 샤드 분할이 어긋나
# 조용한 누락/중복이 생기므로, prompt 전수·무중복을 확인하고 원자적으로 쓴다.
merge_rollouts() {  # merge_rollouts <base이름> <prompt당 K>
  local base="$1" expected_k="$2"
  "$PY" - "$OUT_ROOT" "$base" "$expected_k" <<'PYEOF'
import json, sys
from pathlib import Path
root, base, expected_k = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
merged = root / (base + ".jsonl")
sources = [merged] if merged.exists() else sorted(root.glob(base + ".shard*.jsonl"))
if not sources:
    print(f"[merge-abort] {base}: shard files not found", flush=True)
    sys.exit(1)
n_train = len(json.loads((root / "prompts.json").read_text())["train"])
seen = {}
for s in sources:
    for line in s.open():
        r = json.loads(line)
        seen.setdefault(r["prompt_idx"], []).append((r["rollout_idx"], line))
missing = [i for i in range(n_train) if i not in seen]
unexpected = sorted(i for i in seen if i < 0 or i >= n_train)
dup = [i for i, v in seen.items() if len({j for j, _ in v}) != len(v)]
bad_k = [i for i, v in seen.items() if 0 <= i < n_train
         if sorted(j for j, _ in v) != list(range(expected_k))]
if missing or unexpected or dup or bad_k:
    cleanup = merged if merged.exists() else root / f"{base}.shard*.jsonl"
    print(f"[merge-abort] {base}: 누락 {len(missing)}개(예 {missing[:5]}) "
          f"범위 밖 {len(unexpected)}개(예 {unexpected[:5]}) 중복 {len(dup)}개 "
          f"exact-K 실패 {len(bad_k)}개(예 {bad_k[:5]}) — "
          f"GPU 수 변경 등으로 샤드 분할이 어긋남. 정리 후 재실행:\n"
          f"  rm {cleanup}", flush=True)
    sys.exit(1)
if merged.exists():
    print(f"[validate] {base}: {n_train} prompts x K={expected_k}, "
          f"{sum(len(v) for v in seen.values())} rollouts OK")
    sys.exit(0)
tmp = root / (base + ".jsonl.tmp")
with tmp.open("w") as f:
    for i in range(n_train):
        for _, line in sorted(seen[i]):
            f.write(line)
tmp.rename(root / (base + ".jsonl"))
print(f"[merge] {base}: {n_train} prompts x K={expected_k}, "
      f"{sum(len(v) for v in seen.values())} rollouts OK")
PYEOF
}
verify_code_snapshot() {
  "$PY" - "$OUT_ROOT/run_config.json" <<'PYEOF'
import hashlib, json, subprocess, sys
config = json.load(open(sys.argv[1]))
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
status = subprocess.check_output(
    ["git", "status", "--porcelain", "--", "src", "scripts"], text=True
).strip()
diff = hashlib.sha256(subprocess.check_output(
    ["git", "diff", "HEAD", "--no-ext-diff", "--binary", "--", "src", "scripts"]
)).hexdigest()
if (head != config.get("git") or status != config.get("git_status")
        or diff != config.get("git_diff_sha256")):
    print("[code-abort] repository changed after this run was initialized", flush=True)
    print(f"  initial={config.get('git')} current={head}", flush=True)
    sys.exit(1)
PYEOF
}
run_stage() { local dev="${GPUS[$1]:-$1}" lf="$2"; shift 2
  local t0=$SECONDS
  verify_code_snapshot || return 1
  log "GPU$dev ▶ $*"
  if CUDA_VISIBLE_DEVICES="$dev" "$PY" src/experiment.py "$@" >> "$lf" 2>&1; then
    log "GPU$dev ✔ $1 $2 ($((SECONDS - t0))s)"
  else local rc=$?; log "GPU$dev ✘ $* rc=$rc"; tail -8 "$lf" | tee -a "$LOGS/main.log"; return $rc; fi
}
DRIFT="${DRIFT:-100}"
export DRIFT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 클러스터 노드들의 fused SDPA ULF 이력(C5·C6) — 어떤 경로로 실행해도 eager 기본.
# 빠른 커널을 검증한 노드에서만 OM_ATTN=sdpa 로 명시 해제.
export OM_ATTN="${OM_ATTN:-eager}"
DATASET="${DATASET:-gsm8k}"
export MODEL_14B DATASET OUT_ROOT
# 데이터셋 접미사는 멱등 — 호출자가 v2-s0-dapo-math·v2-27b-dapo-math-s0처럼 데이터셋명이
# 이미 든 경로를 넘기면 그대로 쓴다. 무조건 덧붙이면 v2-s0-dapo-math-dapo-math에 산출물이
# 쌓여 호출자의 DONE 체크(go_v2.sh:98 등)가 완주를 영구 미인식 → 매 루프 전체 재실행.
if [ "$DATASET" != "gsm8k" ]; then
  case "$(basename "$OUT_ROOT")" in
    *"$DATASET"*) ;;
    *) OUT_ROOT="${OUT_ROOT}-${DATASET}" ;;
  esac
  LOGS="$OUT_ROOT/logs"; mkdir -p "$LOGS"
  [ -d "${OUT_ROOT}-${DATASET}" ] && echo "[주의] 구버전 이중 접미사 디렉터리 존재: ${OUT_ROOT}-${DATASET} — DONE·report가 그쪽에 있으면 수동 이관 판단 필요" | tee -a "$LOGS/main.log"
fi
# 데이터 사전 검사 — 오프라인 노드에서 허브 직행으로 죽는 것을 시작 전에 잡는다
# 데이터 사전 검사 — 로더 자신을 그대로 실행 (검사·실제 로드가 같은 코드 경로)
# 실패 시 로더가 '찾아본 위치' 목록을 출력하므로 원인 자가진단됨. GPU 잡기 전에 죽는다.
# 반복 kill이 남긴 HF datasets 캐시 stale lock 청소 — flock 무한 대기가
# "gsm8k만 완주하고 dapo-math에서 조용히 멈춤"의 유력 원인 (gsm8k는 로컬
# jsonl 경로라 datasets 라이브러리를 안 탐). 30분 넘은 lock만 지운다.
find "${HF_HOME:-/nonexistent}" -name '*.lock' -mmin +30 -delete 2>/dev/null || true
# timeout — 데이터 계층이 어떤 이유로든 멈추면 무한 침묵 대신 진단 메시지
timeout 600 "$PY" -c "import sys; sys.path.insert(0, 'src'); from data import load_prompts; \
r = load_prompts('$DATASET', ${N_TRAIN:-256}, ${N_VAL:-50}); \
print('[preflight] $DATASET 데이터 OK — train', len(r['train']), '/ val', len(r['val']))"
rc=$?
if [ "$rc" -ne 0 ]; then
  if [ "$rc" -eq 124 ]; then
    echo "[abort] $DATASET 데이터 로드 600초 초과 — 데이터셋 계층 스톨(HF lock/허브 대기)."
    echo "        진단: bash scripts/check_data.sh $DATASET"
    echo "        해법: 온라인 셸에서 bash scripts/fetch_datasets.sh $DATASET (로컬 jsonl 확보 시 라이브러리 우회)"
  else
    echo "[abort] $DATASET 데이터 로드 실패(스키마·크기 포함) — 위 메시지 확인 / fetch_datasets.sh"
  fi
  exit 1
fi
# run manifest — 재현성 기록 (감사 §17)
"$PY" - "$OUT_ROOT" <<'PYEOF' || exit 1
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True).strip()
    except Exception: return "?"
def digest_file(path):
    p = Path(path) if path else None
    return hashlib.sha256(p.read_bytes()).hexdigest() if p and p.is_file() else None
def env_int(name, default):
    return int(os.environ.get(name, str(default)))
import torch, transformers
model = os.environ["MODEL_14B"]
pool = os.environ.get("OM_POOL_FILE")
pool_manifest = pool + ".manifest.json" if pool else None
git_head = sh("git rev-parse HEAD")
git_status = sh("git status --porcelain -- src scripts")
if git_status and os.environ.get("OM_ALLOW_DIRTY", "0") != "1":
    print("[config-abort] src/scripts worktree is dirty; commit first or set OM_ALLOW_DIRTY=1")
    print(git_status)
    sys.exit(2)
git_diff_hash = hashlib.sha256(
    subprocess.check_output(
        ["git", "diff", "HEAD", "--no-ext-diff", "--binary", "--", "src", "scripts"]
    )
).hexdigest()
config = {
    "git": git_head, "git_diff_sha256": git_diff_hash,
    "git_status": git_status,
    "model": model, "model_resolved": str(Path(model).resolve()),
    "model_config_sha256": digest_file(Path(model) / "config.json"),
    "tokenizer_config_sha256": digest_file(Path(model) / "tokenizer_config.json"),
    "generation_config_sha256": digest_file(Path(model) / "generation_config.json"),
    "dataset": os.environ["DATASET"], "pool": pool,
    "pool_sha256": digest_file(pool),
    "pool_manifest": pool_manifest,
    "pool_manifest_sha256": digest_file(pool_manifest),
    "n_train": env_int("N_TRAIN", 256), "n_val": env_int("N_VAL", 50),
    "behavior_k": env_int("BEHAVIOR_K", 8), "fresh_k": env_int("FRESH_K", 16),
    "val_k": env_int("VAL_K", 8), "micro_group": env_int("MICRO_GROUP", 4),
    "hybrid_prompts": env_int("HYBRID_PROMPTS", 24),
    "k_cell": env_int("K_CELL", 8),
    "seed": env_int("SEED", 0), "drift": env_int("DRIFT", 100),
    "max_new_tokens": env_int("MAX_NEW_TOKENS", 512),
    "proj_dim": env_int("PROJ_DIM", 4096), "grad_layers": env_int("GRAD_LAYERS", 4),
    "clip_cap": float(os.environ.get("CLIP_CAP", "10.0")),
    "temperature": float(os.environ.get("TEMPERATURE", "1.0")),
    "topk_frac": float(os.environ.get("TOPK_FRAC", "0.10")),
    "radius_mode": os.environ.get("RADIUS_MODE", "gaussian"),
    "top_p": float(os.environ.get("OM_TOP_P", "1.0")),
    "thinking": os.environ.get("OM_THINKING", "off"),
    "attn": os.environ.get("OM_ATTN", "eager"),
    "gen_batch": os.environ.get("OM_GEN_BATCH"),
    "lora_targets": os.environ.get("OM_LORA_TARGETS"),
    "skip_hybrid": os.environ.get("OM_SKIP_HYBRID", "0"),
}
encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
config["digest"] = hashlib.sha256(encoded).hexdigest()
lock = root / "run_config.json"
if lock.exists():
    previous = json.loads(lock.read_text())
    if previous.get("digest") != config["digest"]:
        changed = sorted(k for k in set(previous) | set(config)
                         if k != "digest" and previous.get(k) != config.get(k))
        print(f"[config-abort] existing artifacts use a different run config: {changed}")
        sys.exit(2)
else:
    tmp = lock.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(config, indent=1))
    tmp.replace(lock)
m = {"time": time.strftime("%F %T"), **config,
     "torch": torch.__version__, "transformers": transformers.__version__,
     "cuda": torch.version.cuda}
tmp = root / "manifest.json.tmp"
tmp.write_text(json.dumps(m, indent=1))
tmp.replace(root / "manifest.json")
print("[manifest]", git_head[:8], config["dataset"], "n=", config["n_train"],
      "seed=", config["seed"], "config=", config["digest"][:12])
PYEOF

COMMON=(--run "$OUT_ROOT" --model "$MODEL_14B" --dataset "$DATASET"
        --behavior-k "${BEHAVIOR_K:-8}" --fresh-k "${FRESH_K:-16}"
        --val-k "${VAL_K:-8}" --micro-group "${MICRO_GROUP:-4}"
        --hybrid-prompts "${HYBRID_PROMPTS:-24}" --micro-batch 1
        --n-train "${N_TRAIN:-256}" --n-val "${N_VAL:-50}" --seed "${SEED:-0}"
        --max-new-tokens "${MAX_NEW_TOKENS:-512}" --proj-dim "${PROJ_DIM:-4096}"
        --grad-layers "${GRAD_LAYERS:-4}" --clip-cap "${CLIP_CAP:-10.0}"
        --temperature "${TEMPERATURE:-1.0}" --topk-frac "${TOPK_FRAC:-0.10}"
        --radius-mode "${RADIUS_MODE:-gaussian}" --k-cell "${K_CELL:-8}")

KA_DEV=$(IFS=,; echo "${GPUS[*]}")
CUDA_VISIBLE_DEVICES="$KA_DEV" "$PY" scripts/gpu_keepalive.py > "$LOGS/keepalive.log" 2>&1 &
KEEP=$!
printf '%s\n' "$KEEP" > "$OUT_ROOT/keepalive.pid"
# 종료(정상·에러 모두) 시 이 실행이 띄운 모든 자식 정리 — 고아 샤드가 GPU를 점유한 채
# 남아 "계속 실행되는 것처럼" 보이고 재시작 OOM을 일으키는 버그의 수정
cleanup() {
  kill "$KEEP" 2>/dev/null || true
  rm -f "$OUT_ROOT/keepalive.pid"
  pkill -f -- "--run $OUT_ROOT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

log "=== 14B 시작: $MODEL_14B → $OUT_ROOT (GPU ${NGPU}장) ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv | tee -a "$LOGS/main.log" || true
run_stage 0 "$LOGS/prep.log" --stage prep "${COMMON[@]}" || exit 1

# ---- 정합성 사전 검사: 옛 실행 잔재가 이번 실행과 섞이는 것을 원천 차단 ----
# ① π(drift adapter)보다 오래된 π-의존 산출물 격리 — 재학습 후 옛 점수/rollout이
#    병합에 섞여 조용히 오염되는 경로 차단 (B6 확장)
AD=$(ls -t "$OUT_ROOT"/drift_*/adapter_config.json 2>/dev/null | head -1)
if [ -n "${AD:-}" ]; then
  SDIR="$OUT_ROOT/stale-$(date +%s)"; moved=0
  for f in "$OUT_ROOT"/rollouts_fresh_train*.jsonl "$OUT_ROOT"/rollouts_fresh_train*.manifest.json \
           "$OUT_ROOT"/rollouts_fresh_val.jsonl "$OUT_ROOT"/rollouts_fresh_val.manifest.json \
           "$OUT_ROOT"/oracle_micro_groups*.pt "$OUT_ROOT"/scores_oracle.json \
           "$OUT_ROOT"/scores_splithalf.json "$OUT_ROOT"/scores_offpolicy*.json \
           "$OUT_ROOT"/scores_hybrid_*.json "$OUT_ROOT"/rollouts_hybrid_*.jsonl \
           "$OUT_ROOT"/rollouts_hybrid_*.manifest.json \
           "$OUT_ROOT"/val_gradient.pt "$OUT_ROOT"/val_groups.pt \
           "$OUT_ROOT"/score_protocol*.json "$OUT_ROOT"/oracle_protocol.json \
           "$OUT_ROOT"/hybrid_protocol_*.json \
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
    sidecars = []
    for p in stale:
        if p.suffix == ".jsonl":
            sidecar = p.with_name(p.stem + ".manifest.json")
            if sidecar.exists():
                sidecars.append(sidecar)
    stale = list(dict.fromkeys(stale + sidecars))
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
merge_rollouts rollouts_behavior_train "${BEHAVIOR_K:-8}" || exit 1
if [ -n "${OM_POOL_FILE:-}" ]; then
  "$PY" src/qualify_pool.py "$OUT_ROOT" "$OM_POOL_FILE" \
    --topk-frac "${TOPK_FRAC:-0.10}" | tee -a "$LOGS/main.log" || exit 1
fi
run_stage 0 "$LOGS/drift.log" --stage drift "${COMMON[@]}" --drift-steps "$DRIFT" || exit 1
# π fresh N샤딩
pids=(); for i in $(seq 0 $((NGPU - 1))); do
  ( run_stage "$i" "$LOGS/fresh-shard$i.log" --stage rollout-fresh "${COMMON[@]}" --adapter "$OUT_ROOT/drift_$DRIFT" --shard "$i:$NGPU" ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || exit 1; done
merge_rollouts rollouts_fresh_train "${FRESH_K:-16}" || exit 1
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
run_stage 0 "$LOGS/merge.log" --stage merge-grads "${COMMON[@]}" || exit 1
run_stage 0 "$LOGS/report.log" --stage report "${COMMON[@]}" || exit 1
# hybrid 3절단점 — GPU 수만큼 병렬 (모자라면 라운드로빈 순차)
# 주의: hybrid는 π+β 두 모델을 한 GPU에 동시 상주시키는 유일한 스테이지 —
# 27B급(57GB×2>80GB)은 구조적 불가 → OM_SKIP_HYBRID=1로 생략
# (B11 탐색 판정엔 반전율·경보만 필요, C1' 인과는 7B 본편 담당)
if [ "${OM_SKIP_HYBRID:-0}" = "1" ]; then
  log "hybrid 스킵 (OM_SKIP_HYBRID=1 — 대형 모델 π+β 동시 상주 불가)"
else
pids=(); gpu=0
for cut in 0.25 0.5 0.75; do
  ( run_stage "$((gpu % NGPU))" "$LOGS/hybrid-$cut.log" --stage hybrid "${COMMON[@]}" --adapter "$OUT_ROOT/drift_$DRIFT" --cut-frac "$cut" ) & pids+=($!)
  gpu=$((gpu + 1))
  [ $((gpu % NGPU)) -eq 0 ] && for p in "${pids[@]}"; do wait "$p" || exit 1; done && pids=()
done
for p in "${pids[@]}"; do wait "$p" || exit 1; done
fi
required=(prompts.json rollouts_behavior_train.jsonl rollouts_fresh_train.jsonl
          val_gradient.pt val_groups.pt oracle_micro_groups.pt scores_oracle.json
          scores_splithalf.json scores_offpolicy.json score_protocol.json
          oracle_protocol.json report.json)
for artifact in "${required[@]}"; do
  [ -s "$OUT_ROOT/$artifact" ] || { log "[abort] 필수 산출물 누락/빈 파일: $artifact"; exit 1; }
done
printf '%s\n' "completed $(date -Is)" > "$OUT_ROOT/DONE.tmp"
mv "$OUT_ROOT/DONE.tmp" "$OUT_ROOT/DONE"
log "=== 14B 완료 — bash scripts/result.sh 로 판정 (OUT_ROOT=$OUT_ROOT) ==="
