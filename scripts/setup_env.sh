#!/usr/bin/env bash
# 셸마다 한 번 source — failure-atlas/WEASEL/CROPI와 같은 규약:
#   source scripts/setup_env.sh
#
# - 무거운 것(venv·모델·산출물·캐시)은 전부 group-volume, 체크아웃에는 코드만.
# - 경로가 다르면 source 전에 override:
#     export MODELS_DIR=/group-volume/nait-models
#     source scripts/setup_env.sh
# - 없는 경로는 경고만 하고 셸을 죽이지 않는다.

# ---- 볼륨 ---------------------------------------------------------------
export GROUP_VOLUME="${GROUP_VOLUME:-/group-volume}"
export OM_USER="${OM_USER:-minsoo3.kim}"
export OM_REPO="${OM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
export OM_WORK="${OM_WORK:-$GROUP_VOLUME/$OM_USER/offpolicy-misranking}"
export STORAGE_ROOT="$OM_WORK"

# ---- 모델 (failure-atlas와 같은 고정 스냅샷 경로 공유) -------------------
export MODELS_DIR="${MODELS_DIR:-$GROUP_VOLUME/models}"
export MODEL_QWEN25_05B="${MODEL_QWEN25_05B:-$MODELS_DIR/Qwen2.5-0.5B-Instruct}"
export MODEL_QWEN25_7B="${MODEL_QWEN25_7B:-$MODELS_DIR/Qwen2.5-7B-Instruct}"

# ---- venv·캐시 (group-volume, FDMU 검증 torch 2.7.1+cu126 스택) ---------
export VENV_DIR="${VENV_DIR:-$OM_WORK/.venv-cu126}"
export HF_HOME="${HF_HOME:-$OM_WORK/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$OM_WORK/cache/pip}"
export TMPDIR="${TMPDIR:-$OM_WORK/tmp}"
export PYTHONPYCACHEPREFIX="$OM_WORK/cache/pycache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" 2>/dev/null || true

# 클러스터는 HF egress 불안정 — 기본 오프라인, 다운로드 머신에서만:
#   OM_ONLINE=1 source scripts/setup_env.sh
if [[ "${OM_ONLINE:-0}" == "1" ]]; then
  export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
else
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$OM_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# ---- 데이터 (오프라인 노드용 로컬 사본 — provision이 만든다) -------------
export OM_DATA="${OM_DATA:-$OM_WORK/data}"

# ---- 점검 (경고만) ------------------------------------------------------
_om_warn() { echo "[setup_env][warn] $1" >&2; }
[[ -d "$GROUP_VOLUME" ]] || _om_warn "group-volume 없음: $GROUP_VOLUME — 'export GROUP_VOLUME=<mount>' 후 다시 source"
for _v in MODEL_QWEN25_05B MODEL_QWEN25_7B; do
  _p="${!_v}"
  [[ -d "$_p" ]] || _om_warn "$_v 없음: $_p — 'bash scripts/provision.sh'가 다운로드(온라인 머신에서)"
done
[[ -x "$VENV_DIR/bin/python" ]] || _om_warn "venv 없음: $VENV_DIR — 'bash scripts/provision.sh' 실행"

echo "[setup_env] OM_WORK=$OM_WORK VENV=$VENV_DIR"
echo "[setup_env] MODELS: 0.5B=$MODEL_QWEN25_05B 7B=$MODEL_QWEN25_7B (offline=${HF_HUB_OFFLINE:-0})"
