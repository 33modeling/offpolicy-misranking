#!/usr/bin/env bash
# 한 화면 진단 (GPU 안 씀): 코드 버전, 모델 폴더 탐색 결과·파일 대조, venv, 데이터셋.
#   bash scripts/run_qwen35_9b.sh doctor
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
CONFIG=${PROFILE_CONFIG:-configs/qwen35_9b_grpo.json}
echo "code     : $(git rev-parse --short HEAD) ($(git log -1 --format=%s | cut -c1-50))"
echo "MODELS_DIR: $MODELS_DIR"
echo "DATASETS_DIR: $DATASETS_DIR"
echo "venv     : $PY"
"$PY" - <<'PYEOF' 2>&1 | sed 's/^/           /'
import importlib.metadata as m
for name in ("torch", "transformers", "peft", "fla-core", "math-verify"):
    try: print(f"{name}={m.version(name)}")
    except Exception: print(f"{name}=missing")
try:
    from transformers import AutoModelForMultimodalLM  # noqa
    from transformers.models.qwen3_5 import modeling_qwen3_5  # noqa
    print("qwen3_5 classes=OK")
except Exception as e: print(f"qwen3_5 classes=missing ({type(e).__name__})")
PYEOF
echo "--- model ---"
KEY=$("$PY" src/model_matrix.py --config "$CONFIG" list-models | head -1)
if P=$("$PY" src/locate_uploaded_snapshot.py --config "$CONFIG" --model-key "$KEY" --models-dir "$MODELS_DIR" 2>/tmp/doctor.$$); then
  echo "found: $P"
else
  echo "not found"
fi
sed 's/^\[locate\] //' /tmp/doctor.$$; rm -f /tmp/doctor.$$
echo "--- datasets ---"
for d in math500 mbpp; do
  f=$(ls "$DATASETS_DIR"/$d/*.jsonl 2>/dev/null | head -1)
  [ -n "$f" ] && echo "$d: $f ($(wc -l < "$f") rows)" || echo "$d: $DATASETS_DIR/$d 아래 jsonl missing (업로드본은 내용으로 자동 인식됨 — run에서 확인)"
done
