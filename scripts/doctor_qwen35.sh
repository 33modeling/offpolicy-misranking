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
echo "--- points: model-alias contract (the 2026-09-11 matrix stalled here) ---"
# Every Qwen point whose rollout manifests record a different model basename than its run
# configuration fails validate_generation_contract, and the supervisor then repairs the alias
# (src/repair_model_alias.py) or gives up with rc=43. This prints, per point, which of the two
# happens and why, without touching an artifact.
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" - "$OM_WORK" <<'PYEOF' 2>&1 | sed 's/^/  /'
import json, sys
from pathlib import Path
from artifact_contract import PRIMARY_SOURCES, validate_generation_contract
from repair_model_alias import prove_model
work = Path(sys.argv[1])
roots = sorted(p for p in (work / "runs").glob("qwen*") if p.is_dir())
if not roots:
    print("no qwen run root under", work / "runs")
for root in roots:
    runs = sorted(p for p in root.glob("*/family-*/*-d*") if (p / "run_config.json").is_file()) or \
           sorted(p for p in root.glob("family-*/*-d*") if (p / "run_config.json").is_file())
    print(f"{root.name}: {len(runs)} point(s) with a run configuration")
    for run in runs:
        config = json.loads((run / "run_config.json").read_text())
        expected = Path(str(config.get("model", ""))).name
        documents, recorded = [], set()
        for manifest in sorted(run.glob("rollouts_*manifest.json")):
            try:
                document = json.loads(manifest.read_text())
            except (OSError, ValueError):
                continue
            documents.append(document)
            recorded.add(Path(str(document.get("model_name_or_path", ""))).name)
        try:
            validate_generation_contract(run, PRIMARY_SOURCES)
            verdict = "contract OK"
        except Exception as exc:
            verdict = f"contract fails: {str(exc)[:90]}"
            if recorded and recorded != {expected}:
                try:
                    prove_model(config, documents)
                    verdict += " | alias provable -> the supervisor repairs it on the next entry"
                except Exception as why:
                    verdict += f" | alias NOT provable: {type(why).__name__}: {str(why)[:80]}"
        print(f"  {run.parent.name}/{run.name}: expected {expected!r} recorded {sorted(recorded) or '-'} | {verdict}")
PYEOF
