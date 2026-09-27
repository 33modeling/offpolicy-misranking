"""Select the operational Pair/MBPP Python without installing any packages."""

import os
import json
from pathlib import Path
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]


def work_root(environment):
    return Path(environment.get("OM_WORK", f"/group-volume/{environment.get('OM_USER', 'minsoo3.kim')}/offpolicy-misranking"))


def python_path(dataset, environment):
    variable = "SWITCH_PYTHON" if dataset == "mbpp" else "PAIR_PYTHON"
    explicit = environment.get(variable)
    candidate = explicit or str(Path(environment.get("VENV_DIR", str(work_root(environment) / ".venv-cu126"))) / "bin/python")
    found = shutil.which(candidate)
    if found:
        return os.path.abspath(found)
    if explicit:
        raise ValueError(f"existing {variable} interpreter is not executable: {explicit}")
    return sys.executable


def select_python(dataset):
    target = python_path(dataset, os.environ)
    if os.path.abspath(target) != os.path.abspath(sys.executable):
        print(f"[runtime] using existing {dataset} Python: {target}", file=sys.stderr, flush=True)
        os.execv(target, [target, str(REPO / "scripts/run_srgc_rebuttal.py"), *sys.argv[1:]])


def verifier_environment(environment):
    """Reuse Pair's offline, hash-verified bundle; never pip into its venv."""
    result = subprocess.run([sys.executable, str(REPO / "src/bootstrap_math_verify.py"),
        "--cache-root", str(work_root(environment) / "runtime-deps")], env=environment,
        check=True, capture_output=True, text=True, timeout=120)
    path = result.stdout.strip()
    if not Path(path).is_dir():
        raise ValueError("existing verifier bootstrap returned no runtime directory")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (path, environment.get("PYTHONPATH"))))
    environment["OM_MATH_VERIFIER"] = "math_verify"
    sys.path.insert(0, path)


def runtime_packages():
    import importlib.metadata
    sys.path.insert(0, str(REPO / "src"))
    # This is the same model-family admission used by the operational OLMo runner.
    from model_matrix import _require_runtime
    _require_runtime({"key": "olmo3-7b-base", "model_type": "olmo3"})
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    from peft import LoraConfig, get_peft_model  # noqa: F401
    versions = {}
    for name in ("numpy", "torch", "transformers", "peft", "math-verify", "datasets"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def model_path(model, revision, environment):
    """Resolve the same pinned local weights used by the existing OLMo runner."""
    sys.path.insert(0, str(REPO / "src"))
    from model_matrix import validate_snapshot_provenance
    specs = json.loads((REPO / "configs/olmo3_rlzero.json").read_text())["models"]
    spec = next((s for s in specs if s["repository"] == model and s["revision"] == revision), None)
    if spec is None:
        raise ValueError("model/revision is not registered in the existing OLMo configuration")
    group_models = Path(environment.get("GROUP_VOLUME", "/group-volume")) / "models"
    models = Path(environment.get("MODELS_DIR", str(group_models if group_models.is_dir() else work_root(environment) / "models")))
    explicit = environment.get("OM_OLMO3_MODEL_PATH")
    path = Path(explicit) if explicit else models / spec["local_directory"]
    if explicit or path.is_dir():
        validate_snapshot_provenance(spec, path)
        return str(path.resolve())
    from huggingface_hub import snapshot_download
    return snapshot_download(model, revision=revision, local_files_only=True)


def load_model(model, revision, device):
    sys.path.insert(0, str(REPO / "src"))
    from rollout import load_model as operational_load_model
    source = os.environ.get("SRGC_LOCAL_MODEL") or model_path(model, revision, os.environ)
    previous = os.environ.get("OM_ATTN")
    os.environ["OM_ATTN"] = "eager"
    try:
        policy, tokenizer = operational_load_model(source, device=str(device), dtype="bfloat16")
    finally:
        if previous is None:
            os.environ.pop("OM_ATTN", None)
        else:
            os.environ["OM_ATTN"] = previous
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return policy, tokenizer
