"""An isolated HF package overlay; keep the cluster's existing CUDA PyTorch."""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid

from srgc_rebuttal.runtime import atomic_json, lease

from .storage import group_work, inside

MODEL_PACKAGES = {"transformers": "5.19.0", "peft": "0.21.0", "accelerate": "1.10.1"}
PACKAGES = {
    **MODEL_PACKAGES,
    "huggingface-hub": "2.2.0", "tokenizers": "0.23.3", "safetensors": "0.8.0",
    "numpy": "2.5.3", "packaging": "26.3", "pyyaml": "6.0.3", "regex": "2026.9.29",
    "typer": "0.27.3", "tqdm": "4.70.1", "filelock": "4.1.0", "fsspec": "2026.9.0",
    "hf-xet": "1.7.0", "httpx2": "2.13.1", "httpcore2": "2.13.1",
    "typing-extensions": "4.16.0", "click": "8.5.0", "anyio": "4.15.1",
    "idna": "3.20", "truststore": "0.10.4", "h11": "0.16.0",
    "shellingham": "1.5.4", "rich": "15.0.0", "annotated-doc": "0.0.5",
    "markdown-it-py": "4.2.0", "pygments": "2.21.0", "mdurl": "0.1.2",
    "psutil": "7.0.0",
}


def disable_optional_media():
    """A text-only process must not import a broken torchvision/torchaudio ABI."""
    from transformers import utils
    from transformers.utils import import_utils

    for module in (utils, import_utils):
        for name in ("is_torchvision_available", "is_torchvision_v2_available", "is_torchaudio_available", "is_librosa_available"):
            if hasattr(module, name):
                setattr(module, name, lambda: False)


def installed_versions(path):
    return {distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
            for distribution in importlib.metadata.distributions(path=[str(path)])}


def package_overlay(environment):
    """Install pinned non-Torch wheels in shared storage once, with an atomic lease."""
    group, work = group_work(environment)
    protocol = {"packages": PACKAGES, "python": list(sys.version_info[:2]), "machine": platform.machine()}
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:20]
    target = inside(work / "gemma-runtime-cache/packages" / key, group)
    with lease(target.with_suffix(".lock"), wait=True):
        receipt = target / ".runtime.json"
        if target.exists():
            if not receipt.is_file() or json.loads(receipt.read_text()) != protocol:
                raise ValueError(f"Gemma runtime overlay is incomplete/incompatible: {target}")
            actual = installed_versions(target)
            if any(actual.get(name) != version for name, version in PACKAGES.items()):
                raise ValueError(f"Gemma runtime packages changed: {target}")
            return target
        temporary = inside(target.with_name(f".{key}.tmp-{uuid.uuid4().hex}"), group)
        try:
            temporary.mkdir(parents=True)
            command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps",
                       "--only-binary=:all:", "--no-compile", "--target", str(temporary),
                       *(f"{name}=={version}" for name, version in PACKAGES.items())]
            print("GEMMA runtime: preparing isolated packages; existing Torch/venvs stay unchanged", flush=True)
            subprocess.run(command, env=dict(environment), check=True)
            actual = installed_versions(temporary)
            if any(actual.get(name) != version for name, version in PACKAGES.items()):
                raise ValueError("Gemma package installation did not produce the pinned runtime")
            atomic_json(temporary / ".runtime.json", protocol)
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return target


def activate(environment):
    # Every rank imports the overlay via the parent's PYTHONPATH. Only the
    # operational entry installs it; children never pip or download a model.
    target = package_overlay(environment)
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(filter(None, (str(target), environment.get("PYTHONPATH")))))
    sys.path.insert(0, str(target))
    disable_optional_media()
    try:
        import torch
    except ImportError as error:
        raise ValueError("Gemma needs the existing CUDA Torch Python; use the same interpreter as the working Llama job") from error
    from packaging.version import Version

    if Version(torch.__version__.split("+")[0]) < Version("2.5"):
        raise ValueError(f"Gemma requires Torch>=2.5; found {torch.__version__}")
    return target
