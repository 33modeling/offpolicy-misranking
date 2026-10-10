"""Use existing full Llama weights offline and bind their content to each run."""

import hashlib
import json
from pathlib import Path

from srgc_rebuttal.runtime import atomic_json, lease

from .storage import group_work, inside

# Public Hub tree at 0e9e39f249a16976918f6564b8830bc894c89659.
# Gated LFS hashes are not public; record the actual local shard SHA256 below.
OFFICIAL_FILES = {
    "config.json": (855, "0bb6fd75b3ad2fe988565929f329945262c2814e"),
    "generation_config.json": (184, "cc7276afd599de091142c6ed3005faf8a74aa257"),
    "model.safetensors.index.json": (23950, "0fd8120f1c6acddc268ebc2583058efaf699a771"),
    "special_tokens_map.json": (296, "02ee80b6196926a5ad790a004d9efd6ab1ba6542"),
    "tokenizer.json": (9085657, "5cc5f00a5b203e90a27a3bd60d1ec393b07971e8"),
    "tokenizer_config.json": (55351, "db88166e2bc4c799fd5d1ae643b75e84d03ee70e"),
    "model-00001-of-00004.safetensors": (4976698672, None),
    "model-00002-of-00004.safetensors": (4999802720, None),
    "model-00003-of-00004.safetensors": (4915916176, None),
    "model-00004-of-00004.safetensors": (1168138808, None),
}


def find_snapshot(environment):
    from .adapter import REVISION

    group, work = group_work(environment)
    explicit = environment.get("SRGC_LLAMA_MODEL_PATH")
    if explicit:
        path = inside(explicit, group)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Llama model missing: {path}")
        return path
    roots = [
        Path(environment.get("MODELS_DIR", str(group / "models"))),
        work / "models",
    ]
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if environment.get(key):
            roots.append(Path(environment[key]))
    roots.append(work / "llama-runtime-cache/huggingface/hub")
    candidates = []
    for root in roots:
        root = inside(root, group)
        if not root.is_dir():
            continue
        candidates += [
            root / name
            for name in ("Llama-3.1-8B-Instruct", "Meta-Llama-3.1-8B-Instruct")
        ]
        candidates += [
            p
            for p in root.iterdir()
            if p.name.lower()
            in {
                "llama-3.1-8b-instruct",
                "meta-llama-3.1-8b-instruct",
                "llama3.1-8b-instruct",
            }
        ]
        for name in (
            "models--meta-llama--Llama-3.1-8B-Instruct",
            "Llama-3.1-8B-Instruct",
        ):
            snapshots = root / name / "snapshots"
            if snapshots.is_dir():
                candidates.append(snapshots / REVISION)
                candidates += sorted(snapshots.iterdir())
    found = {
        inside(path, group) for path in candidates if (path / "config.json").is_file()
    }
    if len(found) != 1:
        raise ValueError(
            f"Found {len(found)} local Llama snapshots. Set SRGC_LLAMA_MODEL_PATH to the downloaded directory; no download will run"
        )
    return found.pop()


def resolve_snapshot(environment):
    from .adapter import MODEL, REVISION

    group, work = group_work(environment)
    path = find_snapshot(environment)
    files = sorted(OFFICIAL_FILES)
    for name in ("chat_template.jinja", "added_tokens.json"):
        if (path / name).is_file():
            files.append(name)
    for name in files:
        inside(path / name, group)
        if not (path / name).is_file():
            raise ValueError(f"Incomplete Llama snapshot: {path / name}")
        if (
            name in OFFICIAL_FILES
            and (path / name).stat().st_size != OFFICIAL_FILES[name][0]
        ):
            raise ValueError(f"Llama file size differs from pinned snapshot: {name}")
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "llama" or config.get("quantization_config"):
        raise ValueError("Need the unquantized Llama-3.1-8B-Instruct checkpoint")

    def fingerprint():
        return {
            name: [s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
            for name in files
            for s in [(path / name).stat()]
        }

    key = hashlib.sha256(str(path).encode()).hexdigest()
    marker = inside(work / "llama-runtime-cache/verified-models" / f"{key}.json", group)
    with lease(marker.with_suffix(".lock"), wait=True):
        before = fingerprint()
        saved = json.loads(marker.read_text()) if marker.exists() else None
        if saved and (saved.get("model"), saved.get("revision")) != (MODEL, REVISION):
            raise ValueError(
                "Llama verification receipt has a different model identity"
            )
        if saved and saved.get("fingerprint") == before:
            return path, saved["snapshot_sha256"]
        records = {}
        for name in files:
            file = path / name
            expected_blob = OFFICIAL_FILES.get(name, (None, None))[1]
            sha = hashlib.sha256()
            blob = (
                hashlib.sha1(f"blob {file.stat().st_size}\0".encode())
                if expected_blob
                else None
            )
            print(f"[llama-model] verifying {name}", flush=True)
            with file.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    sha.update(chunk)
                    if blob:
                        blob.update(chunk)
            if blob and blob.hexdigest() != expected_blob:
                raise ValueError(
                    f"Llama tokenizer/config differs from pinned snapshot: {name}"
                )
            records[name] = {"size": file.stat().st_size, "sha256": sha.hexdigest()}
        identity = hashlib.sha256(
            json.dumps(
                {"model": MODEL, "revision": REVISION, "files": records}, sort_keys=True
            ).encode()
        ).hexdigest()
        if fingerprint() != before:
            raise ValueError("Llama snapshot changed during verification")
        if saved and saved.get("snapshot_sha256") != identity:
            raise ValueError(
                "Llama weights changed since the first verification; existing results are preserved"
            )
        atomic_json(
            marker,
            {
                "model": MODEL,
                "revision": REVISION,
                "files": records,
                "snapshot_sha256": identity,
                "fingerprint": before,
            },
        )
        return path, identity


class PromptTokenizer:
    """Chat strings already contain BOS; preserve all Llama generation EOS IDs."""

    def __init__(self, tokenizer, eos):
        self.tokenizer = tokenizer
        self.eos_token_id = eos

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **{**kwargs, "add_special_tokens": False})

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **{**kwargs, "add_special_tokens": False})


def load_text_model(source, device, *, attention="eager"):
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    config = LlamaConfig.from_pretrained(source, local_files_only=True)
    if config.model_type != "llama" or getattr(config, "quantization_config", None):
        raise ValueError("Expected an unquantized Llama checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    policy, info = LlamaForCausalLM.from_pretrained(
        source,
        config=config,
        local_files_only=True,
        output_loading_info=True,
        torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
        attn_implementation=attention,
    )
    if any(
        info.get(key)
        for key in ("missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs")
    ):
        raise ValueError(f"Incomplete or incompatible Llama weights: {info}")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Llama tokenizer needs an EOS token")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    eos = policy.generation_config.eos_token_id or tokenizer.eos_token_id
    policy.generation_config.repetition_penalty = 1.0
    policy.eval().to(device)
    from .memory import bounded_generate

    policy.generate = bounded_generate(policy.generate, torch.device(device))
    return policy, PromptTokenizer(tokenizer, eos)
