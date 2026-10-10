"""Validate the downloaded PT snapshot and load only its text model, offline."""

import hashlib
import json
from pathlib import Path

from srgc_rebuttal.runtime import atomic_json, lease

from .storage import group_work, inside

# Public Hub metadata for google/gemma-4-12B at the immutable adapter revision.
# The first hash is a Git blob SHA1 for small JSON files, SHA256 for LFS files.
OFFICIAL_FILES = {
    "config.json": (4383, "6d3994fd0b98fd5eaaaadbf918bd826d778ce5ea", "blob"),
    "generation_config.json": (233, "6e4ef65e8cf563a9177fd933423f89eed2f74d74", "blob"),
    "tokenizer.json": (32170070, "12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6", "sha256"),
    "tokenizer_config.json": (888, "0a3fe0009f6816c5c615c6102eea6599680c1523", "blob"),
    "model.safetensors": (23919549408, "fe054ae05ff7f44318fd8ae90d58992531455c7ed31356704088f0f2d8c8009a", "sha256"),
}


def find_snapshot(environment):
    from .adapter import REVISION

    group, work = group_work(environment)
    explicit = environment.get("SRGC_GEMMA_MODEL_PATH")
    if explicit:
        path = inside(explicit, group)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Gemma model missing: {path}")
        return path
    roots = [
        Path(environment.get("MODELS_DIR", str(group / "models"))),
        work / "models",
        group / environment.get("OM_USER", "minsoo3.kim") / "models",
        work / "gemma-runtime-cache/huggingface/hub",
        *map(Path, json.loads(environment.get("SRGC_GEMMA_ORIGINAL_HUBS", "[]"))),
    ]
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if environment.get(key):
            roots.append(Path(environment[key]))
    found = set()
    for root in set(roots):
        root = inside(root, group)
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if path.name.lower() in {"gemma-4-12b", "gemma4-12b", "gemma-4-12b-pt"} and (path / "config.json").is_file():
                found.add(inside(path, group))
        for name in ("models--google--gemma-4-12B", "gemma-4-12B"):
            snapshots = root / name / "snapshots"
            if snapshots.is_dir():
                for path in snapshots.iterdir():
                    if path.name == REVISION and (path / "config.json").is_file():
                        found.add(inside(path, group))
    if len(found) != 1:
        raise ValueError(
            f"Found {len(found)} local Gemma 4 12B PT snapshots; set SRGC_GEMMA_MODEL_PATH to the downloaded directory. No model download runs"
        )
    return found.pop()


def validate_tokenizer_metadata(path, config):
    from .runtime import disable_optional_media

    disable_optional_media()
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    metadata = json.loads((path / "tokenizer_config.json").read_text())
    if not isinstance(metadata, dict) or metadata.get("auto_map"):
        raise ValueError("expected a local Gemma tokenizer without remote code")
    reference = Tokenizer.from_file(str(path / "tokenizer.json"))
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if tokenizer.get_vocab() != reference.get_vocab() or len(tokenizer) != config["text_config"]["vocab_size"]:
        raise ValueError("Gemma token vocabulary differs from the pinned tokenizer/model")
    if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (2, 1, 0):
        raise ValueError("Gemma PT requires BOS=2, EOS=1, PAD=0; IT uses a different end token")
    for text, token_id in (("<bos>", 2), ("<eos>", 1), ("<pad>", 0)):
        if tokenizer.encode(text, add_special_tokens=False) != [token_id]:
            raise ValueError(f"wrong Gemma special token: {text}")
    probe = "Solve 2 + 2. Answer:"
    if tokenizer.encode(probe, add_special_tokens=False) != reference.encode(probe, add_special_tokens=False).ids:
        raise ValueError("Gemma tokenizer preprocessing differs from tokenizer.json")


def resolve_snapshot(environment):
    from .adapter import MODEL, REVISION

    group, work = group_work(environment)
    path = find_snapshot(environment)
    files = sorted(OFFICIAL_FILES)
    files += [name for name in ("added_tokens.json", "special_tokens_map.json", "chat_template.jinja") if (path / name).is_file()]
    for name in files:
        inside(path / name, group)
        if not (path / name).is_file():
            raise ValueError(f"Incomplete Gemma snapshot: {path / name}")
        if name in OFFICIAL_FILES and name != "tokenizer_config.json" and (path / name).stat().st_size != OFFICIAL_FILES[name][0]:
            raise ValueError(f"Gemma file size differs from the pinned 12B PT snapshot: {name}")
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "gemma4_unified" or config.get("quantization_config"):
        raise ValueError("Need the unquantized Gemma 4 12B pretrained checkpoint")

    def fingerprint():
        return {name: [s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns] for name in files for s in [(path / name).stat()]}

    key = hashlib.sha256(str(path).encode()).hexdigest()
    marker = inside(work / "gemma-runtime-cache/verified-models" / f"{key}.json", group)
    with lease(marker.with_suffix(".lock"), wait=True):
        before = fingerprint()
        saved = json.loads(marker.read_text()) if marker.exists() else None
        if saved and (saved.get("model"), saved.get("revision")) != (MODEL, REVISION):
            raise ValueError("Gemma verification receipt has a different model identity")
        if saved and saved.get("fingerprint") == before:
            return path, saved["snapshot_sha256"]
        records = {}
        for name in files:
            size = (path / name).stat().st_size
            kind = OFFICIAL_FILES.get(name, (None, None, None))[2]
            sha = hashlib.sha256()
            blob = hashlib.sha1(f"blob {size}\0".encode()) if kind == "blob" else None
            print(f"[gemma-model] verifying {name}", flush=True)
            with (path / name).open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    sha.update(chunk)
                    if blob:
                        blob.update(chunk)
            actual = blob.hexdigest() if blob else sha.hexdigest()
            if name in OFFICIAL_FILES and actual != OFFICIAL_FILES[name][1] and name != "tokenizer_config.json":
                raise ValueError(f"Gemma file differs from the pinned pretrained checkpoint: {name}")
            records[name] = {"size": size, "sha256": sha.hexdigest()}
        # Verify what HF actually loads even if metadata was reserialized.
        validate_tokenizer_metadata(path, config)
        identity = hashlib.sha256(json.dumps({"model": MODEL, "revision": REVISION, "files": records}, sort_keys=True).encode()).hexdigest()
        if fingerprint() != before:
            raise ValueError("Gemma snapshot changed during verification")
        if saved and saved.get("snapshot_sha256") != identity:
            raise ValueError("Gemma weights/tokenizer changed since verification; existing results are preserved")
        atomic_json(marker, {"model": MODEL, "revision": REVISION, "files": records, "snapshot_sha256": identity, "fingerprint": before})
    return path, identity


class PromptTokenizer:
    """Plain PT prompts have one explicit BOS, independently of tokenizer defaults."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (2, 1, 0):
            raise ValueError("wrong Gemma PT BOS/EOS/PAD")

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def encode(self, text, *, add_special_tokens=True, **kwargs):
        ids = self.tokenizer.encode(text, add_special_tokens=False, **kwargs)
        if not add_special_tokens:
            return ids
        if not isinstance(ids, list) or 2 in ids:
            raise ValueError("Gemma PT prompt must be plain text without embedded BOS")
        return [2, *ids]

    def __call__(self, text, *, return_tensors="pt", add_special_tokens=True, **kwargs):
        import torch

        if return_tensors != "pt" or not isinstance(text, str) or kwargs:
            raise ValueError("Gemma SRGC tokenization expects one plain text prompt")
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}


def load_text_model(source, device, *, attention="eager"):
    from .runtime import disable_optional_media

    disable_optional_media()
    import torch
    from transformers import (
        AutoTokenizer,
        Gemma4UnifiedForCausalLM,
        Gemma4UnifiedTextConfig,
        GenerationConfig,
    )

    config = Gemma4UnifiedTextConfig.from_pretrained(source, local_files_only=True)
    if config.model_type != "gemma4_unified_text" or getattr(config, "quantization_config", None):
        raise ValueError("Expected the unquantized Gemma 4 unified text model")
    if (config.bos_token_id, config.eos_token_id, config.pad_token_id) != (2, 1, 0):
        raise ValueError("Gemma PT requires BOS=2, EOS=1, PAD=0")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    policy, info = Gemma4UnifiedForCausalLM.from_pretrained(
        source, config=config, local_files_only=True, output_loading_info=True,
        dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
        attn_implementation=attention,
        key_mapping={r"^model\.language_model\.": "model."},
    )
    if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete Gemma text weights: {info}")
    permitted = ("model.embed_vision.", "model.embed_audio.", "model.vision_embedder.")
    unexpected = [key for key in info.get("unexpected_keys", []) if not key.startswith(permitted)]
    if unexpected:
        raise ValueError(f"Unexpected Gemma text weights: {unexpected}")
    # Do not inherit IT stop tokens, top-k/p or any logits processors. This
    # matches the full-vocabulary temperature=1 GRPO/cache sampling protocol.
    policy.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=1, pad_token_id=0, do_sample=True,
        temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, use_cache=True,
    )
    policy.eval().to(device)
    from .memory import bounded_generate

    policy.generate = bounded_generate(policy.generate, torch.device(device))
    return policy, PromptTokenizer(tokenizer)
