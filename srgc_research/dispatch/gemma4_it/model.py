"""Validate the downloaded IT snapshot and load only its text model, offline."""

import hashlib
import json
from pathlib import Path

from srgc_rebuttal.runtime import atomic_json, lease

from .storage import group_work, inside

# Public Hub metadata for google/gemma-4-12B-it at the immutable adapter revision.
# The first hash is a Git blob SHA1 for small JSON files, SHA256 for LFS files.
OFFICIAL_FILES = {
    "config.json": (4423, "324c841b79e36f42e27f08c572c327533469c859", "blob"),
    "generation_config.json": (260, "d09dccf1b9aee197ef72a6e88a1ceff74b842ff3", "blob"),
    "tokenizer.json": (32169626, "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f", "sha256"),
    "tokenizer_config.json": (3089, "6520cdc25462a5eba96fdac6d8cf3caa461feb35", "blob"),
    "chat_template.jinja": (18683, "4741bf6e4132ba23a5537f9d6e74e9a6d613d7cd", "blob"),
    "model.safetensors": (23919549408, "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d", "sha256"),
}


def complete_snapshot(path):
    return all((path / name).is_file() for name in OFFICIAL_FILES)


def find_snapshot(environment):
    from .adapter import REVISION

    group, work = group_work(environment)
    explicit = environment.get("SRGC_GEMMA_IT_MODEL_PATH") or environment.get("SRGC_GEMMA_MODEL_PATH")
    if explicit:
        path = inside(explicit, group)
        if not complete_snapshot(path):
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
            if path.name.lower() in {"gemma-4-12b-it", "gemma4-12b-it"} and complete_snapshot(path):
                found.add(inside(path, group))
        for name in ("models--google--gemma-4-12B-it", "gemma-4-12B-it"):
            snapshots = root / name / "snapshots"
            if snapshots.is_dir():
                for path in snapshots.iterdir():
                    if path.name == REVISION and complete_snapshot(path):
                        found.add(inside(path, group))
    if len(found) != 1:
        error = FileNotFoundError if not found else ValueError
        raise error(f"Found {len(found)} complete local Gemma 4 12B IT snapshots; "
                    "set SRGC_GEMMA_IT_MODEL_PATH to the intended directory")
    return found.pop()



def ensure_snapshot(environment):
    """Fetch the pinned IT snapshot once, before any GPU admission or ranks."""
    try:
        return find_snapshot(environment)
    except FileNotFoundError:
        pass
    from unittest.mock import patch
    from huggingface_hub import snapshot_download
    from huggingface_hub import constants as hub_constants
    from .adapter import MODEL, REVISION

    group, _ = group_work(environment)
    explicit = environment.get("SRGC_GEMMA_IT_MODEL_PATH") or environment.get("SRGC_GEMMA_MODEL_PATH")
    destination = inside(explicit or (Path(environment.get("MODELS_DIR", str(group / "models")))
                                     / "gemma-4-12B-it"), group)
    native = destination / "generation_config.json"
    if native.is_file() and json.loads(native.read_text()).get("eos_token_id") != [1, 106, 50]:
        raise ValueError("Gemma path points to PT or another model; IT weights must use a separate directory")
    with lease(destination.with_suffix(".download.lock"), wait=True):
        try:
            return find_snapshot(environment)
        except FileNotFoundError:
            pass
        print(f"GEMMA downloading {MODEL}@{REVISION} to {destination}; no GPU launched", flush=True)
        with patch.dict("os.environ", {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}), \
                patch.object(hub_constants, "HF_HUB_OFFLINE", False):
            snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=destination,
                              allow_patterns=list(OFFICIAL_FILES), local_files_only=False, token=False)
    return find_snapshot(environment)

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
        raise ValueError("Gemma IT requires BOS=2, EOS=1, PAD=0; the generation configuration additionally stops at IT turn/tool tokens")
    for text, token_id in (("<bos>", 2), ("<eos>", 1), ("<pad>", 0)):
        if tokenizer.encode(text, add_special_tokens=False) != [token_id]:
            raise ValueError(f"wrong Gemma special token: {text}")
    if tokenizer.encode("<turn|>", add_special_tokens=False) != [106]:
        raise ValueError("wrong Gemma IT end-of-turn token")
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": "2 + 2?"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    from .protocol import completion_cue, prompt_prefix
    if (not rendered.startswith(prompt_prefix("math")) or not rendered.endswith(completion_cue("math"))
            or rendered.count("<bos>") != 1):
        raise ValueError("Gemma IT chat template differs from its recorded protocol")
    probe = "Solve 2 + 2. Answer:"
    if tokenizer.encode(probe, add_special_tokens=False) != reference.encode(probe, add_special_tokens=False).ids:
        raise ValueError("Gemma tokenizer preprocessing differs from tokenizer.json")


def resolve_snapshot(environment):
    from .adapter import MODEL, REVISION

    group, work = group_work(environment)
    path = find_snapshot(environment)
    files = sorted(OFFICIAL_FILES)
    files += [name for name in ("added_tokens.json", "special_tokens_map.json") if (path / name).is_file()]
    for name in files:
        inside(path / name, group)
        if not (path / name).is_file():
            raise ValueError(f"Incomplete Gemma snapshot: {path / name}")
        if name in OFFICIAL_FILES and (path / name).stat().st_size != OFFICIAL_FILES[name][0]:
            raise ValueError(f"Gemma file size differs from the pinned 12B IT snapshot: {name}")
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "gemma4_unified" or config.get("quantization_config"):
        raise ValueError("Need the unquantized Gemma 4 12B instruction-tuned checkpoint")

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
            if name in OFFICIAL_FILES and actual != OFFICIAL_FILES[name][1]:
                raise ValueError(f"Gemma file differs from the pinned instruction-tuned checkpoint: {name}")
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
    """The official IT template already supplies BOS; retain all official stops."""

    def __init__(self, tokenizer):
        from .protocol import EOS_TOKENS
        self.tokenizer = tokenizer
        self.eos_token_id = list(EOS_TOKENS)
        if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (2, 1, 0):
            raise ValueError("wrong Gemma IT BOS/EOS/PAD")

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def encode(self, text, *, add_special_tokens=True, **kwargs):
        ids = self.tokenizer.encode(text, add_special_tokens=False, **kwargs)
        if add_special_tokens and (not ids or ids[0] != 2 or ids.count(2) != 1):
            raise ValueError("Gemma IT prompt must contain exactly one template BOS")
        return ids

    def __call__(self, text, *, return_tensors="pt", add_special_tokens=True, **kwargs):
        import torch
        if return_tensors != "pt" or not isinstance(text, str) or kwargs:
            raise ValueError("Gemma SRGC tokenization expects one rendered IT prompt")
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}


def load_text_model(source, device, *, attention="eager", dtype=None):
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
        raise ValueError("Gemma IT requires BOS=2, EOS=1, PAD=0")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    policy, info = Gemma4UnifiedForCausalLM.from_pretrained(
        source, config=config, local_files_only=True, output_loading_info=True,
        dtype=dtype or (torch.bfloat16 if str(device).startswith("cuda") else torch.float32),
        attn_implementation=attention,
        key_mapping={r"^model\.language_model\.": "model."},
    )
    if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete Gemma text weights: {info}")
    permitted = ("model.embed_vision.", "model.embed_audio.", "model.vision_embedder.")
    unexpected = [key for key in info.get("unexpected_keys", []) if not key.startswith(permitted)]
    if unexpected:
        raise ValueError(f"Unexpected Gemma text weights: {unexpected}")
    from .protocol import EOS_TOKENS, SUPPRESS_TOKENS
    native = json.loads((Path(source) / "generation_config.json").read_text())
    if (native.get("eos_token_id") != list(EOS_TOKENS)
            or native.get("suppress_tokens") != list(SUPPRESS_TOKENS)):
        raise ValueError("Gemma IT official stop/suppression tokens differ")
    # Match the comparison's T=1 full-vocabulary policy, retaining IT end tokens.
    policy.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=list(EOS_TOKENS), pad_token_id=0, do_sample=True,
        temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, use_cache=True,
        suppress_tokens=list(SUPPRESS_TOKENS),
    )
    policy.eval().to(device)
    from ..gemma4.memory import bounded_generate

    policy.generate = bounded_generate(policy.generate, torch.device(device))
    return policy, PromptTokenizer(tokenizer)
