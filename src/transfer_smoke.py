#!/usr/bin/env python3
"""Runtime smoke for a pinned transfer model before launching the long matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def fingerprint(model: Path, targets: list[str]) -> dict:
    import torch

    return {
        "schema": "offpolicy-transfer-runtime-smoke/v1",
        "git": git_head(),
        "host": platform.node(),
        "model": str(model.resolve()),
        "model_config_sha256": sha256_file(model / "config.json"),
        "snapshot_manifest_sha256": sha256_file(model / ".om_snapshot.json"),
        "lora_targets": targets,
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else None,
    }


def quick_cuda_health() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    left = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
    value = (left @ left).float().mean()
    if not torch.isfinite(value):
        raise RuntimeError("CUDA bfloat16 matmul produced a non-finite result")
    torch.cuda.synchronize()


def marker_matches(marker: Path, expected: dict) -> bool:
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return recorded == {**expected, "status": "passed"}


def run_smoke(model_path: Path, targets: list[str]) -> None:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    from rollout import chat_ids, load_model

    model, tokenizer = load_model(str(model_path), device="cuda")
    ids = chat_ids(
        tokenizer,
        "Answer with one short sentence: why should an experiment be reproducible?",
    ).unsqueeze(0).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=4,
            pad_token_id=tokenizer.eos_token_id,
        )
    if generated.shape[1] <= ids.shape[1]:
        raise RuntimeError("model generation produced no response token")

    wrapped = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=targets,
            lora_dropout=0.0,
        ),
    )
    wrapped.enable_input_require_grads()
    wrapped.train()
    wrapped.zero_grad(set_to_none=True)
    loss = wrapped(ids, labels=ids).loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"LoRA smoke loss is not finite: {float(loss)}")
    loss.backward()
    gradients = [
        parameter.grad
        for name, parameter in wrapped.named_parameters()
        if "lora_" in name and parameter.requires_grad
    ]
    if not gradients or not all(
        gradient is not None and torch.isfinite(gradient).all() for gradient in gradients
    ):
        raise RuntimeError("LoRA target modules did not receive finite gradients")

    with tempfile.TemporaryDirectory(prefix="offpolicy-transfer-adapter-") as raw_tmp:
        adapter = Path(raw_tmp)
        wrapped.save_pretrained(adapter)
        if not (adapter / "adapter_config.json").is_file():
            raise RuntimeError("PEFT adapter save did not publish adapter_config.json")
        base = wrapped.unload()
        reloaded = PeftModel.from_pretrained(base, adapter)
        merged = reloaded.merge_and_unload()
        merged.eval()
        with torch.no_grad():
            logits = merged(ids).logits
        if not torch.isfinite(logits[:, -1]).all():
            raise RuntimeError("reloaded and merged adapter produced non-finite logits")

    del merged, reloaded, base, wrapped, model, generated, ids
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lora-targets", required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    targets = [target.strip() for target in args.lora_targets.split(",") if target.strip()]
    if not targets:
        parser.error("--lora-targets must contain at least one module name")

    try:
        quick_cuda_health()
        expected = fingerprint(args.model, targets)
        if not args.force and marker_matches(args.marker, expected):
            print(f"[transfer-smoke] cached runtime contract passed: {args.marker}")
            return 0
        run_smoke(args.model, targets)
        args.marker.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.marker.with_name(f"{args.marker.name}.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps({**expected, "status": "passed"}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.marker)
        print(f"[transfer-smoke] generation + LoRA backward/save/reload passed: {args.model}")
        return 0
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[transfer-smoke-abort] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
