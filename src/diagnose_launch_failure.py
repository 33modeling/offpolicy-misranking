#!/usr/bin/env python3
"""Turn a failed launcher session log into one Korean diagnosis + one action.

    python3 src/diagnose_launch_failure.py SESSION_LOG

Reads only the log; no torch. Exit 0 always (diagnosis is advisory).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# (pattern, diagnosis, action). First match from the END of the log wins.
RULES: list[tuple[str, str, str]] = [
    (r"math-verify is required|No module named 'math_verify'",
     "math-verify bundle not on PYTHONPATH (code older than 004f5f8)",
     "git pull --ff-only, then rerun"),
    (r"\[locate-abort\]",
     "no Qwen3.5-9B folder with weights found under the model roots",
     "check the folder is under $MODELS_DIR, or set OM_SNAPSHOT_PATH=/path/to/folder"),
    (r"missing model\.safetensors-0000\d-of-0000\d\.safetensors \(no unique",
     "uploaded shard sizes differ from the official files (truncated or other revision)",
     "see [locate] lines below; re-upload that file, or run with OM_ALLOW_UNPINNED_SNAPSHOT=1"),
    (r"unreadable safetensors header|shards contain only \d+ tensors",
     "a *.safetensors file is truncated or not a safetensors file",
     "see the [locate]/[model] lines for the file; re-upload it"),
    (r"file is not registered for the pinned model|model file missing: |model file (size|hash) mismatch",
     "uploaded model files differ from the pinned Hub revision (downloaded from main?) or are missing",
     "to use them anyway: OM_ALLOW_UNPINNED_SNAPSHOT=1 bash scripts/run_qwen35_9b.sh"),
    (r"safetensors shard set incomplete|no \*\.safetensors in |neither a safetensors index",
     "the checked folder holds no *.safetensors (see the [locate] lines for what IS there)",
     "if the weights sit in another folder: OM_SNAPSHOT_PATH=/that/folder bash scripts/run_qwen35_9b.sh"),
    (r"missing files: .*\.om_snapshot\.json|model snapshot missing|snapshot provenance mismatch|cannot prove local Hub revision",
     "model snapshot not sealed",
     "plain run seals an uploaded folder; download needs internet (prepare on an online machine)"),
    (r"waiting for shared snapshot preparation lock",
     "prepare is blocked by another prepare holding locks/additional-provision.lock (probably hung on a download)",
     "do NOT prepare on the cluster. pkill -f 'run_additional_experiments.sh --prepare'; then: bash scripts/run_qwen35_9b.sh"),
    (r"Qwen3\.5 needs transformers>=5|No module named 'transformers\.models\.qwen3_5'|cannot import name 'AutoModelForMultimodalLM'|Unrecognized configuration class.*qwen3_5",
     "venv transformers lacks the Qwen3.5 classes (needs 5.x)",
     "$VENV_DIR/bin/pip install -U 'transformers>=5' then rerun"),
    (r"expected FLA 0\.5\.2|fla-core .* is not installed|No module named 'fla'|FLA fused recurrent/chunk kernels are unavailable",
     "flash-linear-attention (fla-core 0.5.2) missing or wrong version",
     "$VENV_DIR/bin/pip install 'flash-linear-attention[cuda]==0.5.2' then rerun"),
    (r"CUDA out of memory|OutOfMemoryError",
     "GPU out of memory",
     "check nvidia-smi for other processes; if it repeats lower runtime.generation_batch"),
    (r"unspecified launch failure|illegal memory access|CUBLAS_STATUS|device-side assert",
     "CUDA runtime error (driver/kernel level, not a code bug)",
     "rerun; if it repeats on the same GPU, change node"),
    (r"exactly four H100 GPUs required",
     "this node does not have 4x H100",
     "run on a 4x H100 node"),
    (r"GPUs are already in use|four GPUs did not become idle|GPU memory did not clear",
     "another process holds the GPUs",
     "nvidia-smi; free them or use an idle node"),
    (r"additional suite already queued on this physical node",
     "this launcher is already running on this node",
     "use the running one or stop it first"),
    (r"OLMo primary launcher is running on THIS node",
     "this node is busy with the OLMo experiment; its 4 GPUs are taken for days",
     "run bash scripts/run_qwen35_9b.sh on a node where OLMo is NOT running"),
    (r"queued behind local primary",
     "waiting: the OLMo primary launcher holds this node's lock (not an error)",
     "run on a node where OLMo is not running"),
    (r"checkout is dirty|worktree is dirty",
     "uncommitted changes under src/scripts/configs",
     "git status; commit or git checkout -- . then rerun"),
    (r"venv missing|venv python",
     "no python in VENV_DIR",
     "scripts/provision.sh or set VENV_DIR"),
    (r"OfflineModeIsEnabled|HF_HUB_OFFLINE|Cannot reach|ConnectionError|Repository Not Found|401 Client Error|Max retries exceeded|Name or service not known|no internet",
     "no Hugging Face access from this machine (the cluster is offline)",
     "do NOT use prepare on the cluster; uploaded models/datasets are adopted by plain run"),
    (r"qualification does not match|content fingerprint|official row count|official local dataset not found",
     "MATH-500/MBPP snapshot missing or not matching the official content",
     "upload the official dataset files under $DATASETS_DIR (any folder name)"),
    (r"chat_template missing",
     "tokenizer chat_template missing from the snapshot",
     "upload chat_template.jinja / tokenizer_config.json from the Hub revision"),
    (r"LoRA targets missing",
     "config lora_targets do not exist in the model",
     "check configs/*.json lora_targets"),
    (r"prompts differ from the qualified matrix|prompts\.json differs",
     "prompt split differs from the contract (run initialized with another data copy)",
     "run dir was quarantined automatically; rerun"),
    (r"first policy-loss evaluation is not on-policy",
     "GRPO first ratio != 1 (numerical instability)",
     "rerun; if it repeats check attn=eager and FLA version"),
    (r"invalid policy loss/gradient|zero or non-finite gradient norm",
     "NaN/zero gradient in a GRPO step; optimizer not applied, resumes from checkpoint",
     "rerun the same command"),
]

EXC_RE = re.compile(r"^(?:\S+Error|RuntimeError|ValueError|KeyError|OSError|ImportError|TypeError|AssertionError)\b.*|.*abort\].*")


def diagnose(text: str) -> tuple[str, str, str] | None:
    lines = text.splitlines()
    for line in reversed(lines):
        for pattern, why, what in RULES:
            if re.search(pattern, line):
                return why, what, line.strip()
    return None


def last_exception(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if EXC_RE.match(stripped) and not stripped.startswith("[exit]") and not stripped.startswith("[error]"):
            return stripped[:300]
    return None


def snapshot_listing(text: str) -> list[str]:
    """For weight/snapshot failures, echo the locator's on-disk comparison lines."""
    lines = [line.strip() for line in text.splitlines() if line.startswith("[locate]")]
    return lines[-40:]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: diagnose_launch_failure.py SESSION_LOG", file=sys.stderr)
        return 0
    path = Path(sys.argv[1])
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"DIAGNOSIS: cannot read log ({exc})")
        return 0
    found = diagnose(text)
    if found:
        why, what, line = found
        print(f"DIAGNOSIS: {why}")
        print(f"ACTION:    {what}")
        print(f"EVIDENCE:  {line[:300]}")
        if re.search(r"shard|snapshot|model files|folder with weights|index", why):
            for extra in snapshot_listing(text):
                print(f"  {extra}")
        return 0
    exc = last_exception(text)
    if exc:
        print("DIAGNOSIS: unknown pattern; this exception is the cause")
        print(f"EVIDENCE:  {exc}")
    else:
        print("DIAGNOSIS: no exception/abort line in the log (killed by signal / OOM-killer / external)")
        print("ACTION:    dmesg -T | tail, or rerun the same command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
