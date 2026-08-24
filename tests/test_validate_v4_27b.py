"""The 27B publication gate rejects mixed generation/runtime provenance."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from validate_v4_27b import FIXED_CONFIG, validate_matrix, validate_run


def write_config(
    root: Path, seed: int, dataset: str, git: str, model_hash: str
) -> Path:
    suffix = "" if dataset == "gsm8k" else "-math500"
    run = root / f"v4-27b-s{seed}{suffix}"
    run.mkdir()
    config = {
        **FIXED_CONFIG,
        "git": git,
        "git_status": "",
        "git_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "model_config_sha256": model_hash,
        "seed": seed,
        "dataset": dataset,
        "n_train": 512 if dataset == "gsm8k" else 400,
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config["digest"] = hashlib.sha256(encoded).hexdigest()
    (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    return run


with tempfile.TemporaryDirectory() as raw_tmp:
    root = Path(raw_tmp)
    git = "a" * 40
    model_hash = "b" * 64
    for seed in range(5):
        for dataset in ("gsm8k", "math500"):
            write_config(root, seed, dataset, git, model_hash)

    assert len(validate_matrix(root, git, model_hash)) == 10
    validate_run(root / "v4-27b-s0", git, model_hash, 0, "gsm8k")

    path = root / "v4-27b-s3-math500/run_config.json"
    config = json.loads(path.read_text())
    config["linear_attention_backend"] = "torch"
    path.write_text(json.dumps(config))
    try:
        validate_matrix(root, git, model_hash)
    except ValueError as exc:
        assert "linear_attention_backend" in str(exc)
    else:
        raise AssertionError("mixed backend was accepted")

print("PASS v4 27B publication provenance is fail-closed")


def test_validate_v4_27b() -> None:
    pass
