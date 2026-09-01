"""Legacy math loaders can reproduce the immutable d0 prompt split."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import load_prompts
from materialize_prompt_dataset import materialize


def test_materialized_snapshots_reproduce_source_order(tmp_path: Path) -> None:
    prompts = {
        "train": [
            {"question": f"train-{index}", "answer": str(index)} for index in range(4)
        ],
        "val": [{"question": f"val-{index}", "answer": str(index)} for index in range(2)],
    }
    source = tmp_path / "prompts.json"
    source.write_text(json.dumps(prompts), encoding="utf-8")

    for dataset, env_name in (("gsm8k", "GSM8K_DIR"), ("math500", "MATH500_DIR")):
        root = materialize(source, dataset, tmp_path / "snapshots")
        old = os.environ.get(env_name)
        os.environ[env_name] = str(root)
        try:
            assert load_prompts(dataset, 4, 2, seed=0) == prompts
        finally:
            if old is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = old
