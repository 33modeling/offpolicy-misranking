"""CPU-only tests for independent hard-pool qualification."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qualify_pool import qualify, sha256_file  # noqa: E402


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value))


def fixture(root: Path, mixed: int) -> tuple[Path, Path]:
    run, pool = root / "run", root / "pool.jsonl"
    run.mkdir()
    pool.write_text("".join(json.dumps({"question": str(i), "answer": "0"}) + "\n" for i in range(10)))
    pool_manifest = pool.with_name(pool.name + ".manifest.json")
    write_json(pool_manifest, {"dataset": "fixture", "seed": 104729})
    write_json(
        run / "prompts.json",
        {"train": [{"question": str(i), "answer": "0"} for i in range(10)], "val": []},
    )
    write_json(
        run / "run_config.json",
        {
            "pool_sha256": sha256_file(pool),
            "pool_manifest_sha256": sha256_file(pool_manifest),
            "behavior_k": 4,
            "dataset": "fixture",
            "seed": 0,
        },
    )
    with (run / "rollouts_behavior_train.jsonl").open("w") as stream:
        for prompt_idx in range(10):
            values = [0, 1, 0, 1] if prompt_idx < mixed else [1, 1, 1, 1]
            for rollout_idx, reward in enumerate(values):
                stream.write(json.dumps({
                    "prompt_idx": prompt_idx,
                    "rollout_idx": rollout_idx,
                    "reward": reward,
                }) + "\n")
    return run, pool


with tempfile.TemporaryDirectory() as tmp:
    run, pool = fixture(Path(tmp), mixed=1)
    result = qualify(run, pool)
    assert result["passed"] and result["required_mixed_prompts"] == 1
    assert json.loads((run / "pool_qualification.json").read_text()) == result

with tempfile.TemporaryDirectory() as tmp:
    run, pool = fixture(Path(tmp), mixed=0)
    result = qualify(run, pool)
    assert not result["passed"]

with tempfile.TemporaryDirectory() as tmp:
    run, pool = fixture(Path(tmp), mixed=1)
    pool.write_text(pool.read_text() + "\n")
    try:
        qualify(run, pool)
    except ValueError:
        pass
    else:
        raise AssertionError("mutated pool must fail qualification")

with tempfile.TemporaryDirectory() as tmp:
    run, pool = fixture(Path(tmp), mixed=1)
    config_path = run / "run_config.json"
    config = json.loads(config_path.read_text())
    config["seed"] = 104729
    write_json(config_path, config)
    try:
        qualify(run, pool)
    except ValueError:
        pass
    else:
        raise AssertionError("prescreen and qualification seeds must be disjoint")

print("PASS")
