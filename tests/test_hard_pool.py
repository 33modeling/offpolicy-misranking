"""Regression tests for hard-pool integrity and provenance (CPU only)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from make_hard_pool import build_pool, manifest_path, validate_pool  # noqa: E402


FAIL = 0


def check(name: str, condition: bool) -> None:
    global FAIL
    print(("  ok  " if condition else "FAIL  ") + name)
    if not condition:
        FAIL += 1


def expect_value_error(name: str, fn) -> None:
    try:
        fn()
    except ValueError:
        check(name, True)
    else:
        check(name, False)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value))


def fixture(root: Path, *, missing: bool = False, duplicate: bool = False):
    run, model, out = root / "run", root / "model", root / "pool.jsonl"
    run.mkdir()
    model.mkdir()
    write_json(model / "config.json", {"model_type": "fixture"})
    write_json(
        run / "prompts.json",
        {
            "train": [
                {"question": "mixed", "answer": "1"},
                {"question": "all wrong", "answer": "2"},
                {"question": "all right", "answer": "3"},
            ],
            "val": [],
        },
    )
    write_json(
        run / "rollouts_behavior_train.shard0.manifest.json",
        {
            "k": 4,
            "explicit_kwargs": {"temperature": 1.0, "top_p": 1.0},
            "eos_token_ids": [2],
            "model_name_or_path": str(model),
            "contract": "fixture",
        },
    )
    rows = []
    rewards = ([0, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1])
    for prompt_idx, prompt_rewards in enumerate(rewards):
        for rollout_idx, reward in enumerate(prompt_rewards):
            if missing and (prompt_idx, rollout_idx) == (0, 3):
                continue
            rows.append(
                {
                    "prompt_idx": prompt_idx,
                    "rollout_idx": rollout_idx,
                    "reward": reward,
                }
            )
    if duplicate:
        rows.append(dict(rows[0]))
    with (run / "rollouts_behavior_train.shard0.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return run, model, out


with tempfile.TemporaryDirectory() as tmp:
    run, model, out = fixture(Path(tmp))
    metadata = build_pool(
        run,
        out,
        0.0,
        1.0,
        model=model,
        dataset="fixture",
        expected_k=4,
        seed=7,
    )
    rows = [json.loads(line) for line in out.open()]
    check("only mixed-reward prompts enter the pool", len(rows) == 1)
    check("selected rows retain prescreen statistics", rows[0]["_prescreen"]["pass_rate"] == 0.5)
    check("sidecar records exact K and seed", metadata["behavior_k"] == 4 and metadata["seed"] == 7)
    check("sidecar binds rollout manifests", bool(metadata["source_rollout_manifest_sha256"]))
    check("valid pool passes provenance check", validate_pool(out, model=model, dataset="fixture") == metadata)

    out.write_text(out.read_text() + "\n")
    expect_value_error(
        "pool byte-level mutation is rejected",
        lambda: validate_pool(out, model=model, dataset="fixture"),
    )


with tempfile.TemporaryDirectory() as tmp:
    run, model, out = fixture(Path(tmp), missing=True)
    expect_value_error(
        "missing rollout index is rejected",
        lambda: build_pool(
            run, out, 0.0, 1.0, model=model, dataset="fixture", expected_k=4, seed=0
        ),
    )
    check("failed build leaves no pool or sidecar", not out.exists() and not manifest_path(out).exists())


with tempfile.TemporaryDirectory() as tmp:
    run, model, out = fixture(Path(tmp), duplicate=True)
    expect_value_error(
        "duplicate rollout key is rejected",
        lambda: build_pool(
            run, out, 0.0, 1.0, model=model, dataset="fixture", expected_k=4, seed=0
        ),
    )


print(("PASS" if FAIL == 0 else "FAIL") + f" (failures {FAIL})")
sys.exit(1 if FAIL else 0)
