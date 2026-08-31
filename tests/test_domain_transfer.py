"""Non-math loader contracts and pinned transfer-matrix configuration."""

from __future__ import annotations

import hashlib
import json
import os
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import _dedupe_items, build_user_msg, load_prompts, reward
from qualify_domain_data import qualify

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    dataset_root = root / "arc-challenge"
    dataset_root.mkdir()
    rows = [
        {
            "id": f"q{i}",
            "question": f"Science question {i}?",
            "choices": {"label": ["A", "B", "C"], "text": ["one", "two", "three"]},
            "answerKey": "B",
            "_split": "train",
        }
        for i in range(6)
    ]
    data_file = dataset_root / "arc_challenge.jsonl"
    data_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    revision = "210d026faf9955653af8916fad021475a3f00453"
    (dataset_root / "dataset_manifest.json").write_text(json.dumps({
        "dataset": "arc-challenge",
        "source_repository": "allenai/ai2_arc",
        "source_revision": revision,
        "artifact": "arc_challenge.jsonl",
        "sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
        "rows": len(rows),
    }))

    old = os.environ.get("ARC_CHALLENGE_DIR")
    os.environ["ARC_CHALLENGE_DIR"] = str(dataset_root)
    try:
        split = load_prompts("arc-challenge", 3, 2, seed=7)
    finally:
        if old is None:
            os.environ.pop("ARC_CHALLENGE_DIR", None)
        else:
            os.environ["ARC_CHALLENGE_DIR"] = old
    assert len(split["train"]) == 3 and len(split["val"]) == 2
    assert all(row["answer"] == "ARC:B" for part in split.values() for row in part)
    assert "Choices:" in split["train"][0]["question"]
    assert build_user_msg(split["train"][0]["question"]) == split["train"][0]["question"]
    assert reward("Reasoning\n#### B", "ARC:B") == 1.0
    assert reward("Reasoning\n#### A", "ARC:B") == 0.0
    assert reward("The answer is B", "ARC:B") == 0.0

    report = qualify("arc-challenge", root, 3, 2, [0, 1, 2])
    assert report["status"] == "qualified"
    assert report["snapshot_rows"] == 6
    assert report["prompt_split"]["seed"] == 0
    assert report["experiment_seeds"] == [0, 1, 2]
    assert report["reward_runtime"]["kind"] == "exact-structured-match"

deduped = _dedupe_items([
    {"question": "same", "answer": "one"},
    {"question": "same", "answer": "one"},
    {"question": "other", "answer": "two"},
], "fixture")
assert len(deduped) == 2
os.environ["OM_MATH_VERIFIER"] = "math_verify"
assert reward("Reasoning\n#### 0.5", r"\frac{1}{2}") == 1.0
assert reward("Reasoning\n#### 0.6", r"\frac{1}{2}") == 0.0
os.environ.pop("OM_MATH_VERIFIER")
try:
    _dedupe_items([
        {"question": "same", "answer": "one"},
        {"question": "same", "answer": "different"},
    ], "fixture")
except ValueError as exc:
    assert "conflicting" in str(exc)
else:
    raise AssertionError("conflicting duplicate prompt was accepted")

config = json.loads((ROOT / "configs" / "domain_transfer.json").read_text())
models = {row["key"]: row for row in config["models"]}
assert set(models) == {"mistral7b", "olmo2-7b"}
assert all(len(row["revision"]) == 40 for row in models.values())
assert config["experiment"]["datasets"] == ["gsm8k", "mbpp", "kk", "arc-challenge"]
assert config["experiment"]["fresh_k"] == 32
assert config["experiment"]["fresh_k"] % config["experiment"]["micro_group"] == 0
assert config["experiment"]["grpo"]["world_size"] == 4
assert config["experiment"]["grpo"]["group_size"] == 8
assert config["experiment"]["grpo"]["clip_epsilon"] == 0.2

print("PASS")
