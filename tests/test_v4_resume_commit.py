"""Per-run Git planning for interrupted v4 runs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from v4_resume_commit import resume_plan, shell_environment


def write_config(root: Path, name: str, commit: str) -> Path:
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    path = run / "run_config.json"
    path.write_text(
        json.dumps({
            "git": commit,
            "model": "/models/example",
            "fresh_k": 32,
            "skip_hybrid": "1",
        }),
        encoding="utf-8",
    )
    return path


with tempfile.TemporaryDirectory() as raw_tmp:
    root = Path(raw_tmp)
    current = "c" * 40
    generation_a = "a" * 40
    plan, skipped = resume_plan(root, 1, current)
    assert skipped == 0
    assert len(plan) == 6
    assert {row["commit"] for row in plan} == {current}

    first = write_config(root, "v4-27b-s0", generation_a)
    plan, skipped = resume_plan(root, 1, current)
    by_name = {row["name"]: row for row in plan}
    assert {row["commit"] for row in plan} == {generation_a}
    assert by_name["v4-27b-s0"]["source"] == "recorded run_config"
    assert by_name["v4-27b-s1"]["source"] == "matrix generation commit"

    run = root / "v4-27b-s0"
    for artifact in (
        "DONE",
        "manifest.json",
        "score_protocol.json",
        "oracle_protocol.json",
        "report.json",
    ):
        (run / artifact).write_text("ok\n", encoding="utf-8")
    plan, skipped = resume_plan(root, 1, current)
    assert skipped == 1
    assert "v4-27b-s0" not in {row["name"] for row in plan}

    exports = shell_environment(first)
    assert "export MODEL_14B=/models/example" in exports
    assert "export FRESH_K=32" in exports
    assert "export OM_SKIP_HYBRID=1" in exports
    assert "unset OM_POOL_FILE" in exports

    write_config(root, "v4-27b-s1", "b" * 40)
    try:
        resume_plan(root, 1, current)
    except ValueError as exc:
        mixed_rejected = "mixed v4 generation commits" in str(exc)
    else:
        mixed_rejected = False
    assert mixed_rejected

print("PASS v4 matrices reject mixed generation commits")


def test_v4_resume_commit_selection() -> None:
    pass
