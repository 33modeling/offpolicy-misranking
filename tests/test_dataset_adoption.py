"""Separately uploaded official datasets are adopted without Hub metadata."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import load_prompts
from qualify_domain_data import (
    SPECS,
    _adopt_official_upload,
    _content_sha256,
)


def test_flat_official_upload_replaces_a_broken_standard_shadow(
    tmp_path: Path, monkeypatch
) -> None:
    rows = [
        {"problem": f"problem {index}", "answer": str(index)} for index in range(3)
    ]
    canonical = [
        {"problem": row["problem"], "answer": row["answer"]} for row in rows
    ]
    spec = {
        **SPECS["math500"],
        "rows": len(rows),
        "content_sha256": _content_sha256(canonical),
    }
    monkeypatch.setitem(SPECS, "math500", spec)
    monkeypatch.delenv("MATH500_DIR", raising=False)
    monkeypatch.setenv("OM_DATA", str(tmp_path / "missing-om-data"))

    broken = tmp_path / "math500/math500_test.jsonl"
    broken.parent.mkdir()
    broken.write_text("", encoding="utf-8")
    uploaded = tmp_path / "unrelated-name/deep/tree/test.jsonl"
    uploaded.parent.mkdir(parents=True)
    uploaded.write_text(
        "".join(
            json.dumps(
                {
                    "question": row["problem"],
                    "solution": rf"work \boxed{{{row['answer']}}}",
                    "extra": "allowed",
                }
            )
            + "\n"
            for row in reversed(rows)
        ),
        encoding="utf-8",
    )

    target, adopted_rows = _adopt_official_upload("math500", tmp_path)
    first_manifest = (target.parent / "dataset_manifest.json").read_bytes()
    target_again, _ = _adopt_official_upload("math500", tmp_path)

    assert target == broken
    assert target_again == target
    assert len(adopted_rows) == 3
    assert _content_sha256(adopted_rows) == spec["content_sha256"]
    assert (target.parent / "dataset_manifest.json").read_bytes() == first_manifest


def test_math_loader_accepts_a_flat_datasets_dir_file(
    tmp_path: Path, monkeypatch
) -> None:
    rows = [
        {"problem": f"problem {index}", "answer": str(index)} for index in range(3)
    ]
    (tmp_path / "math500_test.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    monkeypatch.delenv("MATH500_DIR", raising=False)
    monkeypatch.setenv("OM_DATA", str(tmp_path / "missing-om-data"))
    monkeypatch.setenv("DATASETS_DIR", str(tmp_path))

    split = load_prompts("math500", 1, 1, seed=0)

    assert len(split["train"]) == 1
    assert len(split["val"]) == 1
