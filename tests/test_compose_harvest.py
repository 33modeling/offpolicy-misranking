from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compose_harvest import demote_headings, write_document


def test_demote_headings_caps_at_markdown_level_six() -> None:
    assert demote_headings("# one\n## two\n###### six") == (
        "### one\n#### two\n###### six"
    )


def test_write_document_combines_labeled_sections(tmp_path: Path) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    output = tmp_path / "RESULTS.md"
    first.write_text("# First\nbody\n", encoding="utf-8")
    second.write_text("## Second\nmore\n", encoding="utf-8")
    write_document(output, "Results", [("A", first), ("B", second)])
    text = output.read_text(encoding="utf-8")
    assert text.startswith("# Results\n\n## A\n\n### First")
    assert "## B\n\n#### Second" in text
