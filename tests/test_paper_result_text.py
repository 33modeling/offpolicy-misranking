"""Paper exports remain a single bounded, lossless TXT artifact."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paper_result_text import write_export


def test_default_output_overwrites_one_stable_txt(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    first = write_export("rloo", {"complete": False, "rows": [1]}, "arm,score\nbefore,0.5")
    second = write_export("rloo", {"complete": False, "rows": [1, 2]}, "arm,score\nafter,0.7")

    assert first == second == tmp_path / "rloo-results.txt"
    assert list(tmp_path.iterdir()) == [second]
    content = second.read_text()
    assert "after,0.7" in content and "before,0.5" not in content
    assert "Missing values are not zero" in content
    assert json.loads(content.split("DATA_JSON\n", 1)[1]) == {"complete": False, "rows": [1, 2]}


def test_size_limit_preserves_previous_output_without_truncation(tmp_path):
    target = tmp_path / "pair-results.txt"
    write_export("pair", {"rows": [1]}, target=target)
    previous = target.read_bytes()

    with pytest.raises(ValueError, match="1.9 MB TXT limit"):
        write_export("pair", {"rows": ["x" * 1_900_000]}, target=target)

    assert target.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [target]


def test_oversize_first_export_does_not_leave_partial_file(tmp_path):
    target = tmp_path / "pair-results.txt"
    with pytest.raises(ValueError, match="no truncated file"):
        write_export("pair", {}, "x" * 1_900_000, target=target)
    assert list(tmp_path.iterdir()) == []


def test_unicode_size_limit_is_in_bytes(tmp_path):
    with pytest.raises(ValueError, match="1.9 MB TXT limit"):
        write_export("pair", {}, "\u20ac" * 650_000, target=tmp_path / "pair-results.txt")
    assert list(tmp_path.iterdir()) == []


def test_nonfinite_values_do_not_replace_previous_data(tmp_path):
    target = tmp_path / "mbpp-results.txt"
    target.write_text("previous data")
    with pytest.raises(ValueError):
        write_export("mbpp", {"cost": float("inf")}, target=target)
    assert target.read_text() == "previous data"


def test_non_txt_target_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must end in .txt"):
        write_export("pair", {}, target=tmp_path / "results.json")
    assert list(tmp_path.iterdir()) == []
