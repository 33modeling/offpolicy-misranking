"""Pair paper exports regenerate validated reports before packaging partial data."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import selector_pair_results as results


@pytest.mark.parametrize("missing,development,complete", [
    (["s4-t25"], [], False),
    ([], ["s1-t25"], False),
    ([], [], True),
])
def test_exports_current_partial_report_and_curves(tmp_path, monkeypatch, missing, development, complete):
    root = tmp_path / "run"
    root.mkdir()
    target = tmp_path / "selector-pair-results.txt"
    report = {"missing_states": missing, "missing_development_states": development,
              "rows": [{"state": "s3-t25", "score": 0.5}]}
    curves = "state,step,score\ns3-t25,10,0.5\n"
    calls = []

    def regenerate(command, **kwargs):
        calls.append((command, kwargs))
        (root / "report.json").write_text(json.dumps(report))
        (root / "curves.csv").write_text(curves)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(results.subprocess, "run", regenerate)
    monkeypatch.setattr(sys, "argv", ["selector_pair_results", "--root", str(root), "--out", str(target)])
    results.main()

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == ["bash", "scripts/run_selector_pair.sh", "report"]
    assert kwargs["env"]["PAIR_ROOT"] == str(root.resolve())
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    content = target.read_text()
    assert curves in content
    data = json.loads(content.split("DATA_JSON\n", 1)[1])
    assert data == {**report, "source_root": str(root.resolve()), "complete": complete}
    assert list(tmp_path.glob("*.txt")) == [target]


def test_failed_report_never_exports_stale_data(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    (root / "report.json").write_text(json.dumps({"missing_states": [], "missing_development_states": []}))
    (root / "curves.csv").write_text("stale curves")
    target = tmp_path / "selector-pair-results.txt"
    target.write_text("previous valid export")
    monkeypatch.setattr(results.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=80))
    monkeypatch.setattr(sys, "argv", ["selector_pair_results", "--root", str(root), "--out", str(target)])

    with pytest.raises(SystemExit) as failure:
        results.main()

    assert failure.value.code == 80
    assert target.read_text() == "previous valid export"
    assert list(tmp_path.glob("*.txt")) == [target]
