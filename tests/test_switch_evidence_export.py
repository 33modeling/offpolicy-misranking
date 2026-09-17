import importlib.util
import json
import re
from pathlib import Path

import pytest

import selection_gate as core

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("switch_evidence", ROOT / "scripts/switch_evidence_export.py")
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)

# The anchored pattern the paper's importer accepts (v4/scripts/build_switch_evidence.rb).
ACCEPTED = re.compile(
    r"\Astates/s[0-4]-t\d+/(?:points/view-\d+/"
    r"(?:selection_full|selection_reduced|random_full|random_reduced|gated)|mopps|random_online)/"
    r"(?:result\.json|decision\.json|cost\.jsonl|policy/budget_stop\.json)\Z")


def parse(text):
    """The importer's own split: a marker line, then everything up to the next marker."""
    parts = re.split(r"^===== (.+) =====\s*$", text, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


@pytest.fixture
def root(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"schema": "s", "budget_gpu_seconds": 29040, "test_seeds": [3, 4]})
    point = tmp_path / "states/s3-t50/points/view-50"
    for arm in ("selection_full", "random_full", "gated"):
        core.atomic_json(point / arm / "result.json", {"rewards": {"1": .5}, "arm": arm})
        core.atomic_json(point / arm / "decision.json", {"action": "select"})
        core.atomic_json(point / arm / "policy/budget_stop.json", {"completed_steps": 150})
        (point / arm / "cost.jsonl").write_text('{"state": "finished", "event_id": "a"}\n')
    # A state-level arm, which does not live under the point.
    core.atomic_json(tmp_path / "states/s3-t50/mopps/result.json", {"rewards": {"1": .25}})
    # Set aside by a waiver or a reset: never evidence.
    core.atomic_json(point / "gated/discards/old/result.json", {"rewards": {"1": .9}})
    core.atomic_json(point / "gated/waivers/one.json", {"reason": "fault"})
    # Not in the accepted set.
    core.atomic_json(point / "selection_full/execution.json", {"action": "select"})
    return tmp_path


def test_only_the_records_the_importer_accepts_are_written(root):
    blocks = parse(export.report(root))
    assert set(blocks) - {"switch.json"} == {name for name in blocks if ACCEPTED.match(name)} - {"switch.json"}
    assert all(ACCEPTED.match(name) for name in blocks if name != "switch.json")
    assert "states/s3-t50/mopps/result.json" in blocks
    assert not any("discards" in name or "waivers" in name or "execution" in name for name in blocks)
    assert len(blocks) == 3*4 + 1 + 1


def test_every_block_parses_after_the_split_including_the_last(root):
    """A trailer line after the final marker becomes that block's body and breaks JSON."""
    blocks = parse(export.report(root))
    for name, body in blocks.items():
        if name.endswith(".jsonl"):
            assert [json.loads(line) for line in body.splitlines() if line.strip()]
        else:
            assert isinstance(json.loads(body), dict)


def test_header_carries_the_provenance_the_importer_reads(root):
    text = export.report(root)
    assert re.search(r"^UTC: (\S+)", text, flags=re.M) and re.search(r"^COMMIT: (\S+)", text, flags=re.M)
    assert text.splitlines()[0] == "SELECTION SWITCH EVIDENCE"


def test_a_directory_without_a_root_or_without_branches_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        export.report(tmp_path)
    core.atomic_json(tmp_path / "switch.json", {"schema": "s"})
    with pytest.raises(SystemExit):
        export.report(tmp_path)
