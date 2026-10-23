"""Old incidents, failures, and correction events must survive future ledger exports."""
import copy

import pytest

from scripts.record_paper_bugs import preserve


def original():
    return [{"id": "GEMMA-001", "title": "initial 1% report", "cause": "unverified",
             "events": [{"type": "reported", "text": "raw remote evidence unavailable"}]}]


def test_append_fix_and_new_incident_preserves_the_original_report():
    previous = original()
    current = copy.deepcopy(previous)
    current[0]["events"].append({"type": "fixed", "text": "bounded test passed; remote remains unverified"})
    current.append({"id": "GEMMA-002", "title": "next incident", "events": []})
    preserve(previous, current)
    assert previous == original()


@pytest.mark.parametrize("change", ["delete", "rewrite_title", "rewrite_cause", "erase_event", "rewrite_event"])
def test_published_incidents_and_events_cannot_be_deleted_or_overwritten(change):
    previous = original()
    current = copy.deepcopy(previous)
    if change == "delete":
        current.clear()
    elif change == "rewrite_title":
        current[0]["title"] = "all fixed"
    elif change == "rewrite_cause":
        current[0]["cause"] = "pretend known"
    elif change == "erase_event":
        current[0]["events"].clear()
    else:
        current[0]["events"][0]["text"] = "hide original failure"
    with pytest.raises(ValueError, match="cannot"):
        preserve(previous, current)
