import pytest

import selection_gate_gpu as base
from pinned_trainers import released_digest


@pytest.fixture(autouse=True)
def released_pair_trainers(request, monkeypatch):
    # Pair tests exercise the pinned runtime; see tests/pinned_trainers.py.
    if request.module.__name__.startswith("test_selector_pair"):
        monkeypatch.setattr(base, "digest", released_digest(base.digest))
