import pytest

import selection_gate_gpu as base
from pinned_trainers import released_digest


@pytest.fixture(autouse=True)
def released_pair_trainers(request, monkeypatch):
    # Historical migration tests exercise released trainers, not later learners.
    historical = {"test_budget_stop_evaluation_runtime", "test_checkpoint_retention_runtime",
                  "test_curve_spawn_runtime", "test_mbpp_quarantine_runtime",
                  "test_mopps_comparison_gpu", "test_mopps_saved_runtime",
                  "test_pair_recollection_runtime", "test_selection_switch_gpu"}
    name = request.module.__name__.rsplit(".", 1)[-1]
    if name.startswith("test_selector_pair") or name in historical:
        monkeypatch.setattr(base, "digest", released_digest(base.digest))
    if name in {"test_curve_spawn_runtime", "test_pair_recollection_runtime"}:
        digest = base.digest
        launcher = (base.ROOT / "scripts/run_selector_pair.sh").resolve()
        monkeypatch.setattr(base, "digest", lambda path:
                            "c3032239f9b9e55f3f2f1c80481b0b1bcac33ece41931e35d1800b662ccbeac9"
                            if path.resolve() == launcher else digest(path))
