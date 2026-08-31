"""Analysis-only protocol upgrades preserve validated experiment artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import recompute_oracle_scores


def make_upgrade_run(root: Path) -> Path:
    run = root / "run"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({"n_train": 2}))
    (run / "prompts.json").write_text(
        json.dumps({"train": [{"question": "a"}, {"question": "b"}], "val": []})
    )
    torch.save(
        {0: torch.ones(8, 3), 1: torch.ones(8, 3)},
        run / "oracle_micro_groups.pt",
    )
    torch.save(torch.ones(8, 3), run / "val_groups.pt")
    return run


def test_analysis_upgrade_accepts_only_complete_hash_bound_rab_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_upgrade_run(tmp_path)
    monkeypatch.setattr(
        recompute_oracle_scores,
        "validate_generation_contract",
        lambda _: {"generation_hash_missing": [], "validated_rows": 48},
    )
    policy_calls = []
    monkeypatch.setattr(
        recompute_oracle_scores,
        "validate_policy_manifest",
        lambda *args, **kwargs: policy_calls.append((args, kwargs)),
    )
    result = recompute_oracle_scores.validate_analysis_upgrade(run, 25, 4)
    assert result["validated_rows"] == 48
    assert policy_calls == [
        ((run / "policy_step_25",), {"target_steps": 25, "world_size": 4})
    ]

    torch.save({0: torch.ones(6, 3), 1: torch.ones(6, 3)}, run / "oracle_micro_groups.pt")
    with pytest.raises(ValueError, match="candidate R/A/B"):
        recompute_oracle_scores.validate_analysis_upgrade(run, 0, 4)


def test_analysis_upgrade_rejects_unbound_generation_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_upgrade_run(tmp_path)
    monkeypatch.setattr(
        recompute_oracle_scores,
        "validate_generation_contract",
        lambda _: {"generation_hash_missing": ["legacy.manifest.json"]},
    )
    with pytest.raises(ValueError, match="hash-bound"):
        recompute_oracle_scores.validate_analysis_upgrade(run, 0, 4)
