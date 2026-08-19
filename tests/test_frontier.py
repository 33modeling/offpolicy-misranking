"""CPU-only regression tests for frontier accounting and aggregation."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# This test exercises artifact accounting only; the local CPU audit environment
# need not install the large torch dependency used by Run simulation.
if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")
from frontier import aggregate_divergence, to_md  # noqa: E402


def row(policy: str, rollouts: int, precision: float) -> dict:
    return {
        "run": "v2-fixture-s0",
        "policy": policy,
        "fresh_groups": rollouts // 4,
        "fresh_rollouts": rollouts,
        "budget_frac": rollouts / 100,
        "precision": precision,
        "precision_sd": 0.0,
        "regret": 0.0,
    }


rows = [
    row("stale_g00", 0, 0.6),
    row("audit_bnd_p10_m2", 8, 0.7),
    row("fresh_m4", 40, 1.0),
]
md = to_md(rows, [])
assert "| v2-fixture-s0 | 0 | 0.600" in md
assert "| v2-fixture-s0 | 8 | 0.600" in md
assert "0.700 | — | — |" in md
assert "| v2-fixture-s0 | 40 | 0.600" in md and "1.000" in md

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    docs = [
        {
            "token_kl_beta_pi": 1.0,
            "clipfrac_g11": 0.2,
            "traj_ess_frac_g11": 0.8,
            "rollouts": 1,
            "tokens": 10,
            "traj_logw_logsumexp": math.log(2.0),
            "traj_logw2_logsumexp": math.log(4.0),
        },
        {
            "token_kl_beta_pi": 3.0,
            "clipfrac_g11": 0.6,
            "traj_ess_frac_g11": 0.4,
            "rollouts": 3,
            "tokens": 30,
            "traj_logw_logsumexp": math.log(6.0),
            "traj_logw2_logsumexp": math.log(12.0),
        },
    ]
    paths = []
    for i, doc in enumerate(docs):
        path = root / f"divergence_stats.shard{i}.json"
        path.write_text(json.dumps(doc))
        paths.append(path)
    agg = aggregate_divergence(paths)
    assert agg["token_kl_beta_pi"] == 2.5
    assert abs(agg["clipfrac_g11"] - 0.5) < 1e-12
    assert abs(agg["traj_ess_frac_g11"] - 1.0) < 1e-12

print("PASS frontier budget matching and shard aggregation")
