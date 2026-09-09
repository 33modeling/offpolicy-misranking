"""Shared KEY NUMBERS lines: Qwen-style flat roots, OLMo family roots, and parity with the OLMo status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import point_key_numbers as pkn  # noqa: E402
import rlzero_status  # noqa: E402


def scored_point(root: Path, tag: str, dataset: str, seed: int, drift: int, *, floor: float, parked: bool = False,
                 family_layout: bool = False) -> Path:
    parent = root / f"family-{dataset}-s{seed}" if family_layout else root
    run = parent / f"{tag}-s{seed}-{dataset}-d{drift}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "run_config.json").write_text(json.dumps({"n_train": 400, "dataset": dataset}))
    report = {"k": 40, "noise_floor": floor, "certagrad": {"precision_vs_oracle": 0.175},
              "g00": {"precision": 0.225}, "g01": {"precision": 0.2}, "g10": {"precision": 0.15}, "g11": {"precision": 0.1}}
    div = {"token_kl_beta_pi": 0.000327, "traj_ess_frac_g11": 0.421}
    (run / "report.json").write_text(json.dumps(report))
    (run / "divergence_stats.json").write_text(json.dumps(div))
    if parked:
        stamp = run / "pinned-scoring" / "20260908T233347Z"
        stamp.mkdir(parents=True)
        report["noise_floor"] = 0.2
        (stamp / "report.json").write_text(json.dumps(report))
        (stamp / "divergence_stats.json").write_text(json.dumps(div))
    return run


def test_flat_qwen_root_lists_current_and_pinned_lines(tmp_path):
    root = tmp_path / "qwen35-9b-posttrained-math-code-grpo-v1"
    scored_point(root, "qwen35-9b", "math500", 0, 25, floor=0.125, parked=True)
    scored_point(root, "qwen35-9b", "mbpp", 1, 0, floor=0.216)
    (root / "qwen35-9b-s2-math500-d100").mkdir()          # started, not scored: skipped
    points = pkn.discover_points(root)
    assert [(d, s, k) for d, s, k, _ in points] == [("math500", 0, 25), ("mbpp", 1, 0)]
    lines = []
    for dataset, seed, drift, run in points:
        lines.extend(pkn.point_lines(run, dataset, seed, drift))
    assert lines[0].startswith(" math500 s0 d25 current floor=0.125 gate-LOW fresh=0.175 g00=0.225 g01=0.200 g10=0.150 g11=0.100 KL=0.000327 ESS=0.421")
    assert lines[1].startswith(" math500 s0 d25 pinned  floor=0.200 GATE-OK ")
    assert lines[2].startswith(" mbpp s1 d0 current floor=0.216 GATE-OK ")


def test_cli_prints_header_and_handles_empty_root(tmp_path, capsys):
    root = tmp_path / "empty"
    root.mkdir()
    assert pkn.main(["--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(" KEY NUMBERS per scored point") and "(no scored point yet)" in out
    scored_point(root, "qwen35-9b", "mbpp", 3, 400, floor=0.137)
    assert pkn.main(["--root", str(root), "--no-header"]) == 0
    assert capsys.readouterr().out.strip().startswith("mbpp s3 d400 current floor=0.137 gate-LOW")


def test_olmo_status_uses_the_shared_lines(tmp_path):
    tag = "olmo3-1025-7b-base-rlzero-grpo-h100-v2"
    root = tmp_path / tag
    scored_point(root, tag, "math500", 0, 0, floor=0.175, parked=True, family_layout=True)
    args = argparse.Namespace(root=root, model_tag=tag, drifts=[0, 25])
    families = [rlzero_status.Family("math500", 0)]
    olmo = rlzero_status.key_numbers_lines(args, families)
    shared = pkn.point_lines(root / "family-math500-s0" / f"{tag}-s0-math500-d0", "math500", 0, 0)
    assert olmo == shared and len(olmo) == 2
