"""CPU contracts for the gain-versus-reliability analysis and its synthetic calibration."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_additional_experiments import _write_run

import gain_law_simulation as sim
import gain_vs_reliability as gl

ROOT = Path(__file__).resolve().parents[1]


def test_topk_constant_matches_known_order_statistics():
    # E[max of 2 standard normals] = 1/sqrt(pi); E[mean of top 1 of 1] = 0
    assert gl.topk_constant(2, 1, draws=40000) == pytest.approx(1 / math.sqrt(math.pi), abs=0.02)
    assert gl.topk_constant(1, 1, draws=2000) == pytest.approx(0.0, abs=0.05)
    assert 1.7 < gl.topk_constant(400, 40, draws=4000) < 1.85
    with pytest.raises(ValueError):
        gl.topk_constant(4, 5)


def test_cross_half_gain_is_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    theta = rng.standard_normal(400)
    a, b = theta + rng.standard_normal(400), theta + rng.standard_normal(400)
    gain = gl.cross_half_gain(a, b, 40)
    assert gain == pytest.approx(gl.cross_half_gain(b, a, 40))
    rho = float(np.corrcoef(a, b)[0, 1])
    assert abs(gain - rho * gl.topk_constant(400, 40, draws=4000)) < 0.25
    with pytest.raises(ValueError, match="zero variance"):
        gl.cross_half_gain(np.zeros(10), a[:10], 2)


def test_point_rows_read_fresh_and_difficulty_halves(tmp_path):
    root = tmp_path / "matrix"
    for seed in (0, 1):
        run = _write_run(root / "family-math500-s0", n=40, seed=seed, drift=25 * (seed + 1))
        (run / "DONE").write_text("ok")
    rows = gl.collect(root, 0.1)
    fresh = [r for r in rows if r["signal"] == "fresh"]
    assert len(fresh) == 2 and all(r["rho_half"] > 0.9 for r in fresh)
    assert all(abs(r["ratio"] - 1) < 0.35 for r in fresh)  # near the Gaussian prediction at high reliability
    assert {r["signal"] for r in rows} == {"fresh", "difficulty"}  # stale halves absent -> skipped
    summary = gl.pooled(rows)
    assert summary["fresh"]["points"] == 2 and summary["fresh"]["slope_through_origin"] > 0
    gl.write_outputs(rows, tmp_path / "out" / "law")
    assert (tmp_path / "out" / "law.csv").is_file() and "signal dataset" in (tmp_path / "out" / "law.dat").read_text()


def test_gaussian_family_follows_sqrt_rho_and_bernoulli_departs():
    gauss = sim.simulate("gaussian", 0.5, 400, 40, 40, seed=1)
    assert gauss["latent_ratio"] == pytest.approx(math.sqrt(gauss["rho_measured"]), abs=0.06)
    assert gauss["cross_gain"] == pytest.approx(gauss["predicted"], abs=0.12)
    bern = sim.simulate("bernoulli", 0.3, 400, 40, 40, seed=2)
    assert 0 < bern["rho_measured"] < 1 and bern["reps"] == 40
    student = sim.simulate("student3", 0.5, 400, 40, 30, seed=3)
    assert math.isfinite(student["latent_ratio"])


def test_cli_and_script_syntax(tmp_path):
    root = tmp_path / "matrix"
    run = _write_run(root / "family-mbpp-s0", n=40, seed=0, drift=0, dataset="mbpp")
    (run / "DONE").write_text("ok")
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
    result = subprocess.run([sys.executable, str(ROOT / "src/gain_vs_reliability.py"), "--root", str(root), "--out", str(tmp_path / "law")],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0 and "pooled by signal" in result.stdout, result.stdout + result.stderr
    result = subprocess.run([sys.executable, str(ROOT / "src/gain_law_simulation.py"), "--n", "60", "--reps", "5", "--out", str(tmp_path / "syn")],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0 and (tmp_path / "syn.dat").is_file(), result.stdout + result.stderr
    subprocess.run(["bash", "-n", str(ROOT / "scripts/run_gain_law.sh")], check=True)
