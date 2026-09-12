"""CPU contracts for the public-benchmark evaluation of E5 policies."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_evidence_downstream import source_point
from test_gate_arm import _train

import benchmark_eval as be
import evidence_downstream as ed

ROOT = Path(__file__).resolve().parents[1]


def test_conversions_extract_gold_answers():
    assert be.gsm8k_answer("Natalia sold 48 ... #### 1,234") == "1234"
    assert be.gsm8k_answer("no marker") is None
    assert be.boxed("so $x = \\boxed{\\frac{1}{2}}$.") == "\\frac{1}{2}"
    assert be.boxed("unbalanced \\boxed{x") is None
    assert be.convert("aime24", {"problem": "p", "answer": 42, "id": "2024-I-1"})["answer"] == "42"
    assert be.convert("gsm8k", {"question": "q", "answer": "... #### 7"})["answer"] == "7"
    assert be.convert("math_rest", {"problem": "p", "solution": "\\boxed{3}"})["answer"] == "3"
    assert be.convert("amc23", {"question": "", "answer": "1"}) is None
    assert be.normalize_question("What  is\n1+1?") == "What is 1+1?"


def _datasets(tmp_path: Path, n_gsm: int = 30) -> Path:
    d = tmp_path / "benchmarks"
    d.mkdir()
    sets = {"aime24": [{"question": f"aime {i}", "answer": str(i)} for i in range(6)],
            "gsm8k": [{"question": f"gsm {i}", "answer": str(i)} for i in range(n_gsm)]}
    for name, rows in sets.items():
        sha = be.write_set(rows, d / f"{name}.jsonl")
        (d / f"{name}.manifest.json").write_text(json.dumps({"dataset": name, "source_repository": be.SOURCES[name]["repo"],
                                                             "source_revision": "abc", "split": "test", "rows": len(rows), "sha256": sha}))
    return d


def test_prepare_freezes_subsamples_and_rejects_overlap(tmp_path):
    run, evaluation = source_point(tmp_path, drift=400)
    out = tmp_path / "out"
    ed.prepare(run, out, evaluation, 100, 8, ["random", "passrate_beta"])
    datasets = _datasets(tmp_path)
    frozen = be.prepare(out, datasets, ["aime24", "gsm8k"], count=8, eval_k=4)
    assert frozen["sets"]["aime24"]["prompts"] == 6 and frozen["sets"]["gsm8k"]["prompts"] == 8
    assert frozen["eval_k"] == 4 and frozen["eval_seed"] == ed.read(out / "experiment.json")["eval_seed"] + 100_003
    again = be.prepare(out, datasets, ["aime24", "gsm8k"], count=8, eval_k=4)
    assert again == frozen
    with pytest.raises(ValueError, match="contract changed"):
        be.prepare(out, datasets, ["aime24", "gsm8k"], count=8, eval_k=8)
    # a set that repeats a candidate question is refused
    source = ed.read(run / "prompts.json")
    rows = [{"question": source["train"][0]["question"], "answer": "1"}] + [{"question": f"amc {i}", "answer": "1"} for i in range(4)]
    sha = be.write_set(rows, datasets / "amc23.jsonl")
    (datasets / "amc23.manifest.json").write_text(json.dumps({"dataset": "amc23", "source_repository": "x", "source_revision": "r", "split": "test", "rows": 5, "sha256": sha}))
    with pytest.raises(ValueError, match="overlaps"):
        be.prepare(out, datasets, ["amc23"], count=8, eval_k=4)


def _fake_shards(out: Path, arm: str, rate: float) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for shard in range(4):
        result = subprocess.run([sys.executable, str(ROOT / "tests/fake_bench_eval.py"), "evaluate", "--out", str(out),
                                 "--arm", arm, "--shard", str(shard)], capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stdout + result.stderr


def test_bindings_results_and_status(tmp_path, monkeypatch):
    import gate_decision as gd
    rule = tmp_path / "rule.json"
    rule.write_text(json.dumps(gd.default_rule(pilot_size=40, seed=1)))
    monkeypatch.setenv("E5_GATE_RULE", str(rule))
    run, evaluation = source_point(tmp_path, drift=400)
    out = tmp_path / "out"
    ed.prepare(run, out, evaluation, 100, 8, ["random", "passrate_beta", "gate_passrate"])
    for arm in ("random", "passrate_beta"):
        _train(out / "subsets" / f"train-{arm}.args", rho=0.7)
    _train(out / "subsets" / "train-gate_passrate-pilot.args", rho=0.7)
    ed.gate_decide(out, "gate_passrate")
    _train(out / "subsets" / "train-gate_passrate.args", rho=0.7)
    be.prepare(out, _datasets(tmp_path), ["aime24", "gsm8k"], count=8, eval_k=4)
    binding, policy, indices = be.binding_for(out, "before", "aime24", 0)
    assert policy == run / "policy_step_400" and binding["set"] == "aime24" and len(indices) == 1
    binding, policy, _ = be.binding_for(out, "gate_passrate", "gsm8k", 3)
    assert policy == out / "gate_passrate" / "policy" and binding["adapter_sha256"] == ed.digest(policy / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="incomplete"):
        be.summarize(out)
    for arm in ("before", "random", "passrate_beta", "gate_passrate"):
        _fake_shards(out, arm, 0.4)
    report = be.summarize(out)
    assert report["complete"]
    rows = {(r["selector"], r["benchmark"]): r for r in report["rows"]}
    assert set(b for _, b in rows) == {"aime24", "gsm8k", "macro"}
    gate = rows[("gate_passrate", "gsm8k")]
    assert gate["prompts"] == 8 and gate["eval_k"] == 4 and gate["vs_random"] is not None and gate["random_lower"] <= gate["vs_random"] <= gate["random_upper"]
    assert rows[("before", "aime24")]["vs_before"] is None and rows[("random", "aime24")]["vs_random"] is None
    assert rows[("random", "macro")]["reward"] == pytest.approx((rows[("random", "aime24")]["reward"] + rows[("random", "gsm8k")]["reward"]) / 2)
    assert rows[("random", "gsm8k")]["gpu_seconds"] == pytest.approx(12.5 * 8)
    assert (out / "benchmark_results.csv").is_file()
    text = be.status(out)
    assert "gate_passrate" in text and "aime24 done" in text
    # a changed prompt file is refused
    prompts = out / "benchmarks" / "aime24.json"
    original = prompts.read_text()
    prompts.write_text(original.replace("aime 0", "aime zero"))
    with pytest.raises(ValueError, match="changed"):
        be.binding_for(out, "random", "aime24", 0)
    prompts.write_text(original)


def test_cli_and_launcher_syntax(tmp_path):
    run, evaluation = source_point(tmp_path, drift=0)
    out = tmp_path / "out"
    ed.prepare(run, out, evaluation, 100, 8, ["random"])
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    cmd = [sys.executable, str(ROOT / "src/benchmark_eval.py")]
    prepared = subprocess.run(cmd + ["prepare", "--out", str(out), "--datasets-dir", str(_datasets(tmp_path)),
                                     "--sets", "aime24", "gsm8k", "--count", "8", "--eval-k", "2"], capture_output=True, text=True, env=env)
    assert prepared.returncode == 0 and "aime24=6" in prepared.stdout, prepared.stdout + prepared.stderr
    status = subprocess.run(cmd + ["status", "--out", str(out)], capture_output=True, text=True, env=env)
    assert status.returncode == 0 and "before" in status.stdout and "0/4" in status.stdout
    for name in ("scripts/run_e5_bench.sh", "scripts/fetch_benchmarks.sh"):
        subprocess.run(["bash", "-n", str(ROOT / name)], check=True)
    missing = subprocess.run(cmd + ["prepare", "--out", str(out), "--datasets-dir", str(tmp_path / "nowhere")], capture_output=True, text=True, env=env)
    assert missing.returncode == 2 and "fetch_benchmarks" in missing.stderr
