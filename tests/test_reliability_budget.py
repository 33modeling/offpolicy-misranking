"""Reliability-versus-budget sizing on synthetic micro-group artifacts (CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import reliability_budget as rb  # noqa: E402
from select_rules import overlap_under_independent_ties  # noqa: E402

N, K = 400, 40          # registered MATH-500 design so the frozen lookup applies
GROUPS, GSIZE, DIM, VAL = 8, 4, 48, 100


def synthetic_artifacts(noise: float, seed: int = 0, groups: int = GROUPS, val_noise: float = 0.5):
    """Latent utility t_i along one direction u; every micro-group adds isotropic noise."""
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(DIM, generator=generator)
    direction = direction / direction.norm()
    latent = torch.randn(N, generator=generator)
    base = latent[:, None] * direction[None, :] + 0.3 * torch.randn(N, DIM, generator=generator)
    stack = base[:, None, :] + noise * torch.randn(N, groups, DIM, generator=generator)
    val_groups = direction[None, :] + val_noise * torch.randn(VAL, DIM, generator=generator)
    return stack, val_groups


def write_run(tmp_path: Path, stack: torch.Tensor, val_groups: torch.Tensor, *, parked: bool = False,
              behavior: bool = False, fresh_k: int = GROUPS * GSIZE) -> Path:
    run = tmp_path / "point"
    run.mkdir(parents=True, exist_ok=True)
    micro = {idx: stack[idx].clone() for idx in range(stack.shape[0])}
    target = run / "pinned-scoring" / "20260908T233347Z" if parked else run
    target.mkdir(parents=True, exist_ok=True)
    torch.save(micro, target / "oracle_micro_groups.pt")
    torch.save(val_groups, target / "val_groups.pt")
    (run / "run_config.json").write_text(json.dumps(
        {"dataset": "math500", "fresh_k": fresh_k, "val_k": 8, "micro_group": GSIZE, "seed": 0, "drift": 0}
    ))
    if behavior:
        rows = []
        for idx in range(stack.shape[0]):
            rewards = [1.0] * 8 if idx % 10 == 0 else ([0.0] * 8 if idx % 10 == 1 else [1.0, 0.0] * 4)
            rows.extend(json.dumps({"prompt_idx": idx, "rollout_idx": j, "reward": r}) for j, r in enumerate(rewards))
        (run / "rollouts_behavior_train.jsonl").write_text("\n".join(rows) + "\n")
    return run


def test_floor_rises_with_half_size_and_falls_with_noise():
    stack, val = synthetic_artifacts(noise=2.0)
    small = rb.floor_statistic(stack, val, k=K, groups_per_half=1, val_prompts_per_half=25,
                               mode="both", reps=6, pairs=3, seed=1)
    large = rb.floor_statistic(stack, val, k=K, groups_per_half=4, val_prompts_per_half=25,
                               mode="both", reps=6, pairs=3, seed=1)
    assert large.mean > small.mean + 0.05
    assert large.correlation > small.correlation
    noisy_stack, noisy_val = synthetic_artifacts(noise=60.0)
    noisy = rb.floor_statistic(noisy_stack, noisy_val, k=K, groups_per_half=4, val_prompts_per_half=25,
                               mode="both", reps=6, pairs=3, seed=1)
    assert abs(noisy.mean - K / N) < 0.06          # chance = 0.10
    clean_stack, clean_val = synthetic_artifacts(noise=0.01, val_noise=0.01)
    clean = rb.floor_statistic(clean_stack, clean_val, k=K, groups_per_half=1, val_prompts_per_half=25,
                               mode="both", reps=3, pairs=2, seed=1)
    assert clean.mean > 0.9


def test_single_axis_modes_bound_the_registered_construction():
    stack, val = synthetic_artifacts(noise=2.0, val_noise=1.5)
    common = dict(k=K, groups_per_half=2, val_prompts_per_half=25, reps=6, pairs=3, seed=3)
    both = rb.floor_statistic(stack, val, mode="both", **common)
    candidate = rb.floor_statistic(stack, val, mode="candidate", **common)
    validation = rb.floor_statistic(stack, val, mode="validation", **common)
    assert candidate.correlation >= both.correlation - 0.02
    assert validation.correlation >= both.correlation - 0.02


def test_spearman_brown_round_trip_and_lookup_monotone():
    for r_unit in (0.05, 0.2, 0.6):
        for factor in (2, 4, 8):
            assert rb.unit_correlation(rb.spearman_brown(r_unit, factor), factor) == pytest.approx(r_unit, abs=1e-9)
    assert rb.spearman_brown(0.0, 8) == 0.0
    overlaps = [rb.overlap_from_correlation(rho, N, K) for rho in (0.0, 0.1, 0.3, 0.6, 0.9)]
    assert overlaps == sorted(overlaps)
    assert overlaps[0] == pytest.approx(K / N, abs=1e-6)
    assert rb.correlation_from_overlap(rb.overlap_from_correlation(0.3, N, K), N, K) == pytest.approx(0.3, abs=0.02)
    # unregistered design falls back to simulation and stays monotone
    fallback = [rb.overlap_from_correlation(rho, 120, 12) for rho in (0.0, 0.5, 0.95)]
    assert fallback[0] < fallback[1] < fallback[2]


def test_analyze_run_predicts_within_tolerance_and_reports_budget(tmp_path):
    stack, val = synthetic_artifacts(noise=3.0, val_noise=1.0)
    run = write_run(tmp_path, stack, val, behavior=True)
    artifacts = rb.load_run(run)
    assert artifacts.scoring == "current"
    assert artifacts.group_size == GSIZE
    readout = rb.analyze_run(artifacts, reps=8, pairs=3, target=0.20, seed=5)
    assert readout.n == N and readout.k == K and readout.groups == GROUPS
    assert readout.registered is not None
    assert (2, 25) in readout.curves["both"]
    assert readout.coupling is not None and readout.coupling > 0
    assert readout.budget_table[(8, 25)] == pytest.approx(readout.registered.mean, abs=0.06)
    # more responses per half can only raise the prediction
    column = [readout.budget_table[(r, 25)] for r in rb.CANDIDATE_HALF_RESPONSES]
    assert column == sorted(column)
    assert readout.self_check is not None
    predicted, observed = readout.self_check
    assert abs(predicted - observed) < 0.12
    assert readout.strata is not None
    assert readout.strata["mixed"] == 320 and readout.strata["all_right"] == 40 and readout.strata["all_wrong"] == 40
    assert readout.strata["mixed_floor"] is not None
    text = rb.render_report([readout], 0.20, 8, 3)
    assert "KEY point:" in text and "registered geometry" in text and "limiting side now" in text


def test_pure_noise_reports_no_reachable_budget(tmp_path):
    stack, val = synthetic_artifacts(noise=80.0, val_noise=80.0)
    run = write_run(tmp_path, stack, val, parked=True)
    artifacts = rb.load_run(run)
    assert artifacts.scoring == "pinned"
    readout = rb.analyze_run(artifacts, reps=6, pairs=3, target=0.20, seed=9)
    assert readout.registered.mean < 0.16
    assert readout.needed_candidate is None or readout.needed_candidate >= 128
    text = rb.render_report([readout], 0.20, 6, 3)
    assert "KEY point:" in text


def test_cli_writes_report(tmp_path, capsys):
    stack, val = synthetic_artifacts(noise=3.0)
    run = write_run(tmp_path, stack, val)
    out = tmp_path / "exports" / "reliability.txt"
    code = rb.main([str(run), "--out", str(out), "--reps", "4", "--pairs", "2", "--label", "math500/s0 d0"])
    assert code == 0
    assert out.is_file()
    body = out.read_text()
    assert "KEY math500/s0 d0:" in body and "KEY math500:" in body
    assert "reliability-budget" in capsys.readouterr().out


def test_cli_skips_unreadable_runs(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert rb.main([str(empty), "--reps", "2", "--pairs", "1"]) == 1


def test_half_scores_rejects_impossible_geometry():
    stack, val = synthetic_artifacts(noise=1.0)
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        rb.half_scores(stack, val, groups_per_half=5, val_prompts_per_half=25, mode="both", generator=generator)
    with pytest.raises(ValueError):
        rb.half_scores(stack, val, groups_per_half=1, val_prompts_per_half=60, mode="both", generator=generator)
    with pytest.raises(ValueError):
        rb.half_scores(stack, val, groups_per_half=1, val_prompts_per_half=25, mode="nope", generator=generator)


def test_batched_overlap_matches_reference_definition():
    generator = torch.Generator().manual_seed(11)
    a = torch.randn(N, generator=generator)
    b = 0.6 * a + 0.8 * torch.randn(N, generator=generator)
    reference = overlap_under_independent_ties(
        {i: float(v) for i, v in enumerate(a)}, {i: float(v) for i, v in enumerate(b)}, K, seed=3, pairs=20
    ).mean
    batched = rb.topk_overlap_batch(a, b, K, pairs=20, generator=torch.Generator().manual_seed(3))
    assert batched == pytest.approx(reference, abs=1e-9)      # no ties: both definitions are exact
    tied = torch.zeros(N)
    tied_overlap = rb.topk_overlap_batch(tied, tied, K, pairs=200, generator=torch.Generator().manual_seed(5))
    assert abs(tied_overlap - K / N) < 0.03                    # all ties: independent streams give chance
    with pytest.raises(ValueError):
        rb.topk_overlap_batch(a, b, 0, pairs=2, generator=generator)


# ---------------------------------------------------------------------------
# Regression tests inverted from the 2026-09-09 review (six reproduced defects).
# The launcher tests run the real script against a fake src/experiment.py backend
# in a temporary repository copy; no model, no GPU, no registered point.
import fcntl  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

CODE = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch_env(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "configs").mkdir()
    for script in ("run_reliability_budget.sh", "setup_env.sh"):
        shutil.copy2(CODE / "scripts" / script, repo / "scripts" / script)
    shutil.copy2(CODE / "src/model_matrix.py", repo / "src/model_matrix.py")
    shutil.copy2(CODE / "configs/olmo3_rlzero_h100.json", repo / "configs/olmo3_rlzero_h100.json")
    (repo / "src/artifact_contract.py").write_text(
        "import os\ndef cached_rollout_ready(path):\n"
        "    return os.environ.get('FAKE_BLOCK_CHILDREN') != '1'\n"
    )
    (repo / "src/experiment.py").write_text(
        "import json, os, sys, time\nfrom pathlib import Path\n"
        "args = sys.argv[1:]\nrun = Path(args[args.index('--run') + 1])\n"
        "stage = args[args.index('--stage') + 1]\n"
        "record = {'pid': os.getpid(), 'stage': stage, 'args': args}\n"
        "(run / ('child-' + str(os.getpid()) + '.json')).write_text(json.dumps(record))\n"
        "if os.environ.get('FAKE_BLOCK_CHILDREN') == '1':\n    time.sleep(120)\n"
        "if stage == 'val-grads':\n    (run / 'val_groups.pt').write_text('mock')\n"
        "if stage == 'oracle-grads':\n    (run / 'oracle_micro_groups.pt').write_text('mock')\n"
    )
    (repo / "src/reliability_budget.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "Path(sys.argv[sys.argv.index('--out') + 1]).write_text('KEY mock backend completed\\n')\n"
    )
    venv = tmp_path / "venv/bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    model_a, model_b = tmp_path / "model-A", tmp_path / "model-B"
    model_a.mkdir()
    model_b.mkdir()
    tag = "olmo3-1025-7b-base-rlzero-grpo-h100-v2"
    source = tmp_path / "source" / "family-mbpp-s0" / f"{tag}-s0-mbpp-d0"
    source.mkdir(parents=True)
    (source / "prompts.json").write_text(json.dumps({"train": [], "val": []}))
    config = {
        "model": str(model_a), "model_resolved": str(model_a), "dataset": "mbpp",
        "fresh_k": 64, "val_k": 32, "seed": 100, "drift": 0,
        "micro_group": 4, "behavior_k": 8, "max_new_tokens": 2048,
    }
    (source / "run_config.json").write_text(json.dumps(config))
    env = os.environ.copy()
    env.update({
        "OM_REPO": str(repo), "GROUP_VOLUME": str(tmp_path / "absent"),
        "OM_WORK": str(tmp_path / "work"), "VENV_DIR": str(venv.parent),
        "OM_OLMO3_ROOT": str(source.parent.parent), "OM_OLMO3_MODEL_TAG": tag,
        "OM_OLMO3_MODEL_PATH": str(model_a),
        "OM_RLZERO_CONFIG": str(repo / "configs/olmo3_rlzero_h100.json"),
        "OM_LOCAL_LOCK_DIR": str(tmp_path / "node-lock"), "CUDA_VISIBLE_DEVICES": "0,1",
        "TMPDIR": str(tmp_path / "tmp"), "PYTHONPATH": "", "RB_DRY": "0",
    })
    run = Path(env["OM_WORK"]) / "runs/reliability-budget-v1/mbpp-fk64-vk32-s100"
    command = ["bash", str(repo / "scripts/run_reliability_budget.sh"), "mbpp"]
    return env, command, run, model_b


def test_launcher_completes_on_an_idle_node_with_the_fake_backend(launch_env):
    env, command, run, _ = launch_env
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    stages = sorted(json.loads(p.read_text())["stage"] for p in run.glob("child-*.json"))
    assert stages == ["merge-grads", "oracle-grads", "val-grads"]
    assert (run / "RB_DONE").is_file()
    config = json.loads((run / "run_config.json").read_text())
    assert config["fresh_k"] == 64 and config["val_k"] == 32 and config["prompt_format"] == "olmo_rlzero_code"
    assert config["reliability_budget"]["schema"] == "offpolicy-reliability-budget/v1"
    # the same command resumes and reports the matching contract
    again = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert again.returncode == 0 and "effective contract matches" in again.stdout


def test_launcher_refuses_a_node_whose_primary_lock_is_held(launch_env):
    env, command, run, _ = launch_env
    lock = Path(env["OM_LOCAL_LOCK_DIR"]) / "primary.lock"
    lock.parent.mkdir()
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "owns this node's GPUs" in result.stdout + result.stderr
    assert not list(run.glob("child-*.json"))


def test_launcher_refuses_a_second_launcher_for_the_same_run(launch_env):
    env, command, run, _ = launch_env
    run.mkdir(parents=True)
    with (run / ".launcher.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "another launcher already owns" in result.stdout + result.stderr
    assert not list(run.glob("child-*.json"))


def test_launcher_refuses_to_resume_with_a_different_model(launch_env):
    env, command, run, model_b = launch_env
    first = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert first.returncode == 0, first.stdout + first.stderr
    for record in run.glob("child-*.json"):
        record.unlink()
    (run / "RB_DONE").unlink()
    (run / "oracle_micro_groups.pt").unlink()
    env["OM_OLMO3_MODEL_PATH"] = str(model_b)
    second = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert second.returncode != 0
    assert "different effective contract" in second.stdout + second.stderr
    assert "model" in second.stdout
    assert not list(run.glob("child-*.json"))
    assert json.loads((run / "run_config.json").read_text())["model"] != str(model_b)


def test_launcher_sigterm_stops_its_stage_children(launch_env):
    env, command, run, _ = launch_env
    env["FAKE_BLOCK_CHILDREN"] = "1"
    process = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        records = []
        while time.monotonic() < deadline:
            records = list(run.glob("child-*.json"))
            if len(records) == 2:
                break
            time.sleep(0.05)
        assert len(records) == 2, "two rollout shards should be running"
        child_pids = [json.loads(p.read_text())["pid"] for p in records]
        process.terminate()
        process.wait(timeout=15)
        deadline = time.monotonic() + 10
        alive = set(child_pids)
        while alive and time.monotonic() < deadline:
            for pid in list(alive):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    alive.discard(pid)
                else:
                    try:
                        if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
                            alive.discard(pid)
                    except OSError:
                        alive.discard(pid)
            time.sleep(0.1)
        assert not alive, f"stage children survived the launcher: {sorted(alive)}"
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def test_loader_rejects_incomplete_or_inconsistent_points(tmp_path):
    config = {"dataset": "math500", "n_train": 400, "n_val": 100, "fresh_k": 32, "micro_group": 4}
    (tmp_path / "run_config.json").write_text(json.dumps(config))
    micro = {i: torch.ones(8, 3) for i in range(399)}
    micro[0] = torch.ones(4, 3)
    torch.save(micro, tmp_path / "oracle_micro_groups.pt")
    torch.save(torch.ones(100, 3), tmp_path / "val_groups.pt")
    with pytest.raises(ValueError, match="differ in stored geometry"):
        rb.load_run(tmp_path)
    micro[0] = torch.ones(8, 3)
    torch.save(micro, tmp_path / "oracle_micro_groups.pt")
    with pytest.raises(ValueError, match="prompt coverage mismatch"):
        rb.load_run(tmp_path)
    micro[399] = torch.ones(8, 3)
    torch.save(micro, tmp_path / "oracle_micro_groups.pt")
    loaded = rb.load_run(tmp_path)
    assert loaded.prompt_ids == list(range(400))
    config["fresh_k"] = 64
    (tmp_path / "run_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="fresh_k=64"):
        rb.load_run(tmp_path)


def test_stored_scores_are_reproduced_and_a_mismatch_is_flagged(tmp_path):
    stack, val = synthetic_artifacts(noise=3.0)
    run = write_run(tmp_path, stack, val)
    score_a, score_b = rb.registered_half_scores(stack, val)
    stored = {str(i): {"a": float(score_a[i]), "b": float(score_b[i]), "r": 0.0} for i in range(N)}
    (run / "scores_splithalf.json").write_text(json.dumps(stored))
    artifacts = rb.load_run(run)
    assert artifacts.stored_score_max_diff is not None and artifacts.stored_score_max_diff < 1e-5
    stored["7"]["a"] += 0.1
    (run / "scores_splithalf.json").write_text(json.dumps(stored))
    artifacts = rb.load_run(run)
    assert artifacts.stored_score_max_diff > 0.05
    readout = rb.analyze_run(artifacts, reps=4, pairs=3)
    assert not readout.supported and any("not reproduced" in r for r in readout.unsupported_reasons)
    assert "prediction unsupported" in rb.render_report([readout], 0.2, 4, 3)


def test_sparse_tied_pool_yields_no_budget_claim(tmp_path):
    # Two informative prompts and 398 zero vectors: perfect correlation, but the
    # top-40 membership is decided by ties, so no extrapolation may be acted on.
    stack = torch.zeros(400, 8, 3)
    stack[:2, :, 0] = 1
    val = torch.zeros(100, 3)
    val[:, 0] = 1
    artifacts = rb.RunArtifacts(tmp_path, "sparse", "math500", stack, val, 4, {"fresh_k": 32}, None, "current",
                                prompt_ids=list(range(400)), zero_norm_prompts=398)
    result = rb.analyze_run(artifacts, reps=2, pairs=20)
    assert result.registered.mean < 0.2
    assert not result.supported
    assert result.needed_candidate is None and result.needed_candidate_margin is None
    assert result.limiting_side is None
    report = rb.render_report([result], 0.2, 2, 20)
    assert "KEY sparse: prediction unsupported" in report
    assert "expected at" not in report
    assert "zero-norm stored gradient" in report and "tied at the selection boundary" in report


def test_tie_breaker_keeps_distinct_scores_at_any_scale():
    scores = torch.arange(400, dtype=torch.float32) * 1e-16
    exact = overlap_under_independent_ties(
        {i: float(v) for i, v in enumerate(scores)}, {i: float(v) for i, v in enumerate(scores)}, 40, seed=3, pairs=20,
    ).mean
    fast = rb.topk_overlap_batch(scores, scores, 40, pairs=200, generator=torch.Generator().manual_seed(3))
    assert exact == 1.0 and fast == 1.0
    half_tied = torch.cat([torch.zeros(360), torch.ones(40)])
    assert rb.topk_overlap_batch(half_tied, half_tied, 40, pairs=50, generator=torch.Generator().manual_seed(1)) == 1.0
    assert rb.boundary_tie_count(torch.zeros(400), 40) == 400
    assert rb.boundary_tie_count(torch.arange(400.0), 40) == 1


def test_launcher_resumes_a_run_created_by_the_first_launcher_version(launch_env):
    env, command, run, _ = launch_env
    run.mkdir(parents=True)
    source_config = json.loads((Path(env["OM_OLMO3_ROOT"]) / "family-mbpp-s0"
                                / "olmo3-1025-7b-base-rlzero-grpo-h100-v2-s0-mbpp-d0" / "run_config.json").read_text())
    old_style = dict(source_config)
    old_style.update({"gen_batch": "16", "n_train": 512, "n_val": 100, "proj_dim": 4096, "grad_layers": 4,
                      "clip_cap": 10.0, "topk_frac": 0.1, "temperature": 1.0, "top_p": 1.0, "thinking": "off",
                      "prompt_format": "olmo_rlzero_code", "attn": "eager"})
    (run / "run_config.json").write_text(json.dumps(old_style))
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "recorded previously unrecorded fields" in result.stdout
    config = json.loads((run / "run_config.json").read_text())
    assert config["prompts_sha256"] and config["math_verifier"] == "math_verify"
    assert config["fresh_k"] == 64 and config["model"] == old_style["model"]
