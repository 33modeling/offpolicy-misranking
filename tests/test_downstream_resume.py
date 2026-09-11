"""CPU recovery checks and launcher integration for existing E5 outputs."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess

import pytest

import evidence_downstream as ed
from check_downstream_resume import recoverable_checkpoint
from test_evidence_downstream import source_point
from train_policy_grpo import GrpoConfig, _checkpoint_contract

ROOT = Path(__file__).resolve().parents[1]


def repair_fixture(tmp_path, monkeypatch, step=500):
    run, evaluation = source_point(tmp_path)
    out = tmp_path / "output"
    ed.prepare(run, out, evaluation, 100, 8)
    config = ed.read(run / "run_config.json")
    monkeypatch.setenv("OM_PROMPT_FORMAT", config["prompt_format"])
    grpo = GrpoConfig(**{key.removeprefix("grpo_"): config[key]
                        for key in ed.TRAIN_FLAGS.values()
                        if key != "grpo_logprob_micro_batch"}, checkpoint_every=5)
    args = argparse.Namespace(
        objective="grpo", model=config["model"], seed=config["seed"], start_step=400,
        target_steps=500, prompts=str(out / "subsets/subset-g11.json"), max_new_tokens=64,
        resume_adapter=str(run / "policy_step_400"),
        resume_optimizer=str(run / "policy_step_400/optimizer.pt"),
    )
    checkpoint = out / "g11/policy" / f"checkpoint-{step:06d}"
    checkpoint.mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "grpo_stats.jsonl"):
        (checkpoint / name).write_bytes(b"fixture")
    state = {**_checkpoint_contract(args, grpo, 4), "completed_steps": step,
             "adapter_sha256": ed.digest(checkpoint / "adapter_model.safetensors"),
             "optimizer_sha256": ed.digest(checkpoint / "optimizer.pt"),
             "grpo_stats_sha256": ed.digest(checkpoint / "grpo_stats.jsonl")}
    ed.atomic_json(checkpoint / "checkpoint_state.json", state)
    ed.atomic_json(out / "g11/policy/policy_train.json", {})
    return run, evaluation, out, checkpoint


@pytest.mark.parametrize("step", [495, 500])
def test_recovers_a_verified_checkpoint_without_changing_existing_contracts(tmp_path, monkeypatch, step):
    run, evaluation, out, checkpoint = repair_fixture(tmp_path, monkeypatch, step)
    before = {str(path): ed.digest(path) for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="invalid GRPO policy"):
        ed.arm_policy(out, "g11")
    assert recoverable_checkpoint(out, "g11") == (checkpoint, step)
    assert before == {str(path): ed.digest(path) for path in tmp_path.rglob("*") if path.is_file()}
    assert ed.prepare(run, out, evaluation, 100, 8) == ed.read(out / "experiment.json")


@pytest.mark.parametrize("change", ["seed", "target", "parent", "hash", "missing", "before-source"])
def test_repair_rejects_incompatible_or_incomplete_checkpoints(tmp_path, monkeypatch, change):
    _, _, out, checkpoint = repair_fixture(tmp_path, monkeypatch)
    state = ed.read(checkpoint / "checkpoint_state.json")
    if change == "seed":
        state["seed"] += 1
    elif change == "target":
        state["target_steps"] += 1
    elif change == "parent":
        state["resume_adapter"] = "/different/parent"
    elif change == "hash":
        (checkpoint / "optimizer.pt").write_bytes(b"corrupted")
    elif change == "missing":
        (checkpoint / "adapter_model.safetensors").unlink()
    else:
        state["completed_steps"] = 400
    ed.atomic_json(checkpoint / "checkpoint_state.json", state)
    with pytest.raises(ValueError, match="no compatible"):
        recoverable_checkpoint(out, "g11")


@pytest.mark.parametrize("change", ["source", "subset", "code", "model", "arm"])
def test_repair_keeps_source_and_subset_protection(tmp_path, monkeypatch, change):
    run, _, out, _ = repair_fixture(tmp_path, monkeypatch)
    if change == "source":
        (run / "run_config.json").write_text("{}")
    elif change == "subset":
        (out / "subsets/subset-g11.json").write_text("{}")
    elif change == "code":
        contract = ed.read(out / "experiment.json")
        contract["code_hashes"]["src/evidence_downstream.py"] = "changed"
        ed.atomic_json(out / "experiment.json", contract)
    elif change == "model":
        Path(ed.read(run / "run_config.json")["model"], "config.json").write_text("changed")
    with pytest.raises(ValueError):
        recoverable_checkpoint(out, "g00" if change == "arm" else "g11")


@pytest.mark.parametrize("repair_rc", [0, 2])
def test_launcher_repairs_or_preserves_bad_arm_and_continues(tmp_path, repair_rc):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/run_downstream_independent.sh", repo / "scripts")
    (repo / "scripts/setup_env.sh").write_text("true\n")
    bins = tmp_path / "venv/bin"
    bins.mkdir(parents=True)
    python = bins / "python"
    python.write_text('''#!/bin/bash
case "$1" in
  -c) printf 'eager\\n8\\nq_proj,v_proj\\n1.0\\noff\\nolmo_rlzero_math\\n' ;;
  src/bootstrap_math_verify.py) echo /fixture ;;
  src/check_downstream_resume.py) echo repair >> "$CALLS"; exit "$REPAIR_RC" ;;
  -m) echo "train:$3" >> "$CALLS" ;;
  src/evidence_downstream.py)
    case "$2" in
      policy-ready) if [ "${@: -1}" = g11 ]; then exit 2; else exit 1; fi ;;
      evaluate) echo "eval:${@: -3:1}" >> "$CALLS" ;;
      summarize) echo '{"complete": true}' ;;
    esac ;;
  *) exit 99 ;;
esac
''')
    python.chmod(0o755)
    smi = bins / "nvidia-smi"
    smi.write_text("#!/bin/bash\nexit 0\n")
    smi.chmod(0o755)
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_config.json").write_text("{}")
    out = tmp_path / "output"
    (out / "subsets").mkdir(parents=True)
    for arm in ("g11", "fresh_r"):
        (out / "subsets" / f"train-{arm}.args").write_bytes(f"-m\0fixture\0{arm}\0".encode())
    calls = tmp_path / "calls"
    result = subprocess.run(
        ["bash", "scripts/run_downstream_independent.sh", str(run), str(out), "--eval-prompts", "fixture.json"],
        cwd=repo, env={**os.environ, "PATH": str(bins) + os.pathsep + os.environ["PATH"],
                      "VENV_DIR": str(bins.parent), "OM_WORK": str(tmp_path), "CALLS": str(calls),
                      "OM_NODE_LOCK_HELD": "0", "OM_LOCAL_LOCK_DIR": str(tmp_path / "locks"),
                      "CUDA_VISIBLE_DEVICES": "0,1,2,3", "DOWNSTREAM_SELECTORS": "g11 fresh_r",
                      "REPAIR_RC": str(repair_rc)}, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == (0 if repair_rc == 0 else 1), result.stdout + result.stderr
    lines = calls.read_text().splitlines()
    assert "repair" in lines
    assert ("train:g11" in lines) == (repair_rc == 0)
    assert "train:fresh_r" in lines
