"""Regressions for the September 6 operator-path audit."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import grads
import model_matrix


@pytest.mark.parametrize("prep_rc", [0, 1])
def test_actual_prep_branch_preserves_success_and_failure(tmp_path, prep_rc):
    source = (ROOT / "scripts/run_point.sh").read_text()
    start = source.index("  progress 1 prep")
    stop = source.index('\nfi\nif [ -n "${REGIME_MATRIX', start)
    block = source[start:stop]
    script = 'progress() { :; }; run_stage() { return "$TEST_RC"; }; COMMON=();\n' + block
    result = subprocess.run(["bash", "-c", script], env={**os.environ, "TEST_RC": str(prep_rc), "LOGS": str(tmp_path)}, capture_output=True)
    assert result.returncode == prep_rc


def test_partial_backward_oom_restarts_entire_prompt(monkeypatch):
    model = torch.nn.Module()
    model.p = torch.nn.Parameter(torch.tensor(1.))
    sequences = [{"input_ids": torch.tensor([1, 2, 3]), "resp_start": 1} for _ in range(4)]
    weights = [torch.ones(2) for _ in sequences]
    monkeypatch.setattr(grads, "_padded_token_logps", lambda m, ids: [m.p.expand(x.numel()-1) for x in ids])
    monkeypatch.setattr(grads, "project_grads", lambda ps, spec: ps[0].grad.clone())
    normal = grads.prompt_gradient(model, [model.p], sequences, weights, grads.ProjectionSpec(), 2)
    original = torch.Tensor.backward
    calls = 0
    def backward(tensor, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            model.p.grad += .5
            raise torch.cuda.OutOfMemoryError("partial backward")
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "backward", backward)
    retried = grads.prompt_gradient(model, [model.p], sequences, weights, grads.ProjectionSpec(), 2)
    assert retried == normal == 2.


def test_log_writer_exit_one_is_failure(tmp_path):
    from test_generalization_launcher import checkout
    root, env = checkout(tmp_path)
    script = '''set -Eeuo pipefail
source scripts/setup_env.sh
PROFILE=qwen35; MODE=--check
tee() { command tee "$@"; return 1; }
source scripts/launch_logging.sh
echo '[check] synthetic success'
'''
    result = subprocess.run(["bash", "-c", script], cwd=root, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "OK  qwen35" not in result.stdout
    assert "log writer failed" in result.stderr


@pytest.mark.parametrize("mode", ["run", "check", "prepare", "doctor"])
@pytest.mark.parametrize("profile", [None, "qwen35_2b", "qwen35_4b", "olmo3_domains"])
def test_launchers_do_not_mutate_git_or_kill_other_jobs(tmp_path, mode, profile):
    # status gained an explicit fast-forward path on September 6. Exercise
    # actual launch modes instead of rejecting words inside that function.
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    name = "run_followup.sh" if profile else "run_qwen35_9b.sh"
    (scripts / name).write_text((ROOT / "scripts" / name).read_text())
    for child in ("run_additional_experiments.sh", "doctor_qwen35.sh"):
        (scripts / child).write_text("exit 0\n")
    bins = tmp_path / "bin"
    bins.mkdir()
    git = bins / "git"
    git.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$GIT_CALLS"\n[ "$1" = rev-parse ] && { echo test-hash; exit 0; }\nexit 99\n')
    git.chmod(0o755)
    calls = tmp_path / "git-calls"
    args = [profile, mode] if profile else [mode]
    result = subprocess.run(
        ["bash", str(scripts / name), *args],
        env={**os.environ, "PATH": str(bins) + os.pathsep + os.environ["PATH"], "GIT_CALLS": str(calls)},
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls.read_text().splitlines() == ["rev-parse --short HEAD"]
    assert "cleanup_run_processes.py" not in (ROOT / "scripts/run_additional_experiments.sh").read_text()


def test_trust_environment_cannot_seal_unverified_weights(tmp_path, monkeypatch):
    for name, content in {"config.json": b"{}", "tokenizer_config.json": b"{}", "model.safetensors": b"wrong"}.items():
        (tmp_path / name).write_bytes(content)
    spec = {"key": "test", "repository": "test/model", "revision": "a"*40,
            "official_files": {"config.json": {"size": 999, "sha256": "f"*64}}}
    monkeypatch.setenv("OM_TRUST_LOCAL_SNAPSHOT", "1")
    monkeypatch.setenv("OM_ALLOW_UNPINNED_SNAPSHOT", "1")
    with pytest.raises(ValueError):
        model_matrix._seal_local_snapshot(spec, tmp_path)
    assert not (tmp_path / ".om_snapshot.json").exists()


def test_forged_pinned_manifest_is_rejected(tmp_path):
    import hashlib
    files = {"config.json": b"{}", "tokenizer_config.json": b"{}", "model.safetensors": b"wrong"}
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    spec = {"repository": "test/model", "revision": "a"*40,
            "official_files": {name: {"size": len(content), "sha256": "f"*64} for name, content in files.items()}}
    records = {name: {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()} for name, content in files.items()}
    model_matrix._write_manifest(spec, tmp_path, records)
    with pytest.raises(ValueError, match="hash mismatch"):
        model_matrix.validate_snapshot_provenance(spec, tmp_path)


def test_generation_retry_restores_rng_and_preserves_row_order():
    from rollout import generate_with_backoff
    class Model:
        def generate(self, ids, **kwargs):
            if len(ids) > 1:
                torch.rand(100)  # simulate RNG consumed before OOM
                raise torch.cuda.OutOfMemoryError("synthetic generation OOM")
            return torch.cat([ids, torch.randint(1, 100, (1, 1))], dim=1)
    ids = torch.tensor([[1, 2], [3, 4]])
    torch.manual_seed(7)
    expected = torch.randint(1, 100, (2, 1))
    torch.manual_seed(7)
    output = generate_with_backoff(Model(), ids, {"pad_token_id": 0})
    assert torch.equal(output[:, :2], ids)
    assert torch.equal(output[:, 2:], expected)
