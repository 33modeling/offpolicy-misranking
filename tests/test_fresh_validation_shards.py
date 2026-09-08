"""Validation scheduling preserves RNG, exact coverage and interrupted work."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import experiment
import rollout
from artifact_contract import sha256_file, validate_generation_contract
from fresh_validation import validation_layout
from rollout_contract import rollout_seed_base

REPO = Path(__file__).resolve().parents[1]


class Tokenizer:
    eos_token_id = 9

    def apply_chat_template(self, *args, **kwargs):
        return "question"

    def __call__(self, *args, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[5, 6]]))

    def decode(self, ids, **kwargs):
        return str(ids[0].item())


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(calls=0, loads=0, fail_at=None)

    class Model:
        device = "cpu"
        config = SimpleNamespace(_name_or_path="/models/test-model", eos_token_id=9)
        generation_config = SimpleNamespace(eos_token_id=9)

        def generate(self, ids, **kwargs):
            state.calls += 1
            if state.calls == state.fail_at:
                raise RuntimeError("synthetic CUDA error")
            n = ids.shape[0]
            response = torch.cat((torch.randint(10, 99, (n, 2)), torch.full((n, 1), 9)), dim=1)
            return torch.cat((ids, response), dim=1)

    def load(*args, **kwargs):
        state.loads += 1
        model = Model()
        if len(args) > 1 and args[1] is not None:
            adapter = Path(args[1])
            model._om_policy_adapter = {
                "path": str(adapter.resolve()),
                "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                "policy_manifest_sha256": sha256_file(adapter / "policy_train.json"),
            }
        return model, Tokenizer()

    monkeypatch.setenv("OM_GEN_BATCH", "8")
    monkeypatch.setenv("OM_PROMPT_FORMAT", "tokenizer_chat")
    monkeypatch.setattr(experiment, "load_policy", load)
    monkeypatch.setattr(rollout, "reward", lambda text, answer: float(int(text) % 2))
    monkeypatch.setitem(rollout.SAMPLING, "top_p", 1.0)
    state.load = load
    return state


def make_run(path, n_val=7):
    path.mkdir()
    prompts = {
        "train": [{"question": f"train{i}", "answer": "1"} for i in range(8)],
        "val": [{"question": f"val{i}", "answer": "1"} for i in range(n_val)],
    }
    (path / "prompts.json").write_text(json.dumps(prompts))
    (path / "run_config.json").write_text(json.dumps({
        "seed": 5, "drift": 0, "model": "/models/test-model",
        "max_new_tokens": 4, "behavior_k": 2, "fresh_k": 2, "val_k": 2,
        "prompt_format": "tokenizer_chat",
    }))
    args = SimpleNamespace(
        shard=None, model="/models/test-model", adapter=None, fresh_k=2, val_k=2,
        max_new_tokens=4, temperature=1.0, seed=5,
    )
    return args, prompts


def merge(run, source):
    text = (REPO / "scripts/run_point.sh").read_text()
    start = text.index("merge_rollouts() {")
    function = text[start:text.index("\nverify_code_snapshot()", start)]
    return subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + function + '\nmerge_rollouts "$SOURCE" 2'],
        env={**os.environ, "PY": sys.executable, "PYTHONPATH": str(REPO / "src"),
             "OUT_ROOT": str(run), "SOURCE": source},
        text=True, capture_output=True, timeout=10,
    )


def finish(run):
    for source in ("rollouts_fresh_train", "rollouts_fresh_val"):
        result = merge(run, source)
        assert result.returncode == 0, result.stdout + result.stderr
    return validate_generation_contract(
        run, ("rollouts_fresh_train", "rollouts_fresh_val"),
        require_rng_binding=True, require_policy_binding=True,
    )


@pytest.mark.parametrize("n_val", [2, 7, 100])
def test_four_way_validation_matches_serial_bytes_and_seeds(tmp_path, runtime, n_val):
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    serial_args, _ = make_run(serial, n_val)
    args, _ = make_run(parallel, n_val)
    experiment.stage_rollout_fresh(serial_args, serial)
    for shard in (3, 1, 0, 2):
        args.shard = f"{shard}:4"
        experiment.stage_rollout_fresh(args, parallel)
    artifacts = sorted(parallel.glob("rollouts_fresh_val.shard*.jsonl"))
    assert len(artifacts) == min(4, n_val)
    finish(parallel)
    for source in ("rollouts_fresh_train", "rollouts_fresh_val"):
        assert (parallel / f"{source}.jsonl").read_bytes() == (serial / f"{source}.jsonl").read_bytes()
    assert not list(parallel.glob("rollouts_fresh_val.shard*"))
    calls, loads = runtime.calls, runtime.loads
    for shard in range(4):
        args.shard = f"{shard}:4"
        experiment.stage_rollout_fresh(args, parallel)
    assert (runtime.calls, runtime.loads) == (calls, loads)


def test_failed_validation_shard_resumes_only_its_remaining_prompt(tmp_path, runtime):
    run = tmp_path / "run"
    args, _ = make_run(run, 8)
    args.shard = "0:4"
    runtime.fail_at = 4  # two train prompts, one durable validation prompt, failure
    with pytest.raises(RuntimeError, match="synthetic CUDA"):
        experiment.stage_rollout_fresh(args, run)
    partial = run / "rollouts_fresh_val.shard0.partial"
    prefix = partial.read_bytes()
    for shard in (1, 2, 3):
        args.shard = f"{shard}:4"
        experiment.stage_rollout_fresh(args, run)
    completed = {p: p.read_bytes() for p in run.glob("rollouts_fresh_val.shard*.jsonl")}
    calls = runtime.calls
    for shard in range(4):
        args.shard = f"{shard}:4"
        experiment.stage_rollout_fresh(args, run)
    assert runtime.calls == calls + 1
    assert (run / "rollouts_fresh_val.shard0.jsonl").read_bytes().startswith(prefix)
    assert all(p.read_bytes() == content for p, content in completed.items())
    assert finish(run)["validated_rows"] == (8 + 8) * 2
    assert not (run / ".restart-quarantine").exists()


def test_existing_serial_partial_is_preserved_and_resumed_once(tmp_path, runtime):
    run = tmp_path / "run"
    args, prompts = make_run(run)
    model, tok = runtime.load()
    runtime.fail_at = 2
    out = run / "rollouts_fresh_val.jsonl"
    with pytest.raises(RuntimeError):
        rollout.collect_rollouts(
            model, tok, prompts["val"], 2, 4, 1.0, out,
            sampling_seed_base=rollout_seed_base(5, 0, "rollouts_fresh_val"),
        )
    original = out.with_suffix(".partial").read_bytes()
    calls = runtime.calls
    for shard in (2, 1, 0, 3):
        args.shard = f"{shard}:4"
        experiment.stage_rollout_fresh(args, run)
    assert runtime.calls - calls == 8 + 6  # all train, only six missing val prompts
    assert out.read_bytes().startswith(original)
    assert not list(run.glob("rollouts_fresh_val.shard*"))
    assert json.loads((run / ".fresh-val-layout.json").read_text())["mode"] == "serial"
    assert finish(run)["validated_rows"] == 30
    assert not (run / ".restart-quarantine").exists()


def test_changed_gpu_count_preserves_partial_and_fails_before_model_load(tmp_path, runtime):
    run = tmp_path / "run"
    args, _ = make_run(run)
    args.shard = "0:4"
    runtime.fail_at = 4
    with pytest.raises(RuntimeError):
        experiment.stage_rollout_fresh(args, run)
    files = {p: p.read_bytes() for p in run.glob("rollouts_fresh_val.*")}
    loads = runtime.loads
    args.shard = "0:2"
    with pytest.raises(ValueError, match="layout changed"):
        experiment.stage_rollout_fresh(args, run)
    assert runtime.loads == loads
    assert all(p.read_bytes() == content for p, content in files.items())


def test_sharded_validation_binds_the_exact_grpo_adapter(tmp_path, runtime):
    run = tmp_path / "run"
    args, _ = make_run(run)
    config = run / "run_config.json"
    document = json.loads(config.read_text())
    document["drift"] = 100
    config.write_text(json.dumps(document))
    adapter = run / "policy_step_100"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture weights")
    (adapter / "policy_train.json").write_text("{}")
    args.adapter = str(adapter)
    for shard in range(4):
        args.shard = f"{shard}:4"
        experiment.stage_rollout_fresh(args, run)
    assert finish(run)["validated_rows"] == 30
    manifest = json.loads((run / "rollouts_fresh_val.manifest.json").read_text())
    assert manifest["policy_adapter"]["path"] == str(adapter)
    assert manifest["sampling_seed_base"] == rollout_seed_base(5, 100, "rollouts_fresh_val")


def test_serial_publication_interruption_finishes_without_resampling(tmp_path, runtime):
    run = tmp_path / "run"
    args, _ = make_run(run)
    experiment.stage_rollout_fresh(args, run)
    manifest = run / "rollouts_fresh_val.manifest.json"
    manifest.rename(manifest.with_suffix(".json.tmp"))
    calls = runtime.calls
    experiment.stage_rollout_fresh(args, run)
    assert runtime.calls == calls
    assert manifest.exists()


def test_interrupted_validation_merge_recovers_without_generation(tmp_path, runtime):
    run = tmp_path / "run"
    args, _ = make_run(run)
    for shard in range(4):
        args.shard = f"{shard}:4"
        experiment.stage_rollout_fresh(args, run)
    out = run / "rollouts_fresh_val.jsonl"
    out.write_bytes(b"".join(p.read_bytes() for p in sorted(run.glob("rollouts_fresh_val.shard*.jsonl"))))
    original = out.read_bytes()
    result = merge(run, "rollouts_fresh_val")
    assert result.returncode == 0, result.stdout + result.stderr
    calls, loads = runtime.calls, runtime.loads
    for shard in range(4):
        args.shard = f"{shard}:4"
        experiment.stage_rollout_fresh(args, run)
    assert (runtime.calls, runtime.loads) == (calls, loads)
    assert out.read_bytes() == original
    assert finish(run)["validated_rows"] == 30


@pytest.mark.parametrize("failed", [False, True])
def test_shell_launches_all_shards_then_merges_validation(tmp_path, runtime, failed):
    run = tmp_path / "run"
    args, _ = make_run(run)
    for shard in range(4):
        args.shard = f"{shard}:4"
        experiment.stage_rollout_fresh(args, run)
    source = (REPO / "scripts/run_point.sh").read_text()
    merge_start = source.index("merge_rollouts() {")
    merge_function = source[merge_start:source.index("\nverify_code_snapshot()", merge_start)]
    wait_start = source.index("wait_all_stages() {")
    wait_function = source[wait_start:source.index("\n#", wait_start)]
    start = source.index("# Complete an interrupted merge")
    block = source[start:source.index('progress 5 "oracle+val gradients"', start)]
    setup = '''set -euo pipefail
NGPU=4; COMMON=(); POLICY_ARGS=()
artifact_ready() { return 1; }
progress() { :; }
log() { echo "$*"; }
run_stage() {
  touch "$OUT_ROOT/started-$1"
  for _ in {1..100}; do
    files=("$OUT_ROOT"/started-*)
    [ "${#files[@]}" -eq 4 ] && break
    sleep 0.01
  done
  [ "${#files[@]}" -eq 4 ] || return 19
  [ "$FAIL_SHARD" != "$1" ] || return 17
}
'''
    result = subprocess.run(
        ["bash", "-c", "\n".join((setup, merge_function, wait_function, block))],
        env={**os.environ, "OUT_ROOT": str(run), "LOGS": str(run / "logs"),
             "PY": sys.executable, "PYTHONPATH": str(REPO / "src"),
             "FAIL_SHARD": "2" if failed else "none", "FRESH_K": "2", "VAL_K": "2"},
        text=True, capture_output=True, timeout=10,
    )
    assert len(list(run.glob("started-*"))) == 4, result.stdout + result.stderr
    if failed:
        assert result.returncode != 0
        assert not (run / "rollouts_fresh_val.jsonl").exists()
        assert len(list(run.glob("rollouts_fresh_val.shard*.jsonl"))) == 4
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert (run / "rollouts_fresh_val.manifest.json").exists()


def test_missing_validation_shard_cannot_publish_or_remove_other_shards(tmp_path, runtime):
    run = tmp_path / "run"
    args, _ = make_run(run)
    for shard in (0, 1, 2):
        args.shard = f"{shard}:4"
        experiment.stage_rollout_fresh(args, run)
    files = {p: p.read_bytes() for p in run.glob("rollouts_fresh_val.shard*")}
    result = merge(run, "rollouts_fresh_val")
    assert result.returncode != 0
    assert not (run / "rollouts_fresh_val.jsonl").exists()
    assert all(p.read_bytes() == content for p, content in files.items())


def test_corrupted_validation_shard_is_rejected_without_deleting_sources(tmp_path, runtime):
    run = tmp_path / "run"
    args, _ = make_run(run)
    for shard in range(4):
        args.shard = f"{shard}:4"
        experiment.stage_rollout_fresh(args, run)
    shard = run / "rollouts_fresh_val.shard0.jsonl"
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    rows[0]["reward"] = 1 - rows[0]["reward"]
    shard.write_text("".join(json.dumps(row) + "\n" for row in rows))
    files = {p: p.read_bytes() for p in run.glob("rollouts_fresh_val.shard*")}
    result = merge(run, "rollouts_fresh_val")
    assert result.returncode != 0 and "hash mismatch" in result.stderr
    assert not (run / "rollouts_fresh_val.manifest.json").exists()
    assert all(p.read_bytes() == content for p, content in files.items())


def test_layout_is_shared_by_concurrent_workers(tmp_path):
    code = "import json, sys; from pathlib import Path; from fresh_validation import validation_layout; print(json.dumps(validation_layout(Path(sys.argv[1]), 4, 100), sort_keys=True))"
    jobs = [subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) for _ in range(4)]
    outputs = []
    for job in jobs:
        out, err = job.communicate(timeout=10)
        assert job.returncode == 0, err
        outputs.append(out)
    assert len(set(outputs)) == 1
    assert json.loads((tmp_path / ".fresh-val-layout.json").read_text())["shards"] == 4


def test_mixed_validation_layout_is_not_silently_discarded(tmp_path):
    legacy = tmp_path / "rollouts_fresh_val.partial"
    shard = tmp_path / "rollouts_fresh_val.shard0.partial"
    legacy.write_text("legacy")
    shard.write_text("sharded")
    with pytest.raises(ValueError, match="mixed serial/sharded"):
        validation_layout(tmp_path, 4, 100)
    assert legacy.read_text() == "legacy" and shard.read_text() == "sharded"
