"""CPU regressions for resuming the pinned Qwen alias/manifest failure."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from artifact_contract import cached_rollout_ready, sha256_file, validate_generation_contract
from repair_model_alias import repair
from test_artifact_contract import make_run

REPO = Path(__file__).resolve().parents[1]


def alias_run(tmp_path):
    model = tmp_path / "uploaded-snapshot"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "qwen3_5"}')
    alias = tmp_path / "Qwen3.5-9B-pinned"
    alias.symlink_to(model, target_is_directory=True)
    run = tmp_path / "run"
    run.mkdir()
    make_run(run)
    path = run / "run_config.json"
    config = json.loads(path.read_text())
    config.update(model=str(alias), model_resolved=str(model),
                  model_config_sha256=sha256_file(model / "config.json"))
    path.write_text(json.dumps(config))
    for path in run.glob("*.manifest.json"):
        manifest = json.loads(path.read_text())
        manifest["model_name_or_path"] = str(alias)
        path.write_text(json.dumps(manifest))
    return run, model, alias


def test_proven_alias_preserves_config_rollouts_partials_and_audit(tmp_path):
    run, model, _ = alias_run(tmp_path)
    (run / "rollouts_fresh_val.partial").write_text("unfinished generation")
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    with pytest.raises(ValueError, match="model mismatch"):
        validate_generation_contract(run)
    assert repair(run) == 3
    assert validate_generation_contract(run)["validated_rows"] == 14
    assert repair(run) == 0
    for name, data in before.items():
        if not name.endswith(".manifest.json"):
            assert (run / name).read_bytes() == data
        else:
            assert json.loads((run / name).read_text())["model_name_or_path"] == str(model)
            backups = list((run / "logs/model-alias-repair").glob(f"*/{name}"))
            assert len(backups) == 1
            assert backups[0].read_bytes() == data
            receipt = json.loads((backups[0].parent / "receipt.json").read_text())
            assert receipt["canonical_manifest_sha256"] == sha256_file(run / name)
    assert cached_rollout_ready(run / "rollouts_behavior_train.jsonl")


def test_shard_failure_publishes_cached_canonical_without_changing_shards(tmp_path):
    run, _, _ = alias_run(tmp_path)
    prefix = "rollouts_behavior_train"
    manifest = run / f"{prefix}.manifest.json"
    document = json.loads(manifest.read_text())
    document["artifact_file"] = f"{prefix}.shard0.jsonl"
    shard = run / document["artifact_file"]
    shard.write_bytes((run / f"{prefix}.jsonl").read_bytes())
    shard_manifest = run / f"{prefix}.shard0.manifest.json"
    shard_manifest.write_text(json.dumps(document))
    original = shard_manifest.read_bytes()
    manifest.unlink()
    assert repair(run) == 3
    assert shard_manifest.read_bytes() == original
    assert shard.is_file()
    assert cached_rollout_ready(run / f"{prefix}.jsonl")
    assert validate_generation_contract(run)["validated_rows"] == 14


@pytest.mark.parametrize("problem", ["different-model", "missing-alias", "basename-only",
                                     "changed-config", "no-config-hash", "corrupt-rollout",
                                     "sampling", "no-rollout-hash", "merged-tamper", "derived"])
def test_unproven_or_invalid_artifacts_fail_without_any_write(tmp_path, problem):
    run, model, alias = alias_run(tmp_path)
    manifest = run / "rollouts_fresh_val.manifest.json"
    document = json.loads(manifest.read_text())
    if problem == "different-model":
        other = tmp_path / "other-model"
        other.mkdir()
        alias.unlink()
        alias.symlink_to(other, target_is_directory=True)
    elif problem == "missing-alias":
        alias.unlink()
    elif problem == "basename-only":
        document["model_name_or_path"] = alias.name
    elif problem == "changed-config":
        (model / "config.json").write_text("{}")
    elif problem == "no-config-hash":
        path = run / "run_config.json"
        config = json.loads(path.read_text())
        config.pop("model_config_sha256")
        path.write_text(json.dumps(config))
    elif problem == "corrupt-rollout":
        (run / "rollouts_fresh_val.jsonl").write_text("corrupt")
    elif problem == "sampling":
        document["explicit_kwargs"]["top_k"] = 10
    elif problem == "no-rollout-hash":
        document.pop("artifact_sha256")
    elif problem == "merged-tamper":
        shard = run / "rollouts_fresh_val.shard0.jsonl"
        shard.write_bytes((run / "rollouts_fresh_val.jsonl").read_bytes())
        document["artifact_file"] = shard.name
        (run / "rollouts_fresh_val.jsonl").write_text(shard.read_text().replace('"reward": 0.0', '"reward": 1.0'))
        manifest.unlink()
        manifest = run / "rollouts_fresh_val.shard0.manifest.json"
    elif problem == "derived":
        (run / "score_protocol.json").write_text("{}")
    manifest.write_text(json.dumps(document))
    before = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    with pytest.raises((ValueError, OSError)):
        repair(run)
    assert {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()} == before
    assert not (run / "logs/model-alias-repair").exists()
    assert not list(run.glob(".model-alias-validation-*"))


def test_noop_with_missing_local_model_when_names_already_match(tmp_path):
    make_run(tmp_path)
    assert repair(tmp_path) == 0


def test_fresh_alias_repair_does_not_invalidate_reused_behavior(tmp_path):
    run, model, _ = alias_run(tmp_path)
    path = run / "rollouts_behavior_train.manifest.json"
    document = json.loads(path.read_text())
    document["model_name_or_path"] = str(model)
    path.write_text(json.dumps(document))
    (run / "behavior_reuse.json").write_text("{}")
    before = path.read_bytes()
    assert repair(run) == 2
    assert path.read_bytes() == before
    assert (run / "behavior_reuse.json").read_text() == "{}"


@pytest.mark.parametrize("recoveries,expected_rc,attempts", [(0, 43, 1), (1, 0, 2), (99, 43, 4)])
def test_supervisor_alias_recovery_is_bounded_and_skips_cuda_retries(tmp_path, recoveries, expected_rc, attempts):
    source = (REPO / "scripts/run_matrix.sh").read_text()
    start = source.index("run_point_unlocked() {")
    function = source[start:source.index("\n}\n", start) + 2]
    (tmp_path / "logs").mkdir()
    script = function + r'''
run_dir() { echo "$TEST_ROOT"; }
n_train_for_dataset() { echo 400; }
reenter_runtime_fields() { :; }
run_complete() { [ -e "$TEST_ROOT/DONE" ]; }
note_point_accepted() { :; }
short_reason() { cat; }
mock_repair() { [ "$RECOVERIES" -gt 0 ]; }
run_pipeline_watchdog() {
  echo attempt >> "$TEST_ROOT/attempts"
  local count
  count=$(wc -l < "$TEST_ROOT/attempts")
  if [ "$RECOVERIES" = 1 ] && [ "$count" -gt 1 ]; then
    touch "$TEST_ROOT/DONE"
    return 0
  fi
  echo "ValueError: rollouts_behavior_train.shard0.manifest.json: model mismatch: expected 'snapshot', recorded 'Qwen3.5-9B-pinned'" > "$2"
  return 1
}
recover_cuda_rollout() { echo unexpected-cuda; return 1; }
sleep() { echo unexpected-sleep; }
PY=mock_repair
SUPERVISOR_REPO=/unused
CONTRACT=''
DRIFTS=(0 25)
SEEDS=(0)
DATASETS=(math500)
MAX_RETRIES=3
run_point_unlocked math500 0 0 '' '' ''
'''
    result = subprocess.run(["bash", "-c", script], cwd=tmp_path,
                            env={**os.environ, "TEST_ROOT": str(tmp_path), "RECOVERIES": str(recoveries)},
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == expected_rc, result.stdout + result.stderr
    assert len((tmp_path / "attempts").read_text().splitlines()) == attempts
    assert "unexpected-" not in result.stdout
    assert (tmp_path / "logs/regime-attempt-1.log").is_file()
    if attempts > 2:
        assert (tmp_path / "logs/regime-attempt-1-alias-1.log").is_file()


def test_pinned_qwen_pipeline_resumes_each_new_source_without_regeneration(tmp_path):
    run, _, _ = alias_run(tmp_path)
    (run / "logs").mkdir()
    originals = {p.name: p.read_bytes() for p in run.glob("rollouts*")}
    # The real cluster is pinned here. Test against that validator, not a
    # replacement that happens to accept our repaired manifests.
    pinned = tmp_path / "pinned"
    (pinned / "src").mkdir(parents=True)
    for name in ("artifact_contract.py", "rollout_contract.py"):
        payload = subprocess.check_output(["git", "show", f"2e96090:src/{name}"], cwd=REPO)
        (pinned / "src" / name).write_bytes(payload)
    generated = tmp_path / "generated"
    generated.mkdir()
    for path in run.glob("rollouts*"):
        path.rename(generated / path.name)
    pipeline = pinned / "point.sh"
    pipeline.write_text('''#!/usr/bin/env bash
"$REAL_PY" - <<'PY'
import os
from pathlib import Path
from artifact_contract import PRIMARY_SOURCES, cached_rollout_ready, validate_generation_contract
run, generated = Path(os.environ['OUT_ROOT']), Path(os.environ['GENERATED'])
with (run / 'attempts').open('a') as handle:
    handle.write('attempt\\n')
for prefix in PRIMARY_SOURCES:
    if not (run / f'{prefix}.jsonl').exists():
        with (run / 'generations').open('a') as handle:
            handle.write(prefix + '\\n')
        for suffix in ('.jsonl', '.manifest.json'):
            (generated / (prefix + suffix)).rename(run / (prefix + suffix))
    validate_generation_contract(run, (prefix,))
    assert cached_rollout_ready(run / f'{prefix}.jsonl')
validate_generation_contract(run)
(run / 'DONE').write_text('complete')
PY
''')
    source = (REPO / "scripts/run_matrix.sh").read_text()
    start = source.index("run_point_unlocked() {")
    function = source[start:source.index("\n}\n", start) + 2]
    script = function + r'''
run_dir() { echo "$TEST_ROOT"; }
n_train_for_dataset() { echo 2; }
reenter_runtime_fields() { :; }
run_complete() { [ -e "$TEST_ROOT/DONE" ]; }
note_point_accepted() { :; }
short_reason() { cat; }
run_pipeline_watchdog() {
  local log=$2
  shift 2
  "$@" > "$log" 2>&1
}
recover_cuda_rollout() { echo unexpected-cuda; return 1; }
sleep() { echo unexpected-sleep; }
CONTRACT=''
DRIFTS=(0)
SEEDS=(0)
DATASETS=(math500)
MAX_RETRIES=1
run_point_unlocked math500 0 0 '' '' ''
'''
    result = subprocess.run(["bash", "-c", script], cwd=REPO,
                            env={**os.environ, "TEST_ROOT": str(run), "PY": sys.executable,
                                 "REAL_PY": sys.executable, "SUPERVISOR_REPO": str(REPO),
                                 "PIPELINE_REPO": str(pinned), "PIPELINE_SCRIPT": str(pipeline),
                                 "GENERATED": str(generated)},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run / "DONE").is_file()
    assert (run / "attempts").read_text().splitlines() == ["attempt"] * 4
    assert len((run / "generations").read_text().splitlines()) == 3
    assert "unexpected-" not in result.stdout
    assert len(list((run / "logs").glob("regime-attempt-1*.log"))) == 4
    for name, payload in originals.items():
        if name.endswith(".jsonl"):
            assert (run / name).read_bytes() == payload


def test_prompt_rebuild_still_consumes_the_bounded_outer_retry(tmp_path):
    source = (REPO / "scripts/run_matrix.sh").read_text()
    start = source.index("run_point_unlocked() {")
    function = source[start:source.index("\n}\n", start) + 2]
    (tmp_path / "logs").mkdir()
    script = function + r'''
run_dir() { echo "$TEST_ROOT"; }
n_train_for_dataset() { echo 2; }
reenter_runtime_fields() { :; }
run_complete() { return 1; }
short_reason() { cat; }
run_pipeline_watchdog() {
  echo attempt >> "$TEST_ROOT/attempts"
  echo '[abort] prompts need rebuilding' > "$2"
  return 42
}
quarantine_prompt_target() { return 0; }
CONTRACT=''
DRIFTS=(0 25)
SEEDS=(0)
DATASETS=(fixture)
MAX_RETRIES=2
run_point_unlocked fixture 0 25 /missing-source '' ''
'''
    result = subprocess.run(["bash", "-c", script], cwd=tmp_path,
                            env={**os.environ, "TEST_ROOT": str(tmp_path)},
                            capture_output=True, text=True, timeout=3)
    assert result.returncode == 1, result.stdout + result.stderr
    assert (tmp_path / "attempts").read_text().splitlines() == ["attempt"] * 2
