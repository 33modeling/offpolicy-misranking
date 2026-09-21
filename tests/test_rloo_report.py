"""Partial paper exports must not alter training or canonical completion."""

from pathlib import Path
import json
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rloo_report as reporting
from test_rloo_experiment import fixture

# A deterministic synthetic predecessor, not a historical hash that can become
# identical to the current checked-out display file after a revert.
OLD_DISPLAY_SHA256 = "d" * 64


@pytest.fixture
def measured(tmp_path, monkeypatch):
    out = tmp_path / "math500-d0/s0"
    out.mkdir(parents=True)
    (out / "experiment.json").write_text("{}")
    monkeypatch.setattr(reporting.experiment, "validate", lambda out: (
        {"eval_n": 8, "source": {"seed": 0, "drift": 0}}, {}))

    def rows(out, arm, shard):
        return [{"prompt_idx": i, "reward": {"before": 0., "random": .25,
                 "passrate_beta": .5, "fresh_r": .75}[arm]}
                for i in range(shard * 2, (shard + 1) * 2)]

    monkeypatch.setattr(reporting.experiment, "checked_rows", rows)
    return out


def seal(out, arm, shards=range(4)):
    directory = out / arm / "evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        (directory / f"shard-{shard}.done.json").write_text("{}")


def test_missing_before_keeps_cached_comparison(measured):
    seal(measured, "passrate_beta")
    seal(measured, "fresh_r")
    result = reporting.point_report(measured)
    assert result["status"] == "incomplete"
    assert result["missing_arms"] == ["before", "random"]
    fresh = result["rows"][-1]
    assert fresh["vs_passrate_beta"]["mean"] == .25
    assert "vs_before" not in fresh and "vs_random" not in fresh
    assert fresh["missing_references"] == ["before", "random"]
    assert not (measured / "results.json").exists()


def test_partial_shards_are_explicit_and_unsealed_rows_ignored(measured):
    seal(measured, "before", [0])
    (measured / "before/evaluation/shard-1.jsonl").write_text("unfinished")
    result = reporting.point_report(measured)
    partial = result["evaluations"][0]
    assert partial["measured_prompts"] == 2
    assert partial["observed_mean_reward"] == 0.
    assert partial["missing_shards"] == [1, 2, 3]
    assert not partial["complete"] and result["rows"] == []
    assert result["evaluations"][1]["observed_mean_reward"] is None


def test_complete_matches_canonical_report(measured):
    for arm in ("before", *reporting.experiment.ARMS):
        seal(measured, arm)
    expected = reporting.experiment.report(measured)
    before = (measured / "results.json").read_bytes()
    actual = reporting.point_report(measured)
    assert actual["status"] == "complete"
    assert [{k: v for k, v in row.items() if k != "missing_references"}
            for row in actual["rows"]] == expected["rows"]
    assert (measured / "results.json").read_bytes() == before


def test_invalid_point_does_not_hide_other_points(measured, monkeypatch):
    other = measured.parent / "s1"
    seal(other, "random")
    (other / "experiment.json").write_text("{}")
    seal(measured, "fresh_r")
    original = reporting.experiment.checked_rows

    def checked(out, arm, shard):
        if out == measured:
            raise ValueError("evaluation seal mismatch")
        return original(out, arm, shard)

    monkeypatch.setattr(reporting.experiment, "checked_rows", checked)
    result = reporting.report(measured.parent.parent)
    assert result["points"][0]["status"] == "invalid"
    assert result["points"][0]["rows"] == []
    assert result["points"][1]["rows"][0]["arm"] == "random"
    assert result["points"][2]["status"] == "unprepared"
    assert not result["complete"]


def test_real_prepared_point_without_evaluation_is_exportable(tmp_path):
    _, out, _ = fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    result = reporting.point_report(out)
    assert result["missing_arms"] == ["before", *reporting.experiment.ARMS]
    assert result["rows"] == []
    assert before == {str(p): p.read_bytes() for p in out.rglob("*") if p.is_file()}


def test_launcher_exports_incomplete_matrix_without_gpu(tmp_path):
    import os
    root = tmp_path / "rloo"
    root.mkdir()
    process = subprocess.run(["bash", "scripts/run_rloo.sh", "report"],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True,
        env={**os.environ, "HOME": str(tmp_path), "RLOO_ROOT": str(root), "RLOO_PYTHON": sys.executable})
    assert process.returncode == 0, process.stderr
    exports = list(tmp_path.glob("rloo-results.txt"))
    assert len(exports) == 1
    assert "unprepared" in exports[0].read_text()
    assert "TXT saved:" in process.stdout


def test_missing_root_is_not_created(tmp_path):
    with pytest.raises(ValueError, match="no RLOO root"):
        reporting.report(tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_missing_root_replaces_stale_export_with_explicit_error(tmp_path, monkeypatch):
    target = tmp_path / 'rloo-results.txt'
    target.write_text('old completed results')
    root = tmp_path / 'absent'
    monkeypatch.setattr(sys, 'argv', ['report', '--root', str(root), '--out', str(target)])
    with pytest.raises(SystemExit) as exc:
        reporting.main()
    assert exc.value.code == 1
    value = json.loads(target.read_text().split('DATA_JSON\n', 1)[1])
    assert not value['complete'] and value['points'] == []
    assert value['errors'] and str(root) in value['errors'][0]
    assert value['export_status'] == 'failed'
    assert not root.exists()
    assert list(tmp_path.glob('*.txt')) == [target]


def old_display_contract(out):
    ed = reporting.experiment.ed
    assert ed.digest(reporting.experiment.ROOT / 'src/matrix_status.py') != OLD_DISPLAY_SHA256
    contract = ed.read(out / "experiment.json")
    contract["code_hashes"]["src/matrix_status.py"] = OLD_DISPLAY_SHA256
    ed.atomic_json(out / "experiment.json", contract)
    return contract


def measured_policy(out, arm, reward):
    from test_grpo_policy import _policy_artifact
    ed = reporting.experiment.ed
    contract, config = reporting.experiment.validate(out)
    policy = out / arm / "policy"
    _policy_artifact(policy, objective="rloo", completed_steps=100)
    manifest = ed.read(policy / "policy_train.json")
    manifest.update(
        optimizer_sha256=ed.digest(policy / "optimizer.pt"),
        grpo_stats_sha256=ed.digest(policy / "grpo_stats.jsonl"),
        base_model=str(Path(config["model"]).resolve()), seed=config["seed"],
        max_new_tokens=config["max_new_tokens"], prompt_format=config["prompt_format"],
        config=ed._expected_config(config), samples_per_step=32,
        prompts_sha256=ed.digest(out / "subsets" / f"subset-{arm}.json"))
    ed.atomic_json(policy / "policy_train.json", manifest)
    target = out / arm / "evaluation"
    target.mkdir()
    for shard in range(4):
        binding, _, indices = reporting.experiment.binding(out, arm, shard)
        path = target / f"shard-{shard}.jsonl"
        path.write_text("".join(json.dumps({"prompt_idx": i, "rollout_idx": j,
            "reward": reward}) + "\n" for i in indices for j in range(contract["eval_k"])))
        ed.atomic_json(target / f"shard-{shard}.done.json",
                       {"binding": binding, "rollouts_sha256": ed.digest(path)})


def test_reviewed_status_drift_exports_real_sealed_comparison_read_only(tmp_path):
    _, out, _ = fixture(tmp_path)
    old_display_contract(out)
    with pytest.raises(ValueError, match="reviewed runtime receipt missing or changed"):
        reporting.frozen_experiment.validate(out)
    measured_policy(out, "passrate_beta", .5)
    measured_policy(out, "fresh_r", .75)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = reporting.point_report(out)
    assert result["status"] == "incomplete"
    assert result["missing_arms"] == ["before", "random"]
    fresh = result["rows"][-1]
    assert fresh["mean_reward"] == .75
    assert fresh["vs_passrate_beta"] == {"mean": .25, "lower": .25, "upper": .25}
    assert result["report_display_code_changes"]["src/matrix_status.py"]["frozen_sha256"] == (
        OLD_DISPLAY_SHA256)
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="reviewed runtime receipt missing or changed"):
        reporting.frozen_experiment.training_command(out, "fresh_r")


def test_results_launcher_exports_reviewed_drift_to_one_txt(tmp_path):
    import os
    _, prepared, _ = fixture(tmp_path)
    out = tmp_path / "matrix/math500-d0/s0"
    out.parent.mkdir(parents=True)
    prepared.rename(out)
    old_display_contract(out)
    measured_policy(out, "fresh_r", .75)
    process = subprocess.run(["bash", "scripts/run_rloo.sh", "results"],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True,
        env={**os.environ, "HOME": str(tmp_path), "RLOO_ROOT": str(out.parent.parent),
             "RLOO_PYTHON": sys.executable})
    assert process.returncode == 0, process.stderr
    exports = list(tmp_path.glob("rloo-results*.txt"))
    assert len(exports) == 1
    text = exports[0].read_text()
    assert "0\t0\tincomplete\tfresh_r\t0.75\t" in text
    assert '"report_display_code_changes":{"src/matrix_status.py"' in text
    assert not (out / "results.json").exists()


@pytest.mark.parametrize("name", ["src/train_policy_rloo.py", "src/train_policy_grpo.py",
                                 "src/evidence_downstream.py", "src/selector_pair_gpu.py"])
def test_report_rejects_unreviewed_code_even_with_display_upgrade(tmp_path, name):
    _, out, _ = fixture(tmp_path)
    contract = old_display_contract(out)
    contract["code_hashes"][name] = "0" * 64
    reporting.experiment.ed.atomic_json(out / "experiment.json", contract)
    with pytest.raises(ValueError, match=f"code changed since preparation: {name}"):
        reporting.point_report(out)


def test_report_allows_another_isolated_display_revision(tmp_path, monkeypatch):
    _, out, _ = fixture(tmp_path)
    old_display_contract(out)
    ed = reporting.experiment.ed
    digest = ed.digest
    monkeypatch.setattr(ed, "digest", lambda p: "0" * 64
                        if p == reporting.experiment.ROOT / "src/matrix_status.py" else digest(p))
    result = reporting.point_report(out)
    assert result["status"] == "incomplete"
    assert result["report_display_code_changes"]["src/matrix_status.py"]["runtime_sha256"] == "0" * 64
    with pytest.raises(ValueError, match="reviewed runtime receipt missing or changed"):
        reporting.frozen_experiment.validate(out)


@pytest.mark.parametrize("artifact,reason", [
    ("subsets/subset-fresh_r.json", "prepared input changed"),
    ("fresh_r/evaluation/shard-0.jsonl", "evaluation seal mismatch"),
    ("fresh_r/policy/optimizer.pt", "hash"),
])
def test_display_upgrade_keeps_data_policy_and_seal_validation(tmp_path, artifact, reason):
    _, out, _ = fixture(tmp_path)
    old_display_contract(out)
    measured_policy(out, "fresh_r", .75)
    (out / artifact).write_text("tampered")
    with pytest.raises(ValueError, match=reason):
        reporting.point_report(out)


def test_any_recorded_display_revision_preserves_sealed_measurements(tmp_path):
    _, out, _ = fixture(tmp_path)
    contract = old_display_contract(out)
    contract["code_hashes"]["src/matrix_status.py"] = "1" * 64
    reporting.experiment.ed.atomic_json(out / "experiment.json", contract)
    measured_policy(out, "fresh_r", .75)
    result = reporting.point_report(out)
    assert result["rows"][0]["mean_reward"] == .75
    assert result["report_display_code_changes"]["src/matrix_status.py"]["frozen_sha256"] == "1" * 64


def test_display_dependency_scan_is_once_per_point_not_per_shard(tmp_path, monkeypatch):
    _, out, _ = fixture(tmp_path)
    old_display_contract(out)
    measured_policy(out, "fresh_r", .75)
    calls = []
    original = reporting.display_is_isolated
    monkeypatch.setattr(reporting, "display_is_isolated",
                        lambda name: calls.append(name) or original(name))
    reporting.point_report(out)
    assert calls == ["src/matrix_status.py"]
    reporting.point_report(out)
    assert calls == ["src/matrix_status.py"] * 2


def test_display_exception_fails_closed_if_module_is_referenced(tmp_path, monkeypatch):
    _, out, _ = fixture(tmp_path)
    old_display_contract(out)
    monkeypatch.setattr(reporting, "display_is_isolated", lambda name: False)
    with pytest.raises(ValueError, match="code changed since preparation: src/matrix_status.py"):
        reporting.point_report(out)


@pytest.mark.parametrize("reference", ["import matrix_status", "from matrix_status import main",
                                      "importlib.import_module('matrix_status')"])
def test_display_isolation_checks_scientific_source_references(tmp_path, monkeypatch, reference):
    source = tmp_path / "src"
    source.mkdir()
    (source / "matrix_status.py").write_text('"""matrix_status standalone."""\n')
    other = source / "trainer.py"
    other.write_text("print('training')\n")
    monkeypatch.setattr(reporting.frozen_experiment, "ROOT", tmp_path)
    assert reporting.display_is_isolated("src/matrix_status.py")
    other.write_text(reference + "\n")
    assert not reporting.display_is_isolated("src/matrix_status.py")


@pytest.mark.parametrize('reference', ['', 'import matrix_status', 'from matrix_status import main',
    "importlib.import_module('matrix_status')", 'import matrix_status as status',
    "exec(open('src/matrix_status.py').read())", 'for path in DISPLAY_MODULES: exec(open(path).read())',
    'loader(DISPLAY_MODULES)', "DISPLAY_MODULES = ('src/matrix_status.py',)"])
def test_only_exact_validator_metadata_declaration_is_exempt(tmp_path, monkeypatch, reference):
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'matrix_status.py').write_text('pass\n')
    declaration = ("DISPLAY_MODULES = ('src/matrix_status.py', 'src/rlzero_status.py', "
                   "'src/downstream_status.py', 'src/queue_status.py')\n")
    (source / 'rloo_experiment.py').write_text(declaration + 'allowed = name in DISPLAY_MODULES\n' + reference + '\n')
    monkeypatch.setattr(reporting.frozen_experiment, 'ROOT', tmp_path)
    assert reporting.display_is_isolated('src/matrix_status.py') is (not reference)


def test_declaration_exception_does_not_hide_same_line_import(tmp_path, monkeypatch):
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'matrix_status.py').write_text('pass\n')
    (source / 'rloo_experiment.py').write_text(
        "DISPLAY_MODULES = ('src/matrix_status.py', 'src/rlzero_status.py', "
        "'src/downstream_status.py', 'src/queue_status.py'); import matrix_status\n")
    monkeypatch.setattr(reporting.frozen_experiment, 'ROOT', tmp_path)
    assert not reporting.display_is_isolated('src/matrix_status.py')


@pytest.mark.parametrize('damage', [None, 'display_runtime_old', 'missing', 'scientific_runtime',
                                   'display_frozen', 'receipt_binding', 'unknown_file'])
def test_mixed_reviewed_runtime_receipt_retains_only_display_exception(tmp_path, damage):
    _, out, _ = fixture(tmp_path)
    c = old_display_contract(out)
    c['code_hashes']['src/rloo_experiment.py'] = reporting.frozen_experiment.PRE_QUEUE_OBSERVATION_CODE
    reporting.experiment.ed.atomic_json(out / 'experiment.json', c)
    changes = reporting.frozen_experiment.reviewed_code_changes(c['code_hashes'])
    receipt = reporting.frozen_experiment.observation_receipt(out, changes)
    if damage == 'display_runtime_old':
        receipt['changes']['src/matrix_status.py']['runtime_sha256'] = 'e' * 64
    elif damage == 'scientific_runtime':
        receipt['changes']['src/rloo_experiment.py']['runtime_sha256'] = 'e' * 64
    elif damage == 'display_frozen':
        receipt['changes']['src/matrix_status.py']['frozen_sha256'] = 'f' * 64
    elif damage == 'receipt_binding':
        receipt['experiment_sha256'] = 'f' * 64
    elif damage == 'unknown_file':
        receipt['changes']['src/train_policy_rloo.py'] = dict(frozen_sha256='f' * 64, runtime_sha256='e' * 64)
    if damage != 'missing':
        reporting.experiment.ed.atomic_json(out / 'queue-observation-runtime.json', receipt)
    before = {str(p): p.read_bytes() for p in out.rglob('*') if p.is_file()}
    if damage in {None, 'display_runtime_old'}:
        assert reporting.point_report(out)['status'] == 'incomplete'
    else:
        with pytest.raises(ValueError, match='reviewed runtime receipt missing or changed'):
            reporting.point_report(out)
    assert before == {str(p): p.read_bytes() for p in out.rglob('*') if p.is_file()}


def test_invalid_point_includes_exporter_identity_and_all_code_mismatches(tmp_path):
    _, prepared, _ = fixture(tmp_path)
    out = tmp_path / "matrix/math500-d0/s0"
    out.parent.mkdir(parents=True)
    prepared.rename(out)
    contract = old_display_contract(out)
    contract["code_hashes"]["src/train_policy_rloo.py"] = "2" * 64
    reporting.experiment.ed.atomic_json(out / "experiment.json", contract)
    result = reporting.report(out.parent.parent)
    identity = result["exporter"]
    assert identity["version"] == reporting.EXPORTER_VERSION
    assert identity["script_sha256"] == reporting.experiment.ed.digest(Path(reporting.__file__))
    assert len(identity["git_commit"]) == 40
    point = result["points"][0]
    assert point["status"] == "invalid" and point["rows"] == []
    mismatches = point["code_diagnostics"]["mismatches"]
    assert set(mismatches) == {"src/matrix_status.py", "src/train_policy_rloo.py"}
    assert mismatches["src/train_policy_rloo.py"]["frozen_sha256"] == "2" * 64
    assert mismatches["src/train_policy_rloo.py"]["runtime_sha256"] == (
        reporting.experiment.ed.digest(reporting.experiment.ROOT / "src/train_policy_rloo.py"))


def test_bad_contract_diagnostics_do_not_hide_other_points(tmp_path):
    out = tmp_path / "math500-d0/s0"
    out.mkdir(parents=True)
    (out / "experiment.json").write_text("not json")
    result = reporting.report(tmp_path)
    assert result["points"][0]["status"] == "invalid"
    assert "error" in result["points"][0]["code_diagnostics"]
    assert result["points"][1]["status"] == "unprepared"


@pytest.mark.parametrize('payload', ['null', '[]', '{"schema":null}', '{"schema":"x","source":null}'])
def test_malformed_contract_still_writes_one_partial_results_txt(tmp_path, payload):
    out = tmp_path / 'math500-d0/s0'
    out.mkdir(parents=True)
    (out / 'experiment.json').write_text(payload)
    target = tmp_path / 'rloo-results.txt'
    result = subprocess.run([sys.executable, str(Path(reporting.__file__)), '--root', str(tmp_path),
                             '--out', str(target)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 1, result.stderr
    exported = json.loads(target.read_text().split('DATA_JSON\n', 1)[1])
    assert exported['points'][0]['status'] == 'invalid'
    assert exported['points'][1]['status'] == 'unprepared'
    assert not (out / 'results.json').exists()


def test_large_raw_cost_ledger_cannot_hide_measured_results_txt(measured, tmp_path):
    for arm in ('before', *reporting.experiment.ARMS):
        seal(measured, arm)
    path = measured / 'random/cost.jsonl'
    original = (json.dumps({'event_id': 'large-diagnostic', 'note': 'x' * 2_000_000}) + '\n').encode()
    path.write_bytes(original)
    report = reporting.point_report(measured)
    target = tmp_path / 'bounded.txt'
    reporting.write_export('rloo', report, target=target)
    assert target.stat().st_size < 1_900_000
    assert report['status'] == 'complete' and len(report['rows']) == 3
    cost = next(item for item in report['evaluations'] if item['arm'] == 'random')['cost_ledger']
    assert cost['status'] == 'omitted_size_limit' and cost['events'] == []
    assert cost['source_bytes'] == len(original)
    assert cost['source_sha256'] == reporting.experiment.ed.digest(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize('raw', ['{"seconds":NaN}\n', '[]\n', '{"seconds":Infinity}\n'])
def test_invalid_cost_metadata_does_not_prevent_valid_reward_export(measured, tmp_path, raw):
    seal(measured, 'random')
    (measured / 'random/cost.jsonl').write_text(raw)
    report = reporting.point_report(measured)
    assert report['rows'][0]['mean_reward'] == .25
    assert next(item for item in report['evaluations'] if item['arm'] == 'random')['cost_ledger']['status'] == 'unreadable'
    reporting.write_export('rloo', report, target=tmp_path / 'valid-rewards.txt')
