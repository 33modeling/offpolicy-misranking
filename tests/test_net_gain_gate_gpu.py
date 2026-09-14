import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import net_gain_gate as net
import net_gain_gate_gpu as gpu
import selection_gate as core
import selection_gate_gpu as base
from test_net_gain_gate import fixture
from test_selection_gate_gpu import toy_source


def protocol(mode="study", selector="low_order"):
    return {"schema": net.SCHEMA, "mode": mode, "model": None, "role": "development",
            "selector": selector, "schedule": net.SCHEDULE, "max_measurement_fraction": .01,
            "recent_window": 20, "arms": list(gpu.study.BRANCHES if mode == "study" else gpu.TEST_ARMS)}


def source(tmp_path):
    out, c = toy_source(tmp_path)
    c["scope"]["selector"] = "low_order"
    core.atomic_json(out / "contract.json", c)
    core.atomic_json(out / "net_inputs.json", {"cache": "hash"})
    return out, c


def completed(out, arm):
    core.atomic_json(out / arm / "policy/budget_stop.json", {"completed_steps": 101, "stop_reason": "no_block_fits"})
    for i in range(4):
        core.atomic_json(out / arm / "evaluation" / f"shard-{i}.done.json", {})


def test_random_arm_never_measures_and_completed_resume_has_no_cost(tmp_path, monkeypatch):
    out, c = source(tmp_path)
    p = protocol()
    monkeypatch.setattr(base, "verify", lambda _: c)
    monkeypatch.setattr(base, "policy", lambda *a: Path(c["source_run"]))
    monkeypatch.setattr(base, "rewards", lambda *a: {"q0": .5})
    monkeypatch.setattr(gpu, "measure_once", lambda *a: pytest.fail("random measured"))
    monkeypatch.setattr(gpu, "select_once", lambda *a: pytest.fail("random scored"))
    completed(out, "random_full")
    gpu.run_arm(out, {"eval_timeout": 5}, p, "random_full", list("0123"), {})
    paid = base.spent(out / "random_full")
    r = gpu.validate_result(out, p, "random_full")
    assert r["measurement_gpu_seconds"] == 0 and r["action"] == "random"
    gpu.run_arm(out, {}, p, "random_full", [], {})
    assert base.spent(out / "random_full") == paid


@pytest.mark.parametrize("kind", ["stop", "ledger", "subset", "protocol", "hash"])
def test_result_mutation_is_not_done(tmp_path, monkeypatch, kind):
    out, c = source(tmp_path)
    p = protocol()
    monkeypatch.setattr(base, "verify", lambda _: c)
    monkeypatch.setattr(base, "policy", lambda *a: Path(c["source_run"]))
    monkeypatch.setattr(base, "rewards", lambda *a: {"q0": .5})
    completed(out, "random_full")
    gpu.run_arm(out, {"eval_timeout": 5}, p, "random_full", list("0123"), {})
    if kind == "stop":
        core.atomic_json(out / "random_full/policy/budget_stop.json", {"stop_reason": "updates_completed"})
    elif kind == "ledger":
        base.meter(out / "random_full", "unexpected", "H100", action=lambda: None)
    elif kind == "subset":
        core.atomic_json(out / "subsets/subset-random_full.json", {})
    elif kind == "protocol":
        p["recent_window"] = 1
    else:
        core.atomic_json(out / "random_full/result.json", {})
    with pytest.raises(ValueError):
        gpu.validate_result(out, p, "random_full")


def test_update_count_completion_not_accepted_as_equal_budget(tmp_path, monkeypatch):
    out, c = source(tmp_path)
    monkeypatch.setattr(base, "verify", lambda _: c)
    core.atomic_json(out / "random_full/policy/budget_stop.json", {"completed_steps": 200, "stop_reason": "updates_completed"})
    with pytest.raises(ValueError, match="update-count"):
        gpu.run_arm(out, {}, protocol(), "random_full", list("0123"), {})
    assert not (out / "random_full/result.json").exists()


def test_failed_diagnosis_is_charged_and_never_repeated(tmp_path, monkeypatch):
    out, c = source(tmp_path)
    p, directory = protocol(), out / "measurement"
    original = base.meter
    def failed(*a, **kw):
        def action():
            raise ValueError("intentional")
        return original(*a, action=action, ledger=kw["ledger"])
    monkeypatch.setattr(base, "meter", failed)
    first = gpu.measure_once(out, {"measurement_wall_seconds": 30}, p, directory, {})
    assert first["status"] == "failed_no_retry" and first["gpu_seconds"] > 0
    monkeypatch.setattr(base, "meter", lambda *a, **kw: pytest.fail("measurement repeated"))
    assert gpu.measure_once(out, {}, p, directory, {}) == first
    with pytest.raises(ValueError, match="development label"):
        gpu.decision(out, {}, p, "selection_reduced", {})


def test_measurement_success_shared_once_but_charged_to_both_counterfactuals(tmp_path, monkeypatch):
    out, c = source(tmp_path)
    p = protocol()
    original, calls = base.meter, []
    def measurement(*a, **kw):
        calls.append(1)
        return original(*a, action=lambda: base.bind(out / "measurement/measurement.json", {"features": {}}))
    monkeypatch.setattr(base, "meter", measurement)
    suite = {"measurement_wall_seconds": 30}
    r = gpu.decision(out, suite, p, "random_reduced", {})
    s = gpu.decision(out, suite, p, "selection_reduced", {})
    assert len(calls) == 1
    assert r["measurement_gpu_seconds"] == s["measurement_gpu_seconds"] > 0
    assert r["budget_gpu_seconds"] == c["budget_gpu_seconds"]-r["measurement_gpu_seconds"]
    p["recent_window"] = 1
    with pytest.raises(ValueError, match="binding"):
        gpu.decision(out, suite, p, "random_reduced", {})


def test_open_cost_is_not_a_free_retry(tmp_path):
    out, _ = source(tmp_path)
    directory = out / "measurement"
    base.journal(directory / "cost.jsonl", {"event_id": "unknown", "state": "started", "phase": "diagnose",
                  "ledger": "research", "gpus": 4, "gpu_type": "H100", "time": 0.})
    with pytest.raises(ValueError, match="unclosed"):
        gpu.measure_once(out, {}, protocol(), directory, {})


def test_selection_work_is_private_and_billed_to_each_arm(tmp_path, monkeypatch):
    out, c = source(tmp_path)
    c["evaluation"] = {"val": [], "provenance": {}}
    seen = []
    def selected(private, c, directory, cap, env, devices, **kw):
        seen.append((private, directory))
        return [0, 1, 2, 3]
    monkeypatch.setattr(base, "select_once", selected)
    choice = {"budget_gpu_seconds": 1000., "profile_sha256": None}
    for arm in ("selection_full", "gated"):
        assert gpu.select_once(out, c, protocol("test"), arm, choice, {}, list("0123")) == [0, 1, 2, 3]
    assert seen[0][0] != seen[1][0] and seen[0][1] != seen[1][1]


def test_gated_failure_falls_back_without_repeating_selection(tmp_path, monkeypatch):
    out, c = source(tmp_path)
    p = protocol("test")
    choice = {"binding": {"protocol_sha256": core.fingerprint(p), "contract_sha256": base.digest(out / "contract.json")},
              "action": "select", "reason": "predicted", "budget_gpu_seconds": 998., "measurement_gpu_seconds": 2., "profile_sha256": None}
    core.atomic_json(out / "gate_measurement/initial.json", {"gpu_seconds": 2., "report_sha256": None})
    original_spent = base.spent
    monkeypatch.setattr(base, "spent", lambda directory: 2. if directory.name == "gate_measurement" else original_spent(directory))
    core.atomic_json(out / "gated/decision.json", choice)
    monkeypatch.setattr(gpu, "decision", lambda *a: choice)
    monkeypatch.setattr(base, "verify", lambda _: c)
    monkeypatch.setattr(base, "policy", lambda *a: Path(c["source_run"]))
    monkeypatch.setattr(base, "rewards", lambda *a: {"q0": .5})
    def fail(*a):
        base.meter(out / "gated", "score-failed", "H100", action=lambda: None)
        raise ValueError("score failed")
    monkeypatch.setattr(gpu, "select_once", fail)
    completed(out, "gated")
    gpu.run_arm(out, {"eval_timeout": 5}, p, "gated", list("0123"), {})
    r = gpu.validate_result(out, p, "gated")
    assert r["action"] == "random" and r["measurement_gpu_seconds"] == 2
    monkeypatch.setattr(gpu, "select_once", lambda *a: pytest.fail("scoring repeated"))
    gpu.run_arm(out, {}, p, "gated", [], {})


def test_one_failure_does_not_stop_other_gpu_arms(tmp_path, monkeypatch):
    out, c = source(tmp_path)
    p = protocol()
    core.atomic_json(tmp_path / "net_protocol.json", p)
    core.atomic_json(tmp_path / "suite.json", {})
    monkeypatch.setitem(sys.modules, "additive_experiment", SimpleNamespace(model_environment=lambda c: {}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("OM_NODE_LOCK_HELD", "1")
    monkeypatch.setattr(base, "entries", lambda _: [out])
    monkeypatch.setattr(gpu.subprocess, "check_output", lambda *a, **kw: "H100\n"*4)
    monkeypatch.setattr(gpu, "status", lambda _: None)
    calls = []
    def run(out, suite, p, arm, devices, env):
        calls.append(arm)
        if arm == "random_full":
            raise ValueError("intentional")
    monkeypatch.setattr(gpu, "run_arm", run)
    assert gpu.work(tmp_path) == 1
    assert calls == p["arms"]


def test_gpu_rejects_synthetic_fit_and_wrong_selector_before_prepare(tmp_path, monkeypatch):
    model = net.fit(fixture())
    path = tmp_path / "model.json"
    core.atomic_json(path, model)
    monkeypatch.setattr(base, "prepare", lambda _: pytest.fail("invalid deployment prepared"))
    args = SimpleNamespace(mode="test", model=path)
    with pytest.raises(ValueError, match="synthetic"):
        gpu.prepare(args)


def test_measurement_lock_is_nonblocking(tmp_path):
    out, _ = source(tmp_path)
    directory = out / "measurement"
    with base.lease(directory / ".measurement.lock"):
        with pytest.raises(BlockingIOError):
            gpu.measure_once(out, {}, protocol(), directory, {})


def test_failed_subprocess_payload_is_not_accepted_as_success(tmp_path, monkeypatch):
    out, _ = source(tmp_path)
    d = out / "measurement"
    p = protocol()
    original = base.meter
    def interrupted(*a, **kw):
        def write_then_interrupt():
            core.atomic_json(d / "measurement.json", {"features": {}})
            raise KeyboardInterrupt
        return original(*a, action=write_then_interrupt)
    monkeypatch.setattr(base, "meter", interrupted)
    with pytest.raises(KeyboardInterrupt):
        gpu.measure_once(out, {"measurement_wall_seconds": 30}, p, d, {})
    monkeypatch.setattr(base, "meter", lambda *a, **kw: pytest.fail("measurement retried"))
    assert gpu.measure_once(out, {}, p, d, {})["status"] == "failed_no_retry"


def test_sigterm_closes_cost_and_reaps_child(tmp_path):
    code = '''
import os,sys
from pathlib import Path
import net_gain_gate_gpu as gpu
gpu.install_signal_handlers()
gpu.base.meter(Path(sys.argv[1]), 'child', 'H100',
 commands=[([sys.executable, '-c', 'import time; time.sleep(120)'], '')], timeout=120)
'''
    env = {**os.environ, "PYTHONPATH": str(Path(gpu.HERE).parent)}
    proc = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic()+5
        while not (tmp_path / "progress.json").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (tmp_path / "progress.json").exists()
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=10)
        assert proc.returncode != 0
        assert base.cost(tmp_path)["complete"] and base.spent(tmp_path) > 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


def test_four_workers_claim_each_arm_once(tmp_path):
    root = tmp_path / "suite"
    p = protocol()
    core.atomic_json(root / "net_protocol.json", p)
    entries = []
    for seed in range(3):
        out = root / "points" / f"s{seed}"
        core.atomic_json(out / "contract.json", {"config": {"seed": seed}, "scope": {"gpu_type": "H100"}})
        entries.append({"name": out.name, "sha256": base.digest(out / "contract.json")})
    core.atomic_json(root / "suite.json", {"schema": base.SCHEMA, "points": entries})
    code = '''
import sys,time
from pathlib import Path
from types import SimpleNamespace
import net_gain_gate_gpu as g
sys.modules['additive_experiment'] = SimpleNamespace(model_environment=lambda c: {})
g.subprocess.check_output = lambda *a, **kw: 'H100\\n'*4
g.status = lambda root: None
def run(out, suite, protocol, arm, devices, env):
    directory=out/arm
    if (directory/'finished').exists(): return
    with (directory/'started').open('x') as f: f.write('once')
    time.sleep(.03)
    (directory/'finished').touch()
g.run_arm=run
raise SystemExit(g.work(Path(sys.argv[1])))
'''
    env = {**os.environ, "PYTHONPATH": str(gpu.HERE.parent), "CUDA_VISIBLE_DEVICES": "0,1,2,3", "OM_NODE_LOCK_HELD": "1"}
    procs = [subprocess.Popen([sys.executable, "-c", code, str(root)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(4)]
    try:
        for proc in procs:
            stdout, stderr = proc.communicate(timeout=10)
            assert proc.returncode == 0, stdout+stderr
        assert len(list(root.glob("points/*/*/finished"))) == 9
        assert len(list(root.glob("points/*/*/started"))) == 9
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()


def test_shell_export_is_read_only_and_never_overwrites(tmp_path):
    root, output = tmp_path / "suite", tmp_path / "export.txt"
    core.atomic_json(root / "net_protocol.json", protocol())
    core.atomic_json(root / "suite.json", {"schema": base.SCHEMA, "points": []})
    core.atomic_json(root / "study.json", {"test_marker": "present"})
    env = {**os.environ, "NET_GATE_PYTHON": sys.executable, "NET_GATE_ROOT": str(root), "NET_GATE_EXPORT": str(output)}
    command = ["bash", "scripts/run_net_gain_gate.sh", "export"]
    r = subprocess.run(command, cwd=gpu.HERE.parents[1], env=env, capture_output=True, text=True, timeout=10)
    assert r.returncode == 0, r.stderr
    assert 'test_marker' in output.read_text()
    assert core.read(root / "study.json") == {"test_marker": "present"}
    before = output.read_bytes()
    assert subprocess.run(command, cwd=gpu.HERE.parents[1], env=env, capture_output=True, timeout=10).returncode != 0
    assert output.read_bytes() == before


def test_standalone_status_reports_eta_without_writing_suite(tmp_path):
    root = tmp_path / "suite"
    p = protocol()
    core.atomic_json(root / "net_protocol.json", p)
    contract = root / "points/p0/contract.json"
    core.atomic_json(contract, {"config": {"seed": 0, "drift": 100}})
    core.atomic_json(root / "suite.json", {"schema": base.SCHEMA,
        "points": [{"name": "p0", "sha256": base.digest(contract)}], "budget_gpu_seconds": 14400.})
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = subprocess.run(["bash", "scripts/status_net_gain_gate.sh"], cwd=gpu.HERE.parents[1],
                            env={**os.environ, "NET_GATE_ROOT": str(root), "NET_GATE_NODES": "4",
                                 "NET_GATE_PYTHON": sys.executable},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "DONE 0/3" in result.stdout and "QUEUED 3" in result.stdout and "INVALID 0" in result.stdout
    assert "4 nodes x 4 GPUs" in result.stdout
    assert "0.75 h" in result.stdout
    assert before == {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_status_without_jq_still_shows_all_arms(tmp_path):
    root = tmp_path / "suite"
    core.atomic_json(root / "net_protocol.json", protocol())
    contract = root / "points/p0/contract.json"
    core.atomic_json(contract, {"config": {"seed": 2, "drift": 100}})
    core.atomic_json(root / "suite.json", {"schema": base.SCHEMA,
        "points": [{"name": "p0", "sha256": base.digest(contract)}], "budget_gpu_seconds": 14400.})
    arm = root / "points/p0/selection_reduced"
    core.atomic_json(arm / "failure.json", {"error": "score worker failed: [None, None, None, 2]"})
    bins = tmp_path / "bin"
    bins.mkdir()
    for name in ("bash", "dirname", "realpath"):
        (bins / name).symlink_to(shutil.which(name))
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = subprocess.run([str(bins / "bash"), "scripts/run_net_gain_gate.sh", "status"],
        cwd=gpu.HERE.parents[1], env={**os.environ, "PATH": str(bins),
            "NET_GATE_ROOT": str(root), "NET_GATE_PYTHON": sys.executable},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "0/3 DONE" in result.stdout
    assert "selection_reduced" in result.stdout and "FAILED" in result.stdout
    assert "unavailable without jq" in result.stdout and "[abort]" not in result.stdout
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_why_saves_child_error_without_starting_workers(tmp_path):
    root = tmp_path / "suite"
    work = tmp_path / "shared-work"
    report_dir = work / "reports/net-gain-gate-v3"
    arm = root / "points/p0/selection_reduced"
    core.atomic_json(arm / "failure.json", {"error": "score worker failed: [None, None, None, 2]"})
    (arm / "score-3.log").write_text("[abort] example child failure\n")
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = subprocess.run(["bash", "scripts/run_net_gain_gate.sh", "why"],
        cwd=gpu.HERE.parents[1], env={**os.environ, "NET_GATE_ROOT": str(root),
            "HOME": str(tmp_path), "OM_WORK": str(work), "NET_GATE_PYTHON": "/nonexistent/python"},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    reports = list(report_dir.glob("net-gate-errors-*.txt"))
    assert len(reports) == 1
    assert f"[saved] {reports[0]}" in result.stdout
    report = reports[0].read_text()
    assert "score-3.log" in report and "[abort] example child failure" in report
    assert str(root) in report and "COMMIT:" in report
    assert "example child failure" not in result.stdout
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    again = subprocess.run(["bash", "scripts/run_net_gain_gate.sh", "why"],
        cwd=gpu.HERE.parents[1], env={**os.environ, "NET_GATE_ROOT": str(root),
            "HOME": str(tmp_path), "OM_WORK": str(work)}, capture_output=True, text=True, timeout=10)
    assert again.returncode == 0, again.stderr
    assert len(list(report_dir.glob("net-gate-errors-*.txt"))) == 2
    assert not list(tmp_path.glob("net-gate-errors-*.txt"))
    assert reports[0].read_text() == report


def test_why_saves_report_when_no_failures(tmp_path):
    root = tmp_path / "suite"
    root.mkdir()
    result = subprocess.run(["bash", "scripts/run_net_gain_gate.sh", "why"],
        cwd=gpu.HERE.parents[1], env={**os.environ, "NET_GATE_ROOT": str(root),
            "HOME": str(tmp_path), "OM_WORK": str(tmp_path / "shared-work")}, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    reports = list((tmp_path / "shared-work/reports/net-gain-gate-v3").glob("net-gate-errors-*.txt"))
    assert len(reports) == 1 and f"[saved] {reports[0]}" in result.stdout
    assert "no recorded arm failures" in reports[0].read_text()


@pytest.mark.parametrize("mode, prefix", [("why", "net-gate-errors"), ("export", "net-gate-results")])
def test_reports_default_to_group_volume_not_home(tmp_path, mode, prefix):
    group, home = tmp_path / "group-volume", tmp_path / "home"
    group.mkdir()
    home.mkdir()
    work = group / "cluster-user/offpolicy-misranking"
    root = work / "runs/net-gain-gate-v3"
    core.atomic_json(root / "net_protocol.json", protocol())
    core.atomic_json(root / "suite.json", {"schema": base.SCHEMA, "points": []})
    env = {**os.environ, "GROUP_VOLUME": str(group), "OM_USER": "cluster-user", "HOME": str(home),
           "NET_GATE_PYTHON": sys.executable}
    for key in ("OM_WORK", "NET_GATE_ROOT", "NET_GATE_EXPORT"):
        env.pop(key, None)
    command = ["bash", "scripts/run_net_gain_gate.sh", mode]
    for _ in range(2):
        result = subprocess.run(command, cwd=gpu.HERE.parents[1], env=env,
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert str(work / "reports/net-gain-gate-v3") in result.stdout
    reports = list((work / "reports/net-gain-gate-v3").glob(f"{prefix}-*.txt"))
    assert len(reports) == 2 and all(path.stat().st_size for path in reports)
    assert not list(home.iterdir())
