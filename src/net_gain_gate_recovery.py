"""Audited autograd recovery without changing a running suite's frozen source files.

This entrypoint adapts the original runner only at its selection and result hooks.
Original baselines, protocol bindings, task locks and cost ledgers remain in place.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import net_gain_gate_gpu as gpu

base, core = gpu.base, gpu.core
HERE = Path(__file__).resolve()
SCHEMA = "net-gate-autograd-recovery/v1"
_select_once = gpu.select_once
_run_arm = gpu.run_arm
_validate_result = gpu.validate_result
_summarize = gpu.summarize
_status = gpu.status


def calibration_failure(directory):
    """Only a failed score phase with a calibration abort permits this repair."""
    progress = directory / "progress.json"
    if not progress.exists():
        return None
    state = core.read(progress)
    if state.get("state") != "failed" or state.get("phase") != "score":
        return None
    for path in sorted(directory.glob("score-*.log")):
        if path.stat().st_mtime < state.get("updated", 0) - state.get("seconds", 0) - 2:
            continue
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 65536))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
        aborts = [line for line in lines if line.startswith("[abort]")]
        if aborts and aborts[-1].startswith("[abort] finite-difference calibration failed at step="):
            return {"log": path.name, "error": aborts[-1]}
    return None


def private_dir(out, arm):
    return out / "selector-work" / f"{arm}-autograd-recovery"


def validate_recovery(out, p, arm):
    directory = out / arm
    record = core.read(directory / "autograd-recovery.json")
    expected = {"protocol_sha256": core.fingerprint(p),
                "contract_sha256": base.digest(out / "contract.json"),
                "runner_sha256": base.digest(HERE), "arm": arm}
    if record.get("schema") != SCHEMA or record.get("binding") != expected:
        raise ValueError("autograd recovery binding changed")
    with (directory / "cost.jsonl").open("rb") as handle:
        prefix = handle.read(record["original_cost_bytes"])
    if len(prefix) != record["original_cost_bytes"] or hashlib.sha256(prefix).hexdigest() != record["original_cost_sha256"]:
        raise ValueError("pre-recovery cost ledger changed")
    if base.spent(directory) < record["original_gpu_seconds"]:
        raise ValueError("failed finite-difference work must remain charged")
    return record


def begin_recovery(out, c, p, arm, evidence):
    directory = out / arm
    if (directory / "autograd-recovery.json").exists():
        return validate_recovery(out, p, arm)
    if p["mode"] != "study" or arm != "selection_reduced" or p["selector"] not in {"low_order", "pair_u2"}:
        raise ValueError("numerical recovery is restricted to development selection controls")
    if any(path.exists() for path in (directory / "execution.json", directory / "result.json",
                                      directory / "policy", out / "subsets" / f"subset-{arm}.json")):
        raise ValueError("cannot change the scoring backend after a training subset is frozen")
    spent = base.spent(directory)
    prefix = (directory / "cost.jsonl").read_bytes()
    record = {"schema": SCHEMA, "binding": {"protocol_sha256": core.fingerprint(p),
        "contract_sha256": base.digest(out / "contract.json"), "runner_sha256": base.digest(HERE), "arm": arm},
        "from": "finite", "to": "autograd", "evidence": evidence,
        "original_cost_bytes": len(prefix), "original_cost_sha256": hashlib.sha256(prefix).hexdigest(),
        "original_gpu_seconds": spent,
        "cost_policy": "All failed finite work and recovery work remain inside the original branch cap.",
        "reporting": "Numerically amended development result, not an unchanged finite-difference run."}
    base.bind(directory / "autograd-recovery.json", record)
    print(f"[recovery] {out.name}/{arm}: autograd; {spent:.1f} GPU-s already charged", flush=True)
    return record


def exact_selection(out, c, arm, cap, env, devices):
    import low_order_experiment as low

    directory, private = out / arm, private_dir(out, arm)
    selected = private / "selected.json"
    if selected.exists():
        if core.read(selected.with_suffix(".sha256.json")) != {"sha256": base.digest(selected)}:
            raise ValueError("recovered selection hash changed")
        return core.read(selected)["indices"]
    if cap - base.spent(directory) <= 0:
        raise ValueError("branch allocation exhausted; autograd recovery does not reset its budget")
    base.bind(private / "inputs/test.json", {"test": c["evaluation"]["val"],
                                           "provenance": c["evaluation"]["provenance"]})
    scoring = private / "scoring"
    base.meter(directory, "autograd-prepare", c["scope"]["gpu_type"], action=lambda: low.prepare(
        scoring, [Path(c["source_run"])], evaluation=private / "inputs/test.json", steps=1,
        eval_k=c["eval_k"], arms=(c["scope"]["selector"],), derivative="autograd"), ledger="deployment")
    point = next(low.entries(scoring))
    if core.read(point / "experiment.json")["derivative"] != "autograd":
        raise ValueError("recovery requires a separate autograd scoring contract")
    for stage in ("validation", "score"):
        if stage == "validation" and (point / "direction.pt").exists():
            continue
        remaining = (cap - base.spent(directory)) / base.GPUS
        if remaining <= 0:
            raise ValueError("branch allocation exhausted during autograd recovery")
        commands = [([sys.executable, str(base.ROOT / "src/low_order_experiment.py"), "worker",
                      "--out", str(point), "--stage", stage, "--shard", str(i)], devices[i])
                    for i in range(base.GPUS)]
        base.meter(directory, f"autograd-{stage}", c["scope"]["gpu_type"], commands=commands,
                   env=env, timeout=remaining, ledger="deployment")
        base.meter(directory, f"autograd-merge-{stage}", c["scope"]["gpu_type"],
                   action=lambda stage=stage: low.merge_direction(point) if stage == "validation" else low.merge(point),
                   ledger="deployment")
    indices = core.read(point / "selection.json")["selected"][c["scope"]["selector"]]
    base.bind(selected, {"indices": indices, "scoring_sha256": base.digest(point / "selection.json"),
                         "experiment_sha256": base.digest(point / "experiment.json"),
                         "point": str(point.relative_to(private))})
    base.bind(selected.with_suffix(".sha256.json"), {"sha256": base.digest(selected)})
    return indices


def select_once(out, c, p, arm, choice, env, devices):
    directory = out / arm
    recovery = directory / "autograd-recovery.json"
    if recovery.exists():
        validate_recovery(out, p, arm)
        return exact_selection(out, c, arm, choice["budget_gpu_seconds"], env, devices)
    # run_arm has just updated progress while checking inputs. The prior failure
    # is captured by our run hook before that heartbeat is replaced.
    evidence = _pending.get(str(directory))
    if evidence is None:
        try:
            return _select_once(out, c, p, arm, choice, env, devices)
        except Exception:
            evidence = calibration_failure(directory)
            if evidence is None or p["mode"] != "study" or arm != "selection_reduced":
                raise
    begin_recovery(out, c, p, arm, evidence)
    return exact_selection(out, c, arm, choice["budget_gpu_seconds"], env, devices)


def recovery_attestation(out, p, arm):
    directory, private = out / arm, private_dir(out, arm)
    validate_recovery(out, p, arm)
    selected = private / "selected.json"
    selection = core.read(selected)
    if core.read(selected.with_suffix(".sha256.json")) != {"sha256": base.digest(selected)}:
        raise ValueError("recovered selection changed")
    point = private / selection["point"]
    if point.resolve().parent != (private / "scoring/points").resolve():
        raise ValueError("invalid recovery scoring path")
    if base.digest(point / "selection.json") != selection["scoring_sha256"] or base.digest(point / "experiment.json") != selection["experiment_sha256"]:
        raise ValueError("autograd scoring artifacts changed")
    if core.read(point / "experiment.json")["derivative"] != "autograd":
        raise ValueError("recovery backend is not autograd")
    execution = core.read(directory / "execution.json")
    if execution["action"] != "select" or execution["indices"] != selection["indices"]:
        raise ValueError("training did not use the recovered selection")
    return {"schema": SCHEMA, "recovery_sha256": base.digest(directory / "autograd-recovery.json"),
            "result_sha256": base.digest(directory / "result.json"), "selection_sha256": base.digest(selected)}


_pending = {}


def run_arm(out, suite, p, arm, devices, env):
    directory = out / arm
    if (private_dir(out, arm).exists() or (directory / "autograd-recovery-result.json").exists()) and not (directory / "autograd-recovery.json").exists():
        raise ValueError("missing autograd recovery record")
    if (directory / "autograd-recovery.json").exists():
        validate_recovery(out, p, arm)
    if (directory / "autograd-recovery.json").exists() and (directory / "result.json").exists():
        # A crash between the original result write and attestation needs no GPU work.
        _validate_result(out, p, arm)
        base.bind(directory / "autograd-recovery-result.json", recovery_attestation(out, p, arm))
    _pending[str(directory)] = calibration_failure(directory) if p["mode"] == "study" and arm == "selection_reduced" else None
    try:
        _run_arm(out, suite, p, arm, devices, env)
        if (directory / "autograd-recovery.json").exists():
            base.bind(directory / "autograd-recovery-result.json", recovery_attestation(out, p, arm))
    finally:
        _pending.pop(str(directory), None)


def validate_result(out, p, arm):
    result = _validate_result(out, p, arm)
    directory = out / arm
    if (private_dir(out, arm).exists() or (directory / "autograd-recovery-result.json").exists()) and not (directory / "autograd-recovery.json").exists():
        raise ValueError("missing autograd recovery record")
    if (directory / "autograd-recovery.json").exists():
        expected = recovery_attestation(out, p, arm)
        if core.read(directory / "autograd-recovery-result.json") != expected:
            raise ValueError("numerical recovery result attestation changed")
        result = {**result, "numerical_recovery": core.read(directory / "autograd-recovery.json"),
                  "numerical_recovery_attestation": expected}
    return result


def summarize(root):
    result = _summarize(root)
    if any(root.glob("points/*/*/autograd-recovery.json")):
        result["numerical_protocol"] = "finite_with_audited_autograd_recovery/v1"
        result["numerical_protocol_note"] = "Amended development study; failed probes and exact scoring are charged, not refunded."
        # Do not fit a recovery-cost model and deploy it as the old finite-only selector.
        for point in result.get("points", []):
            point["scope"] = {**point["scope"], "selector": point["scope"]["selector"] + ":autograd-recovery-v1"}
        core.atomic_json(root / "study.json", result)
    return result


def status(root):
    _status(root)
    for path in sorted(root.glob("points/*/*/autograd-recovery.json")):
        print(f"[autograd recovery] {path.parent.parent.name}/{path.parent.name}; failed probe costs retained")


def main():
    # Keep the frozen runner byte-for-byte intact; children still use its original
    # measurement/evaluation entrypoint. Only admitted parent execution is adapted.
    gpu.select_once, gpu.run_arm, gpu.validate_result = select_once, run_arm, validate_result
    gpu.summarize, gpu.status = summarize, status
    return gpu.main()


if __name__ == "__main__":
    gpu.install_signal_handlers()
    raise SystemExit(main())
