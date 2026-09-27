"""Reuse the operational Switch four-rank admission before claiming GPU work."""

import json
from pathlib import Path
import sys

from .existing_runtime import model_path, runtime_packages, verifier_environment


def packages():
    return runtime_packages()


def admit(root, environment, run_child, *, pass_fds=(), heartbeat=lambda pid: None,
          should_stop=lambda: False, plan=None):
    root.mkdir(parents=True, exist_ok=False)
    verifier_environment(environment)
    versions = packages()
    if plan is not None:
        environment["SRGC_LOCAL_MODEL"] = model_path(plan["model"], plan["model_revision"], environment)
    script = Path(__file__).resolve().parents[1] / "scripts/selection_nccl_preflight.py"
    code = run_child([sys.executable, str(script), "--root", str(root), "--world-size", "4"],
                     root / "preflight.log", environment, pass_fds=pass_fds,
                     heartbeat=heartbeat, should_stop=should_stop, timeout=600)
    receipts = list((root / "node-preflight").glob("*/admission.json"))
    if code or len(receipts) != 1:
        raise RuntimeError(f"four-GPU admission failed (exit={code}); inspect {root / 'preflight.log'}")
    report = json.loads(receipts[0].read_text())
    if report.get("state") != "passed":
        raise RuntimeError(f"four-GPU admission did not pass: {receipts[0]}")
    environment.update(report["overrides"])
    return {"receipt": str(receipts[0]), "packages": versions, "python": sys.executable,
            "model_path": environment.get("SRGC_LOCAL_MODEL"),
            "runtime_overrides": report["overrides"],
            "allocated_gpu_seconds": report["allocated_gpu_seconds"]}
