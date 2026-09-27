"""Reuse the operational Switch four-rank admission before claiming GPU work."""

import importlib.metadata
import json
from pathlib import Path
import sys


def packages():
    from packaging.requirements import Requirement
    versions = {}
    for line in Path(__file__).with_name("requirements.txt").read_text().splitlines():
        requirement = Requirement(line)
        version = importlib.metadata.version(requirement.name)
        if version not in requirement.specifier:
            raise ValueError(f"{requirement.name}=={version} does not satisfy {requirement}")
        versions[requirement.name] = version
    return versions


def admit(root, environment, run_child, *, pass_fds=(), heartbeat=lambda pid: None,
          should_stop=lambda: False):
    versions = packages()
    root.mkdir(parents=True, exist_ok=False)
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
    return {"receipt": str(receipts[0]), "packages": versions,
            "runtime_overrides": report["overrides"],
            "allocated_gpu_seconds": report["allocated_gpu_seconds"]}
