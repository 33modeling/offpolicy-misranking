"""Regression tests for generation-agnostic run discovery."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_select import describe_skips, iter_runs  # noqa: E402

FAILURES = 0


def check(name: str, condition: bool) -> None:
    global FAILURES
    print(("  ok  " if condition else "FAIL  ") + name)
    if not condition:
        FAILURES += 1


def make(root: Path, name: str, *files: str) -> Path:
    run = root / name
    run.mkdir()
    for filename in files:
        (run / filename).touch()
    return run


with tempfile.TemporaryDirectory() as raw_tmp:
    root = Path(raw_tmp)
    make(root, "v2-old", "DONE", "needed.json")
    make(root, "v3-corrected", "score_protocol.json", "oracle_protocol.json", "needed.json")
    make(root, "v10-incomplete", "needed.json")
    make(root, "gate-14b", "needed.json")
    make(root, "v4-smoke", "DONE", "needed.json")
    make(root, "custom-corrected", "score_protocol.json", "oracle_protocol.json", "needed.json")

    chosen = iter_runs(root, need=("needed.json",), include_legacy=True)
    names = [run.name for run in chosen]
    check("all numeric generations are discovered", "v2-old" in names)
    check("corrected protocol can replace legacy DONE", "v3-corrected" in names)
    check("legacy gate is optional and supported", "gate-14b" in names)
    check("unknown corrected naming is accepted", "custom-corrected" in names)
    check("incomplete and smoke runs are excluded", "v10-incomplete" not in names and "v4-smoke" not in names)

    reasons = describe_skips(
        root,
        chosen,
        need=("needed.json",),
        include_legacy=True,
    )
    text = "\n".join(reasons)
    check("diagnostics use caller requirements", "val_gradient.pt" not in text)
    check("diagnostics explain missing completion marker", "v10-incomplete" in text and "DONE" in text)


with tempfile.TemporaryDirectory() as raw_tmp:
    root = Path(raw_tmp) / "v7-direct"
    root.mkdir()
    (root / "score_protocol.json").touch()
    (root / "oracle_protocol.json").touch()
    (root / "artifact.json").touch()
    chosen = iter_runs(root, need=("artifact.json",))
    check("a direct run path is supported", chosen == [root])


print(("PASS" if FAILURES == 0 else "FAIL") + f" (failures {FAILURES})")


def test_run_selection_regressions() -> None:
    assert FAILURES == 0


if __name__ == "__main__":
    sys.exit(1 if FAILURES else 0)
