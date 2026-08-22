"""CPU-only regression tests for strict readout artifact handling."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from readout_summary import main, precisions  # noqa: E402
from score_artifacts import ScoreArtifactError  # noqa: E402

FAILURES = 0


def check(name: str, condition: bool) -> None:
    global FAILURES
    print(("  ok  " if condition else "FAIL  ") + name)
    if not condition:
        FAILURES += 1


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def make_run(
    root: Path,
    name: str,
    *,
    malformed: bool = False,
    done: bool = True,
) -> Path:
    run = root / name
    run.mkdir()
    if done:
        (run / "DONE").touch()
    write_json(run / "score_protocol.json", {
        "schema": "offpolicy-score-validation-split/v1",
        "generation_validation": {"validated_rows": 20},
    })
    write_json(run / "oracle_protocol.json", {
        "schema": "offpolicy-oracle-validation-split/v1",
        "generation_validation": {"validated_rows": 20},
    })
    scores = {str(i): float(20 - i) for i in range(20)}
    write_json(run / "scores_oracle.json", {
        idx: {"score": score} for idx, score in scores.items()
    })
    write_json(run / "scores_splithalf.json", {
        idx: {"a": score, "b": score} for idx, score in scores.items()
    })
    offpolicy = {
        estimator: {idx: {"score": score} for idx, score in scores.items()}
        for estimator in ("g00", "g10", "g01", "g11")
    }
    if malformed:
        offpolicy["g10"].pop("19")
    write_json(run / "scores_offpolicy.json", offpolicy)
    write_json(run / "report.json", {"source": "test"})
    return run


def run_main(root: Path) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    previous = sys.argv
    sys.argv = ["readout_summary.py", str(root)]
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main()
    finally:
        sys.argv = previous
    return code, stdout.getvalue(), stderr.getvalue()


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    make_run(root, "v2-s0")
    make_run(root, "v3-s2-math500")
    make_run(root, "v4-27b-s1")
    make_run(root, "v4-7b-s1")
    make_run(root, "gate-corrected-no-done", done=False)
    historical = root / "v2-historical"
    historical.mkdir()
    (historical / "DONE").touch()
    code, stdout, stderr = run_main(root)
    check("readout includes corrected v3 runs", code == 0 and "v3-s2-math500" in stdout)
    check("automatic conclusions do not pool generations", "**v2/gsm8k**" in stdout and "**v3/math500**" in stdout)
    check("automatic conclusions do not pool v4 model families", (
        "**v4/27b/gsm8k**" in stdout
        and "**v4/7b/gsm8k**" in stdout
        and "**v4/gsm8k**" not in stdout
    ))
    check("protocol-complete run does not require legacy DONE", "gate-corrected-no-done" in stdout)
    check("historical runs are reported as excluded", "v2-historical" in stdout)
    check("successful readout has no stderr", stderr == "")


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    make_run(root, "v3-good")
    bad = make_run(root, "v3-bad", malformed=True)
    try:
        precisions(bad)
    except ScoreArtifactError:
        mismatch_rejected = True
    else:
        mismatch_rejected = False
    check("precision rejects partial estimator coverage", mismatch_rejected)
    code, stdout, stderr = run_main(root)
    check("one malformed corrected run fails the readout", code == 1)
    check("partial report names the malformed run", "v3-bad" in stdout)
    check("failure count is written to stderr", "산출물 오류 1개" in stderr)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    historical = root / "v2-only-historical"
    historical.mkdir()
    (historical / "DONE").touch()
    code, stdout, stderr = run_main(root)
    check("no corrected runs returns exit 2", code == 2)
    check("no corrected runs still emits a diagnostic report", "제외된 historical" in stdout)
    check("no corrected runs explains abort on stderr", "corrected protocol" in stderr)


print(("PASS" if FAILURES == 0 else "FAIL") + f" (failures {FAILURES})")


def test_readout_summary_regressions() -> None:
    assert FAILURES == 0


if __name__ == "__main__":
    sys.exit(1 if FAILURES else 0)
