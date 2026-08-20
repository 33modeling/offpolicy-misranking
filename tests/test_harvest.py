"""End-to-end shell regressions for atomic harvest publication."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FAILURES = 0


def check(name: str, condition: bool) -> None:
    global FAILURES
    print(("  ok  " if condition else "FAIL  ") + name)
    if not condition:
        FAILURES += 1


def prepare(tmp: Path, *run_names: str) -> tuple[Path, dict[str, str]]:
    work = tmp / "work"
    runs = work / "runs"
    runs.mkdir(parents=True)
    for name in run_names:
        run = runs / name
        run.mkdir()
        for artifact in (
            "scores_oracle.json",
            "score_protocol.json",
            "oracle_protocol.json",
        ):
            (run / artifact).write_text("{}", encoding="utf-8")

    venv = tmp / "venv"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        """#!/usr/bin/env bash
name=$(basename "$1")
case "$name" in
  kcurve_floor.py)
    echo '# kcurve'
    exit "${FAKE_KCURVE_RC:-3}"
    ;;
  readout_summary.py)
    case "${FAKE_READOUT_MODE:-ok}" in
      fail) echo 'readout exploded' >&2; exit 7;;
      empty) exit 0;;
      *) echo '# readout'; exit 0;;
    esac
    ;;
  reversal_freq.py)
    echo '# reversal'
    exit 0
    ;;
  stats_extra.py)
    if [ "$(basename "$2")" = bad ]; then
      echo 'bad stats' >&2
      exit 9
    fi
    echo 'run=good n=20 k=2'
    echo 'g00 1.0'
    echo 'g10 1.0'
    echo 'g01 1.0'
    echo 'g11 1.0'
    exit 0
    ;;
  frontier.py|make_tables.py)
    echo "OM_RESULTS=$OM_RESULTS"
    exit 0
    ;;
  *)
    echo "unexpected script: $1" >&2
    exit 99
    ;;
esac
""",
        encoding="utf-8",
    )
    python.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "GROUP_VOLUME": str(tmp / "missing-group-volume"),
        "OM_REPO": str(REPO),
        "OM_WORK": str(work),
        "VENV_DIR": str(venv),
        "MODELS_DIR": str(tmp / "models"),
    })
    return work, env


def run_harvest(work: Path, env: dict[str, str]) -> tuple[subprocess.CompletedProcess, Path]:
    return run_report_script(work, env, "scripts/harvest.sh")


def run_report_script(
    work: Path,
    env: dict[str, str],
    script: str,
) -> tuple[subprocess.CompletedProcess, Path]:
    readouts = work / "readouts"
    before = set(readouts.iterdir()) if readouts.exists() else set()
    result = subprocess.run(
        ["bash", script],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    created = set(readouts.iterdir()) - before
    if len(created) != 1:
        raise AssertionError(f"harvest created {len(created)} directories: {created}")
    return result, created.pop()


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    work, env = prepare(tmp, "good")
    first, first_dir = run_harvest(work, env)
    second, second_dir = run_harvest(work, env)
    check("scientific kcurve exit 3 is accepted", first.returncode == 0)
    check("successful markdown outputs are nonempty", all(
        (first_dir / name).stat().st_size > 0
        for name in ("KCURVE.md", "READOUT.md", "REVERSAL.md", "STATS.md")
    ))
    check("successful harvest records source status", (first_dir / "HARVEST_STATUS.md").exists())
    check("rapid harvests use distinct directories", second.returncode == 0 and first_dir != second_dir)


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    work, env = prepare(tmp, "good")
    env["FAKE_READOUT_MODE"] = "fail"
    result, output = run_harvest(work, env)
    check("readout process failure aborts harvest", result.returncode == 1)
    check("failed readout is never published", not (output / "READOUT.md").exists())
    check("readout stderr is preserved", "readout exploded" in (output / "READOUT.err").read_text())
    check("harvest failure manifest is written", (output / "HARVEST_FAILURES.md").exists())


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    work, env = prepare(tmp, "good")
    env["FAKE_READOUT_MODE"] = "empty"
    result, output = run_harvest(work, env)
    check("empty successful stdout is rejected", result.returncode == 1)
    check("zero-byte final readout cannot be created", not (output / "READOUT.md").exists())
    check("empty stdout is retained only as partial", (output / "READOUT.partial.md").exists())


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    work, env = prepare(tmp, "good", "bad")
    result, output = run_harvest(work, env)
    partial = (output / "STATS.partial.md").read_text(encoding="utf-8")
    check("one stats run failure aborts harvest", result.returncode == 1)
    check("partial stats are not published as final", not (output / "STATS.md").exists())
    check("successful stats remain diagnostic-only", "## good" in partial and "## bad" not in partial)
    check("failed stats stderr is preserved", "bad stats" in (output / "STATS.err").read_text())


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    work, env = prepare(tmp, "good")
    env["FAKE_READOUT_MODE"] = "empty"
    result, output = run_report_script(work, env, "scripts/read_now.sh")
    check("read_now also rejects an empty readout", result.returncode == 1)
    check("read_now never publishes a zero-byte final", not (output / "READOUT.md").exists())
    check("read_now preserves empty stdout as partial", (output / "READOUT.partial.md").exists())


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    work, env = prepare(tmp, "good")
    result, output = run_report_script(work, env, "scripts/kcurve.sh")
    check("standalone kcurve accepts scientific exit 3", result.returncode == 0)
    check("standalone kcurve publishes nonempty output", (output / "KCURVE.md").stat().st_size > 0)


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    work, env = prepare(tmp, "v3-s0")
    (work / "runs" / "v3-s0" / "DONE").touch()
    frontier = subprocess.run(
        ["bash", "scripts/frontier.sh"],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    tables = subprocess.run(
        ["bash", "scripts/tables.sh"],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    expected = f"OM_RESULTS={work}/results/v3"
    check("frontier output follows its single generation", frontier.returncode == 0 and expected in frontier.stdout)
    check("tables output follows its single generation", tables.returncode == 0 and expected in tables.stdout)


print(("PASS" if FAILURES == 0 else "FAIL") + f" (failures {FAILURES})")


def test_harvest_regressions() -> None:
    assert FAILURES == 0


if __name__ == "__main__":
    sys.exit(1 if FAILURES else 0)
