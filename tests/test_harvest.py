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
  kcurve_all.py)
    case "${FAKE_KCURVE_ALL_MODE:-ok}" in
      fail) echo 'kcurve all exploded' >&2; exit 8;;
      empty) exit 0;;
      *) echo '# kcurve all'; exit 0;;
    esac
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
  frontier.py)
    mkdir -p "$OM_RESULTS"
    echo '# frontier' > "$OM_RESULTS/FRONTIER.md"
    echo '{}' > "$OM_RESULTS/frontier.json"
    echo "OM_RESULTS=$OM_RESULTS"
    exit 0
    ;;
  make_tables.py)
    mkdir -p "$OM_RESULTS"
    echo '# tables' > "$OM_RESULTS/TABLES.md"
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


def complete_v4_matrix(work: Path) -> None:
    for model in ("27b", "7b"):
        for seed in range(5):
            for suffix in ("", "-math500"):
                run = work / "runs" / f"v4-{model}-s{seed}{suffix}"
                run.mkdir(exist_ok=True)
                for artifact in (
                    "DONE",
                    "run_config.json",
                    "manifest.json",
                    "score_protocol.json",
                    "oracle_protocol.json",
                    "report.json",
                    "scores_oracle.json",
                    "scores_offpolicy.json",
                    "scores_splithalf.json",
                    "oracle_micro_groups.pt",
                    "val_groups.pt",
                ):
                    (run / artifact).write_text("{}\n", encoding="utf-8")
                (run / "divergence_stats.json").write_text("{}\n", encoding="utf-8")


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
        for name in (
            "KCURVE.md",
            "KCURVE_ALL.md",
            "READOUT.md",
            "REVERSAL.md",
            "STATS.md",
        )
    ))
    check("preregistered and all-generation kcurves stay separate", (
        first_dir / "KCURVE.md"
    ).read_text().strip() == "# kcurve" and (
        first_dir / "KCURVE_ALL.md"
    ).read_text().strip() == "# kcurve all")
    check("successful harvest records source status", (first_dir / "HARVEST_STATUS.md").exists())
    check("rapid harvests use distinct directories", second.returncode == 0 and first_dir != second_dir)


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    work, env = prepare(tmp, "good")
    partial = work / "runs" / "v4-27b-s1"
    partial.mkdir()
    (partial / "DONE").write_text("complete\n", encoding="utf-8")
    result, output = run_harvest(work, env)
    matrix_error = (output / "V4_MATRIX.err").read_text(encoding="utf-8")
    check("partial v4 matrix aborts final harvest", result.returncode == 1)
    check("v4 matrix failure is recorded", "v4-matrix:incomplete" in (
        output / "HARVEST_FAILURES.md"
    ).read_text(encoding="utf-8"))
    check("v4 matrix diagnostic names missing runs", "v4-7b-s0" in matrix_error)


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    work, env = prepare(tmp)
    complete_v4_matrix(work)
    result, output = run_harvest(work, env)
    check("complete 20-run v4 matrix passes harvest guard", result.returncode == 0)
    check("complete matrix leaves no matrix error file", not (
        output / "V4_MATRIX.err"
    ).exists())


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    work, env = prepare(tmp)
    missing = subprocess.run(
        ["bash", "scripts/collect_v4.sh"],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    check("collect_v4 reports experiments that have no result directory", (
        missing.returncode == 1
        and "[missing-run] v4-27b-s0" in missing.stderr
        and "결과 없는 run=20" in missing.stderr
    ))


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    work, env = prepare(tmp)
    complete_v4_matrix(work)
    result, output = run_report_script(work, env, "scripts/collect_v4.sh")
    check("collect_v4 completes without entering a GPU runner", result.returncode == 0)
    check("collect_v4 publishes separate 27B reports", all(
        (work / "results" / "v4-27b" / name).stat().st_size > 0
        for name in ("TABLES.md", "FRONTIER.md", "frontier.json")
    ))
    check("collect_v4 publishes separate 7B reports", all(
        (work / "results" / "v4-7b" / name).stat().st_size > 0
        for name in ("TABLES.md", "FRONTIER.md", "frontier.json")
    ))
    check("collect_v4 finishes with a valid harvest bundle", (
        output / "HARVEST_STATUS.md"
    ).exists())


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    work, env = prepare(tmp, "good")
    env["FAKE_KCURVE_ALL_MODE"] = "fail"
    result, output = run_harvest(work, env)
    check("all-generation kcurve failure aborts harvest", result.returncode == 1)
    check("failed all-generation kcurve is not published", not (
        output / "KCURVE_ALL.md"
    ).exists())
    check("all-generation kcurve stderr is preserved", "kcurve all exploded" in (
        output / "KCURVE_ALL.err"
    ).read_text())


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    work, env = prepare(tmp, "good")
    result_dir = work / "results" / "v4-27b"
    result_dir.mkdir(parents=True)
    (result_dir / "TABLES.md").touch()
    (result_dir / "FRONTIER.md").write_text("# frontier\n", encoding="utf-8")
    result, output = run_harvest(work, env)
    failures = (output / "HARVEST_FAILURES.md").read_text(encoding="utf-8")
    check("empty existing result report aborts harvest", result.returncode == 1)
    check("empty result report is named in failure manifest", (
        "tables:v4-27b:empty-source" in failures
    ))
    check("valid sibling result report is still copied", (
        output / "FRONTIER-v4-27b.md"
    ).read_text(encoding="utf-8") == "# frontier\n")


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
