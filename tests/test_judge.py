"""Regression tests for the artifact-only gate judge (no model/GPU required).

    PYTHONPATH=src python3 tests/test_judge.py
"""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from judge import judge  # noqa: E402


FAIL = 0


def check(name, condition):
    global FAIL
    print(("  ok  " if condition else "FAIL  ") + name)
    if not condition:
        FAIL += 1


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value))


def report(c2_pass: bool = True, one_sided_fail: bool = True) -> dict:
    precision = 0.5 if one_sided_fail else 0.8
    return {
        "noise_floor": 0.8,
        "k": 1,
        "g00": {"precision": 0.5, "jaccard": 0.3},
        "g10": {"precision": precision, "jaccard": 0.3},
        "g01": {"precision": precision, "jaccard": 0.3},
        "g11": {"precision": 1.0, "jaccard": 1.0},
        "certagrad": {
            "certified": True,
            "fresh_frac_of_uniform": 0.4 if c2_pass else 0.6,
            "precision_vs_oracle": 0.8,
            "uniform_precision_vs_oracle": 0.8,
        },
    }


def run_judge(root: Path) -> dict[str, bool | None]:
    with contextlib.redirect_stdout(io.StringIO()):
        return judge(root)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    run = root / "drift100"
    run.mkdir()
    write_json(run / "report.json", report())
    write_json(run / "scores_oracle.json", {
        "0": {"score": 4.0}, "1": {"score": 3.0},
        "2": {"score": 2.0}, "3": {"score": 1.0},
    })
    recovering_cells = {
        "bb": {"0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0},
        "bp": {"0": 0.0, "1": 1.0, "2": 4.0, "3": 2.0},
        "pb": {"0": 0.0, "1": 4.0, "2": 1.0, "3": 2.0},
        "pp": {"0": 4.0, "1": 3.0, "2": 2.0, "3": 1.0},
    }
    write_json(run / "scores_hybrid_0.5.json", recovering_cells)
    verdicts = run_judge(root)
    check("hybrid requires and accepts recovery of both missing axes",
          verdicts["C1'_hybrid"] is True)

    tied_cells = dict(recovering_cells)
    tied_cells["pp"] = dict(tied_cells["pb"])
    write_json(run / "scores_hybrid_0.5.json", tied_cells)
    verdicts = run_judge(root)
    check("hybrid precision tie is not recovery", verdicts["C1'_hybrid"] is False)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for drift, c2_pass in ((50, True), (100, False)):
        run = root / f"drift{drift}"
        run.mkdir()
        write_json(run / "report.json", report(c2_pass=c2_pass))
    verdicts = run_judge(root)
    check("C1 keeps any-drift semantics", verdicts["C1_g10"] is True)
    check("C2 fails when any completed drift fails", verdicts["C2_certagrad"] is False)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for drift, oracle_acc in ((50, 0.8), (100, 0.6)):
        run = root / f"drift{drift}"
        run.mkdir()
        write_json(run / "report.json", report())
        write_json(run / "downstream_oracle.json", {"base_acc": 0.5, "val_acc": oracle_acc})
        write_json(run / "downstream_random.json", {"base_acc": 0.5, "val_acc": 0.7})
    verdicts = run_judge(root)
    check("C3 fails when any reported downstream comparison fails",
          verdicts["C3_downstream"] is False)


print(("PASS" if FAIL == 0 else "FAIL") + f" (failures {FAIL})")
sys.exit(1 if FAIL else 0)
