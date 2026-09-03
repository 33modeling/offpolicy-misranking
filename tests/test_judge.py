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
from gate_rules import canonical_gate_report  # noqa: E402
from judge import judge  # noqa: E402

FAIL = 0


def check(name, condition):
    global FAIL
    print(("  ok  " if condition else "FAIL  ") + name)
    if not condition:
        FAIL += 1


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value))


def mark_score_protocol(
    run: Path,
    *,
    g10_fail: bool = True,
    g01_fail: bool = True,
) -> None:
    write_json(run / "score_protocol.json", {
        "schema": "offpolicy-score-validation-split/v2",
        "generation_validation": {"validated_rows": 1},
    })
    write_json(run / "oracle_protocol.json", {
        "schema": "offpolicy-oracle-validation-split/v3",
        "generation_validation": {"validated_rows": 1},
    })
    oracle_scores = {str(i): float(20 - i) for i in range(20)}
    reversed_scores = {str(i): float(i) for i in range(20)}
    write_json(run / "scores_oracle.json", {
        idx: {"score": score} for idx, score in oracle_scores.items()
    })
    write_json(run / "scores_splithalf.json", {
        idx: {"a": score, "b": score} for idx, score in oracle_scores.items()
    })
    write_json(run / "scores_offpolicy.json", {
        "g00": {idx: {"score": score} for idx, score in oracle_scores.items()},
        "g10": {
            idx: {"score": score}
            for idx, score in (reversed_scores if g10_fail else oracle_scores).items()
        },
        "g01": {
            idx: {"score": score}
            for idx, score in (reversed_scores if g01_fail else oracle_scores).items()
        },
        "g11": {idx: {"score": score} for idx, score in oracle_scores.items()},
    })


def mark_hybrid_protocol(run: Path, cut: str) -> None:
    write_json(run / f"hybrid_protocol_{cut}.json", {
        "schema": "offpolicy-hybrid-validation-split/v2",
    })


def report(c2_pass: bool = True, one_sided_fail: bool = True,
           g10_fail: bool | None = None, g01_fail: bool | None = None) -> dict:
    g10_fail = one_sided_fail if g10_fail is None else g10_fail
    g01_fail = one_sided_fail if g01_fail is None else g01_fail
    return {
        "noise_floor": 0.8,
        "k": 1,
        "g00": {"precision": 0.5, "jaccard": 0.3},
        "g10": {"precision": 0.5 if g10_fail else 0.8, "jaccard": 0.3},
        "g01": {"precision": 0.5 if g01_fail else 0.8, "jaccard": 0.3},
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
    mark_score_protocol(run)
    write_json(run / "report.json", report())
    recovering_cells = {
        "bb": {"0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0},
        "bp": {"0": 0.0, "1": 1.0, "2": 4.0, "3": 2.0},
        "pb": {"0": 0.0, "1": 4.0, "2": 1.0, "3": 2.0},
        "pp": {"0": 4.0, "1": 3.0, "2": 2.0, "3": 1.0},
    }
    write_json(run / "scores_hybrid_0.5.json", recovering_cells)
    mark_hybrid_protocol(run, "0.5")
    verdicts = run_judge(root)
    check("hybrid requires and accepts recovery of both missing axes",
          verdicts["C1'_hybrid"] is True)
    check("joint C1 is explicit", verdicts["C1_joint"] is True)

    tied_cells = dict(recovering_cells)
    tied_cells["pp"] = dict(tied_cells["pb"])
    write_json(run / "scores_hybrid_0.5.json", tied_cells)
    verdicts = run_judge(root)
    check("hybrid precision tie is not recovery", verdicts["C1'_hybrid"] is False)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    run = root / "drift100"
    run.mkdir()
    mark_score_protocol(run)
    write_json(run / "report.json", report())
    correct = {"0": 4.0, "1": 3.0, "2": 2.0, "3": 1.0}
    wrong = {"0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0}
    write_json(run / "scores_hybrid_0.3.json", {
        "bb": wrong, "bp": correct, "pb": wrong, "pp": correct,
    })
    write_json(run / "scores_hybrid_0.7.json", {
        "bb": wrong, "bp": wrong, "pb": correct, "pp": correct,
    })
    mark_hybrid_protocol(run, "0.3")
    mark_hybrid_protocol(run, "0.7")
    verdicts = run_judge(root)
    check("axis recoveries from different cuts do not form a causal witness",
          verdicts["C1'_hybrid"] is not True)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for drift, c2_pass in ((50, True), (100, False)):
        run = root / f"drift{drift}"
        run.mkdir()
        mark_score_protocol(run)
        write_json(run / "report.json", report(c2_pass=c2_pass))
    verdicts = run_judge(root)
    check("C1 keeps any-drift semantics", verdicts["C1_g10"] is True)
    check("C2 fails when any completed drift fails", verdicts["C2_certagrad"] is False)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    run50 = root / "drift50"
    run100 = root / "drift100"
    run50.mkdir()
    run100.mkdir()
    mark_score_protocol(run50, g10_fail=True, g01_fail=False)
    mark_score_protocol(run100, g10_fail=False, g01_fail=True)
    write_json(run50 / "report.json", report(g10_fail=True, g01_fail=False))
    write_json(run100 / "report.json", report(g10_fail=False, g01_fail=True))
    verdicts = run_judge(root)
    check("axis failures from different runs do not form joint C1",
          verdicts["C1_g10"] is True and verdicts["C1_g01"] is True
          and verdicts["C1_joint"] is False)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for drift, oracle_acc in ((50, 0.8), (100, 0.6)):
        run = root / f"drift{drift}"
        run.mkdir()
        mark_score_protocol(run)
        write_json(run / "report.json", report())
        write_json(run / "downstream_oracle.json", {"base_acc": 0.5, "val_acc": oracle_acc})
        write_json(run / "downstream_random.json", {"base_acc": 0.5, "val_acc": 0.7})
    verdicts = run_judge(root)
    check("C3 fails when any reported downstream comparison fails",
          verdicts["C3_downstream"] is False)


with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp)
    write_json(run / "score_protocol.json", {
        "schema": "offpolicy-score-validation-split/v2",
        "generation_validation": {"validated_rows": 1},
    })
    write_json(run / "report.json", report())
    check("missing oracle protocol fails closed", canonical_gate_report(run) is None)
    (run / "oracle_protocol.json").write_text("{malformed")
    check("malformed oracle protocol fails closed", canonical_gate_report(run) is None)


with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp)
    mark_score_protocol(run)
    write_json(run / "report.json", report())
    offpolicy = json.loads((run / "scores_offpolicy.json").read_text())
    offpolicy["g10"].pop("19")
    write_json(run / "scores_offpolicy.json", offpolicy)
    check("partial score coverage cannot fall back to stored report",
          canonical_gate_report(run) is None)


print(("PASS" if FAIL == 0 else "FAIL") + f" (failures {FAIL})")


def test_judge_regressions() -> None:
    assert FAIL == 0


if __name__ == "__main__":
    sys.exit(1 if FAIL else 0)
