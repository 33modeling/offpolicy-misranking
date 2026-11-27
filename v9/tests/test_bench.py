"""X12 benchmark suites: pinned files match their manifest, HumanEval tests become MBPP-format equality assertions,
integer gold answers are normalized, contamination with the training bundles and failed gold self-checks drop
questions (counted), and the suite file does not depend on the order of the bundle questions."""

import ast

from v9lib import bench


def test_pinned_files_match_manifest_and_sizes():
    sizes = {"math500": 500, "aime24": 30, "aime25": 30, "amc23": 40, "humaneval": 164}
    for suite, n in sizes.items():
        rows, manifest = bench.raw_rows(suite)
        assert len(rows) == n == manifest["rows"]


def test_humaneval_tests_are_literal_equality_on_entry_point():
    src = "METADATA = {}\n\ndef check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n    assert True\n"
    out = bench.humaneval_tests(src, "add")
    assert out == "assert add(1, 2) == 3\nassert add(0, 0) == 0"
    for line in out.splitlines():
        t = ast.parse(line).body[0].test
        assert isinstance(t, ast.Compare) and isinstance(t.ops[0], ast.Eq)
    assert bench.humaneval_tests("def check(candidate):\n    assert abs(candidate(1) - 2) < 1e-6\n", "f") is None
    assert bench.humaneval_tests("def check(candidate):\n    for x in [1]:\n        assert candidate(x) == 1\n", "f") is None


def test_math_gold_normalization():
    assert bench.math_gold("27.0") == "27"
    assert bench.math_gold("025") == "25"
    assert bench.math_gold("-1.0") == "-1"
    assert bench.math_gold("\\frac{1}{2}") == "\\frac{1}{2}"


def test_build_suite_drops_contaminated_and_ungradable():
    rows, _ = bench.raw_rows("aime24")
    calls = []

    def verifier(rec, response):
        calls.append(rec["answer"])
        return 0.0 if rec["question"] == rows[1]["problem"] else 1.0
    s = bench.build_suite("aime24", verifier=verifier, bundle_questions=[rows[0]["problem"]])
    assert s["n_questions"] == 28
    assert s["dropped"] == {"in_training_bundle": 1, "gold_self_check_failed": 1}
    assert all(r["prompt"].startswith("Solve the following problem") for r in s["records"].values())
    assert s["question_ids_hash"] == bench.sha256_json(s["question_ids"])
    again = bench.build_suite("aime24", verifier=verifier, bundle_questions=[rows[0]["problem"]][::-1])
    assert bench.sha256_json(again) == bench.sha256_json(s)


def test_humaneval_suite_keeps_literal_problems():
    s = bench.build_suite("humaneval", verifier=lambda rec, resp: 1.0, bundle_questions=[])
    assert s["n_questions"] == 148 and s["dropped"] == {"tests_not_literal_equality": 16}
    rec = next(iter(s["records"].values()))
    assert rec["answer"].startswith("assert ") and "Your code should satisfy these tests" in rec["question"]
