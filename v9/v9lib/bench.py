"""X12 public-benchmark evaluation of retained snapshots (no training, no optimizer step).

Suites (pinned Hugging Face revisions, stored in v9/bench_data so the cluster downloads nothing):
  MATH cohorts  math500  HuggingFaceH4/MATH-500 test (500), 4 responses per question
                aime24   HuggingFaceH4/aime_2024 (30), 16 responses
                aime25   MathArena/aime_2025 (30), 16 responses
                amc23    math-ai/amc23 test (40), 16 responses
  MBPP cohorts  humaneval openai/openai_humaneval test (164), 8 responses; the tests of check(candidate) are
                rewritten as equality assertions on the entry point (the MBPP verifier format); a problem whose
                tests are not literal equality assertions is left out, and the count is recorded.

Prompts use the cohort's training template (srgc_rebuttal.build_inputs MATH_PROMPT / CODE_PROMPT), the
verifier is the cohort plan's verifier, generation is the evaluation generation of the frozen backend
(temperature 1, 2048 new tokens). Every kept question passes a gold self-check with that verifier at
preparation (MATH: the boxed gold answer; HumanEval: the canonical solution). Questions whose normalized text
equals a question of any input bundle of the cohort are removed (recorded). The suite file is built by the first
benchmark task of a cohort inside the GPU child (the verifier runtime of the plan is ready there) and written once
(v9/<cohort>/bench/suites/<suite>.json); every result records its question-set hash.

Result: v9/<cohort>/bench/results/<snapshot16>__<suite>__<spec12>.json, write-once, integer correct counts per
question, reward_fraction = mean over questions and responses (avg@k).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

BENCH_SPEC = "x12-public-bench-v1"
SUITES = {
    "math500": {"kind": "math", "repo": "HuggingFaceH4/MATH-500", "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
                "file": "test.jsonl", "question": "problem", "answer": "answer", "responses": 4},
    "aime24": {"kind": "math", "repo": "HuggingFaceH4/aime_2024", "revision": "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
               "file": "data/train-00000-of-00001.parquet", "question": "problem",
               "answer": "answer", "responses": 16},
    "aime25": {"kind": "math", "repo": "MathArena/aime_2025", "revision": "c94da77eb22bbd6439e62a323bec18493a421302",
               "file": "data/train-00000-of-00001.parquet", "question": "problem",
               "answer": "answer", "responses": 16},
    "amc23": {"kind": "math", "repo": "math-ai/amc23", "revision": "80815d37005feb82cd7f8fbc6901d5d3eff43057",
              "file": "test-00000-of-00001.parquet", "question": "question", "answer": "answer",
              "responses": 16},
    "humaneval": {"kind": "code", "repo": "openai/openai_humaneval", "revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                  "file": "openai_humaneval/test-00000-of-00001.parquet", "responses": 8},
}
KIND_SUITES = {"math": ("math500", "aime24", "aime25", "amc23"), "code": ("humaneval",)}
FAKE_SIZE = 6


def cohort_kind(cohort):
    return "code" if cohort.endswith("_mbpp") else "math"


def suites_for(cohort):
    return KIND_SUITES[cohort_kind(cohort)]


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------ source rows
DATA_DIR = Path(__file__).resolve().parents[1] / "bench_data"


def raw_rows(suite):
    """Rows of the pinned source file, stored in this repository (v9/bench_data/<suite>.jsonl: the needed columns of
    the Hub file at the pinned revision; MANIFEST.json holds the source file's and the JSONL's SHA-256)."""
    manifest = json.loads((DATA_DIR / "MANIFEST.json").read_text())[suite]
    text = (DATA_DIR / f"{suite}.jsonl").read_text()
    if hashlib.sha256(text.encode()).hexdigest() != manifest["jsonl_sha256"]:
        raise ValueError(f"v9/bench_data/{suite}.jsonl differs from MANIFEST.json")
    return [json.loads(line) for line in text.splitlines() if line.strip()], manifest


def math_gold(ans):
    """Integer answers as integers ('27.0' -> '27', AIME '025' -> '25'); other answers unchanged."""
    ans = str(ans).strip()
    if re.fullmatch(r"-?\d+(\.0+)?", ans):
        return str(int(float(ans)))
    return ans


class _Rename(ast.NodeTransformer):
    def __init__(self, name):
        self.name = name

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self.name, ctx=node.ctx), node) if node.id == "candidate" else node


def humaneval_tests(test_src, entry_point):
    """check(candidate) body -> 'assert f(...) == literal' lines; None if any statement is not of that form."""
    tree = ast.parse(test_src)
    checks = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check"]
    if len(checks) != 1:
        return None
    lines = []
    for st in checks[0].body:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
            continue  # docstring
        if not isinstance(st, ast.Assert):
            return None
        t = st.test
        if isinstance(t, ast.Constant) and t.value is True:
            continue
        if not (isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)):
            return None
        try:
            ast.literal_eval(t.comparators[0])
        except (ValueError, SyntaxError, TypeError, RecursionError):
            return None
        left = _Rename(entry_point).visit(t.left)
        lines.append(f"assert {ast.unparse(left)} == {ast.unparse(t.comparators[0])}")
    return "\n".join(lines) if lines else None


def suite_rows(suite):
    """[(source_index, question, gold answer, self-check response)] and drop counts."""
    from srgc_rebuttal.build_inputs import normalized  # noqa: F401  (same normalization as the bundles)
    s = SUITES[suite]
    rows, manifest = raw_rows(suite)
    out, dropped = [], {}
    for k, r in enumerate(rows):
        if s["kind"] == "math":
            ans = math_gold(r[s["answer"]])
            if not ans:
                dropped["empty_answer"] = dropped.get("empty_answer", 0) + 1
                continue
            gold = "$\\boxed{" + ans + "}$"
            out.append((k, str(r[s["question"]]), gold, f"The answer is below.\nAnswer: {gold}"))
        else:
            tests = humaneval_tests(r["test"], r["entry_point"])
            if tests is None:
                dropped["tests_not_literal_equality"] = dropped.get("tests_not_literal_equality", 0) + 1
                continue
            q = (f"Write a Python function for the task below.\n\n```python\n{r['prompt']}```\n\n"
                 f"Your code should satisfy these tests:\n{tests}\n\n"
                 "Return the complete function in a ```python code block.")
            out.append((k, q, tests, f"```python\n{r['prompt']}{r['canonical_solution']}\n```"))
    return out, dropped, {"repo": s["repo"], "revision": s["revision"], "file": s["file"], "n_source_rows": len(rows),
                          "source_sha256": manifest["source_sha256"], "jsonl_sha256": manifest["jsonl_sha256"]}


def build_suite(suite, *, verifier, bundle_questions):
    """Records of one suite for one cohort: template, contamination filter, gold self-check with the verifier."""
    from srgc_rebuttal.build_inputs import CODE_PROMPT, MATH_PROMPT, normalized
    s = SUITES[suite]
    template = MATH_PROMPT if s["kind"] == "math" else CODE_PROMPT
    rows, dropped, source = suite_rows(suite)
    seen = {normalized(q) for q in bundle_questions}
    records, ids = {}, []
    for k, q, gold, check in rows:
        if normalized(q) in seen:
            dropped["in_training_bundle"] = dropped.get("in_training_bundle", 0) + 1
            continue
        rec = {"question": q, "prompt": template.format(question=q), "answer": gold}
        try:
            ok = float(verifier(rec, check)) == 1.0
        except Exception:  # noqa: BLE001 - an ungradable gold is left out and counted
            ok = False
        if not ok:
            dropped["gold_self_check_failed"] = dropped.get("gold_self_check_failed", 0) + 1
            continue
        rid = f"{suite}-{k:04d}"
        records[rid] = rec
        ids.append(rid)
    return {"suite": suite, "spec": BENCH_SPEC, "kind": s["kind"], "responses": s["responses"], "source": source,
            "dropped": dropped, "n_questions": len(ids), "question_ids": ids, "question_ids_hash": sha256_json(ids),
            "records": records}


def fake_suite(suite, data):
    """Local CPU check only: FAKE_SIZE evaluation questions of the bundle under bench ids (alias = bundle id)."""
    s = SUITES[suite]
    src = list(data["evaluation_ids"])[:FAKE_SIZE]
    records = {f"{suite}-{k:04d}": dict(data["records"][i], alias=i) for k, i in enumerate(src)}
    ids = list(records)
    return {"suite": suite, "spec": BENCH_SPEC, "kind": s["kind"], "responses": s["responses"],
            "source": {"repo": "FAKE", "note": "local software check, bundle evaluation questions"},
            "dropped": {}, "n_questions": len(ids), "question_ids": ids, "question_ids_hash": sha256_json(ids),
            "records": records, "fake": True}


# ------------------------------------------------------------------ GPU task (inside the child)
def result_path(out_root, task):
    return Path(out_root) / "bench" / "results" / (
        f"{task['snapshot_sha256'][:16]}__{task['suite']}__{task['bench_spec_hash'][:12]}.json")


def _attach(backend, records):
    """Make the suite questions known to the backend; returns an undo function."""
    if hasattr(backend, "records"):
        before = backend.records
        backend.records = {**dict(before), **{i: {k: v for k, v in r.items() if k != "alias"} for i, r in records.items()}}

        def undo():
            backend.records = before
        return undo
    if hasattr(backend, "features"):  # ToyBackend (local checks): bench ids alias bundle questions
        for i, r in records.items():
            backend.features[i] = backend.features[r["alias"]]
            backend.answers[i] = backend.answers[r["alias"]]
        return lambda: None
    raise TypeError(f"backend {type(backend).__name__} has no prompt records")


def suite_path(out_root, suite):
    return Path(out_root) / "bench" / "suites" / f"{suite}.json"


def ensure_suite(task, ctx):
    """Rank 0 builds the cohort's suite file when it is missing; every rank then reads the same file."""
    from switching.cluster import load_bundle
    from switching.common import atomic_write_json, read_json
    path = suite_path(ctx.out_root, task["suite"])

    def build():
        if path.exists():
            return True
        if task.get("fake"):
            suite = fake_suite(task["suite"], ctx.data)
        else:
            questions = set()
            for b in task["bundles"]:
                questions.update(str(r.get("question", "")) for r in load_bundle(b)["records"].values())
            suite = build_suite(task["suite"], verifier=ctx.backend.verifier, bundle_questions=questions)
        if not path.exists():  # another node may have written it meanwhile; the first file stays
            atomic_write_json(path, suite)
        return True
    ctx.broadcast(ctx.primary(build))
    return read_json(path)


def execute_bench(task, ctx):
    from switching import snapshots
    from switching.common import new_id, utc_now, write_once_json
    from switching.ledger import Ledger
    from switching.runner import integer_counts
    path = result_path(ctx.out_root, task)
    if ctx.broadcast(ctx.primary(lambda: path.exists())):
        return {"status": "skipped_complete", "path": str(path)}
    suite = ensure_suite(task, ctx)
    ids = list(suite["question_ids"])
    if sha256_json(ids) != suite["question_ids_hash"] or suite["responses"] != task["responses"]:
        raise ValueError("benchmark suite file is inconsistent with its question list or the task")
    state = snapshots.load_engine_state(task["snapshot_path"], task["snapshot_sha256"])
    if state["step"] != task["completed_update"]:
        raise ValueError(f"snapshot holds update {state['step']}, task expects {task['completed_update']}")
    ctx.backend.load_state_dict(state["backend"])
    undo = _attach(ctx.backend, suite["records"])
    attempt = ctx.broadcast(ctx.primary(new_id))
    ledger = Ledger(Path(ctx.out_root) / "bench" / "events" / f"{attempt}.jsonl", run_id=task["run_id"],
                    attempt_id=attempt, gpu_type=ctx.gpu_type, measurement_status=ctx.measurement_status
                    ) if ctx.is_primary else None
    ctx.event_sink = ledger.on_meter_event if ledger else None
    seed = ctx.srgc.stream_seed(task["training_seed"], task["completed_update"], f"x12-{task['suite']}")
    counting = getattr(ctx.backend, "evaluate_counts", None)
    n = task["responses"]

    def run():
        if counting is not None:
            return dict(counting(ids, seed=seed, responses=n)), "integer_counts"
        return dict(ctx.backend.evaluate(ids, seed=seed, responses=n)), "per_question_mean"
    try:
        if ctx.meter is not None:
            with ctx.meter.phase("evaluation", task["completed_update"], ctx.backend.gpu_count):
                values, source = run()
        else:
            values, source = run()
    finally:
        ctx.event_sink = None
        undo()
    if set(values) != set(ids):
        raise ValueError("benchmark evaluation returned a different question set")
    correct = dict(values) if source == "integer_counts" else integer_counts(values, n)
    record = {"run_id": task["run_id"], "checkpoint_hash": task["snapshot_sha256"],
              "completed_update": task["completed_update"], "suite": task["suite"], "bench_spec": BENCH_SPEC,
              "bench_spec_hash": task["bench_spec_hash"], "source": suite["source"], "n_questions": len(correct),
              "question_ids_hash": suite["question_ids_hash"], "dropped": suite["dropped"],
              "responses_per_question": n, "reward_fraction": sum(correct.values()) / (len(correct) * n),
              "per_question_correct": correct, "count_source": source,
              "verifier": ctx.verifier_identity if ctx.verifier_identity is not None else task.get("verifier"),
              "measurement_status": ctx.measurement_status, "fake": bool(suite.get("fake")),
              "attempt_id": attempt, "evaluated_at": utc_now()}
    ctx.primary(lambda: write_once_json(path, record))
    return {"status": "evaluated", "path": str(path), "reward_fraction": record["reward_fraction"]}
