"""Rescoring rewrites rewards consistently, keeps the pinned evidence, and refuses
to touch a family that is held or that it cannot reproduce."""

from __future__ import annotations

import fcntl
import hashlib
import json
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import rescore_rollouts as rr  # noqa: E402

TAG = "tag"


class FakeTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class FakeData:
    @staticmethod
    def extract_answer(text: str):
        return text.rsplit("#### ", 1)[1].strip() if "#### " in text else None


def only_exact(prediction: str, gold: str) -> float:
    return 0.0


def generous(prediction: str, gold: str) -> float:
    return 1.0 if prediction.replace(" ", "") == gold.replace(" ", "") else 0.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_point(root: Path, seed: int, drift: int, rows: list[tuple[int, str, str, float]], *, shards: bool) -> Path:
    """rows: (prompt_idx, text, gold, stored reward). `shards` also writes two shard files."""
    run = root / f"family-math500-s{seed}" / f"{TAG}-s{seed}-math500-d{drift}"
    (run / "logs").mkdir(parents=True)
    (run / "run_config.json").write_text(json.dumps({"dataset": "math500", "model_resolved": str(root / "model")}))
    answers: dict[int, str] = {}
    lines = []
    for prompt_idx, text, gold, stored in rows:
        answers[prompt_idx] = gold
        ids = [ord(c) for c in text]
        lines.append(json.dumps({"prompt_idx": prompt_idx, "rollout_idx": 0, "input_ids": ids,
                                 "resp_start": 0, "resp_end": len(ids), "reward": stored}))
    train = [{"question": f"q{i}", "answer": answers.get(i, "0")} for i in range(max(answers) + 1)]
    (run / "prompts.json").write_text(json.dumps({"train": train, "val": []}))
    merged = run / "rollouts_fresh_train.jsonl"
    merged.write_text("\n".join(lines) + "\n")
    if shards:
        half = len(lines) // 2
        for index, chunk in enumerate((lines[:half], lines[half:])):
            shard = run / f"rollouts_fresh_train.shard{index}.jsonl"
            shard.write_text("\n".join(chunk) + "\n")
            (run / f"rollouts_fresh_train.shard{index}.manifest.json").write_text(
                json.dumps({"artifact_file": shard.name, "artifact_sha256": sha(shard), "k": 1, "n_prompts": len(chunk), "idx_offset": index * half})
            )
    else:
        (run / "rollouts_fresh_train.manifest.json").write_text(
            json.dumps({"artifact_file": merged.name, "artifact_sha256": sha(merged), "k": 1, "n_prompts": len(lines), "idx_offset": 0})
        )
    for name in ("scores_oracle.json", "report.json", "val_gradient.pt", "DONE", ".regime_validated.json"):
        (run / name).write_text("pinned\n")
    return run


@pytest.fixture
def fake_transformers(monkeypatch):
    module = type(sys)("transformers")
    module.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_rewrite_is_consistent_resealed_and_keeps_the_pinned_evidence(tmp_path: Path, fake_transformers) -> None:
    root = tmp_path / "root"
    rows = [
        (0, "#### 2 x", "2x", 0.0),   # corrected verifier accepts: 0 -> 1
        (1, "#### 4", "4", 1.0),      # exact match, unchanged
        (2, "#### 9", "7", 0.0),      # wrong under both
        (3, "#### 1 + x", "1+x", 0.0),
    ]
    for drift in (0, 25):
        write_point(root, 0, drift, rows, shards=(drift == 25))
    (root / "family-math500-s0" / ".family-complete").write_text("stamp\n")

    result = rr.rescore_family(root, "math500", 0, [0, 25], TAG, data=FakeData(),
                               old_verifier=only_exact, new_verifier=generous, apply=True)
    assert result["family"] == "math500/s0"
    for point in result["points"]:
        for entry in point["files"]:
            assert entry["rows"] in (2, 4) and entry["flip_1_to_0"] == 0
        assert "DONE" in point["retired"] and "report.json" in point["retired"]

    d25 = root / "family-math500-s0" / f"{TAG}-s0-math500-d25"
    merged_rows = [json.loads(l) for l in (d25 / "rollouts_fresh_train.jsonl").read_text().splitlines()]
    assert [r["reward"] for r in merged_rows] == [1.0, 1.0, 0.0, 1.0]
    assert [r["reward_pinned"] for r in merged_rows] == [0.0, 1.0, 0.0, 0.0]
    # shards carry the same rows and their manifests bind the rewritten bytes
    shard_rows = []
    for index in (0, 1):
        shard = d25 / f"rollouts_fresh_train.shard{index}.jsonl"
        manifest = json.loads((d25 / f"rollouts_fresh_train.shard{index}.manifest.json").read_text())
        assert manifest["artifact_sha256"] == sha(shard)
        shard_rows += [json.loads(l) for l in shard.read_text().splitlines()]
    assert shard_rows == merged_rows
    sidecar = json.loads((d25 / "rollouts_fresh_train.rescore.json").read_text())
    assert sidecar["verifier"] == rr.VERIFIER_ID
    assert set(sidecar["pinned_artifact_sha256"]) == {"rollouts_fresh_train.shard0.jsonl", "rollouts_fresh_train.shard1.jsonl"}
    # derived artifacts were moved, not deleted; the family is incomplete on purpose
    assert not (d25 / "DONE").exists() and not (d25 / "report.json").exists()
    parking = next((d25 / "pinned-scoring").iterdir())
    assert (parking / "report.json").read_text() == "pinned\n" and (parking / "DONE").exists()
    assert not (root / "family-math500-s0" / ".family-complete").exists()
    # untouched: prompts, run_config, the rollout text itself
    assert [r["input_ids"] for r in merged_rows] == [[ord(c) for c in t] for _, t, _, _ in rows]


def test_second_apply_is_idempotent(tmp_path: Path, fake_transformers) -> None:
    root = tmp_path / "root"
    write_point(root, 1, 0, [(0, "#### 2 x", "2x", 0.0), (1, "#### 9", "7", 0.0)], shards=False)
    args = dict(data=FakeData(), old_verifier=only_exact, new_verifier=generous, apply=True)
    rr.rescore_family(root, "math500", 1, [0], TAG, **args)
    d0 = root / "family-math500-s1" / f"{TAG}-s1-math500-d0"
    first = (d0 / "rollouts_fresh_train.jsonl").read_bytes()
    first_manifest = (d0 / "rollouts_fresh_train.manifest.json").read_text()
    rr.rescore_family(root, "math500", 1, [0], TAG, **args)
    assert (d0 / "rollouts_fresh_train.jsonl").read_bytes() == first
    assert (d0 / "rollouts_fresh_train.manifest.json").read_text() == first_manifest
    rows = [json.loads(l) for l in first.decode().splitlines()]
    assert [r["reward_pinned"] for r in rows] == [0.0, 0.0]   # never overwritten by a corrected value


def test_a_stored_reward_the_pinned_verifier_cannot_reproduce_aborts_untouched(tmp_path: Path, fake_transformers) -> None:
    root = tmp_path / "root"
    run = write_point(root, 2, 0, [(0, "#### 4", "4", 0.0)], shards=False)   # exact match, yet stored 0
    before = (run / "rollouts_fresh_train.jsonl").read_bytes()
    with pytest.raises(ValueError, match="does not reproduce the stored reward"):
        rr.rescore_family(root, "math500", 2, [0], TAG, data=FakeData(),
                          old_verifier=only_exact, new_verifier=generous, apply=True)
    assert (run / "rollouts_fresh_train.jsonl").read_bytes() == before
    assert (run / "DONE").exists() and (run / "report.json").exists()


def test_dry_run_counts_and_changes_nothing(tmp_path: Path, fake_transformers) -> None:
    root = tmp_path / "root"
    run = write_point(root, 3, 0, [(0, "#### 2 x", "2x", 0.0)], shards=False)
    before = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    result = rr.rescore_family(root, "math500", 3, [0], TAG, data=FakeData(),
                               old_verifier=only_exact, new_verifier=generous, apply=False)
    assert result["points"][0]["files"][0]["flip_0_to_1"] == 1
    assert result["points"][0]["retired"] == []
    assert {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()} == before


def test_refuses_a_family_that_is_held(tmp_path: Path, fake_transformers) -> None:
    root = tmp_path / "root"
    write_point(root, 4, 0, [(0, "#### 4", "4", 1.0)], shards=False)
    queue = root / ".families"
    queue.mkdir()
    (queue / "math500-s4.owner.json").write_text('{"worker": "w"}')
    with pytest.raises(SystemExit, match="claimed"):
        rr.rescore_family(root, "math500", 4, [0], TAG, data=FakeData(),
                          old_verifier=only_exact, new_verifier=generous, apply=True)
    (queue / "math500-s4.owner.json").unlink()
    with (queue / "math500-s4.lock").open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        with pytest.raises(SystemExit, match="locked"):
            rr.rescore_family(root, "math500", 4, [0], TAG, data=FakeData(),
                              old_verifier=only_exact, new_verifier=generous, apply=True)


def test_a_mismatch_in_a_later_point_leaves_the_whole_family_untouched(tmp_path: Path, fake_transformers) -> None:
    root = tmp_path / "root"
    good = write_point(root, 5, 0, [(0, "#### 2 x", "2x", 0.0)], shards=False)
    bad = write_point(root, 5, 25, [(0, "#### 4", "4", 0.0)], shards=False)   # cannot be reproduced
    before = {p.name: p.read_bytes() for p in good.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="does not reproduce"):
        rr.rescore_family(root, "math500", 5, [0, 25], TAG, data=FakeData(),
                          old_verifier=only_exact, new_verifier=generous, apply=True)
    assert {p.name: p.read_bytes() for p in good.iterdir() if p.is_file()} == before
    assert (good / "DONE").exists() and (bad / "DONE").exists()


def test_merged_and_shards_are_verified_once_per_unique_response(tmp_path: Path, fake_transformers) -> None:
    run = write_point(tmp_path, 0, 0, [(0, "#### 2 x", "2x", 0.0), (1, "#### 1 + x", "1+x", 0.0)], shards=True)
    calls = {"old": 0, "new": 0}
    def old(prediction, gold):
        calls["old"] += 1
        return only_exact(prediction, gold)
    def new(prediction, gold):
        calls["new"] += 1
        return generous(prediction, gold)
    report = rr.rescore_point(run, FakeData(), old, new, apply=True, stamp="test")
    assert calls == {"old": 2, "new": 2}
    assert sum(f["scored"] for f in report["files"]) == 2
    assert sum(f["reused"] for f in report["files"]) == 2
    assert sum(f["rows"] for f in report["files"]) == 4


@pytest.mark.parametrize("field,value", [("input_ids", [42]), ("reward", 1.0), ("resp_start", 1)])
def test_conflicting_duplicate_aborts_before_any_family_write(tmp_path: Path, fake_transformers, field, value) -> None:
    good = write_point(tmp_path, 0, 0, [(0, "#### 2 x", "2x", 0.0)], shards=False)
    bad = write_point(tmp_path, 0, 25, [(0, "#### 2 x", "2x", 0.0), (1, "#### 9", "7", 0.0)], shards=True)
    shard = bad / "rollouts_fresh_train.shard0.jsonl"
    row = json.loads(shard.read_text())
    row[field] = value
    shard.write_text(json.dumps(row) + "\n")
    before = {p: p.read_bytes() for run in (good, bad) for p in run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="conflicting duplicate"):
        rr.rescore_family(tmp_path, "math500", 0, [0, 25], TAG, data=FakeData(),
                          old_verifier=only_exact, new_verifier=generous, apply=True)
    assert before == {p: p.read_bytes() for run in (good, bad) for p in run.rglob("*") if p.is_file()}


def test_duplicate_cache_does_not_cross_train_and_val(tmp_path: Path, fake_transformers) -> None:
    run = write_point(tmp_path, 0, 0, [(0, "#### 2 x", "2x", 0.0)], shards=False)
    prompts = json.loads((run / "prompts.json").read_text())
    prompts["val"] = [{"answer": "9"}]
    (run / "prompts.json").write_text(json.dumps(prompts))
    (run / "rollouts_fresh_val.jsonl").write_bytes((run / "rollouts_fresh_train.jsonl").read_bytes())
    _, results = rr.scan_point(run, FakeData(), only_exact, generous)
    assert results["rollouts_fresh_train"][(0, 0)] == 1
    assert results["rollouts_fresh_val"][(0, 0)] == 0


def test_duplicate_reuse_matches_independent_real_verifier_results(tmp_path: Path, fake_transformers) -> None:
    pytest.importorskip("math_verify")
    old, new = rr.mmr.verifier_pair()
    pairs = [("#### 2", "2x"), ("#### 0.5", r"\frac{1}{2}")]
    rows = [(i, text, gold, rr.mmr.score(text, gold, FakeData(), old)) for i, (text, gold) in enumerate(pairs)]
    run = write_point(tmp_path, 0, 0, rows, shards=True)
    report, results = rr.scan_point(run, FakeData(), old, new)
    split = json.loads((run / "prompts.json").read_text())["train"]
    independent = {}
    for path, _ in rr.rollout_files(run, "rollouts_fresh_train"):
        rr.scan_rows(path, split, FakeTokenizer(), FakeData(), old, new, independent)
    assert results["rollouts_fresh_train"] == independent
    assert independent[(0, 0)] == 0
    assert sum(f["reused"] for f in report["files"]) == 2


def fake_worker_init(repo, pinned):
    rr._WORKER.update(data=FakeData(), old=only_exact, new=generous)


@pytest.mark.parametrize("mismatch", [False, True])
def test_real_process_pool_keeps_read_before_write_barrier(tmp_path: Path, fake_transformers, monkeypatch, mismatch) -> None:
    import concurrent.futures
    import functools
    import multiprocessing
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fixture inherits its tokenizer through fork")
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", functools.partial(
        concurrent.futures.ProcessPoolExecutor, mp_context=multiprocessing.get_context("fork")))
    monkeypatch.setattr(rr, "_worker_init", fake_worker_init)
    monkeypatch.setenv("OM_RESCORE_REPO", str(REPO))
    monkeypatch.setenv("OM_RESCORE_PINNED", "0e4cd412")
    monkeypatch.setenv("OM_RESCORE_WORKERS", "2")
    good = write_point(tmp_path, 0, 0, [(0, "#### 2 x", "2x", 0.0)], shards=False)
    bad_rows = [(0, "#### 4", "4", 0.0)] if mismatch else [(0, "#### 2 x", "2x", 0.0)]
    second = write_point(tmp_path, 0, 25, bad_rows, shards=False)
    before = {p: p.read_bytes() for run in (good, second) for p in run.rglob("*") if p.is_file()}
    kwargs = dict(data=FakeData(), old_verifier=only_exact, new_verifier=generous, apply=True)
    if mismatch:
        with pytest.raises(ValueError, match="does not reproduce"):
            rr.rescore_family(tmp_path, "math500", 0, [0, 25], TAG, **kwargs)
        assert before == {p: p.read_bytes() for run in (good, second) for p in run.rglob("*") if p.is_file()}
    else:
        report = rr.rescore_family(tmp_path, "math500", 0, [0, 25], TAG, **kwargs)
        assert all("DONE" in p["retired"] for p in report["points"])
        assert all(not (p / "DONE").exists() for p in (good, second))
