"""The math-reward measurement must reproduce the stored rewards before it is believed.

The point of the tool is to decide whether the finished matrix has to be
rescored. A tool that silently mis-replicates the pinned scoring would answer
that question with a made-up number, so the replication check comes first: if the
old verifier cannot reproduce the stored `reward` field, the run is reported
UNRELIABLE and no verdict is printed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import measure_math_reward as mmr  # noqa: E402


class FakeTokenizer:
    """input_ids are code points, so a response decodes back to its text."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class FakeData:
    """data.extract_answer: everything after the last '#### ' marker."""

    @staticmethod
    def extract_answer(text: str):
        if "#### " not in text:
            return None
        return text.rsplit("#### ", 1)[1].strip()


def only_exact(prediction: str, gold: str) -> float:
    """A verifier that agrees with nothing beyond what the caller already tried."""
    return 0.0


def generous(prediction: str, gold: str) -> float:
    """A verifier that accepts a prediction differing only in whitespace."""
    return 1.0 if prediction.replace(" ", "") == gold.replace(" ", "") else 0.0


def _write_run(tmp_path: Path, rows: list[tuple[str, str, float]]) -> Path:
    run = tmp_path / "family-math500-s0" / "tag-s0-math500-d0"
    (run / "logs").mkdir(parents=True)
    run.joinpath("DONE").write_text("done\n")
    run.joinpath("run_config.json").write_text(
        json.dumps({"dataset": "math500", "model_resolved": str(tmp_path / "model")})
    )
    answers = []
    lines = []
    for index, (text, gold, stored) in enumerate(rows):
        answers.append({"question": f"q{index}", "answer": gold})
        lines.append(
            json.dumps(
                {
                    "prompt_idx": index,
                    "rollout_idx": 0,
                    "input_ids": [ord(c) for c in text],
                    "resp_start": 0,
                    "reward": stored,
                }
            )
        )
    run.joinpath("prompts.json").write_text(json.dumps({"train": answers, "val": []}))
    run.joinpath("rollouts_fresh_train.jsonl").write_text("\n".join(lines) + "\n")
    return run


def test_score_follows_the_pinned_pre_verifier_steps() -> None:
    data = FakeData()
    # no answer marker at all
    assert mmr.score("no answer here", "4", data, generous) == 0.0
    # exact match never reaches the verifier
    assert mmr.score("#### 4", "4", data, only_exact) == 1.0
    # numeric match never reaches the verifier
    assert mmr.score("#### 4.0", "4", data, only_exact) == 1.0
    # only what is left is decided by the verifier
    assert mmr.score("#### 2 x", "2x", data, only_exact) == 0.0
    assert mmr.score("#### 2 x", "2x", data, generous) == 1.0


def test_flips_and_flat_groups_are_counted(tmp_path: Path, monkeypatch) -> None:
    run = _write_run(
        tmp_path,
        [
            ("#### 2 x", "2x", 0.0),   # the corrected verifier accepts this: 0 -> 1
            ("#### 4", "4", 1.0),      # exact match, unchanged
            ("#### 9", "7", 0.0),      # wrong under both
        ],
    )
    monkeypatch.setitem(sys.modules, "transformers", type(sys)("transformers"))
    sys.modules["transformers"].AutoTokenizer = FakeTokenizer
    report = mmr.measure_run(run, FakeData(), only_exact, generous, None)
    entry = report["files"][0]
    assert entry["rows"] == 3
    assert entry["replication_mismatch"] == 0
    assert entry["flip_0_to_1"] == 1 and entry["flip_1_to_0"] == 0
    # one rollout per prompt, so every group is trivially flat under both
    assert entry["prompts"] == 3
    assert entry["flat_groups_old"] == 3 and entry["flat_groups_new"] == 3


def test_a_stored_reward_that_cannot_be_reproduced_is_reported(tmp_path: Path, monkeypatch) -> None:
    """If the replication is wrong the measurement must say so, not answer anyway."""
    run = _write_run(tmp_path, [("#### 4", "4", 0.0)])  # stored 0, exact match says 1
    monkeypatch.setitem(sys.modules, "transformers", type(sys)("transformers"))
    sys.modules["transformers"].AutoTokenizer = FakeTokenizer
    report = mmr.measure_run(run, FakeData(), only_exact, generous, None)
    entry = report["files"][0]
    assert entry["replication_mismatch"] == 1
    assert entry["mismatch_examples"] and "stored=0.0" in entry["mismatch_examples"][0]


def test_pinned_module_is_read_from_the_generation_commit() -> None:
    """The working tree may already carry a corrected data.py; the old side must
    come from the commit the matrix was generated with."""
    module = mmr.load_pinned_data_module(REPO, "0e4cd412")
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "parsed_prediction = parse(prediction)" in source
    assert hasattr(module, "extract_answer")


def test_verifier_pair_disagrees_exactly_where_the_bug_is() -> None:
    pytest.importorskip("math_verify")
    old, new = mmr.verifier_pair()
    # bare-text parsing reads "2x" as the number 2 and calls it equal to 2
    assert old("2", "2x") == 1.0
    assert new("2", "2x") == 0.0
    # and both agree on a plain equality
    assert old("0.5", r"\frac{1}{2}") == new("0.5", r"\frac{1}{2}")
