"""Regime runs cannot silently reuse a path from another matrix."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime_contract import (
    COLLECTION_ARTIFACTS,
    MATRIX_SCHEMA,
    collection_is_current,
    config_errors,
    expected_run_config,
    initialize_matrix,
    json_digest,
    mark_collection,
    prepare_run,
    prompt_split_errors,
)


def matrix_document(root: Path) -> dict:
    document = {
        "schema": MATRIX_SCHEMA,
        "git": "a" * 40,
        "config_sha256": "b" * 64,
        "model": {
            "key": "fixture",
            "path": str((root / "model").resolve()),
            "repository": "org/model",
            "revision": "c" * 40,
            "lora_targets": ["q_proj", "v_proj"],
            "config_sha256": "d" * 64,
            "tokenizer_config_sha256": "e" * 64,
            "generation_config_sha256": None,
            "snapshot_manifest_sha256": "f" * 64,
        },
        "qualification_sha256": "1" * 64,
        "datasets": {},
        "experiment": {
            "policy_method": "grpo",
            "datasets": ["mbpp"],
            "seeds": [0],
            "drifts": [0, 25],
            "n_train": 8,
            "n_val": 4,
            "behavior_k": 4,
            "fresh_k": 8,
            "val_k": 4,
            "micro_group": 2,
            "max_new_tokens": 32,
            "proj_dim": 64,
            "grad_layers": 1,
            "clip_cap": 10.0,
            "topk_frac": 0.25,
            "temperature": 1.0,
            "top_p": 1.0,
            "thinking": "off",
            "attn": "eager",
            "skip_hybrid": True,
            "first_bootstrap": 10000,
            "grpo": {
                "world_size": 4,
                "group_size": 8,
                "clip_epsilon": 0.2,
                "learning_rate": 1e-5,
                "reference_kl_beta": 0.0,
                "epochs_per_batch": 2,
                "max_grad_norm": 1.0,
                "advantage_epsilon": 1e-4,
                "lora_rank": 16,
                "lora_alpha": 32,
            },
        },
    }
    document["digest"] = json_digest(document)
    return document


def test_matrix_initialization_is_idempotent_and_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        path = root / "MATRIX.json"
        document = matrix_document(root)
        initialize_matrix(path, document)
        initialize_matrix(path, document)
        changed = json.loads(json.dumps(document))
        changed["experiment"]["drifts"] = [0, 100]
        changed["digest"] = json_digest({k: v for k, v in changed.items() if k != "digest"})
        try:
            initialize_matrix(path, changed)
        except ValueError as exc:
            assert "mismatch" in str(exc)
        else:
            raise AssertionError("different matrix reused the same root")


def test_partial_run_resumes_but_config_mismatch_is_quarantined() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        matrix = matrix_document(root)
        run = root / "run"
        run.mkdir()
        config = expected_run_config(matrix, "mbpp", 0, 0)
        (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        (run / "partial").write_text("keep", encoding="utf-8")
        assert not config_errors(run, matrix, "mbpp", 0, 0, None)
        quarantine = root / "quarantine"
        assert prepare_run(run, matrix, "mbpp", 0, 0, None, quarantine) == "resume"

        config["fresh_k"] = 99
        (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        result = prepare_run(run, matrix, "mbpp", 0, 0, None, quarantine)
        assert result.startswith("quarantined:")
        assert not run.exists()
        moved = next(quarantine.iterdir())
        assert (moved / "partial").read_text(encoding="utf-8") == "keep"


def test_positive_run_config_requires_continuous_checkpoint_lineage() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        matrix = matrix_document(root)
        matrix["experiment"]["drifts"] = [0, 25, 100]
        source = root / "fixture-s0-mbpp-d0"
        run = root / "fixture-s0-mbpp-d100"
        run.mkdir()
        config = expected_run_config(matrix, "mbpp", 0, 100)
        config.update(
            {
                "behavior_source": str(source),
                "grpo_start_step": 25,
                "grpo_resume_adapter": str(
                    root / "fixture-s0-mbpp-d25" / "policy_step_25"
                ),
            }
        )
        (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        assert not config_errors(run, matrix, "mbpp", 0, 100, source)

        config["grpo_start_step"] = 0
        config["grpo_resume_adapter"] = None
        (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        errors = config_errors(run, matrix, "mbpp", 0, 100, source)
        assert any("grpo_start_step" in error for error in errors)
        assert any("grpo_resume_adapter" in error for error in errors)


def test_collection_is_bound_to_runs_matrix_and_outputs() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        matrix = matrix_document(root)
        run = root / "run"
        results = root / "results"
        run.mkdir()
        results.mkdir()
        validation = {
            "schema": "offpolicy-regime-run-validation/v1",
            "matrix_digest": matrix["digest"],
        }
        marker = run / ".regime_validated.json"
        marker.write_text(json.dumps(validation), encoding="utf-8")
        for name in COLLECTION_ARTIFACTS:
            (results / name).write_text(f"{name}\n", encoding="utf-8")

        assert not collection_is_current(results, [run], matrix)
        mark_collection(results, [run], matrix)
        assert collection_is_current(results, [run], matrix)

        report = results / "FINAL_REPORT.md"
        report.write_text("changed\n", encoding="utf-8")
        assert not collection_is_current(results, [run], matrix)
        report.write_text("FINAL_REPORT.md\n", encoding="utf-8")
        assert collection_is_current(results, [run], matrix)

        validation["validated_rows"] = 1
        marker.write_text(json.dumps(validation), encoding="utf-8")
        assert not collection_is_current(results, [run], matrix)


def test_materialized_prompts_must_match_qualified_set_and_order() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        run = root / "run"
        run.mkdir()
        prompts = {
            "train": [{"question": "train one"}, {"question": "train two"}],
            "val": [{"question": "validation one"}],
        }

        def digest(rows: list[dict], *, ordered: bool) -> str:
            hashes = [
                hashlib.sha256(row["question"].encode()).hexdigest() for row in rows
            ]
            payload = "".join(hashes if ordered else sorted(hashes)).encode()
            return hashlib.sha256(payload).hexdigest()

        matrix = matrix_document(root)
        matrix["datasets"] = {
            "mbpp": {
                "prompt_split": {
                    "train_prompt_set_sha256": digest(prompts["train"], ordered=False),
                    "train_prompt_order_sha256": digest(prompts["train"], ordered=True),
                    "validation_prompt_set_sha256": digest(prompts["val"], ordered=False),
                    "validation_prompt_order_sha256": digest(prompts["val"], ordered=True),
                }
            }
        }
        (run / "prompts.json").write_text(json.dumps(prompts), encoding="utf-8")
        assert prompt_split_errors(run, matrix, "mbpp") == []

        prompts["train"].reverse()
        (run / "prompts.json").write_text(json.dumps(prompts), encoding="utf-8")
        assert prompt_split_errors(run, matrix, "mbpp") == [
            "prompts.train order differs from qualified snapshot"
        ]
