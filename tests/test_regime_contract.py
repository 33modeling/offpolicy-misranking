"""Regime runs cannot silently reuse a path from another matrix."""

from __future__ import annotations

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
