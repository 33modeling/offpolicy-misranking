#!/usr/bin/env python3
"""Immutable matrix and completed-run contracts for shared regime workers."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from artifact_contract import validate_generation_contract
from gate_rules import has_valid_analysis_protocol
from model_matrix import _load_config
from score_artifacts import load_complete_score_artifacts
from train_policy_grpo import validate_policy_lineage

MATRIX_SCHEMA = "offpolicy-regime-matrix/v1"
MARKER_SCHEMA = "offpolicy-regime-run-validation/v1"
COLLECTION_SCHEMA = "offpolicy-regime-collection/v2"
COLLECTION_ARTIFACTS = (
    "REGIME.json",
    "REGIME.csv",
    "REGIME_SUMMARY.csv",
    "FINAL_REPORT.md",
)
REQUIRED_ARTIFACTS = (
    "DONE",
    "run_config.json",
    "manifest.json",
    "prompts.json",
    "score_protocol.json",
    "oracle_protocol.json",
    "report.json",
    "scores_oracle.json",
    "scores_offpolicy.json",
    "scores_splithalf.json",
    "divergence_stats.json",
    "oracle_micro_groups.pt",
    "val_groups.pt",
)
SMALL_BOUND_FILES = (
    "DONE",
    "run_config.json",
    "manifest.json",
    "prompts.json",
    "score_protocol.json",
    "oracle_protocol.json",
    "report.json",
    "scores_oracle.json",
    "scores_offpolicy.json",
    "scores_splithalf.json",
    "divergence_stats.json",
)
RUN_CONFIG_FIELDS = (
    "git",
    "git_status",
    "model_resolved",
    "model_config_sha256",
    "tokenizer_config_sha256",
    "generation_config_sha256",
    "model_snapshot_manifest_sha256",
    "dataset",
    "n_train",
    "n_val",
    "behavior_k",
    "fresh_k",
    "val_k",
    "micro_group",
    "max_new_tokens",
    "proj_dim",
    "grad_layers",
    "clip_cap",
    "temperature",
    "topk_frac",
    "top_p",
    "thinking",
    "prompt_format",
    "attn",
    "lora_targets",
    "skip_hybrid",
    "seed",
    "drift",
    "training_objective",
    "policy_update",
    "reward_source",
    "supervised_loss",
    "positive_only_filter",
    "grpo_world_size",
    "grpo_group_size",
    "grpo_clip_epsilon",
    "grpo_learning_rate",
    "grpo_reference_kl_beta",
    "grpo_epochs_per_batch",
    "grpo_max_grad_norm",
    "grpo_advantage_epsilon",
    "grpo_lora_rank",
    "grpo_lora_alpha",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(document: dict) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return document


def optional_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def dataset_n_train(experiment: dict, dataset: str) -> int:
    sizes = experiment.get("n_train_by_dataset")
    if not isinstance(sizes, dict) or dataset not in sizes:
        raise ValueError(f"missing registered candidate count for dataset: {dataset}")
    return int(sizes[dataset])


def build_matrix(
    config_path: Path,
    model_key: str,
    model_path: Path,
    qualification_path: Path,
    git: str,
) -> dict:
    config = _load_config(config_path)
    models = {row["key"]: row for row in config["models"]}
    if model_key not in models:
        raise ValueError(f"model key is not in transfer config: {model_key}")
    spec = models[model_key]
    if model_path.name != spec["local_directory"]:
        raise ValueError(
            f"model path does not match pinned directory: {model_path.name!r} != "
            f"{spec['local_directory']!r}"
        )
    manifest_path = model_path / ".om_snapshot.json"
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("repository") != spec["repository"]
        or manifest.get("revision") != spec["revision"]
    ):
        raise ValueError("model snapshot manifest does not match the transfer config")

    qualification = read_json(qualification_path)
    experiment = config["experiment"]
    qualification_rows = {row["dataset"]: row for row in qualification.get("datasets", [])}
    if qualification.get("status") != "qualified" or set(qualification_rows) != set(
        experiment["datasets"]
    ):
        raise ValueError("dataset qualification does not cover the exact experiment matrix")
    for dataset in experiment["datasets"]:
        row = qualification_rows[dataset]
        split = row.get("prompt_split", {})
        if (
            row.get("status") != "qualified"
            or row.get("n_train") != dataset_n_train(experiment, dataset)
            or row.get("n_val") != experiment["n_val"]
            or row.get("experiment_seeds") != experiment["seeds"]
            or not row.get("reward_runtime")
            or not all(
                isinstance(split.get(key), str) and len(split[key]) == 64
                for key in (
                    "train_prompt_set_sha256",
                    "validation_prompt_set_sha256",
                    "train_prompt_order_sha256",
                    "validation_prompt_order_sha256",
                )
            )
        ):
            raise ValueError(f"{dataset}: qualification does not match matrix dimensions")

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "scripts", "configs"],
        text=True,
    ).strip()
    if dirty:
        raise ValueError("src/scripts/configs worktree is dirty; commit before starting the matrix")
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if current != git:
        raise ValueError(f"requested git {git} differs from checkout {current}")

    document = {
        "schema": MATRIX_SCHEMA,
        "git": git,
        "config_sha256": sha256_file(config_path),
        "model": {
            "key": model_key,
            "path": str(model_path.resolve()),
            "repository": spec["repository"],
            "revision": spec["revision"],
            "lora_targets": spec["lora_targets"],
            "prompt_format": spec.get("prompt_format", "tokenizer_chat"),
            "config_sha256": sha256_file(model_path / "config.json"),
            "tokenizer_config_sha256": sha256_file(model_path / "tokenizer_config.json"),
            "generation_config_sha256": optional_hash(model_path / "generation_config.json"),
            "snapshot_manifest_sha256": sha256_file(manifest_path),
        },
        "qualification_sha256": sha256_file(qualification_path),
        "datasets": qualification_rows,
        "experiment": experiment,
    }
    document["digest"] = json_digest(document)
    return document


def initialize_matrix(path: Path, expected: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            recorded = read_json(path)
            if recorded != expected:
                raise ValueError(
                    f"matrix contract mismatch at {path}; use a new REGIME_ROOT instead "
                    "of mixing models, data, code, or hyperparameters"
                )
            return
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)


def expected_run_config(
    matrix: dict,
    dataset: str,
    seed: int,
    drift: int,
) -> dict:
    experiment = matrix["experiment"]
    if dataset not in experiment["datasets"]:
        raise ValueError(f"dataset is outside matrix: {dataset}")
    if seed not in experiment["seeds"]:
        raise ValueError(f"seed is outside matrix: {seed}")
    if drift not in experiment["drifts"]:
        raise ValueError(f"drift is outside matrix: {drift}")
    model = matrix["model"]
    grpo = experiment["grpo"]
    return {
        "git": matrix["git"],
        "git_status": "",
        "model_resolved": model["path"],
        "model_config_sha256": model["config_sha256"],
        "tokenizer_config_sha256": model["tokenizer_config_sha256"],
        "generation_config_sha256": model["generation_config_sha256"],
        "model_snapshot_manifest_sha256": model["snapshot_manifest_sha256"],
        "dataset": dataset,
        "n_train": dataset_n_train(experiment, dataset),
        "n_val": experiment["n_val"],
        "behavior_k": experiment["behavior_k"],
        "fresh_k": experiment["fresh_k"],
        "val_k": experiment["val_k"],
        "micro_group": experiment["micro_group"],
        "max_new_tokens": experiment["max_new_tokens"],
        "proj_dim": experiment["proj_dim"],
        "grad_layers": experiment["grad_layers"],
        "clip_cap": float(experiment["clip_cap"]),
        "temperature": float(experiment["temperature"]),
        "topk_frac": float(experiment["topk_frac"]),
        "top_p": float(experiment["top_p"]),
        "thinking": experiment["thinking"],
        "prompt_format": model["prompt_format"],
        "attn": experiment["attn"],
        "lora_targets": ",".join(model["lora_targets"]),
        "skip_hybrid": "1" if experiment["skip_hybrid"] else "0",
        "seed": seed,
        "drift": drift,
        "training_objective": (
            "base_control" if drift == 0 else experiment["policy_method"]
        ),
        "policy_update": (
            "none"
            if drift == 0
            else (
                "reinforce_leave_one_out"
                if experiment["policy_method"] == "rloo"
                else "clipped_policy_gradient"
            )
        ),
        "reward_source": "none" if drift == 0 else "verifier",
        "supervised_loss": False,
        "positive_only_filter": False,
        "grpo_world_size": grpo["world_size"],
        "grpo_group_size": grpo["group_size"],
        "grpo_clip_epsilon": float(grpo["clip_epsilon"]),
        "grpo_learning_rate": float(grpo["learning_rate"]),
        "grpo_reference_kl_beta": float(grpo["reference_kl_beta"]),
        "grpo_epochs_per_batch": grpo["epochs_per_batch"],
        "grpo_max_grad_norm": float(grpo["max_grad_norm"]),
        "grpo_advantage_epsilon": float(grpo["advantage_epsilon"]),
        "grpo_lora_rank": grpo["lora_rank"],
        "grpo_lora_alpha": grpo["lora_alpha"],
    }


def prompt_split_errors(run: Path, matrix: dict, dataset: str) -> list[str]:
    """Compare materialized prompts to the qualified ordered snapshot split."""
    prompts = read_json(run / "prompts.json")
    expected = matrix["datasets"][dataset]["prompt_split"]
    errors = []
    for split, prefix in (("train", "train"), ("val", "validation")):
        rows = prompts.get(split)
        if not isinstance(rows, list):
            errors.append(f"prompts.{split} is not a list")
            continue
        hashes = [
            hashlib.sha256(str(row.get("question", "")).strip().encode()).hexdigest()
            for row in rows
            if isinstance(row, dict)
        ]
        if len(hashes) != len(rows):
            errors.append(f"prompts.{split} contains a non-object row")
            continue
        set_digest = hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest()
        order_digest = hashlib.sha256("".join(hashes).encode()).hexdigest()
        if set_digest != expected.get(f"{prefix}_prompt_set_sha256"):
            errors.append(f"prompts.{split} set differs from qualified snapshot")
        if order_digest != expected.get(f"{prefix}_prompt_order_sha256"):
            errors.append(f"prompts.{split} order differs from qualified snapshot")
    return errors


def config_errors(
    run: Path,
    matrix: dict,
    dataset: str,
    seed: int,
    drift: int,
    behavior_source: Path | None,
) -> list[str]:
    config = read_json(run / "run_config.json")
    expected = expected_run_config(matrix, dataset, seed, drift)
    errors = [
        f"{key}: expected={expected[key]!r} recorded={config.get(key)!r}"
        for key in RUN_CONFIG_FIELDS
        if config.get(key) != expected[key]
    ]
    expected_source = str(behavior_source) if drift else None
    if config.get("behavior_source") != expected_source:
        errors.append(
            f"behavior_source: expected={expected_source!r} "
            f"recorded={config.get('behavior_source')!r}"
        )
    drifts = matrix["experiment"]["drifts"]
    drift_index = drifts.index(drift)
    previous_drift = drifts[drift_index - 1] if drift_index > 0 else 0
    expected_resume = None
    if drift > 0 and previous_drift > 0:
        if behavior_source is None or not behavior_source.name.endswith("-d0"):
            errors.append("positive policy lineage has no canonical d0 source")
        else:
            parent_run = behavior_source.with_name(
                f"{behavior_source.name[:-3]}-d{previous_drift}"
            )
            expected_resume = str(parent_run / f"policy_step_{previous_drift}")
    if config.get("grpo_start_step", 0) != previous_drift:
        errors.append(
            f"grpo_start_step: expected={previous_drift!r} "
            f"recorded={config.get('grpo_start_step')!r}"
        )
    if config.get("grpo_resume_adapter") != expected_resume:
        errors.append(
            f"grpo_resume_adapter: expected={expected_resume!r} "
            f"recorded={config.get('grpo_resume_adapter')!r}"
        )
    return errors


def required_artifacts(run: Path) -> None:
    missing = [
        name
        for name in REQUIRED_ARTIFACTS
        if not (run / name).is_file() or (run / name).stat().st_size == 0
    ]
    if missing:
        raise ValueError(f"required artifacts missing or empty: {missing}")


def small_hashes(run: Path) -> dict[str, str]:
    return {name: sha256_file(run / name) for name in SMALL_BOUND_FILES}


def artifact_stats(run: Path, names: list[str]) -> dict[str, dict[str, int]]:
    return {
        name: {
            "size": (run / name).stat().st_size,
            "mtime_ns": (run / name).stat().st_mtime_ns,
        }
        for name in names
    }


def marker_is_current(run: Path, matrix: dict) -> bool:
    try:
        marker = read_json(run / ".regime_validated.json")
        if marker.get("schema") != MARKER_SCHEMA or marker.get("matrix_digest") != matrix["digest"]:
            return False
        if marker.get("small_sha256") != small_hashes(run):
            return False
        for name, expected in marker.get("bound_artifact_stats", {}).items():
            stat = (run / name).stat()
            if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def collection_inputs(runs: list[Path], matrix: dict) -> list[dict[str, str]]:
    inputs = []
    for run in runs:
        marker = run / ".regime_validated.json"
        document = read_json(marker)
        if (
            document.get("schema") != MARKER_SCHEMA
            or document.get("matrix_digest") != matrix["digest"]
        ):
            raise ValueError(f"stale run validation marker: {marker}")
        inputs.append(
            {
                "path": str(run.resolve()),
                "validation_sha256": sha256_file(marker),
            }
        )
    return inputs


def collection_hashes(results: Path) -> dict[str, str]:
    missing = [
        name
        for name in COLLECTION_ARTIFACTS
        if not (results / name).is_file() or (results / name).stat().st_size == 0
    ]
    if missing:
        raise ValueError(f"collection artifacts missing or empty: {missing}")
    return {name: sha256_file(results / name) for name in COLLECTION_ARTIFACTS}


def collection_is_current(results: Path, runs: list[Path], matrix: dict) -> bool:
    try:
        marker = read_json(results / ".regime_collection.json")
        return (
            marker.get("schema") == COLLECTION_SCHEMA
            and marker.get("matrix_digest") == matrix["digest"]
            and marker.get("run_validations") == collection_inputs(runs, matrix)
            and marker.get("output_sha256") == collection_hashes(results)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def mark_collection(results: Path, runs: list[Path], matrix: dict) -> None:
    document = {
        "schema": COLLECTION_SCHEMA,
        "matrix_digest": matrix["digest"],
        "run_validations": collection_inputs(runs, matrix),
        "output_sha256": collection_hashes(results),
    }
    target = results / ".regime_collection.json"
    temporary = results / f"{target.name}.tmp.{os.getpid()}"
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)


def deep_validation(
    run: Path,
    matrix: dict,
    drift: int,
    behavior_source: Path | None,
) -> dict:
    generation = validate_generation_contract(
        run,
        require_policy_binding=True,
        require_rng_binding=True,
    )
    if generation["generation_hash_missing"]:
        raise ValueError("generation manifests are not bound to rollout file hashes")
    if not has_valid_analysis_protocol(run):
        raise ValueError("score/oracle protocol or bound generation hashes are invalid")
    artifacts = load_complete_score_artifacts(run)
    report = read_json(run / "report.json")
    prompts = read_json(run / "prompts.json")
    dataset = read_json(run / "run_config.json").get("dataset")
    split_errors = prompt_split_errors(run, matrix, dataset)
    if split_errors:
        raise ValueError("; ".join(split_errors))
    expected_train = dataset_n_train(matrix["experiment"], dataset)
    if len(prompts.get("train", [])) != expected_train:
        raise ValueError("prompts.json train size differs from matrix")
    if len(prompts.get("val", [])) != matrix["experiment"]["n_val"]:
        raise ValueError("prompts.json validation size differs from matrix")
    if len(artifacts.oracle) != expected_train:
        raise ValueError("score artifact coverage differs from matrix n_train")
    if not report:
        raise ValueError("report.json is empty")
    bound_names = sorted(
        set(generation["artifact_sha256"]) | set(generation["manifest_sha256"])
    )
    if drift > 0:
        policy = run / f"policy_step_{drift}"
        drifts = matrix["experiment"]["drifts"]
        drift_index = drifts.index(drift)
        previous_drift = drifts[drift_index - 1]
        expected_parent = None
        if previous_drift > 0:
            if behavior_source is None or not behavior_source.name.endswith("-d0"):
                raise ValueError("positive policy lineage has no canonical d0 source")
            parent_run = behavior_source.with_name(
                f"{behavior_source.name[:-3]}-d{previous_drift}"
            )
            expected_parent = parent_run / f"policy_step_{previous_drift}"
        validate_policy_lineage(
            policy,
            target_steps=drift,
            world_size=int(matrix["experiment"]["grpo"]["world_size"]),
            training_objective=matrix["experiment"]["policy_method"],
            expected_start_step=previous_drift,
            expected_parent=expected_parent,
            expected_model=Path(matrix["model"]["path"]),
            expected_seed=int(read_json(run / "run_config.json")["seed"]),
            expected_max_new_tokens=int(
                matrix["experiment"]["max_new_tokens"]
            ),
            expected_config={
                "group_size": int(matrix["experiment"]["grpo"]["group_size"]),
                "clip_epsilon": float(
                    matrix["experiment"]["grpo"]["clip_epsilon"]
                ),
                "learning_rate": float(
                    matrix["experiment"]["grpo"]["learning_rate"]
                ),
                "epochs_per_batch": int(
                    matrix["experiment"]["grpo"]["epochs_per_batch"]
                ),
                "max_grad_norm": float(
                    matrix["experiment"]["grpo"]["max_grad_norm"]
                ),
                "advantage_epsilon": float(
                    matrix["experiment"]["grpo"]["advantage_epsilon"]
                ),
                "lora_rank": int(matrix["experiment"]["grpo"]["lora_rank"]),
                "lora_alpha": int(
                    matrix["experiment"]["grpo"]["lora_alpha"]
                ),
                "checkpoint_every": 5,
            },
            expected_prompts=run / "prompts.json",
            require_complete_hashes=True,
        )
        bound_names.extend(
            str(Path(f"policy_step_{drift}") / name)
            for name in (
                "policy_train.json",
                "adapter_config.json",
                "adapter_model.safetensors",
                "optimizer.pt",
                "grpo_stats.jsonl",
            )
        )
    return {
        "generation": generation,
        "bound_artifact_stats": artifact_stats(run, bound_names),
    }


def validate_run(
    run: Path,
    matrix: dict,
    dataset: str,
    seed: int,
    drift: int,
    behavior_source: Path | None,
    *,
    deep: bool,
    mark: bool,
) -> None:
    required_artifacts(run)
    errors = config_errors(run, matrix, dataset, seed, drift, behavior_source)
    if errors:
        raise ValueError("run config mismatch: " + "; ".join(errors))
    if not deep:
        if not marker_is_current(run, matrix):
            raise ValueError("deep-validation marker is missing or stale")
        return
    validation = deep_validation(run, matrix, drift, behavior_source)
    if mark:
        marker = {
            "schema": MARKER_SCHEMA,
            "matrix_digest": matrix["digest"],
            "small_sha256": small_hashes(run),
            "bound_artifact_stats": validation["bound_artifact_stats"],
            "validated_rows": validation["generation"]["validated_rows"],
        }
        temporary = run / f".regime_validated.json.tmp.{os.getpid()}"
        temporary.write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(run / ".regime_validated.json")


def quarantine(run: Path, root: Path, reason: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    destination = root / f"{run.name}-{reason}-{stamp}"
    counter = 1
    while destination.exists():
        destination = root / f"{run.name}-{reason}-{stamp}-{counter}"
        counter += 1
    run.rename(destination)
    return destination


def prepare_run(
    run: Path,
    matrix: dict,
    dataset: str,
    seed: int,
    drift: int,
    behavior_source: Path | None,
    quarantine_root: Path,
) -> str:
    if not run.exists() or not any(run.iterdir()):
        return "new"
    config_path = run / "run_config.json"
    if not config_path.is_file():
        destination = quarantine(run, quarantine_root, "unconfigured")
        return f"quarantined:{destination}"
    try:
        errors = config_errors(run, matrix, dataset, seed, drift, behavior_source)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        errors = ["unreadable run_config.json"]
    if errors:
        destination = quarantine(run, quarantine_root, "config-mismatch")
        return f"quarantined:{destination}"
    if (run / "DONE").exists():
        try:
            if marker_is_current(run, matrix):
                return "complete"
            validate_run(
                run,
                matrix,
                dataset,
                seed,
                drift,
                behavior_source,
                deep=True,
                mark=True,
            )
            return "complete"
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            destination = quarantine(run, quarantine_root, "invalid-complete")
            return f"quarantined:{destination}"
    marker = run / ".regime_validated.json"
    if marker.exists():
        marker.unlink()
    return "resume"


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--drift", type=int, required=True)
    parser.add_argument("--behavior-source", type=Path)


def add_collection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--matrix", type=Path, required=True)
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--model-key", required=True)
    init.add_argument("--model", type=Path, required=True)
    init.add_argument("--qualification", type=Path, required=True)
    init.add_argument("--git", required=True)
    check = sub.add_parser("check-run")
    add_run_arguments(check)
    check.add_argument("--deep", action="store_true")
    check.add_argument("--mark", action="store_true")
    prepare = sub.add_parser("prepare-run")
    add_run_arguments(prepare)
    prepare.add_argument("--quarantine-root", type=Path, required=True)
    collection_check = sub.add_parser("check-collection")
    add_collection_arguments(collection_check)
    collection_mark = sub.add_parser("mark-collection")
    add_collection_arguments(collection_mark)
    prompt_check = sub.add_parser("check-prompts")
    prompt_check.add_argument("--matrix", type=Path, required=True)
    prompt_check.add_argument("--run", type=Path, required=True)
    prompt_check.add_argument("--dataset", required=True)
    args = parser.parse_args()

    try:
        if args.command == "init":
            expected = build_matrix(
                args.config, args.model_key, args.model, args.qualification, args.git
            )
            initialize_matrix(args.matrix, expected)
            print(f"[regime-contract] matrix={args.matrix} digest={expected['digest'][:12]}")
            return 0
        matrix = read_json(args.matrix)
        if matrix.get("schema") != MATRIX_SCHEMA or matrix.get("digest") != json_digest(
            {key: value for key, value in matrix.items() if key != "digest"}
        ):
            raise ValueError("matrix document schema or digest is invalid")
        if args.command == "check-collection":
            return 0 if collection_is_current(args.results, args.runs, matrix) else 1
        if args.command == "mark-collection":
            mark_collection(args.results, args.runs, matrix)
            return 0
        if args.command == "check-prompts":
            errors = prompt_split_errors(args.run, matrix, args.dataset)
            if errors:
                raise ValueError("; ".join(errors))
            return 0
        behavior_source = args.behavior_source.resolve() if args.behavior_source else None
        if args.command == "check-run":
            validate_run(
                args.run,
                matrix,
                args.dataset,
                args.seed,
                args.drift,
                behavior_source,
                deep=args.deep,
                mark=args.mark,
            )
            return 0
        result = prepare_run(
            args.run,
            matrix,
            args.dataset,
            args.seed,
            args.drift,
            behavior_source,
            args.quarantine_root,
        )
        print(f"[regime-contract] {args.run}: {result}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"[regime-contract-abort] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
