"""Plan interrupted v4 runs without requiring one Git commit for the matrix."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from run_provenance import validate_commit

SLOTS = {
    1: (("27b", 0), ("27b", 1), ("7b", 0)),
    2: (("27b", 2), ("27b", 3), ("7b", 1)),
    3: (("27b", 4), ("7b", 2), ("7b", 3), ("7b", 4)),
}
DATASETS = ("gsm8k", "math500")
ENV_KEYS = {
    "MODEL_14B": "model",
    "OM_POOL_FILE": "pool",
    "N_TRAIN": "n_train",
    "N_VAL": "n_val",
    "BEHAVIOR_K": "behavior_k",
    "FRESH_K": "fresh_k",
    "VAL_K": "val_k",
    "MICRO_GROUP": "micro_group",
    "HYBRID_PROMPTS": "hybrid_prompts",
    "K_CELL": "k_cell",
    "DRIFT": "drift",
    "MAX_NEW_TOKENS": "max_new_tokens",
    "PROJ_DIM": "proj_dim",
    "GRAD_LAYERS": "grad_layers",
    "CLIP_CAP": "clip_cap",
    "TEMPERATURE": "temperature",
    "TOPK_FRAC": "topk_frac",
    "RADIUS_MODE": "radius_mode",
    "OM_TOP_P": "top_p",
    "OM_THINKING": "thinking",
    "OM_ATTN": "attn",
    "OM_GEN_BATCH": "gen_batch",
    "OM_LORA_TARGETS": "lora_targets",
    "OM_SKIP_HYBRID": "skip_hybrid",
}


def run_name(model: str, seed: int, dataset: str) -> str:
    suffix = "" if dataset == "gsm8k" else "-math500"
    return f"v4-{model}-s{seed}{suffix}"


def read_config(path: Path) -> dict[str, object]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        commit = config["git"]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"unreadable run config: {path}: {exc}") from exc
    validate_commit(commit, path)
    return config


def all_configs(runs_root: Path) -> dict[tuple[str, int, str], tuple[Path, dict[str, object]]]:
    configs = {}
    for model in ("27b", "7b"):
        for seed in range(5):
            for dataset in DATASETS:
                path = runs_root / run_name(model, seed, dataset) / "run_config.json"
                if path.is_file():
                    configs[(model, seed, dataset)] = (path, read_config(path))
    return configs


def complete(run: Path) -> bool:
    return all(
        (run / name).is_file() and (run / name).stat().st_size > 0
        for name in (
            "DONE",
            "run_config.json",
            "manifest.json",
            "score_protocol.json",
            "oracle_protocol.json",
            "report.json",
        )
    )


def matrix_commit(
    configs: dict[tuple[str, int, str], tuple[Path, dict[str, object]]],
    current: str,
) -> str:
    current = validate_commit(current, "current checkout")
    by_commit: dict[str, list[str]] = {}
    for key, (_, config) in configs.items():
        commit = validate_commit(config.get("git"), run_name(*key))
        by_commit.setdefault(commit, []).append(run_name(*key))
    if len(by_commit) > 1:
        details = ", ".join(
            f"{commit}=[{', '.join(sorted(names))}]"
            for commit, names in sorted(by_commit.items())
        )
        raise ValueError(
            "mixed v4 generation commits cannot share one matrix: " + details
        )
    return next(iter(by_commit), current)


def resume_plan(runs_root: Path, slot: int, current: str) -> tuple[list[dict[str, str]], int]:
    if slot not in SLOTS:
        raise ValueError(f"invalid cluster slot: {slot}")
    configs = all_configs(runs_root)
    commit = matrix_commit(configs, current)
    plan = []
    skipped = 0
    for model, seed in SLOTS[slot]:
        for dataset in DATASETS:
            key = (model, seed, dataset)
            name = run_name(*key)
            run = runs_root / name
            if complete(run):
                skipped += 1
                continue
            if key in configs:
                config_path, config = configs[key]
                run_commit = str(config["git"])
                source = "recorded run_config"
            else:
                config_path = None
                run_commit = commit
                source = "matrix generation commit"
            plan.append({
                "name": name,
                "model": model,
                "seed": str(seed),
                "dataset": dataset,
                "commit": run_commit,
                "config": str(config_path) if config_path else "-",
                "source": source,
            })
    return plan, skipped


def shell_environment(config_path: Path) -> str:
    config = read_config(config_path)
    lines = []
    for env_name, key in ENV_KEYS.items():
        value = config.get(key)
        if value is None:
            lines.append(f"unset {env_name}")
        else:
            lines.append(f"export {env_name}={shlex.quote(str(value))}")
    return "\n".join(lines)


def main() -> int:
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "env":
            print(shell_environment(Path(sys.argv[2])))
            return 0
        if len(sys.argv) != 5 or sys.argv[1] != "plan":
            print(
                "usage: v4_resume_commit.py plan RUNS_ROOT SLOT CURRENT_GIT\n"
                "       v4_resume_commit.py env RUN_CONFIG",
                file=sys.stderr,
            )
            return 2
        plan, skipped = resume_plan(Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
        for row in plan:
            print("\t".join(row[key] for key in (
                "name", "model", "seed", "dataset", "commit", "config"
            )))
            print(
                f"[resume-v4-plan] {row['name']}: {row['commit'][:12]} "
                f"({row['source']})",
                file=sys.stderr,
            )
        print(
            f"[resume-v4-plan] DONE skip={skipped}, resume/start={len(plan)}",
            file=sys.stderr,
        )
    except (ValueError, OSError) as exc:
        print(f"[resume-v4-abort] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
