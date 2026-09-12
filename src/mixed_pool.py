"""Mixed candidate pool: a positive control in which prompt selection should matter.

The candidate pool of a new MATH-500 point is replaced by a mixture of MATH
prompts and off-task prompts (MBPP code tasks by default) while the ranking
validation set and the independent test set stay MATH. A uniform random
subset then spends part of its budget on prompts whose rewards carry no
signal for the target, whereas the difficulty score and the gradient scores
can identify them. The point is created with the matched OLMo configuration
of an existing completed point (scripts/run_point.sh, OM_POOL_FILE), and the
reduced E5 arms and the gate arm run on it unchanged.

    python src/mixed_pool.py build --math-run RUN --other-run RUN --out POOL.jsonl [--math 200 --other 200 --val 100 --seed 0]
    python src/mixed_pool.py env --run RUN            # KEY=VALUE lines for scripts/run_point.sh

``build`` takes the first ``math`` training prompts and the first ``val``
validation prompts of the MATH point (its own order), the first ``other``
training prompts of the off-task point, mixes the training prompts in a
seeded order, and writes a pre-split pool (rows carry ``split`` and
``source``) with a manifest. Rebuilding with the same inputs is a no-op;
different content is refused.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
import shlex
import sys
from pathlib import Path

ENV_KEYS = {
    "MODEL_PATH": "model", "N_TRAIN": "n_train", "N_VAL": "n_val", "BEHAVIOR_K": "behavior_k",
    "FRESH_K": "fresh_k", "VAL_K": "val_k", "MICRO_GROUP": "micro_group", "HYBRID_PROMPTS": "hybrid_prompts",
    "K_CELL": "k_cell", "MAX_NEW_TOKENS": "max_new_tokens", "PROJ_DIM": "proj_dim", "GRAD_LAYERS": "grad_layers",
    "CLIP_CAP": "clip_cap", "TEMPERATURE": "temperature", "TOPK_FRAC": "topk_frac", "RADIUS_MODE": "radius_mode",
    "OM_TOP_P": "top_p", "OM_THINKING": "thinking", "OM_PROMPT_FORMAT": "prompt_format", "OM_ATTN": "attn",
    "OM_GEN_BATCH": "gen_batch", "OM_LORA_TARGETS": "lora_targets", "OM_SKIP_HYBRID": "skip_hybrid",
    "GRPO_WORLD_SIZE": "grpo_world_size", "GRPO_GROUP_SIZE": "grpo_group_size",
    "GRPO_CLIP_EPSILON": "grpo_clip_epsilon", "GRPO_LEARNING_RATE": "grpo_learning_rate",
    "GRPO_EPOCHS_PER_BATCH": "grpo_epochs_per_batch", "GRPO_MAX_GRAD_NORM": "grpo_max_grad_norm",
    "GRPO_ADVANTAGE_EPSILON": "grpo_advantage_epsilon", "GRPO_LORA_RANK": "grpo_lora_rank",
    "GRPO_LORA_ALPHA": "grpo_lora_alpha", "GRPO_LOGPROB_MICRO_BATCH": "grpo_logprob_micro_batch",
    "GRPO_GRADIENT_CHECKPOINTING": "grpo_gradient_checkpointing", "GRADIENT_MICRO_BATCH": "gradient_micro_batch",
}


def read_prompts(run: Path) -> dict:
    path = run / "prompts.json"
    if not path.is_file():
        raise FileNotFoundError(f"prompts.json missing: {run}")
    prompts = json.loads(path.read_text(encoding="utf-8"))
    for split in ("train", "val"):
        if not isinstance(prompts.get(split), list):
            raise ValueError(f"{path}: missing {split} list")
    return prompts


def build_rows(math: dict, other: dict, n_math: int, n_other: int, n_val: int, seed: int,
               math_name: str = "math500", other_name: str = "other") -> list[dict]:
    if n_math < 1 or n_other < 1 or n_val < 1:
        raise ValueError("counts must be positive")
    if len(math["train"]) < n_math or len(math["val"]) < n_val or len(other["train"]) < n_other:
        raise ValueError(f"not enough prompts: math train {len(math['train'])} val {len(math['val'])}, other train {len(other['train'])}")
    def item(row, source, split):
        return {"question": row["question"], "answer": str(row["answer"]), "source": source, "split": split}
    train = [item(r, math_name, "train") for r in math["train"][:n_math]] + \
            [item(r, other_name, "train") for r in other["train"][:n_other]]
    val = [item(r, math_name, "val") for r in math["val"][:n_val]]
    questions = [r["question"] for r in train + val]
    if len(set(questions)) != len(questions):
        raise ValueError("duplicate question across the mixed pool")
    random.Random(seed + 9_973).shuffle(train)
    return train + val


def write_pool(rows: list[dict], out: Path, provenance: dict) -> dict:
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest_path = Path(str(out) + ".manifest.json")
    if out.exists():
        if hashlib.sha256(out.read_bytes()).hexdigest() != sha:
            raise ValueError(f"pool exists with different content: {out}")
        return json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"sha256": sha}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)
    counts = {}
    for r in rows:
        key = f"{r['split']}:{r['source']}"
        counts[key] = counts.get(key, 0) + 1
    manifest = {"schema_version": 1, "kind": "mixed-pool", "rows": len(rows), "counts": counts, "sha256": sha,
                **provenance, "built_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def build(math_run: Path, other_run: Path, out: Path, n_math: int, n_other: int, n_val: int, seed: int) -> dict:
    math, other = read_prompts(math_run), read_prompts(other_run)
    other_name = other_run.name.split("-")[-2] if other_run.name.count("-") >= 2 else "other"
    rows = build_rows(math, other, n_math, n_other, n_val, seed, other_name=other_name)
    provenance = {"math_run": str(math_run), "other_run": str(other_run), "seed": seed,
                  "math_prompts_sha256": hashlib.sha256((math_run / "prompts.json").read_bytes()).hexdigest(),
                  "other_prompts_sha256": hashlib.sha256((other_run / "prompts.json").read_bytes()).hexdigest(),
                  "order": "training prompts shuffled with seed + 9973; validation prompts in the MATH point's order"}
    return write_pool(rows, out, provenance)


def env_lines(run: Path) -> list[str]:
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    lines = []
    for key, field in ENV_KEYS.items():
        value = config.get(field)
        if value is None:
            continue
        if isinstance(value, bool):
            value = "1" if value else "0"
        lines.append(f"{key}={shlex.quote(str(value))}")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--math-run", type=Path, required=True)
    p.add_argument("--other-run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--math", type=int, default=200)
    p.add_argument("--other", type=int, default=200)
    p.add_argument("--val", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("env")
    p.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            manifest = build(args.math_run.resolve(), args.other_run.resolve(), args.out.resolve(),
                             args.math, args.other, args.val, args.seed)
            print(f"[pool] {args.out}: {json.dumps(manifest.get('counts'))} sha256={manifest['sha256'][:12]}")
        else:
            print("\n".join(env_lines(args.run.resolve())))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
