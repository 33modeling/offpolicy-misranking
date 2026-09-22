"""Compare cached-SR and fresh selections at one saved policy, without training.

Saved LOO projections support directional diagnostics, NOT GRPO/Adam reward
increments or target-reaching H. No continuation outcomes are read here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from select_rules import jittered_topk, topk_count

REQUIRED = ("oracle_micro_groups.pt", "val_groups.pt",
            "rollouts_behavior_train.jsonl", "prompts.json", "run_config.json")
SCOPE = (
    "Same-policy saved LOO projected-gradient diagnostic, not GRPO reward gain, "
    "H prediction, or an executed switch. No later training/evaluation outcomes "
    "are inputs. Historical availability before a decision is not certified. "
    "Original full-pool scoring cost is not zero; only reaggregation avoids new GPU work."
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cached_rates(path: Path, n: int) -> dict[int, float]:
    rewards: dict[int, dict[int, float]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            i, j, reward = row["prompt_idx"], row["rollout_idx"], row["reward"]
            if type(i) is not int or type(j) is not int or min(i, j) < 0:
                raise ValueError("invalid cached prompt/rollout identity")
            if not isinstance(reward, (int, float)) or reward not in (0, 1):
                raise ValueError("cached rewards must be binary")
            if j in rewards.setdefault(i, {}):
                raise ValueError("duplicate cached rollout identity")
            rewards[i][j] = float(reward)
    if set(rewards) != set(range(n)):
        raise ValueError("cached rewards do not cover the candidate pool")
    counts = {len(rows) for rows in rewards.values()}
    if len(counts) != 1 or any(sorted(rows) != list(range(len(rows))) for rows in rewards.values()):
        raise ValueError("incomplete cached rollout coverage")
    return {i: sum(rows.values()) / len(rows) for i, rows in rewards.items()}


def cosine_rows(rows: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    denominator = rows.norm(dim=1) * direction.norm()
    return torch.where(denominator > 0, (rows @ direction) / denominator, 0.0)


def compare(micro: dict, validation: torch.Tensor, rates: dict[int, float],
            *, seed: int, frac: float = 0.1) -> dict:
    if not isinstance(micro, dict) or not isinstance(validation, torch.Tensor):
        raise ValueError("expected candidate mapping and validation tensor")
    if any(not isinstance(value, torch.Tensor) for value in micro.values()):
        raise ValueError("candidate gradients must be tensors")
    normalized = {int(i): value for i, value in micro.items()}
    n = len(normalized)
    if len(normalized) != len(micro) or set(normalized) != set(range(n)) or n < 2:
        raise ValueError("invalid candidate gradient IDs")
    if set(rates) != set(normalized) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in rates.values()):
        raise ValueError("invalid cached success-rate coverage")
    shapes = {tuple(value.shape) for value in normalized.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("inconsistent candidate gradient shapes")
    groups, dim = next(iter(shapes))
    if groups < 8 or groups % 4 or dim < 1:
        raise ValueError("candidate R/A/B partition requires at least eight groups, divisible by four")
    if validation.ndim != 2 or validation.shape[1] != dim or validation.shape[0] < 8 or validation.shape[0] % 4:
        raise ValueError("invalid validation R/A/B partition")
    # The experiment reserves first 1/4 candidate groups for matched ranking;
    # final two quarters and disjoint validation prompts evaluate frozen sets.
    full = torch.stack([normalized[i].double() for i in range(n)])
    validation = validation.double()
    if not torch.isfinite(full).all() or not torch.isfinite(validation).all():
        raise ValueError("non-finite projected gradients")
    q, vq = groups // 4, validation.shape[0] // 4
    ranking = cosine_rows(full[:, :q].mean(1), validation[:2 * vq].mean(0))
    k = topk_count(n, frac)
    rng = random.Random(seed + 424_243)
    sets = {
        "on_policy": sorted(jittered_topk(dict(enumerate(ranking.tolist())), k, seed + 1000)),
        "cached_sr": sorted(jittered_topk({i: -abs(p - 0.5) for i, p in rates.items()}, k, seed + 1000)),
        "random": sorted(jittered_topk({i: rng.random() for i in range(n)}, k, seed + 1000)),
    }
    dot_halves, cosine_halves = [], []
    for start, stop, vs, ve in ((2 * q, 3 * q, 2 * vq, 3 * vq),
                                (3 * q, 4 * q, 3 * vq, 4 * vq)):
        candidates = full[:, start:stop].mean(1)
        direction = validation[vs:ve].mean(0)
        dot_halves.append(candidates @ direction)
        cosine_halves.append(cosine_rows(candidates, direction))
    dot = torch.stack(dot_halves).mean(0)
    cosine = torch.stack(cosine_halves).mean(0)
    if not torch.isfinite(dot).all() or not torch.isfinite(cosine).all():
        raise ValueError("non-finite directional statistic")
    rows = {}
    for name, ids in sets.items():
        rows[name] = {
            "selected_ids": ids,
            "cached_success_rate_mean": sum(rates[i] for i in ids) / k,
            "reference_projected_dot": float(dot[ids].mean()),
            "reference_dot_a": float(dot_halves[0][ids].mean()),
            "reference_dot_b": float(dot_halves[1][ids].mean()),
            "reference_mean_cosine": float(cosine[ids].mean()),
            "dot_minus_uniform_expectation": float(dot[ids].mean() - dot.mean()),
        }
    return {
        "n": n, "k": k, "seed": seed, "projection_dim": dim,
        "candidate_groups": groups, "selectors": rows,
        "on_minus_sr_projected_dot": rows["on_policy"]["reference_projected_dot"] - rows["cached_sr"]["reference_projected_dot"],
        "on_sr_overlap_fraction": len(set(sets["on_policy"]) & set(sets["cached_sr"])) / k,
        "h_gpu_seconds": None,
        "h_status": "unavailable: LOO projected gradients are not GRPO/Adam reward increments; prospective cost inputs also required",
        "decision": "not_estimated",
        "uncertainty": "A/B are two disjoint reference estimates, not a confidence interval; projection error is not quantified",
        "per_prompt": [{"prompt_idx": i, "cached_success_rate": rates[i],
                        "ranking_cosine": float(ranking[i]), "reference_projected_dot": float(dot[i]),
                        "reference_mean_cosine": float(cosine[i]),
                        **{f"selected_{name}": i in ids for name, ids in sets.items()}}
                       for i in range(n)],
    }


def audit_point(point: Path, *, seed: int | None = None, frac: float = 0.1) -> dict:
    point = point.resolve()
    paths = [point / name for name in REQUIRED]
    protocol_path = point / "oracle_protocol.json"
    if protocol_path.is_file():
        paths.append(protocol_path)
    for path in paths:
        if not path.is_file():
            raise ValueError(f"missing {path.name}")
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    hashes = {p.name: digest(p) for p in paths}
    config = json.loads((point / "run_config.json").read_text())
    prompts = json.loads((point / "prompts.json").read_text())
    protocol = json.loads(protocol_path.read_text()) if protocol_path in paths else {}
    if seed is None:
        seed = config.get("seed")
        if seed is None:
            match = re.search(r"(?:^|[-/])s(\d+)(?:[-/]|$)", str(point))
            seed = int(match[1]) if match else None
    if type(seed) is not int or seed < 0:
        raise ValueError("selection seed missing; supply --seed explicitly")
    for field in ("artifact_sha256", "manifest_sha256"):
        for name, expected in protocol.get("generation_validation", {}).get(field, {}).items():
            if name in hashes and hashes[name] != expected:
                raise ValueError(f"recorded input hash mismatch: {name}")
    micro = torch.load(point / "oracle_micro_groups.pt", map_location="cpu", weights_only=True)
    validation = torch.load(point / "val_groups.pt", map_location="cpu", weights_only=True)
    if not isinstance(micro, dict) or len(micro) != len(prompts["train"]):
        raise ValueError("candidate gradients differ from prompt pool")
    if not isinstance(validation, torch.Tensor) or validation.ndim != 2 or validation.shape[0] != len(prompts["val"]):
        raise ValueError("validation gradients differ from prompt pool")
    result = compare(micro, validation, cached_rates(point / "rollouts_behavior_train.jsonl", len(micro)), seed=seed, frac=frac)
    if any((p.stat().st_size, p.stat().st_mtime_ns) != before[p] for p in paths):
        raise ValueError("source changed while reading; retry after scoring completes")
    result.update(point=str(point), model=config.get("model"), input_sha256=hashes,
                  recorded_oracle_schema=protocol.get("schema"), scope=SCOPE,
                  metadata=config, historical_availability_certified=False)
    return result


def discover(roots: list[Path]) -> list[Path]:
    points, visited = set(), set()
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"root does not exist: {root}")
        for directory, children, files in os.walk(root, followlinks=True):
            path = Path(directory).resolve()
            if path in visited:
                children[:] = []
                continue
            visited.add(path)
            children[:] = sorted(c for c in children if not c.startswith((".", "checkpoint-"))
                                 and c not in {"policy", "exports", "node_modules", "__pycache__"})
            if "oracle_micro_groups.pt" in files or "scores_splithalf.json" in files:
                points.add(path)
    return sorted(points)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    work = Path(os.environ.get("OM_WORK", "/group-volume/minsoo3.kim/offpolicy-misranking"))
    parser.add_argument("roots", nargs="*", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, help="Override selection seed for every point")
    parser.add_argument("--frac", type=float, default=0.1)
    args = parser.parse_args(argv)
    if not 0 < args.frac <= 1:
        parser.error("--frac must be in (0, 1]")
    roots = [p.resolve() for p in (args.roots or [work / "runs"])]
    out = (args.out or work / "exports" / ("sr-gradient-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))).resolve()
    if any(out == root or root in out.parents for root in roots):
        parser.error("output must be outside the input roots")
    if out.exists():
        parser.error("output already exists; choose a new directory")
    points = discover(roots)
    if not points:
        parser.error("no saved scoring points found under the supplied roots")
    torch.set_num_threads(1)
    started = time.monotonic()
    results, errors = [], []
    for point in points:
        try:
            row = audit_point(point, seed=args.seed, frac=args.frac)
            results.append(row)
            print(f"[point] {point.name}: on-SR dot={row['on_minus_sr_projected_dot']:+.6g}; H=unavailable", flush=True)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            errors.append({"point": str(point), "error": str(exc)})
            print(f"[skip] {point}: {exc}", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    document = {"schema": "cached-sr-gradient-audit/v1", "scope": SCOPE,
                "created_at": datetime.now(timezone.utc).isoformat(), "roots": list(map(str, roots)),
                "analysis_wall_seconds": time.monotonic() - started,
                "new_gpu_work": False, "results": results, "errors": errors}
    (out / "data.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    with (out / "comparison.csv").open("w", newline="") as handle:
        fields = ["point", "seed", "selector", "reference_projected_dot", "reference_dot_a", "reference_dot_b",
                  "dot_minus_uniform_expectation", "on_minus_sr_projected_dot", "h_gpu_seconds", "h_status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            for selector, values in row["selectors"].items():
                writer.writerow({**{k: row[k] for k in ("point", "seed", "on_minus_sr_projected_dot", "h_gpu_seconds", "h_status")},
                                 "selector": selector, **{k: values[k] for k in fields if k in values}})
    print(f"REPORT {out / 'data.json'}\nTABLE  {out / 'comparison.csv'}\n{len(results)} calculated; {len(errors)} skipped. No H predictions generated.")
    return 0 if results and not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
