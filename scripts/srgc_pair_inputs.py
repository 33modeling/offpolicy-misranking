"""Run seeds 5–9 on the seed 3/4 source splits with their existing initial-policy caches.

The completed seed 3/4 runs already hold ``prompts.json`` and
``rollouts_behavior_train.jsonl`` (eight initial-policy rewards per candidate).
Their ``switch.json`` lists them under ``sources`` with the frozen independent
``evaluation``. The bundles built here carry complete ``cached_rewards``, so the
queue marks every cache task complete and the shared prefix starts at once.

MATH reads ``SRGC_PAIR_ROOT`` (default ``$OM_WORK/runs/selector-pair-v1``) and
builds bundles with the existing ``import_pair_inputs.build_bundle``. MBPP reads
``SRGC_MBPP_PAIR_ROOT`` (default ``$OM_WORK/runs/selection-switch-mbpp-v1``).
Without a source root the prepared plan and GPU cache generation are unchanged.
"""

import json
import os
from pathlib import Path
import sys

from srgc_rebuttal.build_inputs import CODE_PROMPT, SCHEMA, record_id
from srgc_rebuttal.import_pair_inputs import (RESPONSES, build_bundle, cached_rewards, parse_assignment, sha256,
                                              write_bundle)
from srgc_rebuttal.plan import input_path, load_plan, validate_inputs

PLANS = {"math": ("pair_seeds.json", "additional_seeds.json", "SRGC_PAIR_ROOT", "selector-pair-v1",
                  "olmo_rlzero_math"),
         "mbpp": ("mbpp_pair_seeds.json", "mbpp_seeds.json", "SRGC_MBPP_PAIR_ROOT", "selection-switch-mbpp-v1",
                  "olmo_rlzero_code")}
EVALUATION = 300


def say(message):
    print(f"[inputs] {message}", file=sys.stderr, flush=True)


def source_root(dataset, environment):
    _, _, variable, default, _ = PLANS[dataset]
    if environment.get(variable):
        return Path(environment[variable])
    group = Path(environment.get("GROUP_VOLUME", "/group-volume"))
    work = Path(environment.get("OM_WORK", str(group / environment.get("OM_USER", "minsoo3.kim") / "offpolicy-misranking")))
    return work / "runs" / default


def manifest(root):
    """``switch.json`` of a selection-switch root, or of the first selector-pair branch."""
    found = [root / "switch.json"] if (root / "switch.json").is_file() else sorted(root.glob("branches/*/switch.json"))
    return found[0] if found else None


def check_source(run, prompt_format):
    config = run / "run_config.json"
    if config.is_file():
        value = json.loads(config.read_text())
        if value.get("prompt_format") != prompt_format or value.get("behavior_k", RESPONSES) != RESPONSES:
            raise ValueError(f"{run} is not a {prompt_format} source with {RESPONSES} cached responses")


def code_bundle(seed, origin, run, evaluation):
    """MBPP counterpart of ``import_pair_inputs.build_bundle`` (same checks, code prompt and verifier)."""
    prompts = json.loads((run / "prompts.json").read_text())
    train, val = prompts["train"], prompts["val"]
    test = evaluation["test"]
    if len(test) < EVALUATION:
        raise ValueError(f"source evaluation has {len(test)} questions; the frozen plan needs {EVALUATION}")
    test = test[:EVALUATION]
    if len(val) % 2:
        raise ValueError("validation pool must split into two equal halves (R and A/B)")
    rewards = cached_rewards(run, len(train))
    records, groups = {}, {}
    for name, items in (("candidate_ids", train), ("validation_pool_ids", val), ("evaluation_ids", test)):
        ids = []
        for item in items:
            answer = str(item["answer"]).strip()
            if not answer.startswith("assert"):
                raise ValueError("MBPP answers must be executable assert tests")
            rid = record_id("mbpp", item["question"])
            records[rid] = {"question": item["question"], "prompt": CODE_PROMPT.format(question=item["question"]),
                            "answer": answer}
            ids.append(rid)
        groups[name] = ids
    bundle = {"schema": SCHEMA, "dataset": "mbpp", "records": records, **groups,
              "ranking_validation_ids": groups["validation_pool_ids"][: len(val) // 2],
              "cached_rewards": dict(zip(groups["candidate_ids"], rewards)),
              "provenance": {"experiment_seed": seed, "reused_from_seed": origin, "source_run": str(run),
                             "split_seed": origin, "prompt_format": "olmo_rlzero_code",
                             "ranking_validation": "first half of the source validation pool (manuscript R split)",
                             "cache": {"file": str(run / "rollouts_behavior_train.jsonl"),
                                       "sha256": sha256(run / "rollouts_behavior_train.jsonl"),
                                       "responses": RESPONSES,
                                       "source": "initial-policy behavior rollouts of the source run"},
                             "prompts_sha256": sha256(run / "prompts.json"),
                             "evaluation": {**evaluation.get("provenance", {}), "first": EVALUATION},
                             "verifier": "srgc_rebuttal.verifiers:code_reward"}}
    validate_inputs(bundle)
    return bundle


def import_bundles(dataset, plan_path, plan, root):
    switch = json.loads(manifest(root).read_text())
    sources = {int(seed): Path(entry["path"]) for seed, entry in switch["sources"].items()}
    assignment = parse_assignment(None)
    built = {}
    for seed in plan["seeds"]:
        origin = assignment.get(seed)
        if origin not in sources:
            raise ValueError(f"seed {seed}: source seed {origin} missing under {root}; available {sorted(sources)}")
        check_source(sources[origin], PLANS[dataset][4])
        build = build_bundle if dataset == "math" else code_bundle
        built[seed] = (origin, build(seed, origin, sources[origin], switch["evaluation"]))
    for seed, (origin, bundle) in built.items():
        write_bundle(input_path(plan_path, plan, seed), bundle)
        solved = sum(1 for rewards in bundle["cached_rewards"].values() if any(rewards))
        say(f"seed {seed} <- seed {origin} split and cache ({solved}/{len(bundle['candidate_ids'])} "
            "candidates with a cached success)")


def active(dataset, environment):
    try:
        from srgc_shared_storage import storage_root
        _, root = storage_root(environment)
    except ValueError:
        return False
    return (root / f".{Path(PLANS[dataset][0]).stem}-active.json").is_file()


def default_plan(root, dataset, environment, *, writing):
    experiments = Path(root) / "srgc_rebuttal/experiments"
    pair_name, prepared_name = PLANS[dataset][:2]
    pair, prepared = experiments / pair_name, experiments / prepared_name
    plan = load_plan(pair)
    if all(input_path(pair, plan, seed).is_file() for seed in plan["seeds"]) or active(dataset, environment):
        return pair
    source = source_root(dataset, environment)
    if not writing:
        return prepared
    if manifest(source) is None:
        say(f"no seed 3/4 {dataset} source at {source}; generating caches with the prepared plan")
        return prepared
    try:
        import_bundles(dataset, pair, plan, source)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        say(f"seed 3/4 {dataset} reuse unavailable ({exc}); generating caches with the prepared plan")
        return prepared
    say(f"reusing seed 3/4 {dataset} splits and caches from {source}; no cache generation")
    return pair
