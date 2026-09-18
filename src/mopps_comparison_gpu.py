"""Read-only import of switch checkpoints; isolated MoPPS continuation queue."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import time

import mopps
import selection_gate as core
import selection_gate_gpu as base
import selection_switch as rule
import selection_switch_gpu as switch

HERE = Path(__file__).resolve()
CODE = (*switch.CODE, "src/mopps.py", "src/mopps_comparison_gpu.py", "src/train_mopps_grpo.py")
_base_policy = base.policy
_base_verify = base.verify
PRE_CODE_COMPAT_CODE = "585e2e9efc8cb79433d90daabf589112c2bb64968bcd6b3fceef7a849d286def"
PRE_LIFECYCLE_CODE = "ef5eb15dbd1d646adff3c515932d7b03171fcce937d8d8d69c44ee5e2e41da40"
PRE_QUEUE_FAILURE_CODE = "0bb0f42a82215281267a5f8741f1e9b87d16ccaaf945ace32c81f50c6400f33a"
PRE_CACHE_GUARD_CODE = "7a4e2b957080e1af5158e9802005dfcd13a51234e3400c9011c35ed6a9c1e5f9"
PRE_NONBLOCKING_RETRY_CODE = "46f08b629491f95ec180ead37440dd92c1161d19d34a81dfd14ffebfcdeb9756"
# Exact 89c26af runtime before Switch held-out controls could run ahead of the gate.
PRE_TEST_PARALLEL_CODE = "b49a7417bf1f00c8c164c6bd9d0aa480dd4d7438e2de4d6f8d505d53a3e2dab2"
# Exact 091ae20 runtime before Switch workers survived a failing gate fit.
PRE_FIT_RESILIENCE_CODE = "ca5486eda6ba169460ea15378986a6c514d76c421d05137d98bf179a7f9a9f2b"
# Exact b8d90c0 runtime before Switch variant roots could import certified prefixes.
PRE_VARIANT_ROOT_CODE = "2989e14c8cfb391158cb00171881afbc37cca0682c58447625fa8dffcc01f21c"
# Exact 47339ca runtime before Switch prepare could build an MBPP root.
PRE_DATASET_CODE = "c12c6e7956f4a61e648c29b3da2762bc71fc0713b68bd77012aed59004ecdddd"
# Exact released shared runtimes; MoPPS's own selector and trainer are unchanged.
PRE_SHARED_ALLOCATION_CODE = "3b6ad4bb17b2e0c7cec64c535f86430bb5dd2af0f7940cec9b173634c9d462e6"
PRE_SHARED_RECOVERY_CODE = "1e57e00d07645d49e28cbacc693917d1c00a41f9d3b2df16e818c9756ce30419"
PRE_CHECKPOINT_RETENTION_CODE = "af93d909a6f2afc3545ea23e7b3c43939d2163e29811ee652e6ce52cf53299e1"
RECOVERY_PREDECESSORS = {PRE_SHARED_ALLOCATION_CODE, PRE_SHARED_RECOVERY_CODE, PRE_CHECKPOINT_RETENTION_CODE}
PRIOR_RUNTIME_CODES = {PRE_CODE_COMPAT_CODE, PRE_LIFECYCLE_CODE, PRE_QUEUE_FAILURE_CODE,
                       PRE_CACHE_GUARD_CODE, PRE_NONBLOCKING_RETRY_CODE, PRE_TEST_PARALLEL_CODE,
                       PRE_FIT_RESILIENCE_CODE, PRE_VARIANT_ROOT_CODE, PRE_DATASET_CODE,
                       *RECOVERY_PREDECESSORS}


def hashes():
    return {name: base.digest(base.ROOT / name) for name in CODE}


def read_parent(root):
    p = core.read(root / "switch.json")
    if (p["schema"] != rule.SCHEMA or p["test_seeds"] != list(rule.TEST_SEEDS)
            or p["steps"] != list(rule.STEPS)):
        raise ValueError("requires the registered selected-prefix switch experiment")
    switch.validate_code_hashes(p["code_hashes"])
    return p


def prepare(root, parent):
    root, parent = root.resolve(), parent.resolve()
    p = read_parent(parent)
    protected = [parent, base.ROOT, *(Path(v["path"]).resolve() for v in p["sources"].values()),
                 *(Path(v["config"]["model"]).resolve() for v in p["sources"].values())]
    if any(root == path or path in root.parents or root in path.parents for path in protected):
        raise ValueError("comparison output must be separate from source runs, models and repository")
    value = {"schema": mopps.SCHEMA, "parent": str(parent),
             "parent_switch_sha256": base.digest(parent / "switch.json"), "code_hashes": hashes(),
             "paper": mopps.PAPER, "config": mopps.specification("mopps", 3, 25, 4)["config"],
             "arms": list(mopps.ARMS), "seeds": list(rule.TEST_SEEDS), "steps": list(rule.STEPS),
             "budget_gpu_seconds": p["budget_gpu_seconds"], "gpu_type": p["gpu_type"],
             "eval_timeout": p["eval_timeout"],
             "primary_comparison": "executed_gated_minus_online_mopps",
             "design": "online selector replacement from fresh_r prefixes; not a MoPPS-prefix stopping study",
             "initialization": "uniform prior at each branch; no cached, validation or future rewards",
             "sampling": "full original training pool; four prompts without replacement per update; K=8",
             "cost": "all selection, feedback, training, startup and retries charged; evaluation separate"}
    with base.lease(root / ".prepare.lock", blocking=True):
        if (root / "mopps.json").exists():
            recorded = core.read(root / "mopps.json")
            if recorded != {**value, "code_hashes": recorded.get("code_hashes")}:
                raise ValueError(f"frozen contract changed: {root / 'mopps.json'}")
            value = protocol(root)
        else:
            base.bind(root / "mopps.json", value)
    print(f"[prepared] MoPPS (KDD 2026) + random_online: 12 continuations; B={p['budget_gpu_seconds']:.0f} GPU-s")
    return value


def protocol(root):
    p = core.read(root / "mopps.json")
    current = hashes()
    recorded = p.get("code_hashes")
    if recorded != current:
        if (not isinstance(recorded, dict) or core.fingerprint(recorded) not in PRIOR_RUNTIME_CODES
                or set(recorded) != set(current)
                or current["src/net_gate_memory_worker.py"] not in {switch.PRE_CACHE_GUARD_WORKER, switch.CACHE_GUARD_WORKER}
                or any(current[name] not in allowed for name, allowed in switch.PUBLICATION_PATCH_HASHES.items())
                or any(recorded[name] != sha for name, sha in current.items()
                       if name not in {"src/selection_switch_gpu.py", "src/mopps_comparison_gpu.py", "src/selection_gate_gpu.py", "src/net_gate_memory_worker.py", "src/net_gain_gate_gpu.py", "src/train_selection_gate_grpo.py", "src/train_policy_grpo.py"})):
            raise ValueError("frozen MoPPS experiment changed: unreviewed code hashes")
        switch.validate_code_hashes({name: recorded[name] for name in switch.CODE})
    if (p["schema"] != mopps.SCHEMA
            or p["arms"] != list(mopps.ARMS) or p["seeds"] != list(rule.TEST_SEEDS)
            or p["steps"] != list(rule.STEPS) or p["paper"] != mopps.PAPER
            or p["primary_comparison"] != "executed_gated_minus_online_mopps"
            or p["config"] != mopps.specification("mopps", 3, 25, 4)["config"]):
        raise ValueError("frozen MoPPS experiment changed")
    parent = Path(p["parent"])
    if base.digest(parent / "switch.json") != p["parent_switch_sha256"]:
        raise ValueError("parent switch manifest changed")
    source = read_parent(parent)
    if any(p[key] != source[key] for key in ("budget_gpu_seconds", "gpu_type", "eval_timeout")):
        raise ValueError("comparison allocation differs from original branches")
    if recorded != current:
        with base.lease(root / ".code-compat-runtime.lock", blocking=True):
            receipt = {
                "schema": "mopps-code-compat-runtime/v1", "protocol_sha256": base.digest(root / "mopps.json"),
                "original_code_hashes": recorded, "runtime_code_hashes": current,
                "change": "switch frozen-runtime compatibility and diagnostics only",
                "cost_policy": "no change to selectors, training, frozen artifacts or budgets",
            }
            path = root / "code-compat-runtime.json"
            if path.exists():
                previous = core.read(path)
                previous_code = previous.get("runtime_code_hashes")
                if (previous != receipt and
                        (previous != {**receipt, "runtime_code_hashes": previous_code}
                         or core.fingerprint(previous_code) not in {PRE_LIFECYCLE_CODE, PRE_QUEUE_FAILURE_CODE, PRE_CACHE_GUARD_CODE, PRE_NONBLOCKING_RETRY_CODE, PRE_TEST_PARALLEL_CODE, PRE_FIT_RESILIENCE_CODE, PRE_VARIANT_ROOT_CODE, PRE_DATASET_CODE, *RECOVERY_PREDECESSORS})):
                    raise ValueError(f"frozen contract changed: {path}")
            else:
                base.bind(path, receipt)
            lifecycle_path = root / "worker-lifecycle-runtime.json"
            lifecycle_receipt = {
                "schema": "mopps-worker-lifecycle-runtime/v1",
                "protocol_sha256": base.digest(root / "mopps.json"),
                "compat_runtime_sha256": base.digest(path), "runtime_code_hashes": current,
                "change": "signal cleanup and independent certified-prefix import; Gate evidence checked at comparison",
                "cost_policy": "same policies, sampling, optimizer, evaluation and budgets; no source writes",
            }
            if lifecycle_path.exists():
                previous = core.read(lifecycle_path)
                previous_code = previous.get("runtime_code_hashes")
                if (previous != lifecycle_receipt and
                        (previous != {**lifecycle_receipt, "runtime_code_hashes": previous_code}
                         or core.fingerprint(previous_code) not in {PRE_QUEUE_FAILURE_CODE, PRE_CACHE_GUARD_CODE, PRE_NONBLOCKING_RETRY_CODE, PRE_TEST_PARALLEL_CODE, PRE_FIT_RESILIENCE_CODE, PRE_VARIANT_ROOT_CODE, PRE_DATASET_CODE, *RECOVERY_PREDECESSORS})):
                    raise ValueError(f"frozen contract changed: {lifecycle_path}")
            else:
                base.bind(lifecycle_path, lifecycle_receipt)
            queue_path = root / "queue-failure-runtime.json"
            queue_receipt = {
                "schema": "mopps-queue-failure-runtime/v1",
                "protocol_sha256": base.digest(root / "mopps.json"),
                "lifecycle_runtime_sha256": base.digest(lifecycle_path), "runtime_code_hashes": current,
                "change": "distinguish failed prefix dependencies from live producers; drain independent ready tasks",
                "cost_policy": "no automatic retries, source writes, policy changes or cost waivers",
            }
            if queue_path.exists():
                previous = core.read(queue_path)
                previous_code = previous.get("runtime_code_hashes")
                if (previous != queue_receipt and
                        (previous != {**queue_receipt, "runtime_code_hashes": previous_code}
                         or core.fingerprint(previous_code) not in {PRE_CACHE_GUARD_CODE, PRE_NONBLOCKING_RETRY_CODE, PRE_TEST_PARALLEL_CODE, PRE_FIT_RESILIENCE_CODE, PRE_VARIANT_ROOT_CODE, PRE_DATASET_CODE, *RECOVERY_PREDECESSORS})):
                    raise ValueError(f"frozen contract changed: {queue_path}")
            else:
                base.bind(queue_path, queue_receipt)
            if current["src/net_gate_memory_worker.py"] == switch.CACHE_GUARD_WORKER:
                cache_path = root / "cache-guard-runtime.json"
                cache_receipt = {
                    "schema": "mopps-cache-guard-runtime/v1",
                    "protocol_sha256": base.digest(root / "mopps.json"),
                    "queue_runtime_sha256": base.digest(queue_path), "runtime_code_hashes": current,
                    "change": "checkpointed decoder cache guard in shared Switch scoring; MoPPS trainer unchanged",
                    "cost_policy": "same policies, gradients and budgets; no parent writes or cost waivers",
                }
                if cache_path.exists():
                    previous = core.read(cache_path)
                    previous_code = previous.get("runtime_code_hashes")
                    if (previous != cache_receipt and
                            (previous != {**cache_receipt, "runtime_code_hashes": previous_code}
                             or core.fingerprint(previous_code) not in {PRE_NONBLOCKING_RETRY_CODE, PRE_TEST_PARALLEL_CODE, PRE_FIT_RESILIENCE_CODE, PRE_VARIANT_ROOT_CODE, PRE_DATASET_CODE, *RECOVERY_PREDECESSORS})):
                        raise ValueError(f"frozen contract changed: {cache_path}")
                else:
                    base.bind(cache_path, cache_receipt)
                retry_path = root / "nonblocking-retry-runtime.json"
                retry_receipt = {
                    "schema": "mopps-nonblocking-retry-runtime/v1",
                    "protocol_sha256": base.digest(root / "mopps.json"),
                    "cache_guard_runtime_sha256": base.digest(cache_path), "runtime_code_hashes": current,
                    "change": "explicit zero idle timeout skips busy branches, including active peers",
                    "cost_policy": "same per-branch locks, policies and budgets; no parent writes or cost waivers",
                }
                if retry_path.exists():
                    previous = core.read(retry_path)
                    previous_code = previous.get("runtime_code_hashes")
                    if (previous != retry_receipt and
                            (previous != {**retry_receipt, "runtime_code_hashes": previous_code}
                             or core.fingerprint(previous_code) not in {PRE_TEST_PARALLEL_CODE, PRE_FIT_RESILIENCE_CODE, PRE_VARIANT_ROOT_CODE, PRE_DATASET_CODE, *RECOVERY_PREDECESSORS})):
                        raise ValueError(f"frozen contract changed: {retry_path}")
                else:
                    base.bind(retry_path, retry_receipt)
                parallel_path = root / "test-parallel-runtime.json"
                parallel_receipt = {
                    "schema": "mopps-test-parallel-runtime/v1",
                    "protocol_sha256": base.digest(root / "mopps.json"),
                    "retry_runtime_sha256": base.digest(retry_path), "runtime_code_hashes": current,
                    "change": "Switch held-out states bind the gate in gate.json with separate control and gate "
                              "barriers; origin binding requires both; certified-prefix imports unchanged",
                    "cost_policy": "same selectors, policies and budgets; no parent writes or cost waivers",
                }
                if parallel_path.exists():
                    previous = core.read(parallel_path)
                    previous_code = previous.get("runtime_code_hashes")
                    if (previous != parallel_receipt and
                            (previous != {**parallel_receipt, "runtime_code_hashes": previous_code}
                             or core.fingerprint(previous_code) not in {PRE_FIT_RESILIENCE_CODE, PRE_VARIANT_ROOT_CODE, PRE_DATASET_CODE, *RECOVERY_PREDECESSORS})):
                        raise ValueError(f"frozen contract changed: {parallel_path}")
                else:
                    base.bind(parallel_path, parallel_receipt)
                resilience_path = root / "fit-resilience-runtime.json"
                resilience_receipt = {
                    "schema": "mopps-fit-resilience-runtime/v1",
                    "protocol_sha256": base.digest(root / "mopps.json"),
                    "test_parallel_runtime_sha256": base.digest(parallel_path), "runtime_code_hashes": current,
                    "change": "shared Switch worker records a failing gate fit instead of exiting; MoPPS unchanged",
                    "cost_policy": "same selectors, policies and budgets; no parent writes or cost waivers",
                }
                if resilience_path.exists():
                    previous = core.read(resilience_path)
                    previous_code = previous.get("runtime_code_hashes")
                    if (previous != resilience_receipt and
                            (previous != {**resilience_receipt, "runtime_code_hashes": previous_code}
                             or core.fingerprint(previous_code) not in {PRE_VARIANT_ROOT_CODE, PRE_DATASET_CODE, *RECOVERY_PREDECESSORS})):
                        raise ValueError(f"frozen contract changed: {resilience_path}")
                else:
                    base.bind(resilience_path, resilience_receipt)
                variant_path = root / "variant-root-runtime.json"
                variant_receipt = {
                    "schema": "mopps-variant-root-runtime/v1",
                    "protocol_sha256": base.digest(root / "mopps.json"),
                    "fit_resilience_runtime_sha256": base.digest(resilience_path), "runtime_code_hashes": current,
                    "change": "shared Switch prepare can import certified prefixes for a variant root; MoPPS unchanged",
                    "cost_policy": "same selectors, policies and budgets; no parent writes or cost waivers",
                }
                if variant_path.exists():
                    previous = core.read(variant_path)
                    previous_code = previous.get("runtime_code_hashes")
                    if (previous != variant_receipt and
                            (previous != {**variant_receipt, "runtime_code_hashes": previous_code}
                             or core.fingerprint(previous_code) not in {PRE_DATASET_CODE, *RECOVERY_PREDECESSORS})):
                        raise ValueError(f"frozen contract changed: {variant_path}")
                else:
                    base.bind(variant_path, variant_receipt)
                dataset_path = root / "dataset-runtime.json"
                dataset_receipt = {
                    "schema": "mopps-dataset-runtime/v1",
                    "protocol_sha256": base.digest(root / "mopps.json"),
                    "variant_root_runtime_sha256": base.digest(variant_path), "runtime_code_hashes": current,
                    "change": "shared Switch prepare accepts --dataset mbpp; this MATH comparison is unchanged",
                    "cost_policy": "same selectors, policies and budgets; no parent writes or cost waivers",
                }
                if dataset_path.exists():
                    previous = core.read(dataset_path)
                    previous_code = previous.get("runtime_code_hashes")
                    if (previous != dataset_receipt and
                            (previous != {**dataset_receipt, "runtime_code_hashes": previous_code}
                             or core.fingerprint(previous_code) not in RECOVERY_PREDECESSORS)):
                        raise ValueError(f"frozen contract changed: {dataset_path}")
                else:
                    base.bind(dataset_path, dataset_receipt)
                if core.fingerprint(current) not in PRIOR_RUNTIME_CODES - {PRE_CHECKPOINT_RETENTION_CODE}:
                    recovery_path = root / "shared-recovery-runtime.json"
                    recovery_receipt = {
                        "schema": "mopps-shared-recovery-runtime/v1",
                        "protocol_sha256": base.digest(root / "mopps.json"),
                        "dataset_runtime_sha256": base.digest(dataset_path), "runtime_code_hashes": current,
                        "change": "reviewed shared Switch saved-policy/checkpoint publication recovery only; "
                                  "MoPPS selector and training implementation unchanged",
                        "cost_policy": "preserve all prior receipts, policies, costs, frozen choices and budgets",
                    }
                    if recovery_path.exists():
                        previous = core.read(recovery_path)
                        previous_code = previous.get("runtime_code_hashes")
                        if (previous != recovery_receipt and
                                (previous != {**recovery_receipt, "runtime_code_hashes": previous_code}
                                 or core.fingerprint(previous_code) != PRE_CHECKPOINT_RETENTION_CODE)):
                            raise ValueError(f"frozen contract changed: {recovery_path}")
                    else:
                        base.bind(recovery_path, recovery_receipt)
                    if core.fingerprint(current) != PRE_CHECKPOINT_RETENTION_CODE:
                        base.bind(root / "checkpoint-retention-runtime.json", {
                            "schema": "mopps-checkpoint-retention-runtime/v1",
                            "protocol_sha256": base.digest(root / "mopps.json"),
                            "shared_recovery_runtime_sha256": base.digest(recovery_path), "runtime_code_hashes": current,
                            "change": "shared checkpoint retention validates older checkpoint contracts and never prunes the newly committed checkpoint",
                            "cost_policy": "preserve all policies, results, costs, choices and budgets; no MoPPS selector or optimizer change",
                        })
    return p


def point(root, seed, step):
    return root / "states" / f"s{seed}-t{step}"


def original_point(p, seed, step):
    return switch.child_root(Path(p["parent"]), seed, step) / "points" / f"view-{step}"


def origin_frozen(out):
    """The original held-out state with its control and gate decisions both frozen."""
    return all((out / name).is_file() for name in ("contract.json", "decisions-frozen.json", "gate-frozen.json"))


def ready(p, seed, step):
    out = original_point(p, seed, step)
    return ((switch.prefix_dir(Path(p["parent"]), seed) / f"prefix-{step}.json").is_file()
            or origin_frozen(out))


def prefix_dependency(p, seed, step):
    """Inspect the earliest missing segment without creating source files."""
    directory = switch.prefix_dir(Path(p["parent"]), seed)
    with contextlib.ExitStack() as stack:
        try:
            handle = stack.enter_context((directory / ".prefix.lock").open("rb"))
        except FileNotFoundError:
            handle = None
        if handle is not None:
            try:
                # A shared probe works with a read-only descriptor on NFS too.
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"active": True, "failure": None}
        for current in rule.STEPS:
            if current > step:
                break
            if not (directory / f"prefix-{current}.json").is_file():
                path = directory / f"segment-{current}" / "failure.json"
                if path.is_file():
                    error = str(core.read(path).get("error", "unknown prefix failure"))
                    return {"active": False, "failure": f"{path}: {error[:500]}"}
                break
    return {"active": False, "failure": None}


def verify_prefix(p, seed, step):
    """Validate the immutable checkpoint without requiring a fitted Gate."""
    import evidence_downstream as ed
    from train_policy_grpo import validate_policy_lineage
    parent = Path(p["parent"])
    source_protocol = read_parent(parent)
    item = source_protocol["sources"][str(seed)]
    for name, digest in item["hashes"].items():
        if base.digest(Path(item["path"]) / name) != digest:
            raise ValueError("initial selected-prefix source changed")
    directory = switch.prefix_dir(parent, seed)
    cfg = item["config"]
    if "model_sha256" in item and base.digest(Path(cfg["model"]) / "config.json") != item["model_sha256"]:
        raise ValueError("initial selected-prefix model changed")
    if core.read(directory / "subset.json") != item["subset"]:
        raise ValueError("prefix selected subset changed")
    previous = 0
    for current in rule.STEPS:
        policy_path = directory / f"policy_step_{current}"
        validate_policy_lineage(policy_path, target_steps=current, world_size=4, training_objective="grpo",
            expected_start_step=previous, expected_parent=directory / f"policy_step_{previous}" if previous else None,
            expected_model=Path(cfg["model"]), expected_seed=seed, expected_max_new_tokens=cfg["max_new_tokens"],
            expected_prompt_format=cfg["prompt_format"], expected_config=ed._expected_config(cfg),
            expected_prompts=directory / "subset.json", require_complete_hashes=True)
        cert = {"schema": rule.SCHEMA, "seed": seed, "step": current, "previous": previous,
                "selector": "fresh_r", "subset_sha256": base.digest(directory / "subset.json"),
                "source_sha256": core.fingerprint(item),
                "policy_hashes": {name: base.digest(policy_path / name) for name in ed.POLICY_FILES}}
        if core.read(directory / f"prefix-{current}.json") != cert:
            raise ValueError("unverified selected-prefix history")
        if current == step:
            break
        previous = current
    else:
        raise ValueError("unregistered checkpoint")
    return source_protocol, item, directory, cert


def verify_origin(p, seed, step):
    """Validate legacy imports and final Gate evidence without source writes."""
    import evidence_downstream as ed
    source_protocol, item, directory, cert = verify_prefix(p, seed, step)
    parent, cfg = Path(p["parent"]), item["config"]
    origin = original_point(p, seed, step)
    c = _base_verify(origin)
    source = directory / f"view-{step}"
    prompts = core.read(Path(item["path"]) / "prompts.json")
    expected_hashes = {name: base.digest(source / name) for name in
                       ["run_config.json", "prompts.json", "selected-prefix.json", "rollouts_behavior_train.jsonl"]
                       + [f"policy_step_{step}/{name}" for name in ed.POLICY_FILES]}
    if (c["selected_prefix"] != {"schema": rule.SCHEMA, "root": str(parent),
                                 "certificate_sha256": core.fingerprint(cert)}
            or c["config"] != {**cfg, "drift": step} or c["budget_gpu_seconds"] != p["budget_gpu_seconds"]
            or c["scope"]["gpu_type"] != p["gpu_type"] or c["role"] != "test"
            or c["scope"]["selector"] != "fresh_r" or Path(c["source_run"]) != source
            or c["source_hashes"] != expected_hashes
            or (source / f"policy_step_{step}").resolve() != (directory / f"policy_step_{step}").resolve()
            or c["n"] != len(prompts["train"]) or c["eval_k"] != source_protocol["eval_k"]
            or c["eval_seed"] != 701_000_003 + seed*1_000_003 or c["max_steps"] != 100000
            or c["evaluation"] != {"val": ed.independent_test(prompts, source_protocol["evaluation"]),
                                   "provenance": source_protocol["evaluation"]["provenance"]}):
        raise ValueError("source state differs from registered comparison")
    net = core.read(origin.parent.parent / "net_protocol.json")
    gate = core.read(switch.gate_path(origin.parent.parent))
    barrier = core.read(origin / "decisions-frozen.json")
    gate_barrier = core.read(origin / "gate-frozen.json")
    controls = [arm for arm in rule.TEST_ARMS if arm != "gated"]
    if (net["schema"] != rule.SCHEMA or net["mode"] != "test" or net["arms"] != list(rule.TEST_ARMS)
            or net["model"] is not None or net["code_hashes"] != source_protocol["code_hashes"]
            or gate["protocol_sha256"] != core.fingerprint(net) or gate["model"] != core.read(parent / "model.json")
            or gate["model_sha256"] != base.digest(parent / "model.json")
            or barrier["protocol_sha256"] != core.fingerprint(net)
            or barrier["decisions"] != {arm: base.digest(origin / arm / "decision.json") for arm in controls}
            or gate_barrier["protocol_sha256"] != core.fingerprint(net)
            or gate_barrier["gate_sha256"] != base.digest(switch.gate_path(origin.parent.parent))
            or gate_barrier["controls_sha256"] != base.digest(origin / "decisions-frozen.json")
            or gate_barrier["decision"] != base.digest(origin / "gated/decision.json")):
        raise ValueError("original gate must be frozen before comparison training")
    rule.validate_model(gate["model"])
    return c


def prefix_contract(out, p, seed, step, *, publish=False):
    source_protocol, item, directory, cert = verify_prefix(p, seed, step)
    source = out / "source"
    cfg = {**item["config"], "drift": step}
    links = {name: Path(item["path"]) / name for name in ("prompts.json", "rollouts_behavior_train.jsonl")}
    links[f"policy_step_{step}"] = directory / f"policy_step_{step}"
    if publish:
        base.bind(source / "run_config.json", cfg)
        base.bind(source / "selected-prefix.json", cert)
        for name, target in links.items():
            switch.link(source / name, target)
    if (core.read(source / "run_config.json") != cfg or core.read(source / "selected-prefix.json") != cert
            or any((source / name).resolve() != target.resolve() for name, target in links.items())):
        raise ValueError("comparison private prefix view changed")
    c = base.source_contract(source, source_protocol["evaluation"], budget=p["budget_gpu_seconds"],
        gpu_type=p["gpu_type"], role="test", selector="mopps_comparison",
        eval_k=source_protocol["eval_k"], max_steps=100000)
    c["selected_prefix"] = {"schema": rule.SCHEMA, "root": p["parent"],
                            "certificate_sha256": core.fingerprint(cert)}
    c["source_hashes"].update({name: base.digest(source / name)
                              for name in ("selected-prefix.json", "rollouts_behavior_train.jsonl")})
    c["comparison"] = {"protocol_sha256": core.fingerprint(p), "origin": str(original_point(p, seed, step)),
                       "source_kind": "certified_prefix", "prefix_sha256": base.digest(directory / f"prefix-{step}.json")}
    return c


def import_point(root, p, seed, step):
    out = point(root, seed, step)
    def publish():
        origin = original_point(p, seed, step)
        if (out / "contract.json").exists():
            independent = core.read(out / "contract.json")["comparison"].get("source_kind") == "certified_prefix"
        else:
            independent = not origin_frozen(origin)
        if independent:
            c = prefix_contract(out, p, seed, step, publish=True)
        else:
            original = verify_origin(p, seed, step)
            c = {**original, "scope": {**original["scope"], "selector": "mopps_comparison"},
                 "comparison": {"protocol_sha256": core.fingerprint(p), "origin": str(origin),
                                "contract_sha256": base.digest(origin / "contract.json"),
                                "decisions_sha256": base.digest(origin / "decisions-frozen.json"),
                                "gate_decision_sha256": base.digest(origin / "gate-frozen.json")}}
        base.bind(out / "contract.json", c)
        base.bind(out / "evaluation.json", c["evaluation"])
        prompts = core.read(Path(c["source_run"]) / "prompts.json")
        for arm in mopps.ARMS:
            base.bind(out / "subsets" / f"subset-{arm}.json", prompts)
            base.bind(out / arm / "selector.json", mopps.specification(arm, seed, step, len(prompts["train"])))
        base.bind(out / "import.done.json", {"contract_sha256": base.digest(out / "contract.json")})
    with base.lease(out / ".import.lock"):
        base.meter(out / "import-cost", "import", p["gpu_type"], action=publish, ledger="research",
                   devices=4 if os.environ.get("OM_NODE_LOCK_HELD") == "1" else 0)
    return out


def verify(out):
    p = protocol(out.parent.parent)
    c = _base_verify(out)
    if c["comparison"].get("source_kind") == "certified_prefix":
        expected = prefix_contract(out, p, c["config"]["seed"], c["config"]["drift"])
    else:
        original = verify_origin(p, c["config"]["seed"], c["config"]["drift"])
        origin = Path(c["comparison"]["origin"])
        expected = {**original, "scope": {**original["scope"], "selector": "mopps_comparison"},
                    "comparison": {"protocol_sha256": core.fingerprint(p), "origin": str(original_point(p, c["config"]["seed"], c["config"]["drift"])),
                                   "contract_sha256": base.digest(origin / "contract.json"),
                                   "decisions_sha256": base.digest(origin / "decisions-frozen.json"),
                                   "gate_decision_sha256": base.digest(origin / "gate-frozen.json")}}
    if c != expected or core.read(out / "import.done.json") != {"contract_sha256": base.digest(out / "contract.json")}:
        raise ValueError("comparison input binding changed")
    prompts = core.read(Path(c["source_run"]) / "prompts.json")
    for arm in mopps.ARMS:
        if core.read(out / "subsets" / f"subset-{arm}.json") != prompts:
            raise ValueError("online comparison must use the full original training pool")
        if core.read(out / arm / "selector.json") != mopps.specification(
                arm, c["config"]["seed"], c["config"]["drift"], len(prompts["train"])):
            raise ValueError("online selector specification changed")
    return c


def policy(out, c, arm):
    path = _base_policy(out, c, arm)
    if arm in mopps.ARMS and path == out / arm / "policy":
        mopps.validate_policy_evidence(path, core.read(out / arm / "selector.json"))
    return path


def train_command(out, c, arm, remaining):
    command = base.train_command(out, c, arm, remaining)
    command[command.index(str(base.ROOT / "src/train_selection_gate_grpo.py"))] = str(base.ROOT / "src/train_mopps_grpo.py")
    return [*command, "--selector-config", str(out / arm / "selector.json")]


def result_paths(out, arm):
    return [out / "contract.json", out / "import.done.json", out / arm / "selector.json",
            out / "subsets" / f"subset-{arm}.json", out / arm / "policy/budget_stop.json"]


def validate_result(out, arm, *, recover_receipt=False):
    c = verify(out)
    directory = out / arm
    path = directory / "result.json"
    result = core.read(path)
    receipt = directory / "result.sha256.json"
    expected_receipt = {"sha256": base.digest(path)}
    if (not receipt.exists() and not recover_receipt) or (receipt.exists() and core.read(receipt) != expected_receipt):
        raise ValueError("comparison result receipt changed")
    if (result["schema"] != mopps.SCHEMA or not result["complete"] or result["arm"] != arm
            or result["artifact_hashes"] != {str(path.relative_to(out)): base.digest(path) for path in result_paths(out, arm)}):
        raise ValueError("comparison result binding changed")
    costs = base.cost(directory)
    used = sum(v["gpu_seconds"] for key, v in costs["ledgers"].items() if key != "reporting")
    if (not costs["complete"] or used != result["used_gpu_seconds"] or costs != result["cost"]
            or result["budget_gpu_seconds"] != c["budget_gpu_seconds"] or used > c["budget_gpu_seconds"]):
        raise ValueError("comparison cost is unknown, changed or exceeds allocation")
    if result["rewards"] != base.rewards(out, c, arm):
        raise ValueError("comparison evaluation changed")
    stop = core.read(directory / "policy/budget_stop.json")
    if (result["completed_steps"] != stop["completed_steps"] or result["stop_reason"] != stop["stop_reason"]
            or stop["stop_reason"] not in {"no_block_fits", "budget_exhausted"}):
        raise ValueError("comparison is not a fixed-budget continuation")
    if recover_receipt and not receipt.exists():
        base.bind(receipt, expected_receipt)
    return result


def run_arm(out, p, arm, devices, env):
    directory = out / arm
    if (directory / "result.json").exists():
        validate_result(out, arm, recover_receipt=True)
        return
    base.spent(directory)
    c = base.meter(directory, "verify-inputs", p["gpu_type"], action=lambda: verify(out), ledger="deployment")
    stop_path = directory / "policy/budget_stop.json"
    if not stop_path.exists():
        remaining = c["budget_gpu_seconds"] - base.spent(directory)
        if remaining <= 0:
            raise ValueError("comparison allocation exhausted")
        if remaining/4 <= 30:
            base.bind(stop_path, {"completed_steps": c["config"]["drift"], "stop_reason": "no_block_fits",
                                 "use_parent_policy": True, "requested_target_steps": c["config"]["drift"]+c["max_steps"]})
        else:
            base.meter(directory, "train", p["gpu_type"], commands=[(train_command(out, c, arm, remaining), ",".join(devices))],
                       env=env, timeout=remaining/4, ledger="deployment")
    stop = core.read(stop_path)
    if stop["stop_reason"] not in {"budget_exhausted", "no_block_fits"}:
        raise ValueError("update-count termination is not a fixed-budget result")
    policy(out, c, arm)
    commands = [([sys.executable, str(HERE), "worker", "--root", str(out.parent.parent),
                  "--seed", str(c["config"]["seed"]), "--step", str(c["config"]["drift"]),
                  "--arm", arm, "--shard", str(i)], devices[i]) for i in range(4)
                if not (directory / "evaluation" / f"shard-{i}.done.json").exists()]
    if commands:
        base.meter(directory, "evaluate", p["gpu_type"], commands=commands, env=env,
                   timeout=p["eval_timeout"], ledger="reporting")
    used = base.spent(directory)
    if used > c["budget_gpu_seconds"]:
        raise ValueError("comparison exceeded fixed allocation")
    result = {"schema": mopps.SCHEMA, "complete": True, "arm": arm, "rewards": base.rewards(out, c, arm),
              "budget_gpu_seconds": c["budget_gpu_seconds"], "used_gpu_seconds": used, "cost": base.cost(directory),
              "completed_steps": stop["completed_steps"], "stop_reason": stop["stop_reason"],
              "artifact_hashes": {str(path.relative_to(out)): base.digest(path) for path in result_paths(out, arm)}}
    base.bind(directory / "result.json", result)
    base.bind(directory / "result.sha256.json", {"sha256": base.digest(directory / "result.json")})
    validate_result(out, arm)
    (directory / "failure.json").unlink(missing_ok=True)


def work(root, idle_timeout=600., only=None):
    import additive_experiment as ae
    p = protocol(root)
    core.number(idle_timeout, "idle timeout", 0.)
    devices = switch.admitted_devices(p)
    attempted, failures, last_progress = set(), 0, time.monotonic()
    tasks = [(s, t, a) for s in p["seeds"] for t in p["steps"] for a in (p["arms"] if s % 2 == 0 else p["arms"][::-1])]
    if only:
        if only not in tasks:
            raise ValueError("unregistered retry task")
        tasks = [only]
    reported_blockers = set()
    while True:
        progressed, busy, waiting, blocked = False, [], [], []
        active_prefix = False
        for seed, step, arm in tasks:
            out, key = point(root, seed, step), (seed, step, arm)
            directory = out / arm
            if key in attempted or (directory / "result.sha256.json").exists():
                continue
            if (directory / "failure.json").exists() and only is None:
                failures = 1
                blocked.append(f"{directory / 'failure.json'}: explicit retry required")
                continue
            if not ready(p, seed, step):
                dependency = prefix_dependency(p, seed, step)
                active_prefix |= dependency["active"]
                if dependency["failure"]:
                    failures = 1
                    blocked.append(dependency["failure"])
                    continue
                waiting.append(f"s{seed}/t{step}: missing {switch.prefix_dir(Path(p['parent']), seed) / f'prefix-{step}.json'}")
                continue
            try:
                if not (out / "import.done.json").exists():
                    import_point(root, p, seed, step)
                with base.lease(directory / ".task.lock"):
                    if (directory / "result.sha256.json").exists() or ((directory / "failure.json").exists() and only is None):
                        continue
                    attempted.add(key)
                    print(f"[claimed] host={socket.gethostname()} pid={os.getpid()} task=s{seed}/t{step}/{arm}", flush=True)
                    run_arm(out, p, arm, devices, ae.model_environment(core.read(out / "contract.json")["config"]))
                    progressed = True
            except BlockingIOError:
                busy.append(switch.busy_task(f"s{seed}/t{step}/{arm}", directory))
            except Exception as exc:
                attempted.add(key)
                failures += 1
                switch.record_failure(directory, exc)
        for detail in dict.fromkeys(blocked):
            if detail not in reported_blockers:
                print(f"[blocked] {detail}; independent ready tasks remain eligible", flush=True)
                reported_blockers.add(detail)
        if progressed:
            last_progress = time.monotonic()
        elif busy and idle_timeout > 0 and switch.wait_for_peers(busy, last_progress=last_progress, idle_timeout=idle_timeout):
            continue
        elif waiting and (active_prefix or not failures) and time.monotonic()-last_progress < idle_timeout:
            print("[waiting] " + "; ".join(dict.fromkeys(waiting)) + "; original experiment is read-only", flush=True)
            time.sleep(15)
        else:
            if failures:
                print("[blocked] no claimable work remains; exiting instead of reserving idle GPUs; "
                      "inspect the recorded failures before retrying", flush=True)
            break
    status(root)
    return int(bool(failures))


def status(root):
    p = core.read(root / "mopps.json")
    counts, rows = {}, []
    for seed in p["seeds"]:
        for step in p["steps"]:
            for arm in p["arms"]:
                directory = point(root, seed, step) / arm
                state, detail = "READY", ""
                try:
                    progress = core.read(directory / "progress.json") if (directory / "progress.json").exists() else {}
                    if (directory / "result.json").exists():
                        if not (directory / "result.sha256.json").exists():
                            state = "SAVING"
                        else:
                            state = "DONE" if core.read(directory / "result.sha256.json") == {"sha256": base.digest(directory / "result.json")} else "INVALID"
                    elif progress.get("state") == "running":
                        state = "RUNNING" if 0 <= time.time()-progress.get("updated", 0) < 60 else "STALE"
                        detail = f"{progress.get('host', '?')} {progress.get('phase', '?')} {progress.get('seconds', 0):.0f}s"
                    elif (directory / "failure.json").exists():
                        state, detail = "FAILED", core.read(directory / "failure.json")["error"].splitlines()[0]
                    elif not ready(p, seed, step):
                        dependency = prefix_dependency(p, seed, step)
                        if dependency["failure"]:
                            state, detail = "BLOCKED", dependency["failure"]
                        else:
                            state, detail = "WAIT", f"prefixes/seed-{seed}/prefix-{step}.json missing (Gate not required)"
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    state, detail = "INVALID", str(exc)
                counts[state] = counts.get(state, 0)+1
                rows.append(f"s{seed} t{step:<3} {arm:<13} {state:<7} {detail[:70]}")
    print("Gate vs MoPPS (KDD 2026) | new continuation jobs | " + " ".join(f"{k}={v}" for k, v in counts.items()))
    print("\n".join(rows))
    return counts


def primary_comparison(gated, reward_selector, budget, seed):
    import evidence_downstream as ed
    import numpy as np
    for result in (gated, reward_selector):
        if result.get("complete") is not True:
            raise ValueError("primary comparison requires both executed results")
        diagnostic = core.number(result.get("measurement_gpu_seconds", 0.), "diagnostic cost", 0.)
        if not math.isclose(result["budget_gpu_seconds"] + diagnostic, budget, rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError("Gate vs MoPPS total allocations differ")
        core.number(result["used_gpu_seconds"] + diagnostic, "total deployment cost", 0., budget)
    left, right = gated["rewards"], reward_selector["rewards"]
    if not left or left.keys() != right.keys():
        raise ValueError("Gate vs MoPPS evaluation questions differ")
    differences = {key: 100*(core.number(left[key], "gate reward", 0., 1.)
                            - core.number(right[key], "MoPPS reward", 0., 1.)) for key in left}
    low, high = ed.paired_interval(np.array(list(differences.values())), seed)
    return {"gate_reward_pp": 100*statistics.fmean(left.values()),
            "mopps_reward_pp": 100*statistics.fmean(right.values()),
            "gate_minus_mopps_pp": statistics.fmean(differences.values()),
            "per_question_gate_minus_mopps_pp": differences, "conditional_prompt_ci95_pp": [low, high],
            "interval_scope": "paired prompt bootstrap conditional on trained policies; not across-seed uncertainty",
            "total_allocation_gpu_seconds": budget,
            "gate_total_gpu_seconds": gated["used_gpu_seconds"] + gated["measurement_gpu_seconds"],
            "mopps_total_gpu_seconds": reward_selector["used_gpu_seconds"],
            "gate_executed_action": gated["action"]}


def summarize(root):
    p = protocol(root)
    rows = []
    for seed in p["seeds"]:
        for step in p["steps"]:
            out = point(root, seed, step)
            row = {"seed": seed, "step": step, "arms": {}, "errors": {}, "original_errors": {},
                   "shared_import_cost": base.cost(out / "import-cost")}
            for arm in p["arms"]:
                if not (out / arm / "result.json").exists():
                    row["errors"][arm] = "missing result"
                    continue
                try:
                    row["arms"][arm] = validate_result(out, arm)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row["errors"][arm] = str(exc)
            if all(arm in row["arms"] for arm in mopps.ARMS):
                left, right = (row["arms"][arm]["rewards"] for arm in mopps.ARMS)
                if left.keys() != right.keys():
                    raise ValueError("paired evaluation questions differ")
                row["mopps_minus_random_online_pp"] = 100*statistics.fmean(left[key]-right[key] for key in left)
            origin = original_point(p, seed, step)
            for arm in ("gated", "selection_full", "random_full"):
                try:
                    verify_origin(p, seed, step)
                    row["arms"][arm] = original_result(origin, arm)
                    if "mopps" in row["arms"]:
                        left, right = row["arms"]["mopps"]["rewards"], row["arms"][arm]["rewards"]
                        if left.keys() != right.keys():
                            raise ValueError("original comparison evaluation questions differ")
                        row[f"mopps_minus_{arm}_pp"] = 100*statistics.fmean(left[key]-right[key] for key in left)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row["original_errors"][arm] = str(exc)
            if "gated" in row["arms"] and "mopps" in row["arms"]:
                try:
                    row["primary"] = primary_comparison(row["arms"]["gated"], row["arms"]["mopps"],
                                                        p["budget_gpu_seconds"], seed*100003+step)
                    row["gate_minus_mopps_pp"] = row["primary"]["gate_minus_mopps_pp"]
                except (ValueError, KeyError, TypeError) as exc:
                    row["errors"]["primary"] = str(exc)
            rows.append(row)
    report = {"schema": mopps.SCHEMA, "paper": mopps.PAPER, "states": rows,
              "primary_comparison": "executed GATE vs online MoPPS; positive Gate-minus-MoPPS favors GATE",
              "independent_seeds": len(p["seeds"]),
              "complete": all("primary" in row and not row["errors"] and not row["original_errors"] for row in rows),
              "primary_complete": all("primary" in row for row in rows),
              "new_continuations_complete": all(not row["errors"] for row in rows),
              "original_comparisons_complete": all(not row["original_errors"] for row in rows),
              "per_seed_mean_contrasts_pp": {
                  str(seed): {name: statistics.fmean(row[name] for row in rows if row["seed"] == seed)
                              for name in ("gate_minus_mopps_pp", "mopps_minus_random_online_pp", "mopps_minus_selection_full_pp",
                                           "mopps_minus_random_full_pp", "mopps_minus_gated_pp")
                              if all(name in row for row in rows if row["seed"] == seed)} for seed in p["seeds"]},
              "scope": "six checkpoint pairs clustered within two seeds; not six independent replicates"}
    core.atomic_json(root / "comparison-report.json", report)
    print("PRIMARY: executed Gate vs MoPPS | reward in pp | costs include gate diagnosis")
    print("STATE       GATE    MoPPS   GATE-MoPPS       95% prompt CI       GATE GPU-s  MoPPS GPU-s")
    for row in rows:
        primary = row.get("primary")
        label = f"s{row['seed']} t{row['step']}"
        if primary:
            low, high = primary["conditional_prompt_ci95_pp"]
            print(f"{label:<10} {primary['gate_reward_pp']:6.2f}  {primary['mopps_reward_pp']:6.2f} "
                  f"{primary['gate_minus_mopps_pp']:+11.3f}   [{low:+8.3f}, {high:+8.3f}] "
                  f"{primary['gate_total_gpu_seconds']:12.1f} {primary['mopps_total_gpu_seconds']:12.1f}")
        else:
            print(f"{label:<10} INCOMPLETE: " + ", ".join([*row["errors"], *row["original_errors"]]))
    print("Two independent seeds; prompt intervals are conditional on each trained policy pair.")
    return report


def original_result(out, arm):
    """Reuse the original validator only after proving no ledger repair is needed."""
    net = core.read(out.parent.parent / "net_protocol.json")
    directories = [out / arm] + ([out / "gate_measurement"] if arm == "gated" else [])
    for directory in directories:
        if not base.cost(directory)["complete"]:
            raise ValueError("original branch has unknown cost; source was not repaired")
    result = switch.runtime.validate_result(out, net, arm)
    c = core.read(out / "contract.json")
    execution = core.read(out / arm / "execution.json")
    stop = core.read(out / arm / "policy/budget_stop.json")
    if result["action"] != execution["action"] or result["completed_steps"] != stop["completed_steps"]:
        raise ValueError("original result does not match its executed policy")
    if result["rewards"] != base.rewards(out, c, arm):
        raise ValueError("original result evaluation changed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "retry", "worker", "status", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--seed", type=int, choices=rule.TEST_SEEDS)
    parser.add_argument("--step", type=int, choices=rule.STEPS)
    parser.add_argument("--arm", choices=mopps.ARMS)
    parser.add_argument("--shard", type=int, choices=range(4))
    parser.add_argument("--idle-timeout", type=float, default=600.)
    args = parser.parse_args()
    args.root = args.root.resolve()
    base.verify, base.policy = verify, policy
    if args.command == "prepare":
        if not args.parent_root:
            parser.error("prepare requires --parent-root")
        prepare(args.root, args.parent_root)
    elif args.command == "status":
        status(args.root)
    elif args.command == "summarize":
        summarize(args.root)
    elif args.command == "worker":
        if None in (args.seed, args.step, args.arm, args.shard) or os.environ.get("OM_NODE_LOCK_HELD") != "1":
            parser.error("worker requires admitted node, seed, step, arm and shard")
        base.evaluate(point(args.root, args.seed, args.step), args.arm, args.shard)
    else:
        core.number(args.idle_timeout, "idle timeout", 0.)
        only = None
        if args.command == "retry":
            if None in (args.seed, args.step, args.arm):
                parser.error("retry requires --seed, --step and --arm")
            only = (args.seed, args.step, args.arm)
        raise SystemExit(work(args.root, args.idle_timeout, only))


if __name__ == "__main__":
    from light_selection_gate_gpu import install_signal_handlers
    install_signal_handlers()
    main()
