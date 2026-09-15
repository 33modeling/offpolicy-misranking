"""Selected-prefix switching experiment, isolated from all existing run roots."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import net_gain_gate_gpu as runtime
import selection_gate as core
import selection_gate_gpu as base
import selection_switch as rule
import selection_switch_score as scoring

HERE = Path(__file__).resolve()
CODE = tuple(dict.fromkeys((*runtime.CODE_FILES, "src/selection_switch.py", "src/selection_switch_gpu.py",
    "src/selection_switch_score.py", "src/grads.py", "src/experiment.py", "src/rollout.py",
    "src/rollout_contract.py", "src/artifact_contract.py", "src/select_rules.py", "src/data.py",
    "src/score_artifacts.py", "src/downstream_compare.py", "src/additive_experiment.py",
    "src/net_gate_memory_worker.py", "src/bootstrap_math_verify.py")))
_verify = base.verify
# cb01401 froze valid v3 inputs before legacy preparation recovery was added.
PRE_INITIAL_SCORE_CODE = "220a651698983460ea5e39d7f53d60127623cf824b5ffa5ab2309351ff9fe019"
# Exact a63e69d runtime before the teacher-forced KV-cache fix.
PRE_KV_CACHE_CODE = "8cb0b16a8c2e4229674c0165212a36ee916d9a2cbaea8dcc7adefc0deb9819ae"
PRE_COST_CODE = "7c2480d74d8c4b2b109570ddab68d513961db60517acf1dad9d8794834ab6f7a"
PRE_PREFIX_RESUME_CODE = "0e1bc0c39315258210b2ed0a003fe777b468993d60e9797f68972f739949de76"
PRE_WORKER_LOGS_CODE = "803be77868081423affe81d073b0b5b566889c7d43cfc99608ad444ebb3dd4b9"
PRE_CODE_COMPAT_CODE = "113afc51b2544e23d8389d6da5ad5f10e67b9407d54f835ba4935a1dd8523512"
PRIOR_RUNTIME_CODES = {PRE_INITIAL_SCORE_CODE, PRE_KV_CACHE_CODE, PRE_COST_CODE,
                       PRE_PREFIX_RESUME_CODE, PRE_WORKER_LOGS_CODE, PRE_CODE_COMPAT_CODE}
RUNTIME_PATCH_FILES = {"src/grads.py", "src/selection_switch_gpu.py", "src/selection_gate_gpu.py"}
KV_CACHE_GRADS = "6640be340a42fc79ba521a19440703fbb91d3fb6b9a11f3c5f152fa2e8a20bfe"
COST_METER = "58fd87dfdc00c3ee66e6903e12a53b31c4d2798f7594352ee9aa894d23525a99"


def code_hashes():
    return {name: base.digest(base.ROOT / name) for name in CODE}


def validate_code_hashes(recorded):
    current = code_hashes()
    if recorded == current:
        return current
    if (not isinstance(recorded, dict) or core.fingerprint(recorded) not in PRIOR_RUNTIME_CODES
            or set(recorded) != set(current)
            or current["src/grads.py"] != KV_CACHE_GRADS
            or current["src/selection_gate_gpu.py"] != COST_METER
            or any(recorded[name] != sha for name, sha in current.items() if name not in RUNTIME_PATCH_FILES)):
        old = recorded if isinstance(recorded, dict) else {}
        changed = {name: {"frozen": old.get(name), "current": current.get(name)}
                   for name in sorted(set(old) | set(current)) if old.get(name) != current.get(name)}
        raise ValueError("switch protocol or scientific code changed; preserve the frozen run; "
                         + json.dumps({"frozen_code": core.fingerprint(recorded),
                                       "current_code": core.fingerprint(current), "changed_files": changed}, sort_keys=True)
                         + "; do not rewrite switch.json; run the updated launcher after old workers have stopped")
    return current


def check_code(root):
    """Read-only preflight, before acquiring GPUs or writing migration receipts."""
    p = core.read(root / "switch.json")
    if p.get("schema") != rule.SCHEMA:
        raise ValueError(f"switch protocol changed: frozen schema={p.get('schema')!r}, "
                         f"expected={rule.SCHEMA!r}; preserve the frozen run")
    current = validate_code_hashes(p.get("code_hashes"))
    print(json.dumps({"status": "exact" if current == p["code_hashes"] else "compatible",
                      "frozen_code": core.fingerprint(p["code_hashes"]),
                      "current_code": core.fingerprint(current), "repository": str(base.ROOT),
                      "changed_files": [name for name in current if current[name] != p["code_hashes"][name]]},
                     sort_keys=True), flush=True)
    return p


def manifest(root):
    p = core.read(root / "switch.json")
    if p["schema"] != rule.SCHEMA:
        raise ValueError("switch protocol or scientific code changed; preserve the frozen run")
    current = validate_code_hashes(p["code_hashes"])
    if p["code_hashes"] != current:
        with base.lease(root / ".kv-cache-runtime.lock", blocking=True):
            path = root / "kv-cache-runtime.json"
            receipt = {
                "schema": "selection-switch-kv-cache-runtime/v1",
                "switch_sha256": base.digest(root / "switch.json"),
                "original_code_hashes": p["code_hashes"], "runtime_code_hashes": current,
                "change": "teacher-forced scoring forwards explicitly disable KV cache",
                "cost_policy": "retain all previous costs and the original branch allocation",
            }
            if path.exists():
                previous = core.read(path)
                previous_code = previous.get("runtime_code_hashes")
                if (previous != receipt and
                        (previous != {**receipt, "runtime_code_hashes": previous_code}
                         or core.fingerprint(previous_code) not in {PRE_COST_CODE, PRE_PREFIX_RESUME_CODE, PRE_WORKER_LOGS_CODE, PRE_CODE_COMPAT_CODE})):
                    raise ValueError(f"frozen contract changed: {path}")
            else:
                base.bind(path, receipt)
            cost_receipt = {
                "schema": "selection-switch-cost-runtime/v1",
                "switch_sha256": base.digest(root / "switch.json"),
                "kv_cache_runtime_sha256": base.digest(path), "runtime_code_hashes": current,
                "change": "protect phase startup, recover atomic finish receipts, skip busy publication tasks",
                "cost_policy": "recover only from completion evidence or operator-reported termination duration",
            }
            cost_path = root / "cost-runtime.json"
            if cost_path.exists():
                previous = core.read(cost_path)
                previous_code = previous.get("runtime_code_hashes")
                if (previous != cost_receipt and
                        (previous != {**cost_receipt, "runtime_code_hashes": previous_code}
                         or core.fingerprint(previous_code) not in {PRE_PREFIX_RESUME_CODE, PRE_WORKER_LOGS_CODE, PRE_CODE_COMPAT_CODE})):
                    raise ValueError(f"frozen contract changed: {cost_path}")
            else:
                base.bind(cost_path, cost_receipt)
            prefix_receipt = {
                "schema": "selection-switch-prefix-resume-runtime/v1",
                "switch_sha256": base.digest(root / "switch.json"),
                "cost_runtime_sha256": base.digest(cost_path), "runtime_code_hashes": current,
                "change": "resume research prefixes with explicitly unknown historical costs; keep active queue peers",
                "cost_policy": "preserve open research events; never waive deployment budget accounting",
            }
            prefix_path = root / "prefix-resume-runtime.json"
            if prefix_path.exists():
                previous = core.read(prefix_path)
                previous_code = previous.get("runtime_code_hashes")
                if (previous != prefix_receipt and
                        (previous != {**prefix_receipt, "runtime_code_hashes": previous_code}
                         or core.fingerprint(previous_code) not in {PRE_WORKER_LOGS_CODE, PRE_CODE_COMPAT_CODE})):
                    raise ValueError(f"frozen contract changed: {prefix_path}")
            else:
                base.bind(prefix_path, prefix_receipt)
            worker_receipt = {
                "schema": "selection-switch-worker-logs-runtime/v1",
                "switch_sha256": base.digest(root / "switch.json"),
                "prefix_runtime_sha256": base.digest(prefix_path), "runtime_code_hashes": current,
                "change": "include failed child stderr in supervisor exceptions",
                "cost_policy": "no change to phase costs or branch budgets",
            }
            worker_path = root / "worker-logs-runtime.json"
            if worker_path.exists():
                previous = core.read(worker_path)
                previous_code = previous.get("runtime_code_hashes")
                if (previous != worker_receipt and
                        (previous != {**worker_receipt, "runtime_code_hashes": previous_code}
                         or core.fingerprint(previous_code) != PRE_CODE_COMPAT_CODE)):
                    raise ValueError(f"frozen contract changed: {worker_path}")
            else:
                base.bind(worker_path, worker_receipt)
            base.bind(root / "code-compat-runtime.json", {
                "schema": "selection-switch-code-compat-runtime/v1",
                "switch_sha256": base.digest(root / "switch.json"),
                "worker_runtime_sha256": base.digest(worker_path), "runtime_code_hashes": current,
                "change": "accept the exact original switch runtime; diagnose other code mismatches",
                "cost_policy": "no change to frozen inputs, policies, phase costs or branch budgets",
            })
    return p


def link(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    target = target.resolve()
    if path.is_symlink() or path.exists():
        if path.resolve() != target:
            raise ValueError(f"existing input link differs: {path}")
    else:
        path.symlink_to(target, target_is_directory=target.is_dir())


def prefix_dir(root, seed):
    return root / "prefixes" / f"seed-{seed}"


def verify_source(root, seed):
    p = manifest(root)
    item = p["sources"][str(seed)]
    source = Path(item["path"])
    for name, sha in item["hashes"].items():
        if base.digest(source / name) != sha:
            raise ValueError(f"initial source changed: {source/name}")
    if base.digest(Path(item["config"]["model"]) / "config.json") != item["model_sha256"]:
        raise ValueError("initial base model changed")
    return item


def validate_prefix(root, seed, step):
    import evidence_downstream as ed
    from train_policy_grpo import validate_policy_lineage
    item = verify_source(root, seed)
    directory = prefix_dir(root, seed)
    cfg = item["config"]
    subset = directory / "subset.json"
    if core.read(subset) != item["subset"]:
        raise ValueError("initial selected subset changed")
    previous = 0
    for current in rule.STEPS:
        policy = directory / f"policy_step_{current}"
        validate_policy_lineage(policy, target_steps=current, world_size=4, training_objective="grpo",
            expected_start_step=previous, expected_parent=directory / f"policy_step_{previous}" if previous else None,
            expected_model=Path(cfg["model"]), expected_seed=seed, expected_max_new_tokens=cfg["max_new_tokens"],
            expected_prompt_format=cfg["prompt_format"], expected_config=ed._expected_config(cfg),
            expected_prompts=subset, require_complete_hashes=True)
        expected = {"schema": rule.SCHEMA, "seed": seed, "step": current, "previous": previous,
                    "selector": "fresh_r", "subset_sha256": base.digest(subset),
                    "source_sha256": core.fingerprint(item),
                    "policy_hashes": {name: base.digest(policy / name) for name in ed.POLICY_FILES}}
        if core.read(directory / f"prefix-{current}.json") != expected:
            raise ValueError("checkpoint is not a certified selected-training prefix")
        if current == step:
            return expected
        previous = current
    raise ValueError("unregistered prefix checkpoint")


def verify(out):
    c = _verify(out)
    selected = c.get("selected_prefix")
    if not selected or selected["schema"] != rule.SCHEMA:
        raise ValueError("a generic GRPO checkpoint is not a selected-training prefix")
    cert = validate_prefix(Path(selected["root"]), c["config"]["seed"], c["config"]["drift"])
    if core.fingerprint(cert) != selected["certificate_sha256"]:
        raise ValueError("selected history changed")
    return c


def initial_fresh_scores(source, cfg, prompts, generation):
    """Recover legacy scalar metadata from saved gradients, without modifying inputs."""
    if not isinstance(generation, dict) or core.number(generation.get("validated_rows", 0), "validated rollout rows", 1) <= 0:
        raise ValueError("initial selection requires freshly validated rollout inputs")
    oracle = core.read(source / "oracle_protocol.json")
    recorded = oracle.get("generation_validation") or {}
    for key in ("manifest_sha256", "artifact_sha256"):
        for name, sha in recorded.get(key, {}).items():
            path = source / name
            if path.is_file() and base.digest(path) != sha:
                raise ValueError(f"initial score input changed since scoring: {path}")
    n = len(prompts["train"])
    info = {"recorded_schema": oracle.get("schema"), "recorded_validated_rows": recorded.get("validated_rows"),
            "live_validated_rows": generation["validated_rows"], "input_hashes": {}}
    if (oracle.get("schema") == "offpolicy-oracle-validation-split/v3"
            and isinstance(recorded.get("validated_rows"), (int, float)) and recorded["validated_rows"] > 0):
        rows = core.read(source / "scores_splithalf.json")
        scores = {int(i): core.number(row["r"], "fresh_r score", -1.00001, 1.00001) for i, row in rows.items()}
        if len(scores) != len(rows) or set(scores) != set(range(n)):
            raise ValueError("initial fresh_r prompt coverage differs")
        info["method"] = "verified_v3_scalar_scores"
        return scores, info
    import torch
    from experiment import score_oracle_microgroups, split_validation_directions
    paths = [source / "oracle_micro_groups.pt", source / "val_groups.pt"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"{source}: legacy initial score metadata (schema={oracle.get('schema')!r}, "
                         f"validated_rows={recorded.get('validated_rows')!r}); CPU repair needs {missing}")
    info["input_hashes"] = {path.name: base.digest(path) for path in paths}
    micro = torch.load(paths[0], map_location="cpu", weights_only=True)
    validation = torch.load(paths[1], map_location="cpu", weights_only=True)
    normalized = {int(i): value for i, value in micro.items()}
    if len(normalized) != len(micro) or set(normalized) != set(range(n)):
        raise ValueError("saved candidate gradients have invalid prompt coverage")
    if tuple(validation.shape) != (len(prompts["val"]), cfg["proj_dim"]) or not torch.isfinite(validation).all():
        raise ValueError("saved validation gradients have invalid shape or values")
    directions = split_validation_directions(validation.float())
    scores = {}
    for i, groups in normalized.items():
        if tuple(groups.shape) != (8, cfg["proj_dim"]) or not torch.isfinite(groups).all():
            raise ValueError(f"saved candidate {i}: expected eight finite LOO4 projected gradients")
        scores[i] = score_oracle_microgroups(groups.float(), *directions)[1]["r"]
    info.update(method="cpu_reconstructed_fresh_r_from_saved_gradients",
                scores_sha256=core.fingerprint(scores), original_artifacts_modified=False)
    print(f"[initial-fresh-r] {source.name}: legacy metadata; recovered {n} scores from saved gradients on CPU; no GPU generation", flush=True)
    return scores, info


def prepare(args):
    import additive_experiment as ae
    import evidence_downstream as ed
    from artifact_contract import validate_generation_contract
    from select_rules import jittered_topk, topk_count

    root = args.root.resolve()
    for value, name in ((args.eval_timeout, "evaluation timeout"), (args.prefix_timeout, "prefix timeout")):
        core.number(value, name, 1.)
    core.integer(args.eval_k, "evaluation responses", 1)
    with base.lease(root / ".prepare.lock", blocking=True):
        if (root / "switch.json").exists():
            manifest(root)
            print(f"[prepared] frozen experiment already exists: {root}")
            return
        runs = ae.resolve_runs(args.matrix, (*rule.DEV_SEEDS, *rule.TEST_SEEDS), 0)
        ed.require_separate_output(root, runs)
        if any(root in run.parents for run in runs):
            raise ValueError("output must not contain existing matrix data")
        sources = {}
        for source in runs:
            cfg = core.read(source / "run_config.json")
            prompts = core.read(source / "prompts.json")
            if (cfg["drift"] != 0 or cfg["dataset"] != "math500" or cfg["prompt_format"] != "olmo_rlzero_math"
                    or cfg["grpo_world_size"] != 4 or cfg["grpo_group_size"] != 8
                    or cfg["grpo_epochs_per_batch"] != 1 or cfg["topk_frac"] != .1
                    or cfg["temperature"] != 1. or cfg.get("top_p", 1.) != 1.
                    or not (source / "DONE").is_file()):
                raise ValueError("expected complete base-policy OLMo MATH source with registered GRPO recipe")
            scoring.layout(cfg, prompts)
            generation = validate_generation_contract(source)
            scores, score_provenance = initial_fresh_scores(source, cfg, prompts, generation)
            if set(scores) != set(range(len(prompts["train"]))):
                raise ValueError("initial fresh_r prompt coverage differs")
            indices = sorted(jittered_topk(scores, topk_count(len(scores), .1), cfg["seed"]+1000))
            names = ["run_config.json", "prompts.json", "scores_splithalf.json", "oracle_protocol.json",
                     "rollouts_behavior_train.jsonl"]
            names += list(score_provenance["input_hashes"])
            sources[str(cfg["seed"])] = {"path": str(source), "config": cfg,
                "hashes": {name: base.digest(source / name) for name in names}, "generation_validation": generation,
                "initial_score_provenance": score_provenance,
                "model_sha256": base.digest(Path(cfg["model"]) / "config.json"),
                "subset": {**prompts, "train": [prompts["train"][i] for i in indices],
                           "selector": "fresh_r", "selected_idx": indices, "k": len(indices)}}
        budget = args.budget_gpu_seconds
        budget_source = {"kind": "explicit", "gpu_seconds": budget}
        if budget is None:
            reference = ae.resolve_runs(args.matrix, [0], 100)[0] / "policy_step_100/grpo_stats.jsonl"
            timings = [core.number(json.loads(line)["step_seconds"], "step duration", 1e-12)
                       for line in reference.read_text().splitlines() if line.strip()]
            budget = math.ceil(statistics.median(timings)*4*100/60)*60
            budget_source = {"kind": "100-update equivalent; development seed-0 timings only", "path": str(reference),
                             "sha256": base.digest(reference), "median_update_wall_seconds": statistics.median(timings)}
        core.number(budget, "budget", 120.)
        if args.eval_prompts:
            evaluation = core.read(args.eval_prompts)
        else:
            if not args.pool or not args.pool_manifest:
                raise ValueError("independent evaluation requires --eval-prompts or --pool/--pool-manifest")
            evaluation = ed.prepare_test(args.pool, runs, root / "test.json", args.test_count, 20260914,
                                         "EleutherAI/hendrycks_math", core.read(args.pool_manifest)["source_revision"], "train")
        for source in runs:
            if len(ed.independent_test(core.read(source / "prompts.json"), evaluation)) < 4:
                raise ValueError("too few independent evaluation questions")
        base.bind(root / "test.json", evaluation)
        p = {"schema": rule.SCHEMA, "code_hashes": code_hashes(), "sources": sources,
             "budget_gpu_seconds": budget, "budget_source": budget_source, "steps": list(rule.STEPS),
             "development_seeds": list(rule.DEV_SEEDS), "test_seeds": list(rule.TEST_SEEDS),
             "gpu_type": args.gpu_type, "evaluation": evaluation, "eval_k": args.eval_k,
             "eval_timeout": args.eval_timeout, "prefix_timeout": args.prefix_timeout,
             "measurement_config": rule.MEASUREMENT,
             "historical_scoring_cost": "reused verified d0 fresh_r scores; historical cost unknown, not zero",
             "prefix_cost": "shared research work, recorded separately from continuation allocation"}
        base.bind(root / "switch.json", p)
        print(f"[prepared] {root}; 18 development + 30 held-out continuations, five selected prefixes; B={budget:.0f} GPU-s")


def prefix_cost(segment, gpu_type):
    """Called under the seed lease; unknown research cost is not a branch budget."""
    result = base.cost(segment)
    if not result["complete"]:
        result = base.recover_cost_receipts(segment)
    with base.lease(segment / ".cost.lock"):
        path = segment / "cost.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        result = core.cost_summary(events)
        if result["missing_starts"]:
            raise ValueError(f"prefix cost has missing start records: {segment}")
        if any(row.get("ledger") != "research" or row.get("phase") != "prefix-train"
               or row.get("gpus") != 4 or row.get("gpu_type") != gpu_type for row in events):
            raise ValueError(f"unexpected prefix research allocation: {segment}")
        for event_id in result["incomplete_events"]:
            start = next(row for row in events if row["event_id"] == event_id)
            archive = segment / "pending-costs" / f"{event_id}.json"
            saved = core.read(archive) if archive.exists() else None
            if saved is not None and saved.get("start") != start:
                raise ValueError(f"pending prefix cost evidence changed: {archive}")
            progress_path = segment / "progress.json"
            progress = core.read(progress_path) if progress_path.exists() else {}
            if progress.get("event_id") != event_id:
                progress = saved.get("progress") if saved else None
            if progress and any(progress.get(key) != start.get(key) for key in
                                ("event_id", "phase", "ledger", "gpus", "gpu_type", "host")):
                raise ValueError(f"prefix progress allocation changed: {segment}")
            pid = (progress or {}).get("pid", start.get("pid"))
            if start.get("host") == socket.gethostname() and type(pid) is int and pid > 0:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError(f"prefix owner PID {pid} is still alive: {segment}")
            if saved is None:
                core.atomic_json(archive, {"start": start, "progress": progress,
                    "total_gpu_seconds": None, "reason": "interrupted research prefix; duration unknown"})
    if not result["complete"]:
        print(f"[prefix-cost pending] {segment}: events={','.join(result['incomplete_events'])}; "
              "historical research total UNKNOWN; resuming prefix, deployment budgets unchanged", flush=True)
    return result


def prefix_cost_report(root):
    segments = {str(path.parent.relative_to(root)): base.cost(path.parent)
                for path in sorted((root / "prefixes").glob("seed-*/segment-*/cost.jsonl"))}
    complete = all(value["complete"] for value in segments.values())
    known = sum(value["ledgers"]["research"]["gpu_seconds"] for value in segments.values())
    return {"complete": complete, "known_gpu_seconds": known,
            "total_gpu_seconds": known if complete else None, "segments": segments}


def build_prefix(root, seed, step, devices, env):
    import evidence_downstream as ed
    from train_policy_grpo import validate_policy_lineage
    p, item = manifest(root), verify_source(root, seed)
    directory = prefix_dir(root, seed)
    previous = (0, *rule.STEPS)[rule.STEPS.index(step)]
    if previous:
        validate_prefix(root, seed, previous)
    base.bind(directory / "subset.json", item["subset"])
    segment = directory / f"segment-{step}"
    base.bind(segment / "subsets/subset-fresh_r.json", item["subset"])
    cfg = {**item["config"], "drift": previous}
    policy = segment / "fresh_r/policy"
    prefix_cost(segment, p["gpu_type"])
    if not (policy / "policy_train.json").exists():
        command = ed.train_args(cfg, directory, segment, "fresh_r", step-previous)
        command = [x for x in command if x != "--reliability-log"]
        base.meter(segment, "prefix-train", p["gpu_type"], commands=[([sys.executable, *command], ",".join(devices))],
                   env=env, timeout=p["prefix_timeout"], ledger="research")
    link(directory / f"policy_step_{step}", policy)
    validate_policy_lineage(policy, target_steps=step, world_size=4, training_objective="grpo",
        expected_start_step=previous, expected_parent=directory / f"policy_step_{previous}" if previous else None,
        expected_model=Path(cfg["model"]), expected_seed=seed, expected_max_new_tokens=cfg["max_new_tokens"],
        expected_prompt_format=cfg["prompt_format"], expected_config=ed._expected_config(cfg),
        expected_prompts=directory / "subset.json", require_complete_hashes=True)
    cert = {"schema": rule.SCHEMA, "seed": seed, "step": step, "previous": previous,
            "selector": "fresh_r", "subset_sha256": base.digest(directory / "subset.json"),
            "source_sha256": core.fingerprint(item),
            "policy_hashes": {name: base.digest(policy / name) for name in ed.POLICY_FILES}}
    base.bind(directory / f"prefix-{step}.json", cert)
    validate_prefix(root, seed, step)
    (segment / "failure.json").unlink(missing_ok=True)


def child_root(root, seed, step):
    return root / "states" / f"s{seed}-t{step}"


def publish_state(root, seed, step):
    p, item = manifest(root), verify_source(root, seed)
    cert = validate_prefix(root, seed, step)
    held_out = seed in rule.TEST_SEEDS
    model = core.read(root / "model.json") if held_out else None
    directory = prefix_dir(root, seed)
    source = directory / f"view-{step}"
    cfg = {**item["config"], "drift": step}
    base.bind(source / "run_config.json", cfg)
    link(source / "prompts.json", Path(item["path"]) / "prompts.json")
    link(source / f"policy_step_{step}", directory / f"policy_step_{step}")
    link(source / "rollouts_behavior_train.jsonl", Path(item["path"]) / "rollouts_behavior_train.jsonl")
    base.bind(source / "selected-prefix.json", cert)
    child = child_root(root, seed, step)
    c = base.source_contract(source, p["evaluation"], budget=p["budget_gpu_seconds"], gpu_type=p["gpu_type"],
        role="test" if held_out else "development", selector="fresh_r", eval_k=p["eval_k"], max_steps=100000)
    c["selected_prefix"] = {"schema": rule.SCHEMA, "root": str(root), "certificate_sha256": core.fingerprint(cert)}
    c["source_hashes"]["selected-prefix.json"] = base.digest(source / "selected-prefix.json")
    c["source_hashes"]["rollouts_behavior_train.jsonl"] = base.digest(source / "rollouts_behavior_train.jsonl")
    out = child / "points" / source.name
    base.bind(out / "contract.json", c)
    base.bind(out / "evaluation.json", c["evaluation"])
    base.bind(out / "net_inputs.json", {name: base.digest(source / name) for name in (
        "rollouts_behavior_train.jsonl", f"policy_step_{step}/grpo_stats.jsonl")})
    base.bind(child / "suite.json", {"schema": base.SCHEMA, "points": [{"name": source.name, "sha256": base.digest(out / "contract.json")}],
        "budget_gpu_seconds": p["budget_gpu_seconds"], "measurement_wall_seconds": 30., "eval_timeout": p["eval_timeout"]})
    protocol_value = {"schema": rule.SCHEMA, "schedule": rule.SCHEDULE, "mode": "test" if held_out else "study",
        "model": model, "role": c["role"], "selector": "fresh_r", "arms": list(rule.TEST_ARMS if held_out else rule.DEV_ARMS),
        "recent_window": 20, "max_measurement_fraction": .01, "code_hashes": p["code_hashes"]}
    if model:
        runtime.check_model(model, c)
    base.bind(child / "net_protocol.json", protocol_value)
    return child


def protocol(root):
    value = core.read(root / "net_protocol.json")
    if value.get("schema") != rule.SCHEMA or value.get("schedule") != rule.SCHEDULE:
        raise ValueError("not a selected-prefix switch suite")
    arms = list(rule.DEV_ARMS) if value["mode"] == "study" else list(rule.TEST_ARMS)
    if value["mode"] not in {"study", "test"} or value["arms"] != arms or value["selector"] != "fresh_r":
        raise ValueError("invalid switch experimental design")
    if value["mode"] == "test":
        rule.validate_model(value["model"])
        if value["role"] != "test" or value["model"]["data_kind"] != "observed":
            raise ValueError("actual gate testing needs an observed frozen model and held-out trajectories")
    elif value["model"] is not None or value["role"] != "development":
        raise ValueError("study collects development labels; it does not run a fitted gate")
    core.number(value["max_measurement_fraction"], "measurement fraction", 1e-12, .1)
    core.integer(value["recent_window"], "recent window", 1)
    validate_code_hashes(value.get("code_hashes"))
    if value["code_hashes"] != manifest(root.parent.parent)["code_hashes"]:
        raise ValueError("state code binding differs from the switch manifest")
    return value


def measurement_worker(out, arm, *, window, wall_cap, scoring_only=False):
    c = core.read(out / "contract.json")
    run = Path(c["source_run"])
    step = c["config"]["drift"]
    report = rule.measure(run / "rollouts_behavior_train.jsonl", stats=run / f"policy_step_{step}/grpo_stats.jsonl",
        step=step, prompts=c["n"], responses=8, seed=c["config"]["seed"], window=window, wall_cap=wall_cap)
    expected = core.read(out / "net_inputs.json")
    if expected != {"rollouts_behavior_train.jsonl": report["source_sha256"],
                    f"policy_step_{step}/grpo_stats.jsonl": report["stats_sha256"]}:
        raise ValueError("pre-decision inputs changed")
    if arm == "gate_measurement":
        model = protocol(out.parent.parent)["model"]
        runtime.check_model(model, c)
        report["choice"] = rule.choose(model, report["features"])
        report["checkpoint_only"] = rule.choose(model, report["features"], checkpoint_only=True)
    base.bind(out / arm / "measurement.json", report)


def decision(out, suite, p, arm, env):
    c = core.read(out / "contract.json")
    binding = {"protocol_sha256": core.fingerprint(p), "contract_sha256": base.digest(out / "contract.json")}
    value = {"binding": binding, "action": "select" if arm.startswith("selection_") else "random",
             "reason": "control_arm", "profile_sha256": None, "measurement_gpu_seconds": 0.,
             "budget_gpu_seconds": c["budget_gpu_seconds"], "start_step": c["config"]["drift"], "schedule": rule.SCHEDULE}
    if arm in {*rule.DEV_ARMS, "gated"}:
        measured = out / ("measurement" if p["mode"] == "study" else "gate_measurement")
        first = runtime.measure_once(out, suite, p, measured, env)
        value.update(measurement_gpu_seconds=first["gpu_seconds"], profile_sha256=first["report_sha256"])
        value["budget_gpu_seconds"] -= first["gpu_seconds"]
        if arm == "gated":
            runtime.check_model(p["model"], c)
            value.update(core.read(measured / "measurement.json")["choice"] if first["status"] == "complete" else
                         {"action": "random", "reason": "measurement_failed_no_retry", "prediction": None})
        elif first["status"] != "complete":
            if p["mode"] == "study":
                raise ValueError("failed development measurement cannot form a feature/label pair")
            value["reason"] = "control_with_failed_diagnostic_charge"
    if value["budget_gpu_seconds"] <= 0:
        raise ValueError("diagnosis exhausted the branch allocation")
    base.bind(out / arm / "decision.json", value)
    return value


def freeze_decisions(out, suite, p, env):
    """No control may start before the held-out gate decision is durably frozen."""
    path = out / "decisions-frozen.json"
    with base.lease(out / ".decision-barrier.lock"):
        if path.exists():
            value = core.read(path)
            if value["protocol_sha256"] != core.fingerprint(p) or value["decisions"] != {
                    arm: base.digest(out / arm / "decision.json") for arm in p["arms"]}:
                raise ValueError("frozen decision barrier changed")
            return value
        if any((out / arm / name).exists() for arm in p["arms"] for name in ("execution.json", "result.json")):
            raise ValueError("continuation artifacts precede the decision barrier")
        # Paid controls share this exact diagnostic. Test gate is evaluated first.
        order = (["gated"] if p["mode"] == "test" else []) + [a for a in p["arms"] if a != "gated"]
        for arm in order:
            runtime.decision(out, suite, p, arm, env)
        value = {"protocol_sha256": core.fingerprint(p), "frozen_at": time.time(),
                 "decisions": {arm: base.digest(out / arm / "decision.json") for arm in p["arms"]}}
        base.bind(path, value)
        return value


def select_once(out, c, p, arm, choice, env, devices):
    directory = out / arm
    private = directory / "fresh-r"
    run = Path(c["source_run"])
    parent = run / f"policy_step_{c['config']['drift']}"
    base.bind(private / "scoring.json", {"config": c["config"], "parent": str(parent),
        "adapter_sha256": base.digest(parent / "adapter_model.safetensors"), "prompts": str(run / "prompts.json"),
        "prompts_sha256": base.digest(run / "prompts.json"), "sampling_seed": 701000003+c["config"]["seed"]*1000003+c["config"]["drift"]*7919,
        "contract_sha256": base.digest(out / "contract.json"), "protocol_sha256": core.fingerprint(p)})
    for stage in ("validation", "candidate"):
        commands = [([sys.executable, str(base.ROOT / "src/selection_switch_score.py"), "--root", str(private),
                      "--stage", stage, "--shard", str(i)], devices[i]) for i in range(4)
                    if not (private / f"{stage}-{i}.done.json").exists()]
        if commands:
            base.meter(directory, f"fresh-r-{stage}", c["scope"]["gpu_type"], commands=commands, env=env,
                timeout=(choice["budget_gpu_seconds"]-base.spent(directory))/4, ledger="deployment")
        base.meter(directory, f"fresh-r-merge-{stage}", c["scope"]["gpu_type"],
                   action=lambda: scoring.merge(private, stage), ledger="deployment")
    if core.read(private / "selected.sha256.json") != {"sha256": base.digest(private / "selected.json")}:
        raise ValueError("selection changed")
    return core.read(private / "selected.json")["indices"]


def collect(root, *, development):
    p, rows, missing = manifest(root), [], []
    seeds = rule.DEV_SEEDS if development else rule.TEST_SEEDS
    for seed in seeds:
        for step in rule.STEPS:
            child = child_root(root, seed, step)
            try:
                protocol_value = protocol(child)
                out = next(base.entries(child))
                c = verify(out)
                freeze = core.read(out / "decisions-frozen.json")
                if freeze["decisions"] != {a: base.digest(out / a / "decision.json") for a in protocol_value["arms"]}:
                    raise ValueError("decision evidence changed")
                results = {arm: runtime.validate_result(out, protocol_value, arm) for arm in protocol_value["arms"]}
                for arm, result in results.items():
                    base.policy(out, c, arm)
                    if result["rewards"] != base.rewards(out, c, arm):
                        raise ValueError("reported rewards differ from evaluation artifacts")
                if len({tuple(sorted(r["rewards"])) for r in results.values()}) != 1:
                    raise ValueError("paired evaluation identities differ")
                measured_dir = out / ("measurement" if development else "gate_measurement")
                initial = core.read(measured_dir / "initial.json")
                profile = core.read(measured_dir / "measurement.json") if initial["status"] == "complete" else None
                means = {arm: statistics.fmean(r["rewards"].values()) for arm, r in results.items()}
                trajectory, parent = runtime.identity(c)
                row = {"seed": seed, "step": step, "role": c["role"], "complete": True,
                    "scope": c["scope"], "trajectory_id": trajectory, "parent": list(parent),
                    "budget_gpu_seconds": p["budget_gpu_seconds"], "features": profile["features"] if profile else None,
                    "means": means, "branches": results, "decision_frozen_at": freeze["frozen_at"],
                    "measurement_gpu_seconds": initial["gpu_seconds"]}
                if not development:
                    decision = core.read(out / "gated/decision.json")
                    row["intended_action"] = decision["action"]
                    row["actual_action"] = results["gated"]["action"]
                    row["fallback"] = (decision["reason"] == "measurement_failed_no_retry" or
                                       core.read(out / "gated/execution.json")["reason"] == "selector_failed")
                    row["audit"] = rule.decision_audit(means, decision["action"])
                    if profile:
                        row["checkpoint_only_audit"] = rule.decision_audit(means, profile["checkpoint_only"]["action"])
                    row["paired_question_differences"] = {i: results["selection_reduced"]["rewards"][i]-results["random_reduced"]["rewards"][i]
                                                          for i in results["selection_reduced"]["rewards"]}
                rows.append(row)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                missing.append({"seed": seed, "step": step, "reason": str(exc)})
    return {"schema": rule.SCHEMA, "complete": not missing, "rows": rows, "missing_or_failed": missing}


def fit_once(root):
    # Missing labels are not a fitting job; other nodes need not contend for its lock.
    if not (root / "model.json").exists() and any(
            not list(child_root(root, s, t).glob(f"points/*/{arm}/result.json"))
            for s in rule.DEV_SEEDS for t in rule.STEPS for arm in rule.DEV_ARMS):
        return False
    with base.lease(root / ".fit.lock"):
        if (root / "model.json").exists():
            rule.validate_model(core.read(root / "model.json"))
            return True
        # Do not repeatedly hash large model/evaluation files until every dev arm has a result.
        if any(not list(child_root(root, s, t).glob(f"points/*/{arm}/result.json"))
               for s in rule.DEV_SEEDS for t in rule.STEPS for arm in rule.DEV_ARMS):
            return False
        started = time.monotonic()
        data = collect(root, development=True)
        if not data["complete"]:
            raise ValueError(f"development labels are invalid: {data['missing_or_failed']}")
        model = rule.fit(data["rows"])
        base.bind(root / "development.json", data)
        base.bind(root / "model.json", model)
        elapsed = time.monotonic()-started
        allocated = 4 if os.environ.get("OM_NODE_LOCK_HELD") == "1" else 0
        base.bind(root / "fit-cost.json", {"wall_seconds": elapsed,
            "gpu_seconds": allocated*elapsed, "allocated_gpus": allocated,
            "ledger": "offline research; not a deployment diagnostic"})
        print(f"[frozen] {root/'model.json'}; held-out branches now eligible", flush=True)
        return True


def admitted_devices(p):
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 4 or len(set(devices)) != 4 or not all(devices) or os.environ.get("OM_NODE_LOCK_HELD") != "1":
        raise ValueError("requires one admitted node with four distinct GPUs")
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", ",".join(devices)], text=True, timeout=20).splitlines()
    if len(hardware) != 4 or set(map(str.strip, hardware)) != {p["gpu_type"]}:
        raise ValueError("hardware differs from frozen allocation")
    return devices


def smoke(root):
    """One registered prefix and paid continuation; full run reuses these artifacts."""
    import additive_experiment as ae
    p = manifest(root)
    devices = admitted_devices(p)
    env = ae.model_environment(p["sources"]["0"]["config"])
    directory = prefix_dir(root, 0)
    with base.lease(directory / ".prefix.lock"):
        if not (directory / "prefix-25.json").exists():
            build_prefix(root, 0, 25, devices, env)
    child = child_root(root, 0, 25)
    with base.lease(child / ".publish.lock", blocking=True):
        if not (child / "net_protocol.json").exists():
            publish_state(root, 0, 25)
    out = next(base.entries(child))
    with base.lease(out / "selection_reduced/.task.lock"):
        suite, p = core.read(child / "suite.json"), protocol(child)
        freeze_decisions(out, suite, p, env)
        runtime.run_arm(out, suite, p, "selection_reduced", devices, env)
    print(f"[smoke complete] registered s0/t25 CONTINUE_D; reused by full run: {out}")


def busy_task(label, directory):
    paths = [directory / "progress.json"]
    if directory.parent.parent.name == "points":
        paths += [directory.parent / name / "progress.json" for name in ("measurement", "gate_measurement")]
    for path in paths:
        if path.exists():
            progress = core.read(path)
            if progress.get("state") == "running" and 0 <= time.time()-progress.get("updated", 0) < 60:
                return {"task": label, "active": True, "host": progress.get("host", "unknown"),
                        "pid": progress.get("pid"), "phase": progress.get("phase", "unknown")}
    return {"task": label, "active": False}


def wait_for_peers(busy, *, last_progress, idle_timeout):
    if not busy:
        return False
    active = [item for item in busy if item["active"]]
    if not active and time.monotonic()-last_progress >= idle_timeout:
        return False
    owners = {(item["host"], item["pid"]) for item in active}
    detail = ",".join(item["task"] + (f"@{item['host']}:{item['pid']}({item['phase']})" if item["active"] else "(locked)")
                      for item in busy[:8])
    print(f"[waiting] no claimable task; active_peers={len(owners)}; busy={detail}; retry in 15s", flush=True)
    time.sleep(15)
    return True


def work(root, *, idle_timeout=600.):
    import additive_experiment as ae
    p = manifest(root)
    core.number(idle_timeout, "idle timeout", 0.)
    devices = admitted_devices(p)
    attempted, failures = set(), 0
    last_progress = time.monotonic()
    while True:
        progress, busy = False, []
        try:
            fit_once(root)
        except BlockingIOError:
            busy.append(busy_task("gate-fit", root))
        for seed in (*rule.DEV_SEEDS, *rule.TEST_SEEDS):
            env = ae.model_environment(p["sources"][str(seed)]["config"])
            for step in rule.STEPS:
                cert = prefix_dir(root, seed) / f"prefix-{step}.json"
                if not cert.exists() or (seed in rule.TEST_SEEDS and not (root / "model.json").exists()):
                    continue
                child = child_root(root, seed, step)
                if not (child / "net_protocol.json").exists():
                    try:
                        with base.lease(child / ".publish.lock"):
                            if not (child / "net_protocol.json").exists():
                                publish_state(root, seed, step)
                    except BlockingIOError:
                        busy.append(busy_task(f"s{seed}/t{step}/publish", child))
                        continue
                out = next(base.entries(child))
                protocol_value, suite = protocol(child), core.read(child / "suite.json")
                for arm in protocol_value["arms"] if seed % 2 == 0 else protocol_value["arms"][::-1]:
                    key = (seed, step, arm)
                    directory = out / arm
                    if key in attempted or (directory / "result.json").exists():
                        continue
                    try:
                        with base.lease(directory / ".task.lock"):
                            if (directory / "result.json").exists():
                                continue
                            freeze_decisions(out, suite, protocol_value, env)
                            attempted.add(key)
                            print(f"[claimed] host={socket.gethostname()} pid={os.getpid()} task=s{seed}/t{step}/{arm}", flush=True)
                            runtime.run_arm(out, suite, protocol_value, arm, devices, env)
                            progress = True
                    except BlockingIOError:
                        busy.append(busy_task(f"s{seed}/t{step}/{arm}", directory))
                    except Exception as exc:
                        attempted.add(key)
                        failures += 1
                        record_failure(directory, exc)
            # One segment per pass publishes ready work without waiting for the whole prefix.
            for step in rule.STEPS:
                directory = prefix_dir(root, seed)
                key = (seed, step, "prefix")
                if (directory / f"prefix-{step}.json").exists():
                    continue
                if key in attempted:
                    break
                previous = (0, *rule.STEPS)[rule.STEPS.index(step)]
                if previous and not (directory / f"prefix-{previous}.json").exists():
                    break
                try:
                    with base.lease(directory / ".prefix.lock"):
                        if not (directory / f"prefix-{step}.json").exists():
                            attempted.add(key)
                            print(f"[claimed] host={socket.gethostname()} pid={os.getpid()} task=s{seed}/t{step}/prefix", flush=True)
                            build_prefix(root, seed, step, devices, env)
                            progress = True
                except BlockingIOError:
                    busy.append(busy_task(f"s{seed}/t{step}/prefix", directory / f"segment-{step}"))
                except Exception as exc:
                    attempted.add(key)
                    failures += 1
                    record_failure(directory / f"segment-{step}", exc)
                break
        if progress:
            last_progress = time.monotonic()
        else:
            if failures:
                print(f"[queue] failed_on_this_node={failures}; failed tasks need attention, not more nodes", flush=True)
            if not wait_for_peers(busy, last_progress=last_progress, idle_timeout=idle_timeout):
                break
    status(root)
    return int(bool(failures))


def record_failure(directory, exc):
    traceback.print_exc()
    core.atomic_json(directory / "failure.json", {"error": str(exc), "host": socket.gethostname(), "time": time.time()})
    print(f"[failed] {directory}: {exc}; trying other tasks (no automatic retry loop)", flush=True)


def status(root):
    if not (root / "switch.json").exists():
        print(f"[not prepared] {root}")
        return
    manifest(root)
    done, total = 0, 48
    print("STATE        ARM                  STATUS     DETAIL")
    for seed in (*rule.DEV_SEEDS, *rule.TEST_SEEDS):
        prefix = prefix_dir(root, seed)
        reached = [t for t in rule.STEPS if (prefix / f"prefix-{t}.json").exists()]
        prefix_state = "DONE" if 100 in reached else "QUEUED"
        detail = ""
        for t in rule.STEPS:
            if t in reached:
                continue
            segment = prefix / f"segment-{t}"
            progress = core.read(segment / "progress.json") if (segment / "progress.json").exists() else {}
            if progress.get("state") == "running" and time.time()-progress.get("updated", 0) < 60:
                prefix_state, detail = "RUNNING", f"to {t}: {progress.get('seconds', 0):.0f}s"
            elif (segment / "failure.json").exists():
                prefix_state, detail = "FAILED", core.read(segment / "failure.json")["error"]
            break
        print(f"s{seed} prefix   fresh_r              {prefix_state:10} {max(reached, default=0):3}/100 {detail}")
        for step in rule.STEPS:
            child = child_root(root, seed, step)
            points = list((child / "points").glob("*")) if (child / "points").exists() else []
            arms = rule.DEV_ARMS if seed in rule.DEV_SEEDS else rule.TEST_ARMS
            for arm in arms:
                state, detail = "QUEUED", "prefix pending" if step not in reached else "development/model pending"
                if points:
                    directory = points[0] / arm
                    if (directory / "result.json").exists():
                        try:
                            r = runtime.validate_result(points[0], protocol(child), arm)
                            state, detail = "DONE", f"reward={statistics.fmean(r['rewards'].values()):.4f} updates={r['completed_steps']-step}"
                            done += 1
                        except (OSError, ValueError, KeyError) as exc:
                            state, detail = "INVALID", str(exc)
                    else:
                        prog = core.read(directory / "progress.json") if (directory / "progress.json").exists() else {}
                        failure = core.read(directory / "failure.json") if (directory / "failure.json").exists() else {}
                        if prog.get("state") == "running" and time.time()-prog.get("updated", 0) < 60:
                            state, detail = "RUNNING", prog["phase"]
                        elif failure:
                            state, detail = "FAILED", failure["error"]
                        else:
                            detail = "ready"
                print(f"s{seed}/t{step:<6} {arm:20} {state:10} {detail}")
    print(f"[switch] {done}/{total} DONE; logs/results: {root}")
    prefix_costs = prefix_cost_report(root)
    if not prefix_costs["complete"]:
        print("[prefix-cost pending] research total UNKNOWN; prefix execution is allowed; "
              "inspect with bash scripts/run_selection_switch.sh recover-cost")


def summarize(root):
    for development, name in ((True, "development-report.json"), (False, "test-report.json")):
        report = collect(root, development=development)
        report["prefix_research_cost"] = prefix_cost_report(root)
        if not development:
            report["summary"] = rule.clustered_summary(report["rows"])
        core.atomic_json(root / name, report)
        print(f"[saved] {root/name}: {len(report['rows'])} complete states, {len(report['missing_or_failed'])} missing/failed")


def install_runtime():
    # Reuse the frozen learner/ledger implementation without changing its legacy entry points.
    runtime.net, runtime.HERE = rule, HERE
    runtime.TEST_ARMS, runtime.SELECTORS, runtime.CODE_FILES = rule.TEST_ARMS, ("fresh_r",), CODE
    runtime.study = SimpleNamespace(BRANCHES=rule.DEV_ARMS, reward_mean=runtime.study.reward_mean)
    runtime.protocol, runtime.select_once, runtime.measurement_worker = protocol, select_once, measurement_worker
    runtime.decision = decision
    base.verify = verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "smoke", "worker", "status", "summarize", "fit", "check-code"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--budget-gpu-seconds", type=float)
    parser.add_argument("--gpu-type", default="NVIDIA H100 80GB HBM3")
    parser.add_argument("--eval-prompts", type=Path)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--pool-manifest", type=Path)
    parser.add_argument("--test-count", type=int, default=300)
    parser.add_argument("--eval-k", type=int, default=8)
    parser.add_argument("--eval-timeout", type=float, default=14400.)
    parser.add_argument("--prefix-timeout", type=float, default=14400.)
    parser.add_argument("--idle-timeout", type=float, default=600.)
    parser.add_argument("--phase", choices=("measure", "evaluate"))
    parser.add_argument("--arm")
    parser.add_argument("--shard", type=int, choices=range(4))
    parser.add_argument("--recent-window", type=int, default=20)
    parser.add_argument("--measurement-wall-seconds", type=float, default=30.)
    args = parser.parse_args()
    install_runtime()
    if args.command == "prepare":
        if not args.matrix:
            parser.error("prepare requires --matrix")
        prepare(args)
    elif args.command == "check-code":
        check_code(args.root)
    elif args.command == "worker":
        if os.environ.get("OM_NODE_LOCK_HELD") != "1" or not args.phase or not args.arm:
            parser.error("worker requires an admitted node, phase and arm")
        if args.phase == "measure":
            measurement_worker(args.root, args.arm, window=args.recent_window, wall_cap=args.measurement_wall_seconds)
        elif args.shard is None:
            parser.error("evaluation requires a shard")
        else:
            base.evaluate(args.root, args.arm, args.shard)
    elif args.command == "run":
        return work(args.root, idle_timeout=args.idle_timeout)
    elif args.command == "smoke":
        smoke(args.root)
    elif args.command == "fit":
        if not fit_once(args.root):
            raise ValueError("all 18 development continuations must finish first")
    elif args.command == "status":
        status(args.root)
    else:
        summarize(args.root)
    return 0


if __name__ == "__main__":
    from light_selection_gate_gpu import install_signal_handlers
    install_signal_handlers()
    raise SystemExit(main())
