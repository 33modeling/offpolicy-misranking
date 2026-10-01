#!/usr/bin/env python3
"""SR with refreshed success rates: the "cache refresh" experiment named in the Limitations.

The recorded SR arm ranks the whole candidate pool once, by success rates cached
from the initial policy, and never updates that ranking. This arm keeps SR's
rule (train the four prompts whose success rate is closest to 0.5) but re-measures
the success rates under the CURRENT policy on the same schedule On-policy uses:
every 25 updates it draws 40 candidates, generates eight fresh responses per
candidate (320 rollouts, without scoring gradients or validation generation),
ranks them by |success rate - 0.5| and trains the top four until
the next refresh. ``--scope pool`` instead re-measures the entire input pool at
every refresh (3200 rollouts for 400 candidates; eight times the actual pool size).

It forks from the seed's verified shared prefix in the existing run root and
writes ``sr_refresh-latest.pt`` / ``sr_refresh-progress.json`` /
``sr_refresh-endpoint.json`` (plus ``sr_refresh-pool-*`` for the pool scope)
next to the four recorded arms. The hashed ``srgc_rebuttal`` package is not
modified: the engine variant lives here and is launched through torchrun.

``sr_hold`` is the matching cached-SR control: the same 25-update draw of 40
candidates and the same four-prompt batch retained until the next refresh, but
ranked by the frozen cache instead of fresh success rates. ``sr_refresh`` minus
``sr_hold`` isolates the refresh; ``sr_hold`` minus ``sr`` isolates the batch
retention interval.

The same entry point runs the other extra arms: ``switch_repeat``,
``switch_fixed<N>`` (``srgc_switch_fixed``), the matched direction ablations
``direction_removed`` / ``direction_magnitude`` / ``direction_replaced``
(``srgc_direction_ablation``) and independent replicates ``replicate<k>-<arm>``
of a recorded arm or fixed control on a separate post-prefix sampling stream
(``srgc_replicate``; outputs under ``seed-N/replicate-<k>/``).

    sh scripts/run_srgc_sr_refresh.sh math 5              # candidates scope, seed 5
    sh scripts/run_srgc_sr_refresh.sh math 5 pool         # full-pool refresh
    sh scripts/run_srgc_sr_refresh.sh math 5 sr_hold      # cached SR, 25-update batch retention
    sh scripts/run_srgc_sr_refresh.sh math 5 direction_removed
    sh scripts/run_srgc_sr_refresh.sh math 5 replicate1-switch
    sh scripts/run_srgc_sr_refresh.sh math results
"""

import argparse
from contextlib import ExitStack
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srgc_rebuttal.srgc import Engine, cached_sr_set, stream_seed  # noqa: E402

SCOPES = ("candidates", "pool", "cached")
SCOPE_ARMS = {"candidates": "sr_refresh", "pool": "sr_refresh-pool", "cached": "sr_hold"}


def arm_name(scope):
    return SCOPE_ARMS[scope]


def scope_of(arm):
    return next((scope for scope, name in SCOPE_ARMS.items() if name == arm), None)


DIRECTION_ARMS = ("direction_removed", "direction_magnitude", "direction_replaced")
EXTRA_ARMS = ("sr_refresh", "sr_refresh-pool", "sr_hold", "switch_repeat", "switch_fixed100", "switch_fixed125",
              *DIRECTION_ARMS)


def replicate_of(name):
    from scripts.srgc_replicate import parse_replicate_arm
    return parse_replicate_arm(name)


def extra_arm(name):
    """argparse type: a listed extra arm, any switch_fixed<N>, or replicate<k>-<recorded arm|switch_fixed<N>>."""
    if name in EXTRA_ARMS or re.fullmatch(r"switch_fixed\d+", name):
        return name
    try:
        if replicate_of(name):
            return name
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    raise argparse.ArgumentTypeError(f"unknown extra arm {name!r}; use one of {EXTRA_ARMS}, switch_fixed<N> "
                                     "or replicate<k>-<random|sr|on_policy|switch|switch_fixed<N>>")


def make_engine(arm, backend, data, config):
    """Engine for an extra arm (its engine label is the recorded arm name it extends)."""
    if arm in SCOPE_ARMS.values():
        return SRRefreshEngine(backend, data["candidate_ids"], data["ranking_validation_ids"], data["cached_rewards"],
                               arm="sr_refresh", config=config, refresh_scope=scope_of(arm)), "sr_refresh"
    if arm in DIRECTION_ARMS:
        from scripts.srgc_direction_ablation import DirectionAblationEngine, mode_of
        return DirectionAblationEngine(backend, data["candidate_ids"], data["ranking_validation_ids"],
                                       data["cached_rewards"], arm="direction_ablation", config=config,
                                       mode=mode_of(arm)), "direction_ablation"
    replicate = replicate_of(arm)
    if replicate:
        from scripts.srgc_replicate import make_engine as make_replicate_engine
        return make_replicate_engine(replicate[0], replicate[1], backend, data, config)
    if arm == "switch_repeat":
        from scripts.srgc_switch_repeat import SwitchRepeatEngine
        return SwitchRepeatEngine(backend, data["candidate_ids"], data["ranking_validation_ids"],
                                  data["cached_rewards"], arm="switch_repeat", config=config), "switch_repeat"
    if re.fullmatch(r"switch_fixed\d+", arm):
        from scripts.srgc_switch_fixed import SwitchFixedEngine, fixed_step_of
        return SwitchFixedEngine(backend, data["candidate_ids"], data["ranking_validation_ids"],
                                 data["cached_rewards"], arm="switch_fixed", config=config,
                                 fixed_step=fixed_step_of(arm)), "switch_fixed"
    raise ValueError(f"unknown extra arm {arm!r}")


class SRRefreshEngine(Engine):
    """Engine whose ``sr_refresh`` arm re-measures success rates with the current policy.

    The ``cached`` scope (arm ``sr_hold``) keeps the refresh schedule and batch
    retention but ranks the drawn 40 by the frozen cache, without any rollout.
    """

    ARMS = Engine.ARMS | {"sr_refresh"}

    def __init__(self, *args, refresh_scope="candidates", **kwargs):
        if refresh_scope not in SCOPES:
            raise ValueError("refresh scope must be candidates, pool or cached")
        self.refresh_scope = refresh_scope
        super().__init__(*args, **kwargs)

    def _refresh_ranking(self):
        c = self.config
        if self.refresh_scope == "cached":
            ids = self._draw_candidates()
            with self._timing_scope("candidate_sampling_and_ranking"):
                eligible = set(ids)
                ranked = tuple(i for i in self.sr_ranked_ids if i in eligible)  # the SR arm's rule on this draw
            return ids, None, ranked
        ids = self._draw_candidates() if self.refresh_scope == "candidates" else self.candidates
        with self._timing_scope("sr_refresh_rollouts", section=True):
            means = dict(self.backend.evaluate(ids, seed=stream_seed(c.seed, self.step, "sr-refresh"),
                                               responses=c.responses))
        if set(means) != set(ids):
            raise ValueError("backend must return one success rate per refreshed prompt")
        rewards = {}
        for i in ids:
            successes = int(round(float(means[i]) * c.responses))
            if not 0 <= successes <= c.responses or abs(successes / c.responses - float(means[i])) > 1e-9:
                raise ValueError(f"{i}: success rate is not a multiple of 1/{c.responses}")
            rewards[i] = [1.0] * successes + [0.0] * (c.responses - successes)
        with self._timing_scope("sr_refresh_ranking"):
            ranked = cached_sr_set(ids, rewards, len(ids), c.seed, c.responses)
        return ids, means, ranked

    def update(self):
        if self.arm != "sr_refresh":
            return super().update()
        c = self.config
        before = dict(self.costs)
        refresh = self.step % c.selection_interval == 0
        record = {"checkpoint": self.step, "d": None, "switched": False, "selection_refreshed": refresh,
                  "refresh_scope": self.refresh_scope}
        if not refresh and self.active_selection is None:
            raise ValueError("mid-block continuation requires the saved selected prompts")
        if refresh:
            started = self._begin("selection")
            ids, means, ranked = self._refresh_ranking()
            train_ids = ranked[:c.training_prompts]
            # The saved block keeps On-policy's shape (40 ids containing the 4 trained ones) so
            # checkpoints resume through the unchanged validation in Engine.load_state_dict.
            block = list(ranked[:c.scoring_prompts]) if self.refresh_scope == "pool" else list(ids)
            self.active_selection = {"step": self.step, "on_ids": block, "train_ids": list(train_ids)}
            self._end("selection", started)
            record.update(refreshed_ids=list(ids), ranked_ids=list(ranked[:c.scoring_prompts]),
                          selected_ids=list(train_ids), scored_distinct_prompts=0 if means is None else len(ids),
                          scoring_responses_per_prompt=0 if means is None else c.responses,
                          refreshed_success_rates=None if means is None else {i: float(means[i]) for i in ids})
        train_ids = tuple(self.active_selection["train_ids"])
        record["selection_step"] = self.active_selection["step"]
        used, cycle = set(self.used_training_ids), self.sampling_cycle
        started = self._begin("training")
        used.update(train_ids)
        if len(used) == len(self.candidates):
            used.clear()
            cycle += 1
        metrics = dict(self.backend.train(train_ids, responses=c.responses, objective=c.objective,
                                          seed=stream_seed(c.seed, self.step, "training")))
        self._end("training", started)
        self.used_training_ids, self.sampling_cycle = used, cycle
        self.step += 1
        record.update(completed_updates=self.step, train_ids=list(train_ids),
                      selection_gpu_seconds=self.costs["selection_gpu_seconds"] - before["selection_gpu_seconds"],
                      training_gpu_seconds=self.costs["training_gpu_seconds"] - before["training_gpu_seconds"],
                      selector="sr_hold" if self.refresh_scope == "cached" else "sr_refresh", metrics=metrics)
        self.history.append(record)
        return record

    def state_dict(self):
        state = super().state_dict()
        state["refresh_scope"] = self.refresh_scope
        return state

    def load_state_dict(self, state, *, fork_arm=None):
        if fork_arm is None and state.get("refresh_scope", "candidates") != self.refresh_scope:
            raise ValueError("checkpoint refresh scope differs; use a new arm name for a different scope")
        super().load_state_dict(state, fork_arm=fork_arm)


def extra_checkpoint_policy(state, *, resuming, previous_run=None):
    """Keep the kernel of the saved continuation, or inherit its shared prefix.

    Older extra runners ignored SRGC_ATTENTION and always loaded eager, even
    when their prefix used SDPA. Preserve that kernel on an interrupted run.
    """
    from scripts.srgc_child_tuning import ATTENTION_CHOICES
    from srgc_rebuttal.plan import digest
    if resuming:
        attention = state.get("checkpoint_policy", {}).get("attention", "eager")
        source = "continuation"
    elif previous_run is not None:
        attention = previous_run.get("checkpoint_policy", {}).get("attention", "eager")
        source = "previous-attempt"
    else:
        attention = state.get("checkpoint_policy", {}).get("attention", "eager")
        source = "shared-prefix"
    if attention not in ATTENTION_CHOICES:
        raise ValueError(f"unsupported saved attention kernel: {attention!r}")
    return {"interval_updates": 1, "attention": attention, "attention_source": source,
            "storage_adapter_sha256": digest(Path(__file__))}


def continue_updates(current, out, arm, launch_arm, total_updates, extras, policy):
    from scripts.srgc_step_checkpoints import save_checkpoint
    from srgc_rebuttal.distributed import primary
    from srgc_rebuttal.progress import record as progress
    from srgc_rebuttal.runtime import atomic_json
    while current.step < total_updates:
        primary(lambda: print(f"TRAIN seed={current.config.seed} phase={launch_arm} arm={launch_arm} "
                              f"step={current.step + 1}/{total_updates} status=running "
                              f"completed={current.step}/{total_updates}", flush=True))
        current.update()
        save_checkpoint(current, out, arm, metadata={"checkpoint_policy": policy})
        primary(lambda: atomic_json(out / f"{arm}-progress.json", {
            "seed": current.config.seed, "arm": arm, **extras, "step": current.step,
            "switched_at": current.switched_at, "transitions": getattr(current, "transitions", None),
            "sampling_protocol": current.SAMPLING_PROTOCOL, "costs": current.costs,
            "checkpoint_policy": policy, "history": current.history}))
        progress("update", arm=launch_arm, step=current.step)
        primary(lambda: print(f"TRAIN seed={current.config.seed} phase={launch_arm} arm={launch_arm} "
                              f"step={current.step}/{total_updates} status=completed "
                              f"completed={current.step}/{total_updates}", flush=True))


def run(args):
    """Inside torchrun: fork the arm from the shared prefix and run it to the plan's total."""
    prepare_run_storage(args, verify_checkpoint=False)
    prepare_verifier_runtime()
    from srgc_rebuttal.plan import digest, input_path, load_plan, validate_inputs
    from srgc_rebuttal.runtime import atomic_json, identity, lease, matches, run_root
    from srgc_rebuttal.srgc import Config
    from srgc_rebuttal.cost_ledger import PhaseLedger
    from srgc_rebuttal.timing import invocation, torch_meter
    from srgc_rebuttal.distributed import initialize, primary
    from srgc_rebuttal.progress import record as progress
    from srgc_rebuttal.existing_runtime import load_model
    try:
        from srgc_resumable_rollouts import install as install_resumable_rollouts
    except ImportError:
        from scripts.srgc_resumable_rollouts import install as install_resumable_rollouts
    try:
        from srgc_verifier_fallback import install as install_tolerant_verifier
        install_tolerant_verifier()
    except ImportError:
        pass
    invocation_started = time.perf_counter()
    plan = load_plan(args.plan)
    if args.seed not in plan["seeds"]:
        raise SystemExit("seed is not in the frozen plan")
    launch_arm = args.arm if getattr(args, "arm", None) else arm_name(args.scope)
    replicate = replicate_of(launch_arm)
    arm = replicate[1] if replicate else launch_arm  # the name in checkpoint, progress and endpoint files
    if re.fullmatch(r"switch_fixed\d+", arm):
        from scripts.srgc_switch_fixed import fixed_step_of
        step = fixed_step_of(arm)
        if not plan["shared_prefix_updates"] <= step < plan["total_updates"] or step % plan["selection_interval"]:
            raise ValueError("fixed step must be a refresh boundary from the shared prefix to before the endpoint")
    scope = scope_of(arm)
    folder = run_root(args.plan, plan) / f"seed-{args.seed}"
    # A replicate keeps its files apart from the recorded arm it repeats.
    out = folder / f"replicate-{replicate[0]}" if replicate else folder
    TorchBackend = install_resumable_rollouts(out / "rollout-cache" / arm)
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    world = plan["world_size"]
    rank, local_rank = initialize(world)
    with ExitStack() as locks:
        locks.push(lambda exc_type, *_: dist.destroy_process_group() if exc_type is None else None)

        def startup():
            data = json.loads(input_path(args.plan, plan, args.seed).read_text())
            validate_inputs(data)
            expected, _ = result_identity(args.plan, plan, args.seed)
            return data, expected, extra_complete(args.plan, plan, args.seed, launch_arm)
        data, expected, complete = primary(startup)
        if complete:
            if rank == 0:
                print(f"PASS: {launch_arm} already complete for seed {args.seed}")
            return
        prefix_hash = json.loads((folder / "prefix-ready.json").read_text())["checkpoint_sha256"]
        result = [None, None]
        if rank == 0:
            try:
                locks.enter_context(lease(out / f".{arm}.execution.lock"))
                with lease(folder / ".manifest.lock", wait=True):
                    marker = json.loads((folder / "run.json").read_text())
                    if not matches(marker, expected):
                        raise ValueError("the seed's run manifest belongs to a different experiment")
                    if replicate:
                        replicate_manifest(out, expected, args.seed, replicate[0], prefix_hash)
                    marker_path = out / f"{arm}-run.json"
                    if marker_path.exists():
                        result[1] = json.loads(marker_path.read_text())
                        if not matches(result[1], expected) or result[1].get("arm") != arm:
                            raise ValueError("the extra arm manifest belongs to a different experiment")
            except Exception as exc:  # noqa: BLE001 - report on rank 0, abort every rank
                result[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(result, src=0)
        if result[0]:
            raise RuntimeError(result[0])
        meter = torch_meter(lambda event: PhaseLedger(out / "cost-receipts" / arm).record(event))
        locks.enter_context(invocation(PhaseLedger(out / "invocations" / arm), meter, world, invocation_started))
        checkpoint_path = out / f"{arm}-latest.pt"
        resuming = primary(checkpoint_path.exists)
        with meter.phase("checkpoint_load", gpu_count=world), meter.stage("read"):
            state = torch.load(checkpoint_path if resuming else folder / "prefix.pt",
                               map_location="cpu", weights_only=False)
        policy = extra_checkpoint_policy(state, resuming=resuming, previous_run=result[1])
        primary(lambda: atomic_json(out / f"{arm}-run.json", {**expected, "arm": arm, "launch_arm": launch_arm,
            "refresh_scope": scope, "replicate": replicate[0] if replicate else None,
            "checkpoint_policy": policy, "status": "running",
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "math-verify")}}))
        primary(lambda: print(f"ATTENTION {policy['attention']} source={policy['attention_source']} "
                              "(saved run/prefix kernel takes precedence over SRGC_ATTENTION)", flush=True))
        from scripts.srgc_child_tuning import count_progress
        count_progress()
        with meter.phase("startup", gpu_count=world):
            torch.manual_seed(args.seed)
            with meter.stage("model_and_tokenizer_load"):
                model, tokenizer = load_model(plan["model"], plan["model_revision"], torch.device("cuda", local_rank),
                                              attention=policy["attention"])
            with meter.stage("adapter_load"):
                model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                                                        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
            module, function = plan["verifier"].split(":", 1)
            backend = TorchBackend(model, tokenizer, data["records"], getattr(importlib.import_module(module), function),
                projection_dim=plan["projection_dim"], projection_seed=plan["projection_seed"],
                max_new_tokens=plan["max_new_tokens"], logprob_micro_batch=plan.get("logprob_micro_batch", 2),
                logit_chunk_tokens=plan.get("logit_chunk_tokens", 512), cost_meter=meter)
        progress("model_ready")
        config = Config(seed=args.seed, objective=plan["objective"],
                        selection_interval=plan["selection_interval"], check_interval=plan["check_interval"],
                        first_check=plan["first_check"], scoring_prompts=plan["scoring_prompts_per_set"],
                        training_prompts=plan["training_prompts"], responses=plan["responses"],
                        projection_dim=plan["projection_dim"])
        with meter.phase("preparation", gpu_count=world), meter.stage("selector_setup"):
            current, engine_arm = make_engine(launch_arm, backend, data, config)
        if replicate and current.replicate_record() != json.loads((out / "replicate.json").read_text())["replicate"]:
            raise ValueError("the replicate manifest and the engine disagree on the sampling stream")

        def restore(engine, state, **kwargs):
            with meter.phase("checkpoint_load", state["step"], world), meter.stage("restore"):
                engine.load_state_dict(state, **kwargs)

        if resuming:
            if state["arm"] != engine_arm or not plan["shared_prefix_updates"] <= state["step"] <= plan["total_updates"]:
                raise ValueError("resume checkpoint has the wrong arm or update count")
            restore(current, state)
        else:
            if state["arm"] != "on_policy" or state["step"] != plan["shared_prefix_updates"]:
                raise ValueError("shared prefix has the wrong arm or update count")
            restore(current, state, fork_arm=engine_arm)
        del state
        extras = {"refresh_scope": scope, "launch_arm": launch_arm,
                  "replicate": current.replicate_record() if replicate else None,
                  "ablation": getattr(current, "mode", None) if engine_arm == "direction_ablation" else None}
        continue_updates(current, out, arm, launch_arm, plan["total_updates"], extras, policy)
        with meter.phase("evaluation", current.step, world):
            # The endpoint evaluation keeps the recorded rule and the base seed for every replicate.
            per_question = backend.evaluate(data["evaluation_ids"],
                seed=stream_seed(args.seed, current.step, "reporting-evaluation"), responses=8)

        def endpoint():
            ledger = PhaseLedger(out / "cost-receipts" / arm).totals()
            measured = {**current.costs, **ledger["known_gpu_seconds"]}
            atomic_json(out / f"{arm}-endpoint.json", {**expected, "arm": arm, **extras,
                "prefix_checkpoint_sha256": prefix_hash, "total_updates": current.step,
                "shared_prefix_updates": plan["shared_prefix_updates"], "switched_at": current.switched_at,
                "transitions": getattr(current, "transitions", None),
                "checks": [{"step": r["checkpoint"], "d": r["d"]} for r in current.history if r.get("d") is not None],
                "sampling_protocol": current.SAMPLING_PROTOCOL,
                "checkpoint_policy": policy,
                "reward": sum(per_question.values()) / len(per_question), "per_question_reward": per_question,
                "costs": measured, "cost_measurement_complete": ledger["complete"], "cost_receipts": ledger,
                "evaluation_gpu_seconds": ledger["known_gpu_seconds"]["evaluation_gpu_seconds"],
                "selection_interval": plan["selection_interval"],
                "selection_steps": [r["checkpoint"] for r in current.history if r["selection_refreshed"]],
                "refreshed_prompts_per_selection": (current.config.scoring_prompts if scope == "candidates"
                                                    else len(current.candidates) if scope == "pool"
                                                    else 0 if scope == "cached" else None)})
            atomic_json(out / f"{arm}-run.json", {**json.loads((out / f"{arm}-run.json").read_text()),
                                                    "status": "complete"})
        primary(endpoint)
        if rank == 0:
            print(f"PASS: {launch_arm} seed {args.seed} reward={sum(per_question.values()) / len(per_question):.4f}")


def replicate_manifest(out, expected, seed, replicate, prefix_hash):
    """Write ``replicate.json`` once per replicate folder; a later launch must describe the same replicate."""
    from scripts.srgc_replicate import REPLICATE_PROTOCOL, sampling_seed
    from srgc_rebuttal.runtime import atomic_json, matches
    manifest = {**expected, "prefix_checkpoint_sha256": prefix_hash,
                "replicate": {"protocol": REPLICATE_PROTOCOL, "id": replicate, "base_seed": seed,
                              "sampling_seed": sampling_seed(seed, replicate)}}
    path = out / "replicate.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if not matches(saved, manifest):
            raise ValueError("the replicate folder belongs to a different experiment, prefix or sampling stream")
        return saved
    atomic_json(path, manifest)
    return manifest


def prepare_verifier_runtime():
    """Use P0's pinned offline verifier before package receipts or CUDA startup."""
    from srgc_rebuttal.existing_runtime import verifier_environment
    verifier_environment(os.environ)


def prepare_run_storage(args, *, verify_checkpoint=True):
    """Set cache paths in the GPU process, not just the shell's plan lookup child."""
    from scripts.srgc_shared_storage import route_plan, storage_root
    from srgc_rebuttal.plan import load_plan
    from srgc_rebuttal.runtime import run_root
    print(f"PLAN reading {args.plan}", file=sys.stderr, flush=True)
    plan = load_plan(args.plan)
    if args.seed not in plan["seeds"]:
        raise ValueError("seed is not in the frozen plan")
    print("STORAGE resolving shared run directory", file=sys.stderr, flush=True)
    group, _ = storage_root(os.environ)
    folder = run_root(args.plan, plan) / f"seed-{args.seed}"
    if not folder.is_relative_to(group):
        raise ValueError("extra arms require an existing group-volume run; start the main queue first")
    result_identity(args.plan, plan, args.seed, verify_checkpoint=verify_checkpoint)
    args.plan = route_plan(args.plan, writing=True)


def result_identity(plan_path, plan, seed, *, recorded=False, verify_checkpoint=True):
    """Recorded identities are for read-only reports, never admission or resume."""
    from srgc_rebuttal.runtime import code_digest, matches
    from srgc_rebuttal.runtime import run_root
    from srgc_rebuttal.plan import input_path
    from scripts.srgc_prefix_check import stream_digest, verify_prefix
    print(f"INPUT resolving bundle for seed {seed}", file=sys.stderr, flush=True)
    bundle = input_path(plan_path, plan, seed)
    expected = {"plan_sha256": stream_digest(plan_path, label="PLAN"),
                "input_sha256": stream_digest(bundle),
                "implementation_sha256": code_digest(), "seed": seed}
    folder = run_root(plan_path, plan) / f"seed-{seed}"
    if recorded:
        marker = json.loads((folder / "run.json").read_text())
        stable = {k: v for k, v in expected.items() if k != "implementation_sha256"}
        if (not isinstance(marker, dict) or not matches(marker, stable) or not isinstance(marker.get("implementation_sha256"), str)
                or not marker["implementation_sha256"]):
            raise ValueError(f"{folder}: run identity differs from plan or inputs")
        expected["implementation_sha256"] = marker["implementation_sha256"]
    prefix_hash = verify_prefix(folder, expected, plan["shared_prefix_updates"],
                                verify_checkpoint=verify_checkpoint)
    return expected, prefix_hash


def _endpoint(path, plan_path, plan, seed, folder, arm, *, verified=None):
    """Validate the result's provenance, evaluation and finite nonnegative costs."""
    from srgc_rebuttal.plan import input_path
    from srgc_rebuttal.runtime import matches
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an endpoint JSON object")
    expected, prefix_hash = verified or result_identity(plan_path, plan, seed)
    if (not matches(value, expected) or value.get("arm") != arm or
            value.get("total_updates") != plan["total_updates"] or
            value.get("shared_prefix_updates") != plan["shared_prefix_updates"]):
        raise ValueError(f"{path}: endpoint identity, prefix or update count differs")
    if value.get("prefix_checkpoint_sha256") != prefix_hash:
        raise ValueError(f"{path}: endpoint used a different shared prefix")
    def finite_number(number):
        return type(number) in (float, int) and math.isfinite(number)
    evaluation_ids = json.loads(input_path(plan_path, plan, seed).read_text())["evaluation_ids"]
    rewards = value.get("per_question_reward")
    reward = value.get("reward")
    if (not isinstance(rewards, dict) or not rewards or set(rewards) != set(evaluation_ids)
            or any(not finite_number(v) or not 0 <= v <= 1 for v in rewards.values())
            or not finite_number(reward) or not 0 <= reward <= 1
            or abs(sum(rewards.values()) / len(rewards) - reward) > 1e-10):
        raise ValueError(f"{path}: invalid or mismatched evaluation rewards")
    costs = value.get("costs")
    if not isinstance(costs, dict) or any(v is not None and (not finite_number(v) or v < 0)
                                         for v in costs.values()):
        raise ValueError(f"{path}: costs must be finite nonnegative measurements or null")
    return {"reward_percent": 100 * value["reward"],
            "selection_gpu_seconds": costs.get("selection_gpu_seconds"),
            "training_gpu_seconds": costs.get("training_gpu_seconds"),
            "cost_measurement_complete": value.get("cost_measurement_complete"),
            "implementation_sha256": expected["implementation_sha256"],
            "checkpoint_policy": value.get("checkpoint_policy"),
            "transitions": value.get("transitions"), "switched_at": value.get("switched_at")}


def checked_replicate_manifest(out, seed, replicate, verified):
    from scripts.srgc_replicate import REPLICATE_PROTOCOL, sampling_seed
    from srgc_rebuttal.runtime import matches
    manifest = out / "replicate.json"
    saved = json.loads(manifest.read_text())
    if not isinstance(saved, dict):
        raise ValueError(f"{manifest}: expected a replicate JSON object")
    expected, prefix_hash = verified
    record = {"protocol": REPLICATE_PROTOCOL, "id": replicate, "base_seed": seed,
              "sampling_seed": sampling_seed(seed, replicate)}
    if not matches(saved, {**expected, "replicate": record}):
        raise ValueError(f"{manifest}: replicate manifest identity or sampling stream differs")
    if saved.get("prefix_checkpoint_sha256") != prefix_hash:
        raise ValueError(f"{manifest}: replicate used a different shared prefix")
    return record


def extra_complete(plan_path, plan, seed, name):
    """Use the same strict result checks before skipping work or publishing success."""
    from srgc_rebuttal.runtime import run_root
    replicate = replicate_of(name)
    arm = replicate[1] if replicate else name
    folder = run_root(plan_path, plan) / f"seed-{seed}"
    out = folder / f"replicate-{replicate[0]}" if replicate else folder
    path = out / f"{arm}-endpoint.json"
    if not path.exists():
        return False
    verified = result_identity(plan_path, plan, seed)
    _endpoint(path, plan_path, plan, seed, folder, arm, verified=verified)
    if replicate:
        record = checked_replicate_manifest(out, seed, replicate[0], verified)
        if json.loads(path.read_text()).get("replicate") != record:
            raise ValueError(f"{path}: endpoint replicate record differs from the manifest")
    return True


def replicate_rows(plan_path, plan, root, *, errors=None, context=None):
    """Independent replicates: one row per ``seed-N/replicate-<k>/`` with its arms' endpoints."""
    context = context or (lambda seed: result_identity(plan_path, plan, seed, recorded=True))
    def invalid(path, exc):
        if errors is None:
            raise exc
        errors.append(f"{path}: {type(exc).__name__}: {exc}")
    rows = []
    for seed in plan["seeds"]:
        folder = root / f"seed-{seed}"
        for manifest in sorted(folder.glob("replicate-*/replicate.json")):
            try:
                match = re.fullmatch(r"replicate-([1-9]\d*)", manifest.parent.name)
                if not match:
                    raise ValueError("invalid replicate directory name")
                replicate = int(match.group(1))
                verified = context(seed)
                record = checked_replicate_manifest(manifest.parent, seed, replicate, verified)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                invalid(manifest, exc)
                continue
            row = {"seed": seed, "replicate": replicate, "sampling_seed": record["sampling_seed"]}
            for path in sorted(manifest.parent.glob("*-endpoint.json")):
                arm = path.name.removesuffix("-endpoint.json")
                try:
                    value = _endpoint(path, plan_path, plan, seed, folder, arm, verified=verified)
                    if json.loads(path.read_text()).get("replicate") != record:
                        raise ValueError(f"{path}: endpoint replicate record differs from the manifest")
                    row[arm] = value
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    invalid(path, exc)
            rows.append(row)
    return sorted(rows, key=lambda row: (row["seed"], row["replicate"]))


def results(args):
    """Per-seed endpoint rewards of the extra arms next to the recorded arms, plus costs and replicates."""
    from srgc_rebuttal.plan import load_plan
    from srgc_rebuttal.runtime import run_root
    plan = load_plan(args.plan)
    root = run_root(args.plan, plan)
    errors, warnings, identities = [], [], {}
    def context(seed):
        from srgc_rebuttal.runtime import code_digest
        if seed not in identities:
            identities[seed] = result_identity(args.plan, plan, seed, recorded=True)
            if identities[seed][0]["implementation_sha256"] != code_digest():
                warnings.append(f"seed {seed}: code changed; viewing recorded results only, resume is blocked")
        return identities[seed]
    fixed = {p.name.removesuffix("-endpoint.json")
             for seed in plan["seeds"] for p in (root / f"seed-{seed}").glob("switch_fixed*-endpoint.json")
             if re.fullmatch(r"switch_fixed\d+-endpoint.json", p.name)}
    arms = [*plan["arms"], *EXTRA_ARMS,
            *sorted(fixed - set(EXTRA_ARMS), key=lambda name: int(name.removeprefix("switch_fixed")))]
    rows = []
    for seed in plan["seeds"]:
        row = {"seed": seed}
        folder = root / f"seed-{seed}"
        for arm in arms:
            path = folder / f"{arm}-endpoint.json"
            if path.exists():
                try:
                    row[arm] = _endpoint(path, args.plan, plan, seed, folder, arm, verified=context(seed))
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append(f"{path}: {type(exc).__name__}: {exc}")
        rows.append(row)
    replicates = replicate_rows(args.plan, plan, root, errors=errors, context=context)
    if args.json:
        print(json.dumps({"dataset": plan["dataset"], "output_root": str(root), "rows": rows,
                          "replicates": replicates, "errors": errors, "warnings": warnings}, indent=2, allow_nan=False))
        if errors:
            raise ValueError("; ".join(errors))
        return
    width = max(16, *(len(arm) for arm in arms))
    print(f"SR refresh · {plan['dataset']} · {root}")
    header = "seed  " + "  ".join(f"{arm:>{width}}" for arm in arms)
    print(header)
    for row in rows:
        cells = []
        for arm in arms:
            value = row.get(arm)
            cells.append(f"{value['reward_percent']:{width - 1}.2f}%" if value else f"{'-':>{width}}")
        print(f"{row['seed']:>4}  " + "  ".join(cells))
    for row in rows:
        for arm in (a for a in arms if a in {"switch", "switch_repeat"} or a.startswith("switch_fixed")):
            value = row.get(arm)
            if value and (value.get("transitions") or value.get("switched_at") is not None):
                moves = value.get("transitions") or [{"step": value["switched_at"], "to": "sr"}]
                print(f"  seed {row['seed']} {arm}: " + ", ".join(f"step {m['step']} -> {m['to']}" for m in moves))
    print("selection GPU-seconds (refresh rollouts for sr_refresh; ranking only for sr_hold; "
          "scoring for on_policy/switch/switch_repeat/switch_fixed/direction_*):")
    for row in rows:
        cells = []
        for arm in arms:
            value = row.get(arm)
            cost = value.get("selection_gpu_seconds") if value else None
            cells.append(f"{cost:{width}.0f}" if cost is not None else f"{'unknown' if value else '-':>{width}}")
        print(f"{row['seed']:>4}  " + "  ".join(cells))
    for row in rows:
        for arm in arms:
            value = row.get(arm)
            if value and value["cost_measurement_complete"] is not True:
                print(f"  seed {row['seed']} {arm}: cost measurement incomplete or unverified")
    if replicates:
        print("independent replicates (same verified prefix, separate post-prefix sampling stream; "
              "the recorded run is stream 0):")
        for row in replicates:
            arms_done = [k for k in row if k not in {"seed", "replicate", "sampling_seed"}]
            cells = ", ".join(f"{arm} {row[arm]['reward_percent']:.2f}%" for arm in arms_done) or "no endpoint yet"
            line = f"  seed {row['seed']} replicate {row['replicate']} (sampling seed {row['sampling_seed']}): {cells}"
            if "switch" in row and "sr" in row:
                line += f"; switch - sr = {row['switch']['reward_percent'] - row['sr']['reward_percent']:+.2f} pp"
            print(line)
            for arm in arms_done:
                value = row[arm]
                if value.get("transitions") or value.get("switched_at") is not None:
                    moves = value.get("transitions") or [{"step": value["switched_at"], "to": "sr"}]
                    print(f"    {arm}: " + ", ".join(f"step {m['step']} -> {m['to']}" for m in moves))
                if value["cost_measurement_complete"] is not True:
                    print(f"    {arm}: cost measurement incomplete or unverified")
        for row in rows:
            if "switch" in row and "sr" in row:
                print(f"  seed {row['seed']} recorded (stream 0): switch - sr = "
                      f"{row['switch']['reward_percent'] - row['sr']['reward_percent']:+.2f} pp")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        raise ValueError("; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "results"):
        p = sub.add_parser(name)
        p.add_argument("--plan", type=Path, required=True)
        if name == "run":
            p.add_argument("--seed", type=int, required=True)
            p.add_argument("--scope", choices=SCOPES, default="candidates")
            p.add_argument("--arm", type=extra_arm,
                           help="overrides --scope: sr_hold, switch_repeat, switch_fixed<N>, "
                                "direction_removed|magnitude|replaced, "
                                "or replicate<k>-<random|sr|on_policy|switch|switch_fixed<N>>")
        else:
            p.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        try:
            results(args)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"INVALID RESULTS: {exc}", file=sys.stderr)
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
