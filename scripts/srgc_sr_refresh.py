#!/usr/bin/env python3
"""SR with refreshed success rates: the "cache refresh" experiment named in the Limitations.

The recorded SR arm ranks the whole candidate pool once, by success rates cached
from the initial policy, and never updates that ranking. This arm keeps SR's
rule (train the four prompts whose success rate is closest to 0.5) but re-measures
the success rates under the CURRENT policy on the same schedule On-policy uses:
every 25 updates it draws 40 candidates, generates eight fresh responses per
candidate (320 rollouts, without scoring gradients or validation generation),
ranks them by |success rate - 0.5| and trains the top four until
the next refresh. ``--scope pool`` instead re-measures all 400 candidates at
every refresh (3200 rollouts per refresh; roughly one cache build per refresh).

It forks from the seed's verified shared prefix in the existing run root and
writes ``sr_refresh-latest.pt`` / ``sr_refresh-progress.json`` /
``sr_refresh-endpoint.json`` (plus ``sr_refresh-pool-*`` for the pool scope)
next to the four recorded arms. The hashed ``srgc_rebuttal`` package is not
modified: the engine variant lives here and is launched through torchrun.

    sh scripts/run_srgc_sr_refresh.sh math 5              # candidates scope, seed 5
    sh scripts/run_srgc_sr_refresh.sh math 5 pool         # full-pool refresh
    sh scripts/run_srgc_sr_refresh.sh math results
"""

import argparse
from contextlib import ExitStack
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srgc_rebuttal.srgc import Engine, cached_sr_set, stream_seed  # noqa: E402

SCOPES = ("candidates", "pool")


def arm_name(scope):
    return "sr_refresh" if scope == "candidates" else "sr_refresh-pool"


EXTRA_ARMS = ("sr_refresh", "sr_refresh-pool", "switch_repeat", "switch_fixed100", "switch_fixed125")


def extra_arm(name):
    """argparse type: a listed extra arm or any switch_fixed<N>."""
    if name in EXTRA_ARMS or re.fullmatch(r"switch_fixed\d+", name):
        return name
    raise argparse.ArgumentTypeError(f"unknown extra arm {name!r}; use one of {EXTRA_ARMS} or switch_fixed<N>")


def make_engine(arm, backend, data, config):
    """Engine for an extra arm (its engine label is the recorded arm name it extends)."""
    if arm in {"sr_refresh", "sr_refresh-pool"}:
        return SRRefreshEngine(backend, data["candidate_ids"], data["ranking_validation_ids"], data["cached_rewards"],
                               arm="sr_refresh", config=config,
                               refresh_scope="candidates" if arm == "sr_refresh" else "pool"), "sr_refresh"
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
    """Engine whose ``sr_refresh`` arm re-measures success rates with the current policy."""

    ARMS = Engine.ARMS | {"sr_refresh"}

    def __init__(self, *args, refresh_scope="candidates", **kwargs):
        if refresh_scope not in SCOPES:
            raise ValueError("refresh scope must be candidates or pool")
        self.refresh_scope = refresh_scope
        super().__init__(*args, **kwargs)

    def _refresh_ranking(self):
        c = self.config
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
            record.update(refreshed_ids=list(ids), refreshed_success_rates={i: float(means[i]) for i in ids},
                          ranked_ids=list(ranked[:c.scoring_prompts]), selected_ids=list(train_ids),
                          scored_distinct_prompts=len(ids), scoring_responses_per_prompt=c.responses)
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
                      selector="sr_refresh", metrics=metrics)
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


def run(args):
    """Inside torchrun: fork the arm from the shared prefix and run it to the plan's total."""
    prepare_run_storage(args)
    from srgc_rebuttal.plan import digest, input_path, load_plan, validate_inputs
    from srgc_rebuttal.runtime import arm_complete, atomic_json, identity, lease, matches, prefix_ready, run_root
    from srgc_rebuttal.srgc import Config
    from srgc_rebuttal.cost_ledger import PhaseLedger
    from srgc_rebuttal.timing import invocation, torch_meter
    from srgc_rebuttal.distributed import initialize, primary
    from srgc_rebuttal.progress import record as progress
    from srgc_rebuttal.existing_runtime import load_model
    from srgc_rebuttal.torch_backend import TorchBackend
    try:
        from srgc_verifier_fallback import install as install_tolerant_verifier
        install_tolerant_verifier()
    except ImportError:
        pass
    invocation_started = time.perf_counter()
    plan = load_plan(args.plan)
    if args.seed not in plan["seeds"]:
        raise SystemExit("seed is not in the frozen plan")
    arm = args.arm if getattr(args, "arm", None) else arm_name(args.scope)
    if re.fullmatch(r"switch_fixed\d+", arm):
        from scripts.srgc_switch_fixed import fixed_step_of
        step = fixed_step_of(arm)
        if not plan["shared_prefix_updates"] <= step < plan["total_updates"] or step % plan["selection_interval"]:
            raise ValueError("fixed step must be a refresh boundary from the shared prefix to before the endpoint")
    scope = "candidates" if arm == "sr_refresh" else "pool" if arm == "sr_refresh-pool" else None
    folder = run_root(args.plan, plan) / f"seed-{args.seed}"
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
            expected = identity(args.plan, plan, args.seed)
            if not prefix_ready(folder, expected, plan["shared_prefix_updates"]):
                raise ValueError("the verified shared prefix must finish before this arm can start")
            return data, expected, arm_complete(folder, expected, arm, plan["total_updates"])
        data, expected, complete = primary(startup)
        if complete:
            if rank == 0:
                print(f"PASS: {arm} already complete for seed {args.seed}")
            return
        result = [None]
        if rank == 0:
            try:
                locks.enter_context(lease(folder / f".{arm}.execution.lock"))
                with lease(folder / ".manifest.lock", wait=True):
                    marker = json.loads((folder / "run.json").read_text())
                    if not matches(marker, expected):
                        raise ValueError("the seed's run manifest belongs to a different experiment")
                    atomic_json(folder / f"{arm}-run.json", {**expected, "arm": arm, "refresh_scope": scope,
                        "status": "running", "packages": {p: importlib.metadata.version(p)
                                                          for p in ("torch", "transformers", "peft", "math-verify")}})
            except Exception as exc:  # noqa: BLE001 - report on rank 0, abort every rank
                result[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(result, src=0)
        if result[0]:
            raise RuntimeError(result[0])
        meter = torch_meter(lambda event: PhaseLedger(folder / "cost-receipts" / arm).record(event))
        locks.enter_context(invocation(PhaseLedger(folder / "invocations" / arm), meter, world, invocation_started))
        with meter.phase("startup", gpu_count=world):
            torch.manual_seed(args.seed)
            with meter.stage("model_and_tokenizer_load"):
                model, tokenizer = load_model(plan["model"], plan["model_revision"], torch.device("cuda", local_rank))
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
            current, engine_arm = make_engine(arm, backend, data, config)

        def save(path, engine):
            with meter.phase("checkpoint_save", engine.step, world):
                with meter.stage("state_snapshot"):
                    state = engine.state_dict()
                def write():
                    with meter.stage("write"):
                        temporary = path.with_suffix(".tmp")
                        torch.save(state, temporary)
                        temporary.replace(path)
                primary(write)

        def load(path):
            with meter.phase("checkpoint_load", gpu_count=world), meter.stage("read"):
                return torch.load(path, map_location="cpu", weights_only=False)

        def restore(engine, state, **kwargs):
            with meter.phase("checkpoint_load", state["step"], world), meter.stage("restore"):
                engine.load_state_dict(state, **kwargs)

        prefix_hash = json.loads((folder / "prefix-ready.json").read_text())["checkpoint_sha256"]
        checkpoint_path = folder / f"{arm}-latest.pt"
        if primary(checkpoint_path.exists):
            state = load(checkpoint_path)
            if state["arm"] != engine_arm or not plan["shared_prefix_updates"] <= state["step"] <= plan["total_updates"]:
                raise ValueError("resume checkpoint has the wrong arm or update count")
            restore(current, state)
        else:
            shared = load(folder / "prefix.pt")
            if shared["arm"] != "on_policy" or shared["step"] != plan["shared_prefix_updates"]:
                raise ValueError("shared prefix has the wrong arm or update count")
            restore(current, shared, fork_arm=engine_arm)
        while current.step < plan["total_updates"]:
            current.update()
            progress("update", arm=arm, step=current.step)
            if rank == 0:
                print(f"TRAIN seed={args.seed} phase={arm} arm={arm} step={current.step}/{plan['total_updates']} "
                      f"status=completed completed={current.step}/{plan['total_updates']}", flush=True)
            if current.step % 25 == 0 or current.step == plan["total_updates"]:
                save(checkpoint_path, current)
                primary(lambda: atomic_json(folder / f"{arm}-progress.json", {"seed": args.seed, "arm": arm,
                        "refresh_scope": scope, "step": current.step, "switched_at": current.switched_at,
                        "transitions": getattr(current, "transitions", None),
                        "sampling_protocol": current.SAMPLING_PROTOCOL, "costs": current.costs,
                        "history": current.history}))
        with meter.phase("evaluation", current.step, world):
            per_question = backend.evaluate(data["evaluation_ids"],
                seed=stream_seed(args.seed, current.step, "reporting-evaluation"), responses=8)

        def endpoint():
            ledger = PhaseLedger(folder / "cost-receipts" / arm).totals()
            measured = {**current.costs, **ledger["known_gpu_seconds"]}
            atomic_json(folder / f"{arm}-endpoint.json", {**expected, "arm": arm, "refresh_scope": scope,
                "prefix_checkpoint_sha256": prefix_hash, "total_updates": current.step,
                "shared_prefix_updates": plan["shared_prefix_updates"], "switched_at": current.switched_at,
                "transitions": getattr(current, "transitions", None),
                "checks": [{"step": r["checkpoint"], "d": r["d"]} for r in current.history if r.get("d") is not None],
                "sampling_protocol": current.SAMPLING_PROTOCOL,
                "reward": sum(per_question.values()) / len(per_question), "per_question_reward": per_question,
                "costs": measured, "cost_measurement_complete": ledger["complete"], "cost_receipts": ledger,
                "evaluation_gpu_seconds": ledger["known_gpu_seconds"]["evaluation_gpu_seconds"],
                "selection_interval": plan["selection_interval"],
                "selection_steps": [r["checkpoint"] for r in current.history if r["selection_refreshed"]],
                "refreshed_prompts_per_selection": (current.config.scoring_prompts if scope == "candidates"
                                                    else len(current.candidates) if scope == "pool" else None)})
            atomic_json(folder / f"{arm}-run.json", {**json.loads((folder / f"{arm}-run.json").read_text()),
                                                       "status": "complete"})
        primary(endpoint)
        if rank == 0:
            print(f"PASS: {arm} seed {args.seed} reward={sum(per_question.values()) / len(per_question):.4f}")


def prepare_run_storage(args):
    """Set cache paths in the GPU process, not just the shell's plan lookup child."""
    from scripts.srgc_shared_storage import route_plan, storage_root
    from srgc_rebuttal.plan import load_plan
    from srgc_rebuttal.runtime import identity, prefix_ready, run_root
    plan = load_plan(args.plan)
    if args.seed not in plan["seeds"]:
        raise ValueError("seed is not in the frozen plan")
    group, _ = storage_root(os.environ)
    folder = run_root(args.plan, plan) / f"seed-{args.seed}"
    if not folder.is_relative_to(group):
        raise ValueError("extra arms require an existing group-volume run; start the main queue first")
    if not prefix_ready(folder, identity(args.plan, plan, args.seed), plan["shared_prefix_updates"]):
        raise ValueError("the verified shared prefix must finish before this arm can start")
    args.plan = route_plan(args.plan, writing=True)


def results(args):
    """Per-seed endpoint rewards of the refresh arm next to the recorded arms, plus costs."""
    from srgc_rebuttal.plan import load_plan
    from srgc_rebuttal.runtime import identity, matches, prefix_ready, run_root
    plan = load_plan(args.plan)
    root = run_root(args.plan, plan)
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
                value = json.loads(path.read_text())
                expected = identity(args.plan, plan, seed)
                if (not matches(value, expected) or value.get("arm") != arm or
                        value.get("total_updates") != plan["total_updates"] or
                        not prefix_ready(folder, expected, plan["shared_prefix_updates"])):
                    raise ValueError(f"{path}: endpoint identity, prefix or update count differs")
                prefix = json.loads((folder / "prefix-ready.json").read_text())
                if value.get("prefix_checkpoint_sha256") != prefix["checkpoint_sha256"]:
                    raise ValueError(f"{path}: endpoint used a different shared prefix")
                row[arm] = {"reward_percent": 100 * value["reward"],
                            "selection_gpu_seconds": value["costs"].get("selection_gpu_seconds"),
                            "training_gpu_seconds": value["costs"].get("training_gpu_seconds"),
                            "cost_measurement_complete": value.get("cost_measurement_complete"),
                            "transitions": value.get("transitions"), "switched_at": value.get("switched_at")}
        rows.append(row)
    if args.json:
        print(json.dumps({"dataset": plan["dataset"], "rows": rows}, indent=2))
        return
    print(f"SR refresh · {plan['dataset']} · {root}")
    header = "seed  " + "  ".join(f"{arm:>16}" for arm in arms)
    print(header)
    for row in rows:
        cells = []
        for arm in arms:
            value = row.get(arm)
            cells.append(f"{value['reward_percent']:15.2f}%" if value else f"{'-':>16}")
        print(f"{row['seed']:>4}  " + "  ".join(cells))
    for row in rows:
        for arm in (a for a in arms if a in {"switch", "switch_repeat"} or a.startswith("switch_fixed")):
            value = row.get(arm)
            if value and (value.get("transitions") or value.get("switched_at") is not None):
                moves = value.get("transitions") or [{"step": value["switched_at"], "to": "sr"}]
                print(f"  seed {row['seed']} {arm}: " + ", ".join(f"step {m['step']} -> {m['to']}" for m in moves))
    print("selection GPU-seconds (refresh rollouts for sr_refresh; scoring for on_policy/switch/switch_repeat):")
    for row in rows:
        cells = []
        for arm in arms:
            value = row.get(arm)
            cost = value.get("selection_gpu_seconds") if value else None
            cells.append(f"{cost:16.0f}" if cost is not None else f"{'unknown' if value else '-':>16}")
        print(f"{row['seed']:>4}  " + "  ".join(cells))
    for row in rows:
        for arm in arms:
            value = row.get(arm)
            if value and value["cost_measurement_complete"] is not True:
                print(f"  seed {row['seed']} {arm}: cost measurement incomplete or unverified")


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
                           help="overrides --scope: switch_repeat (repeated transitions) or switch_fixed<N> (fixed schedule)")
        else:
            p.add_argument("--json", action="store_true")
    args = parser.parse_args()
    (run if args.command == "run" else results)(args)


if __name__ == "__main__":
    main()
