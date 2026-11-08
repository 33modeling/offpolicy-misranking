#!/usr/bin/env python3
"""torchrun child for one v8 task on a 4xH100 node (one process per GPU).

    torchrun --standalone --nproc_per_node=4 cluster_child.py --cohort olmo_math \
        --matrix $V8_ROOT/olmo_math/matrix/bank-branches.rebuttal_bank.json --task <run_id|task_id>

Model loading, LoRA, TorchBackend/QwenBackend, verifier runtime and plan resolution are
the code repository's own functions (V8_CODE_ROOT = the frozen checkout that produced
the seed runs; implementation_sha256 must match run.json).

Exit 75 = task busy elsewhere, or stopped (SIGTERM/SIGINT) and resumable from latest.pt.
torchrun reports every nonzero worker exit as its own exit 1, so rank 0 also writes
{"exit_code": 75, "status": "busy"|"interrupted", ...} to $V8_EXIT_FILE (default
$SRGC_PROGRESS_DIR/exit.json); launchers read that file, not torchrun's code.
A stop abandons the in-flight update at once (torchrun SIGKILLs 30 s after it is
signalled, the v8 supervisor after 10 s); a rank blocked in a collective is ended by
a watchdog after V8_STOP_GRACE_SECONDS (7).
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from switching import coderepo  # noqa: E402
from switching.cluster import load_bundle, v8_root  # noqa: E402
from switching.common import read_json  # noqa: E402
from switching.runner import install_stop_handlers, run_child_task, start_stop_watchdog  # noqa: E402


def _stamp():
    import time
    return time.strftime("%H:%M:%S")


def install_diagnostics(root, task_id, rank):
    """Per-rank stack dumps: on SIGUSR1, on fatal signals and every V8_STACK_DUMP_SECONDS (default 1800)."""
    import faulthandler
    import signal
    stacks = Path(root) / ".multi-node" / "stacks"
    stacks.mkdir(parents=True, exist_ok=True)
    handle = open(stacks / f"{task_id}-rank{rank}.txt", "a", buffering=1)
    handle.write(f"== {_stamp()} pid={os.getpid()} rank={rank} task={task_id}\n")
    faulthandler.enable(file=handle, all_threads=True)
    try:
        faulthandler.register(signal.SIGUSR1, file=handle, all_threads=True, chain=False)
    except (AttributeError, ValueError, RuntimeError):
        pass
    seconds = int(os.environ.get("V8_STACK_DUMP_SECONDS", "1800") or 0)
    if seconds > 0:
        faulthandler.dump_traceback_later(seconds, repeat=True, file=handle)
    return handle


def install_progress(rank, interval=None):
    """Print the frozen backend's own progress records (rank 0, throttled) so long refreshes are visible."""
    import time
    interval = float(os.environ.get("V8_PROGRESS_SECONDS", "60") if interval is None else interval)
    from srgc_rebuttal import progress as progress_module
    targets = []
    for name in ("torch_backend", "run_experiment"):
        try:
            targets.append(importlib.import_module(f"srgc_rebuttal.{name}"))
        except Exception:  # noqa: BLE001 - optional module in older frozen engines
            pass
    state = {"last": 0.0, "count": 0}

    def wrap(original):
        if getattr(original, "_v8_progress", False):
            return original

        def record(stage, **details):
            original(stage, **details)
            state["count"] += 1
            now = time.monotonic()
            if rank == 0 and now - state["last"] >= interval:
                state["last"] = now
                text = " ".join(f"{k}={v}" for k, v in details.items())
                print(f"PROGRESS {_stamp()} stage={stage} records={state['count']} {text}".rstrip(), flush=True)
        record._v8_progress = True
        return record

    for module in targets:
        if hasattr(module, "progress"):
            module.progress = wrap(module.progress)
    progress_module.record = wrap(progress_module.record)
    return state


def install_verifier_fallback(code_root, rank):
    """Same tolerant math verifier the seed 5-9 children installed (srgc_step_checkpoints / srgc_sr_refresh)."""
    sys.path.insert(0, str(Path(code_root) / "scripts"))
    try:
        from srgc_verifier_fallback import install
    except ImportError as exc:
        if rank == 0:
            print(f"WARN verifier fallback unavailable: {exc}", flush=True)
        return False
    install()
    return True


def _gather(value):
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, value)
    return parts


def all_ranks(label, fn, *, gather=_gather):
    """Run `fn` on every rank; if it fails anywhere, every rank raises the same error.

    One symmetric collective after the call, so a failure on one rank can neither be skipped silently
    nor leave the other ranks waiting in a later collective.
    """
    try:
        value, error = fn(), None
    except Exception as exc:  # noqa: BLE001 - reported identically on every rank below
        value, error = None, f"{type(exc).__name__}: {exc}"
    errors = {rank: e for rank, e in enumerate(gather(error)) if e}
    if errors:
        raise RuntimeError(f"{label} failed on rank(s) {sorted(errors)}: " +
                           "; ".join(f"rank {r}: {e}" for r, e in sorted(errors.items())))
    return value


def prepare_runtime(profile, *, gather=_gather):
    """Seed-run startup checks, fatal and identical on every rank.

    srgc_sr_refresh.run() aborts when the pinned offline math-verify bundle cannot be attached; the
    Qwen seed runs also enforced their package pins (transformers 5.14.1, peft 0.20.0, fla-core 0.5.2:
    without fla, transformers silently falls back to the slow torch DeltaNet kernel).
    """
    def verifier():
        from scripts.srgc_sr_refresh import prepare_verifier_runtime
        prepare_verifier_runtime()
    all_ranks("verifier runtime (pinned offline math-verify bundle)", verifier, gather=gather)
    if profile == "qwen":
        import srgc_qwen35
        return all_ranks("Qwen runtime packages (transformers/peft/fla-core pins)", srgc_qwen35.runtime_packages,
                         gather=gather)
    return None


def verifier_identity(plan, fallback_installed):
    """What scores the rollouts: recorded with every evaluation (and compared on reuse)."""
    import importlib.metadata as metadata
    packages = {}
    for name in ("math-verify", "sympy", "latex2sympy2_extended", "antlr4-python3-runtime"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"plan_verifier": plan["verifier"], "tolerant_math_fallback": bool(fallback_installed),
            "packages": packages}


def protected_backend_class(base_class):
    """The seed runs' rollout protections (scripts/srgc_resumable_rollouts.py) inside scoring/evaluation
    blocks: the CUDA cache is released after every rollout and one CUDA out-of-memory is retried once
    after freeing it (same per-rank draw seed, so the retried rollout is the same draw). Rollouts are
    not persisted for reuse across attempts: reused work would vanish from the surviving attempt's
    algorithm cost (the killed attempt's events are dropped from the lineage)."""
    import gc
    import time
    import torch
    oom = (torch.OutOfMemoryError,) if hasattr(torch, "OutOfMemoryError") else (torch.cuda.OutOfMemoryError,)

    class Protected(base_class):
        _v8_protect = False
        _oom_retry_sleep = 5.0

        def _release_cuda(self):
            if getattr(getattr(self, "device", None), "type", "") == "cuda":
                torch.cuda.empty_cache()

        def _rollout(self, prompt_id, responses, seed):
            if not self._v8_protect:
                return super()._rollout(prompt_id, responses, seed)
            message = None
            try:
                result = super()._rollout(prompt_id, responses, seed)
            except oom as exc:
                message = str(exc)
            if message is not None:
                # outside the except scope: its traceback owns the failed rollout's tensors
                gc.collect()
                self._release_cuda()
                print(f"V8 CUDA out of memory on {prompt_id}; freed cache, retrying once ({message[:200]})",
                      flush=True)
                time.sleep(self._oom_retry_sleep)
                result = super()._rollout(prompt_id, responses, seed)
            self._release_cuda()
            return result

        def score_gradients(self, *args, **kwargs):
            self._v8_protect = True
            try:
                return super().score_gradients(*args, **kwargs)
            finally:
                self._v8_protect = False

        def evaluate(self, *args, **kwargs):
            self._v8_protect = True
            try:
                return super().evaluate(*args, **kwargs)
            finally:
                self._v8_protect = False
    Protected.__name__ = f"Protected{base_class.__name__}"
    return Protected


def runtime_seed(seed_loc, task):
    """Seed whose prefix receipt selects the archived engine (a fresh seed uses its source seed's)."""
    return int(seed_loc.get("runtime_source_seed", task["training_seed"]))


def attention_prefix(seed_loc):
    return seed_loc.get("prefix_path") or seed_loc.get("attention_prefix_path")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--matrix", required=True, type=Path)
    ap.add_argument("--task", required=True)
    ap.add_argument("--v8-root", default=None)
    args = ap.parse_args()
    install_stop_handlers()
    primary_rank = os.environ.get("RANK", "0") == "0"
    start_stop_watchdog(is_primary=primary_rank)
    run_child_task(lambda: run(args), is_primary=primary_rank, task_id=args.task)


def run(args):
    code_root = coderepo.activate()
    root = v8_root(args) / args.cohort
    m = read_json(args.matrix)
    task = next(t for t in m["tasks"] if (t.get("task_id") or t["run_id"]) == args.task)
    loc = read_json(root / "locator.json")
    seed_loc = loc["seeds"][str(task["training_seed"])]
    from switching import spec
    cfg = spec.resolve(args.cohort, role=task.get("role", "rebuttal_bank"))
    profile = cfg["runtime"]["profile"]
    if not seed_loc.get("implementation_matches_code_root"):
        raise SystemExit("V8_CODE_ROOT srgc_rebuttal digest differs from the seed run's implementation_sha256")
    verified = coderepo.source_runtime(code_root, profile, seed_loc['run_implementation_sha256'])
    if seed_loc.get('runtime_selection') != verified:
        raise SystemExit('Verified source runtime selection changed before child startup')
    if verified['mode'] == 'saved':
        coderepo.activate_source_runtime(code_root, seed_loc['plan_path'], runtime_seed(seed_loc, task), verified)

    import torch
    import torch.distributed as dist
    from srgc_rebuttal.distributed import initialize, primary
    from srgc_rebuttal.plan import load_plan, validate_inputs
    from srgc_rebuttal.srgc import Config
    from srgc_rebuttal import srgc
    from srgc_rebuttal.timing import torch_meter
    from switching.engine import make_engine_class
    from switching.runner import Context, execute_eval, execute_train
    from switching.cache import CacheStore

    plan = load_plan(Path(seed_loc["plan_path"]))
    world = plan["world_size"]
    rank, local_rank = initialize(world)
    install_diagnostics(root, args.task, rank)
    if rank == 0:
        print(f"V8 {_stamp()} ranks ready world={world} task={args.task} kind={task.get('kind')} "
              f"seed={task.get('training_seed')} (progress every {os.environ.get('V8_PROGRESS_SECONDS', '60')}s; "
              f"stack dumps: {root / '.multi-node' / 'stacks'}; kill -USR1 <pid> dumps now)", flush=True)
    with ExitStack() as stack:
        stack.push(lambda exc_type, *_: dist.destroy_process_group() if exc_type is None else None)
        if profile == "qwen":
            sys.path.insert(0, str(code_root / "scripts"))
            import srgc_qwen35
            stack.enter_context(srgc_qwen35.training_adapter())
            load_model = srgc_qwen35.load_model
        else:
            from srgc_rebuttal.existing_runtime import load_model
        prepare_runtime(profile)
        data = primary(lambda: load_bundle(seed_loc["input_bundle"]))
        validate_inputs(data)
        attention = os.environ.get("V8_ATTENTION", "eager")
        if attention_prefix(seed_loc):
            prefix_state = torch.load(attention_prefix(seed_loc), map_location="cpu", weights_only=False)
            # Every branch inherits its prefix's attention kernel; a failed import must not silently fall
            # back to eager for sdpa prefixes, so errors propagate.
            from scripts.srgc_step_checkpoints import saved_attention
            attention = saved_attention(prefix_state)
            del prefix_state
            if rank == 0:
                print(f"V8 {_stamp()} attention={attention} (from prefix checkpoint)", flush=True)
        ctx_holder = {}

        def on_meter_event(event):
            if rank == 0 and isinstance(event, dict) and event.get("phase"):
                seconds = event.get("seconds") or event.get("wall_seconds")
                print(f"PHASE {_stamp()} {event.get('phase')} {event.get('state', 'started')}"
                      + (f" {float(seconds):.0f}s" if isinstance(seconds, (int, float)) else ""), flush=True)
            ctx_holder["ctx"].on_meter_event(event)
        meter = torch_meter(on_meter_event)
        from peft import LoraConfig, get_peft_model
        from srgc_rebuttal import torch_backend
        torch.manual_seed(task["training_seed"])  # LoRA initialization protocol of run_experiment.py
        fallback = install_verifier_fallback(code_root, rank)
        install_progress(rank)
        model, tokenizer = load_model(plan["model"], plan["model_revision"], torch.device("cuda", local_rank),
                                      attention=attention)
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                                                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
        module, function = plan["verifier"].split(":", 1)
        Backend = protected_backend_class(torch_backend.TorchBackend)
        backend = Backend(
            model, tokenizer, data["records"], getattr(importlib.import_module(module), function),
            projection_dim=plan["projection_dim"], projection_seed=plan["projection_seed"],
            max_new_tokens=plan["max_new_tokens"], logprob_micro_batch=plan.get("logprob_micro_batch", 2),
            logit_chunk_tokens=plan.get("logit_chunk_tokens", 512), cost_meter=meter)
        config = Config(seed=task["training_seed"], objective=plan["objective"],
                        selection_interval=plan["selection_interval"], check_interval=plan["check_interval"],
                        first_check=plan["first_check"], scoring_prompts=plan["scoring_prompts_per_set"],
                        training_prompts=plan["training_prompts"], responses=plan["responses"],
                        projection_dim=plan["projection_dim"])
        identity = verifier_identity(plan, fallback)
        if len({json.dumps(x, sort_keys=True) for x in _gather(identity)}) != 1:
            raise RuntimeError("ranks score rollouts with different verifier packages/fallbacks")
        ctx = Context(srgc=srgc, engine_class=make_engine_class(srgc), backend=backend, data=data, config=config,
                      out_root=root, meter=meter, measurement_status="measured",
                      gpu_type=os.environ.get("V8_GPU_TYPE", cfg["runtime"]["gpu_type"]), is_primary=rank == 0,
                      primary=primary, broadcast=lambda v: primary(lambda: v),
                      cache_store=CacheStore(root / "caches"),
                      cache_context={"prompt_hash": plan.get("prompt_format") or seed_loc["plan_sha256"],
                                     "generation_spec": f"max_new_tokens={plan['max_new_tokens']},T=1,top_p=1",
                                     "verifier_hash": plan["verifier"]},
                      verifier_identity=identity)
        ctx_holder["ctx"] = ctx
        if rank == 0:
            print(f"V8 {_stamp()} model+LoRA ready; starting {task.get('kind')} run "
                  f"{task.get('run_id') or task.get('task_id')} — the first update includes the selection "
                  f"refresh and can take 15-30 min (OLMo) or 2-3 h (Qwen); PROGRESS lines follow", flush=True)
        return execute_eval(task, ctx) if task["kind"] == "evaluate" else execute_train(task, ctx)


if __name__ == "__main__":
    main()
