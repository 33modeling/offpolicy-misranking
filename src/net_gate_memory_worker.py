"""Checkpoint decoder activations in eval mode; drain independent scoring shards."""

from __future__ import annotations

import argparse
import contextlib
import functools
import inspect
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import selection_gate as core
import selection_gate_gpu as base

HERE = Path(__file__).resolve()


def checkpoint_decoder_layers(model):
    import torch
    from torch.utils.checkpoint import checkpoint

    causal = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = getattr(causal, "model", None)
    layers = getattr(decoder, "layers", None)
    if layers is None or not len(layers):
        raise ValueError("memory-safe scoring requires explicit transformer decoder layers")
    if any(module.training for module in model.modules()):
        raise ValueError("scoring must retain evaluation mode, including attention and LoRA dropout")
    if not getattr(decoder, "_net_gate_cache_guarded", False):
        decoder_forward = decoder.forward
        signature = inspect.signature(decoder_forward)

        @functools.wraps(decoder_forward)
        def uncached_forward(*args, **kwargs):
            if not torch.is_grad_enabled():
                return decoder_forward(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            if bound.arguments.get("past_key_values") is not None:
                raise ValueError("checkpointed scoring requires full sequences with past_key_values=None")
            # Disable cache creation before masks/layers capture a mutable cache.
            # Setting use_cache=False only at a layer does not stop cache.update.
            bound.arguments["use_cache"] = False
            # Transformers' config-default decorator reads keyword arguments.
            call_kwargs = {}
            for name, value in bound.arguments.items():
                if signature.parameters[name].kind == inspect.Parameter.VAR_KEYWORD:
                    call_kwargs.update(value)
                else:
                    call_kwargs[name] = value
            return decoder_forward(**call_kwargs)

        decoder.forward = uncached_forward
        decoder._net_gate_cache_guarded = True
    for layer in layers:
        if getattr(layer, "_net_gate_checkpointed", False):
            continue
        forward = layer.forward

        @functools.wraps(forward)
        def recompute(*args, _forward=forward, **kwargs):
            if not torch.is_grad_enabled():
                return _forward(*args, **kwargs)
            # Non-reentrant checkpointing supports autograd.grad and frozen inputs.
            return checkpoint(_forward, *args, use_reentrant=False, **kwargs)

        layer.forward = recompute
        layer._net_gate_checkpointed = True
    return len(layers)


def install_backend(low):
    load = low.backend.load_current

    def memory_load(*args, **kwargs):
        model, tokenizer = load(*args, **kwargs)
        count = checkpoint_decoder_layers(model)
        print(f"[memory] {count} decoder layers checkpointed; eval/dropout modes unchanged", flush=True)
        return model, tokenizer

    low.backend.load_current = memory_load
    exact, logps = low.backend.exact_directional, low.sequence_logprobs_batch

    def explain(fn, phase, model, rows, *args, **kwargs):
        try:
            return fn(model, rows, *args, **kwargs)
        except Exception:
            print(f"[memory-context] phase={phase} tokens={[int(r['input_ids'].numel()) for r in rows]} "
                  f"response_starts={[r['resp_start'] for r in rows]}", file=sys.stderr, flush=True)
            raise

    low.backend.exact_directional = functools.partial(explain, exact, "exact-gradient")
    low.sequence_logprobs_batch = functools.partial(explain, logps, "forward-logps")


def worker(out, stage, shard):
    import torch
    import low_order_experiment as low

    state_path = out / f"memory-{stage}-{shard}.json"
    binding = {"experiment_sha256": base.digest(out / "experiment.json"), "worker_sha256": base.digest(HERE)}
    c = low.verify(out, inputs=False)
    if c["derivative"] != "autograd":
        raise ValueError("memory worker only accepts an explicit autograd scoring contract")
    if state_path.exists():
        previous = core.read(state_path)
        prompt = previous.get("prompt")
        pending = prompt is None or not (out / "scores" / f"p{prompt}.json").exists()
        if previous.get("binding") == binding and previous.get("oom") and pending:
            print(f"[abort] unchanged checkpointed worker already OOM at prompt={prompt}; "
                  f"no automatic GPU retry; see {state_path}", file=sys.stderr)
            return 2
    install_backend(low)
    try:
        (low.validation_worker if stage == "validation" else low.score_worker)(out, shard)
    except Exception as exc:
        progress_path = out / f"progress-{shard}.json"
        progress = core.read(progress_path) if progress_path.exists() else {}
        core.atomic_json(state_path, {"binding": binding, "exit_code": 2,
            "oom": isinstance(exc, torch.OutOfMemoryError) or "CUDA out of memory" in str(exc),
            "error": str(exc), "prompt": progress.get("prompt"), "time": time.time()})
        traceback.print_exc()
        print(f"[abort] {exc}", file=sys.stderr, flush=True)
        return 2
    core.atomic_json(state_path, {"binding": binding, "exit_code": 0, "oom": False, "time": time.time()})
    return 0


def drain_workers(commands, logs, env):
    """Keep healthy siblings running; the parent's metered deadline still applies."""
    processes = []
    try:
        with contextlib.ExitStack() as stack:
            for command, visible, log in zip(*commands, logs, strict=True):
                handle = stack.enter_context(log.open("a"))
                # Same process group: the original meter can terminate the whole tree.
                processes.append(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                    env={**env, "CUDA_VISIBLE_DEVICES": visible}))
            reported = set()
            while True:
                codes = [process.poll() for process in processes]
                for shard, code in enumerate(codes):
                    if code is not None and shard not in reported:
                        print(f"[shard] {shard} exit={code}; unfinished siblings continue", flush=True)
                        reported.add(shard)
                if all(code is not None for code in codes):
                    return codes
                time.sleep(.2)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "supervise"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("validation", "score"), required=True)
    parser.add_argument("--shard", type=int, choices=range(4))
    parser.add_argument("--logs", type=Path)
    args = parser.parse_args()
    if os.environ.get("OM_NODE_LOCK_HELD") != "1":
        parser.error("an admitted node is required")
    if args.command == "worker":
        if args.shard is None:
            parser.error("worker requires --shard")
        return worker(args.out, args.stage, args.shard)
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 4 or len(set(devices)) != 4 or not all(devices) or args.logs is None:
        parser.error("supervision requires four distinct GPUs and --logs")
    args.logs.mkdir(parents=True, exist_ok=True)
    commands = [[sys.executable, str(HERE), "worker", "--out", str(args.out),
                 "--stage", args.stage, "--shard", str(i)] for i in range(4)]
    logs = [args.logs / f"autograd-{args.stage}-shard-{i}.log" for i in range(4)]
    codes = drain_workers((commands, devices), logs, os.environ)
    core.atomic_json(args.logs / f"autograd-{args.stage}-workers.json", {
        "codes": codes, "logs": [str(path) for path in logs], "time": time.time(),
        "worker_sha256": base.digest(HERE), "experiment_sha256": base.digest(args.out / "experiment.json")})
    if any(codes):
        print(f"[abort] {args.stage} shard exits={codes}; see {args.logs}/autograd-{args.stage}-shard-*.log",
              file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    raise SystemExit(main())
