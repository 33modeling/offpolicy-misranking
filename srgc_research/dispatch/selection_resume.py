"""Keep a finished scoring block until the policy moves past it (2026-10-31).

The pinned ``ResumableRolloutMixin`` deletes a scoring block's rollout cache as
soon as that block returns. A switch check scores 80 candidate/SR prompts and
then 50 validation prompts before update 25 trains and saves a checkpoint; a
restart anywhere after the first block (validation, the update itself) restarts
from the update-25 checkpoint and regenerates all 640 candidate responses. On
Gemma that block takes hours, so repeated attempts never reached update 26.

This operational patch is installed by the rank wrapper, outside the pinned
adapter/engine digests. It keeps each scoring block's per-prompt rollouts and
also saves the gathered projected gradients, bound to a fingerprint of the
trainable weights. A restart at the same weights reloads the gathered result
(and replays the cached rollouts so raw traces stay complete) instead of
generating again. Blocks whose fingerprint no longer matches are removed at the
next training update, after the earlier update's checkpoint exists. Sampling,
seeds, scores and the selected prompts are unchanged.
"""

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

RESULT = "gathered-gradients.npz"
PARTIAL = "partial-gradients.npz"
# Ranks shard a scoring block by prompt and wait for each other at its gather.
# A switch check gives each Gemma rank 20 candidate/SR prompts of very uneven
# length; the pinned 1800 s NCCL limit then aborts every rank while one is still
# generating. A rank that really hangs stops writing progress and the worker's
# own 1800 s stall check still ends the attempt; a dead rank ends it at once.
COLLECTIVE_TIMEOUT_SECONDS = 6 * 3600
POLICY = "policy.sha256"
MODELS = ("gemma4", "gemma4_text", "gemma4_it")


def policy_fingerprint(backend):
    value = hashlib.sha256()
    for name, parameter in backend.train_parameters:
        if hasattr(parameter, "detach"):
            parameter = parameter.detach().float().cpu().numpy()
        value.update(name.encode())
        value.update(np.ascontiguousarray(parameter, dtype=np.float32).tobytes())
    return value.hexdigest()


def agreed(backend, value):
    """Rank 0's decision on every rank, so all ranks take the same path."""
    if getattr(backend, "world", 1) == 1:
        return value
    import torch.distributed as dist
    box = [value]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def saved_policy(directory):
    try:
        return (directory / POLICY).read_text().strip()
    except OSError:
        return None


def write_atomic(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    os.close(handle)
    try:
        with open(temporary, "wb") as output:
            write(output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def release_memory(backend):
    import gc

    gc.collect()
    device = getattr(backend, "device", None)
    if device is not None and getattr(device, "type", "") == "cuda":
        import torch
        torch.cuda.empty_cache()


def load_result(directory, ids, fingerprint, responses, group_size, *, name=None, subset=False):
    path = directory / (name or RESULT)
    if not path.is_file() or saved_policy(directory) != fingerprint:
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            saved_ids = [str(i) for i in data["ids"]]
            vectors = np.asarray(data["vectors"], dtype=np.float64)
            if (str(data["policy"]) != fingerprint or int(data["responses"]) != responses
                    or int(data["group_size"]) != group_size):
                return None
    except (OSError, ValueError, KeyError, EOFError):
        return None
    if ((not set(saved_ids) <= set(ids) if subset else sorted(saved_ids) != sorted(ids))
            or len(set(saved_ids)) != len(saved_ids)
            or vectors.ndim != 2 or vectors.shape[0] != len(saved_ids) or not np.isfinite(vectors).all()):
        return None
    return {i: vectors[n].copy() for n, i in enumerate(saved_ids)}


def save_result(directory, result, fingerprint, responses, group_size, *, name=None):
    ids = list(result)
    vectors = np.stack([np.asarray(result[i], dtype=np.float64) for i in ids])
    write_atomic(directory / (name or RESULT), lambda output: np.savez(
        output, ids=np.asarray(ids), vectors=vectors, policy=np.asarray(fingerprint),
        responses=np.int64(responses), group_size=np.int64(group_size)))


def drop_stale(root, fingerprint):
    """Remove finished or abandoned scoring blocks of earlier weights."""
    if not root.is_dir():
        return []
    removed = []
    for directory in sorted(root.glob("score-*")):
        saved = saved_policy(directory)
        if directory.is_dir() and saved is not None and saved != fingerprint:
            shutil.rmtree(directory, ignore_errors=True)
            removed.append(directory.name)
    return removed


def patch_mixin(mixin):
    if getattr(mixin, "selection_resume_installed", False):
        return mixin

    def score_gradients(self, ids, *, responses, group_size, seed):
        directory = self._resumable("score", seed)
        base = super(mixin, self).score_gradients
        if directory is None:
            return base(ids, responses=responses, group_size=group_size, seed=seed)
        fingerprint = policy_fingerprint(self)
        hit = None
        if self.rank == 0:
            hit = load_result(directory, ids, fingerprint, responses, group_size) is not None
            if not hit:
                if saved_policy(directory) not in (None, fingerprint):
                    # Rollouts drawn by other weights are never reused.
                    shutil.rmtree(directory, ignore_errors=True)
                write_atomic(directory / POLICY, lambda output: output.write(fingerprint.encode()))
        if agreed(self, hit):
            result = load_result(directory, ids, fingerprint, responses, group_size)
            if result is None:
                raise ValueError(f"saved scoring block is unreadable on rank {self.rank}: {directory}")
            # Replay this rank's cached rollouts so the raw trace export still
            # receives every scoring group; a missing file is regenerated.
            self._resumable_dir = directory
            try:
                for prompt_id in list(ids)[self.rank::self.world]:
                    self._rollout(prompt_id, responses, seed)
            finally:
                self._resumable_dir = None
            print(f"RESUME scoring block seed={seed}: reused {len(result)} saved gradients from {directory}",
                  flush=True)
            return result
        # Score one prompt per rank at a time and keep every finished slice, so
        # an attempt that runs out of memory part way resumes after it instead
        # of recomputing the whole block. Each rank still scores ids[rank::world]
        # in the original order; sampling seeds depend on (seed, rank, prompt).
        partial = None
        if self.rank == 0:
            partial = load_result(directory, ids, fingerprint, responses, group_size, name=PARTIAL, subset=True)
        done = agreed(self, sorted(partial) if partial else [])
        result = dict(partial or {})
        world = max(1, getattr(self, "world", 1))
        before = self.resumed_rollouts
        self._resumable_dir = directory
        try:
            for start in range(0, len(ids), world):
                part = list(ids)[start:start + world]
                if set(part) <= set(done):
                    # Replay this rank's cached rollout so raw traces stay complete.
                    if self.rank < len(part):
                        self._rollout(part[self.rank], responses, seed)
                    continue
                result.update(base(part, responses=responses, group_size=group_size, seed=seed))
                if self.rank == 0:
                    save_result(directory, result, fingerprint, responses, group_size, name=PARTIAL)
                release_memory(self)
        finally:
            self._resumable_dir = None
        if done:
            print(f"RESUME scoring block seed={seed}: {len(done)} saved prompt gradient(s) reused", flush=True)
        if self.resumed_rollouts > before:
            print(f"RESUME scoring block seed={seed}: {self.resumed_rollouts - before} rollout(s) reused from "
                  f"{directory} on this rank", flush=True)
        if self.rank == 0:
            save_result(directory, result, fingerprint, responses, group_size)
        return result

    def train(self, *args, **kwargs):
        root = self._cache_root()
        if root is not None and self.rank == 0:
            removed = drop_stale(root, policy_fingerprint(self))
            if removed:
                print(f"RESUME removed finished scoring blocks: {', '.join(removed)}", flush=True)
        return super(mixin, self).train(*args, **kwargs)

    mixin.score_gradients = score_gradients
    mixin.train = train
    mixin.selection_resume_installed = True
    return mixin


def extend_collective_timeout(seconds=COLLECTIVE_TIMEOUT_SECONDS):
    """Raise the pinned NCCL wait limit for this rank process only."""
    from datetime import timedelta

    import torch.distributed as dist

    original = dist.init_process_group
    if getattr(original, "selection_resume_timeout", None):
        return original

    def init_process_group(*args, **kwargs):
        timeout = kwargs.get("timeout")
        if timeout is None or timeout < timedelta(seconds=seconds):
            kwargs["timeout"] = timedelta(seconds=seconds)
        return original(*args, **kwargs)

    init_process_group.selection_resume_timeout = seconds
    dist.init_process_group = init_process_group
    return init_process_group


def expandable_segments(environment=None):
    """Keep the CUDA allocator's expandable segments on in every Gemma rank.

    The launch script sets it only when PYTORCH_CUDA_ALLOC_CONF is unset; an
    inherited value without it leaves long scoring blocks to fragmentation.
    Allocation behaviour only; results are unchanged."""
    environment = os.environ if environment is None else environment
    value = environment.get("PYTORCH_CUDA_ALLOC_CONF", "")
    options = [item for item in value.split(",") if item.strip()]
    if not any(item.replace(" ", "").startswith("expandable_segments:") for item in options):
        options.append("expandable_segments:True")
    environment["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(options)
    return environment["PYTORCH_CUDA_ALLOC_CONF"]


def install():
    """Patch the rollout mixin under both of its import names."""
    from scripts import srgc_resumable_rollouts as module
    sys.modules.setdefault("srgc_resumable_rollouts", module)
    patched = []
    for name in ("scripts.srgc_resumable_rollouts", "srgc_resumable_rollouts"):
        loaded = sys.modules.get(name)
        if loaded is not None and loaded.ResumableRolloutMixin not in patched:
            patched.append(patch_mixin(loaded.ResumableRolloutMixin))
    return patched
