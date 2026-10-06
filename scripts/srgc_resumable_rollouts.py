"""Save every scored/evaluated rollout as it is produced so an interrupted block resumes, not restarts.

The frozen runner checkpoints the policy after every optimizer update, but a
selection refresh (about 130 prompts, eight 2048-token responses each) and the
final 300-prompt evaluation had no save point inside them: a CUDA OOM, an NCCL
failure or a pre-empted GPU anywhere in those one-to-three hours threw the
whole block away, and a run that failed repeatedly never left ``update 0/25``.

``ResumableRolloutMixin`` wraps ``TorchBackend``: while ``score_gradients`` or
``evaluate`` runs, each rank writes every finished rollout (token sequences,
rewards, prompt length) to ``<seed folder>/rollout-cache/<task>/<phase>-<seed>/``
on the shared run root; a restarted attempt loads those files instead of
generating again and only computes what is missing. The sampling seed of a
prompt depends on (seed, rank, prompt), and ranks shard prompts by position in
the same list, so a resumed rank regenerates exactly the prompts it would have
drawn. The directory is removed once the block completes. Between rollouts the
CUDA cache is released, and a single CUDA out-of-memory on one prompt is
retried once after freeing the cache before it is reported.
"""

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import numpy as np

CACHE_ROOT_ENV = "SRGC_ROLLOUT_CACHE"


def _key(prompt_id, responses):
    return f"{hashlib.sha256(prompt_id.encode()).hexdigest()[:24]}-{responses}.npz"


def _to_numpy(sequence):
    if hasattr(sequence, "detach"):
        return sequence.detach().cpu().numpy()
    return np.asarray(sequence)


def _oom_types():
    try:
        import torch
        return (torch.OutOfMemoryError,) if hasattr(torch, "OutOfMemoryError") else (torch.cuda.OutOfMemoryError,)
    except ImportError:
        return ()


class ResumableRolloutMixin:
    """Mix in before the backend: ``class B(ResumableRolloutMixin, TorchBackend)``."""

    rollout_cache_root = None  # Path; set by make_resumable or the SRGC_ROLLOUT_CACHE environment variable
    _resumable_dir = None
    resumed_rollouts = 0
    generated_rollouts = 0

    def _cache_root(self):
        root = self.rollout_cache_root or os.environ.get(CACHE_ROOT_ENV)
        return Path(root) if root else None

    def _release_cuda(self):
        device = getattr(self, "device", None)
        if device is not None and getattr(device, "type", "") == "cuda":
            import torch
            torch.cuda.empty_cache()

    def _load_rollout(self, path):
        with np.load(path, allow_pickle=False) as data:
            count_value, start_value = data["count"], data["start"]
            if (count_value.shape or start_value.shape or count_value.dtype.kind not in "iu"
                    or start_value.dtype.kind not in "iu"):
                raise ValueError("invalid cached rollout count or prompt length")
            count, start = int(count_value), int(start_value)
            if count < 1 or start < 1 or count > len(data.files):
                raise ValueError("invalid cached rollout dimensions")
            sequences = [data[f"seq{i}"] for i in range(count)]
            rewards = data["rewards"].astype(np.float64)
            if rewards.shape != (count,) or not np.isfinite(rewards).all() or not np.isin(rewards, [0., 1.]).all():
                raise ValueError("invalid cached rollout rewards")
            if any(s.ndim != 1 or s.dtype.kind not in "iu" or len(s) < start or (s < 0).any()
                   for s in sequences):
                raise ValueError("invalid cached token sequences")
        device = getattr(self, "device", None)
        if device is not None and getattr(device, "type", "") in {"cuda", "cpu"}:
            try:
                import torch
            except ImportError:
                return sequences, rewards, start
            target = device if isinstance(device, torch.device) else torch.device(getattr(device, "type", "cpu"))
            sequences = [torch.as_tensor(s, dtype=torch.long, device=target) for s in sequences]
        return sequences, rewards, start

    def _save_rollout(self, path, sequences, rewards, start):
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"seq{i}": _to_numpy(s).astype(np.int64) for i, s in enumerate(sequences)}
        handle, temporary = tempfile.mkstemp(prefix=".writing-", suffix=".npz", dir=path.parent)
        os.close(handle)
        try:
            with open(temporary, "wb") as output:
                np.savez(output, count=np.int64(len(sequences)), rewards=np.asarray(rewards, dtype=np.float64),
                         start=np.int64(start), **arrays)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _rollout(self, prompt_id, responses, seed):
        directory = self._resumable_dir
        if directory is None:
            return super()._rollout(prompt_id, responses, seed)
        path = directory / _key(prompt_id, responses)
        if path.exists():
            try:
                sequences, rewards, start = self._load_rollout(path)
                if len(sequences) != responses:
                    raise ValueError("cached rollout response count differs")
                self.resumed_rollouts += 1
                return sequences, rewards, start
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
                print(f"RESUME unreadable cached rollout {path.name}: {exc}; regenerating", flush=True)
        attempts = 0
        while True:
            try:
                sequences, rewards, start = super()._rollout(prompt_id, responses, seed)
                break
            except _oom_types() as exc:
                attempts += 1
                self._release_cuda()
                if attempts > 1:
                    raise
                print(f"RESUME CUDA out of memory on {prompt_id}; freed cache, retrying once ({exc})", flush=True)
                time.sleep(5)
        self._save_rollout(path, sequences, rewards, start)
        self.generated_rollouts += 1
        self._release_cuda()
        return sequences, rewards, start

    def _resumable(self, label, seed):
        root = self._cache_root()
        return None if root is None else root / f"{label}-{seed}"

    def score_gradients(self, ids, *, responses, group_size, seed):
        directory = self._resumable("score", seed)
        self._resumable_dir = directory
        before = self.resumed_rollouts
        try:
            result = super().score_gradients(ids, responses=responses, group_size=group_size, seed=seed)
        finally:
            self._resumable_dir = None
        if directory is not None:
            if self.resumed_rollouts > before:
                print(f"RESUME scoring block seed={seed}: {self.resumed_rollouts - before} rollout(s) reused from "
                      f"{directory} on this rank", flush=True)
            shutil.rmtree(directory, ignore_errors=True)
        return result

    def evaluate(self, ids, *, seed, responses=8):
        directory = self._resumable("eval", seed)
        self._resumable_dir = directory
        before = self.resumed_rollouts
        try:
            result = super().evaluate(ids, seed=seed, responses=responses)
        finally:
            self._resumable_dir = None
        if directory is not None:
            if self.resumed_rollouts > before:
                print(f"RESUME evaluation seed={seed}: {self.resumed_rollouts - before} rollout(s) reused on this rank",
                      flush=True)
            shutil.rmtree(directory, ignore_errors=True)
        return result


def make_resumable(backend_class, cache_root):
    """A subclass of ``backend_class`` that persists rollouts under ``cache_root``."""
    class ResumableBackend(ResumableRolloutMixin, backend_class):
        rollout_cache_root = Path(cache_root)
    ResumableBackend.__name__ = f"Resumable{backend_class.__name__}"
    return ResumableBackend


def install(cache_root):
    """Replace ``srgc_rebuttal.torch_backend.TorchBackend`` for runners that import it at call time."""
    from srgc_rebuttal import torch_backend
    original = torch_backend.TorchBackend
    if getattr(original, "rollout_cache_root", None) is not None:
        original.rollout_cache_root = Path(cache_root)
        return original
    torch_backend.TorchBackend = make_resumable(original, cache_root)
    return torch_backend.TorchBackend
