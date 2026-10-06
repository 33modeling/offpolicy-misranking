#!/usr/bin/env python3
"""Isolate rank-local compiler caches without changing the frozen Qwen runtime."""

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]


@contextmanager
def isolated_storage():
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    storage = importlib.import_module("srgc_qwen35_storage")
    original = storage.setup_storage

    def setup_storage(root, environment, *, scan_tree=True):
        result = original(root, environment, scan_tree=scan_tree)
        try:
            rank = int(environment["LOCAL_RANK"])
            world = int(environment["WORLD_SIZE"])
            local_world = int(environment["LOCAL_WORLD_SIZE"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Qwen rank caches require a four-rank torchrun environment") from exc
        if world != 4 or local_world != 4 or not 0 <= rank < 4:
            raise ValueError("Qwen rank caches require LOCAL_RANK 0..3 and four local/world ranks")
        group, _ = result
        node_cache = Path(environment["TRITON_CACHE_DIR"]).parent
        rank_cache = storage.inside(node_cache / "rank-runtime-v1" / f"rank-{rank}", group)
        paths = {name: storage.inside(rank_cache / directory, group) for name, directory in (
            ("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
            ("TORCH_EXTENSIONS_DIR", "extensions"), ("CUDA_CACHE_PATH", "cuda"), ("TMPDIR", "tmp"))}
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        environment.update({name: str(path) for name, path in paths.items()})
        print(f"[qwen-rank-cache] rank={rank} triton={environment['TRITON_CACHE_DIR']}", flush=True)
        return result

    with patch.object(storage, "setup_storage", setup_storage):
        yield


def main():
    with isolated_storage():
        from srgc_qwen35_rank import main as rank_main
        return rank_main()


if __name__ == "__main__":
    main()
