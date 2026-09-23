"""Validate and export SR-GC Pair results without launching GPU work."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import selector_pair_gpu as worker
import selector_pair_srgc as srgc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    worker.install_runtime()
    with worker.queue_lease(root / ".pair.lock", shared=True):
        p = worker.manifest(root)
        with srgc.activated(root, p, None, initialize=False):
            with worker.queue_lease(root / ".pair-barrier.lock"):
                with worker.completed_state_leases(root, ("development", "test"), require_complete=False):
                    srgc.report(root, p)


if __name__ == "__main__":
    main()
