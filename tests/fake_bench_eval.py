"""Stand-in for ``src/benchmark_eval.py evaluate`` in the local fake cluster:
writes valid shard files (bindings, rewards, done records) without a model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import benchmark_eval as be  # noqa: E402
import evidence_downstream as ed  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--sets", nargs="*")
    args = p.parse_args(argv)
    if args.command != "evaluate":
        raise SystemExit("fake bench eval only implements evaluate")
    out = args.out.resolve()
    frozen = be.benchmark_contract(out)
    rng = random.Random(hash((args.arm, args.shard)) & 0xFFFF)
    for name in args.sets or frozen["sets"]:
        binding, _, indices = be.binding_for(out, args.arm, name, args.shard)
        target = out / args.arm / "benchmark" / name
        target.mkdir(parents=True, exist_ok=True)
        completed = target / f"shard-{args.shard}.done.json"
        if completed.exists():
            print(f"[reuse] {args.arm} {name} shard {args.shard}")
            continue
        path = target / f"shard-{args.shard}.jsonl"
        rate = 0.35 if args.arm == "before" else 0.45
        rows = [{"prompt_idx": i, "rollout_idx": j, "reward": 1.0 if rng.random() < rate else 0.0}
                for i in indices for j in range(frozen["eval_k"])]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        ed.bind(target / f"shard-{args.shard}.contract.json", binding)
        be.atomic_done(completed, binding, path, 12.5 * len(indices), len(indices))
        print(f"[done] {args.arm} {name} shard {args.shard}: {len(indices)} prompts (fake)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
