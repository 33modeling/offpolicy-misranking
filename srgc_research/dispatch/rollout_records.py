"""Persist successful scoring/training groups independently of resume caches."""

import gzip
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from srgc_rebuttal.distributed import primary

SCHEMA = "srgc-rollout-trace-v1"


@contextmanager
def capture(backend):
    """Retain returned tensors until export, without an extra generation call."""
    groups = []
    if not hasattr(backend, "_rollout"):
        yield groups
        return
    original_rollout = backend._rollout
    original_train = backend.train
    original_score = backend.score_gradients
    phase = [None]

    def rollout(prompt_id, responses, seed):
        result = original_rollout(prompt_id, responses, seed)
        if phase[0] is not None:
            sequences, rewards, start = result
            groups.append((phase[0], prompt_id, sequences, rewards, start))
        return result

    def within(label, function, *args, **kwargs):
        previous, phase[0] = phase[0], label
        try:
            return function(*args, **kwargs)
        finally:
            phase[0] = previous

    with patch.object(backend, "_rollout", rollout), \
            patch.object(backend, "train", lambda *a, **kw: within("training", original_train, *a, **kw)), \
            patch.object(backend, "score_gradients", lambda *a, **kw: within("selection", original_score, *a, **kw)):
        yield groups


def as_tokens(sequence, start):
    sequence = sequence[start:]
    if hasattr(sequence, "detach"):
        sequence = sequence.detach().cpu()
    values = sequence.tolist()
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("rollout must contain nonempty response-only integer tokens")
    return values


def publish(path, document):
    """Atomic, deterministic gzip; a conflicting completed record is an error."""
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    path = Path(path)
    if path.exists():
        if gzip.decompress(path.read_bytes()) != payload:
            raise ValueError(f"saved rollout record differs from this update: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(gzip.compress(payload, mtime=0))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save(engine, local, folder, task, plan, data, protocol, record):
    if not hasattr(engine.backend, "_rollout"):
        return  # Scripted CPU backends do not generate raw responses.
    from srgc_rebuttal.cost_ledger import PhaseLedger
    from srgc_rebuttal.timing import torch_meter

    directory = Path(folder) / "reward-curves" / task
    meter = torch_meter(lambda event: PhaseLedger(directory / "rollout-costs").record(event))
    with meter.phase("preparation", engine.step, engine.backend.gpu_count):
        candidates, comparisons = set(record.get("on_ids", [])), set(record.get("sr_ids", []))
        reference = set(record.get("validation_ids", []))
        selected = set(record.get("train_ids", []))
        local_rows = {}
        for phase, prompt, sequences, rewards, start in local:
            identity = (phase, prompt)
            if identity in local_rows:
                raise ValueError("duplicate successful rollout within one update")
            if (len(sequences) != engine.config.responses or len(sequences) != len(rewards) or
                    any(float(r) not in (0., 1.) for r in rewards)):
                raise ValueError("invalid rollout rewards during trace export")
            scope = [name for name, ids in (("on_candidates", candidates), ("sr_comparison", comparisons),
                                           ("reference", reference)) if prompt in ids]
            current_phase = "diagnostic" if phase == "selection" and prompt in reference else phase
            row = {"arm": task, "phase": current_phase, "update": engine.step if phase == "training" else record["checkpoint"],
                   "prompt_id": prompt, "draw": 0, "scoring_scope": scope,
                   "selected_for_training": prompt in selected, "selector": record.get("selector"),
                   "responses": [{"reward": float(r), "token_ids": as_tokens(sequence, start)}
                                 for sequence, r in zip(sequences, rewards, strict=True)]}
            cached = data.get("cached_rewards", {}).get(prompt)
            if cached is not None and len(cached):
                row["cached_success_rate"] = sum(cached) / len(cached)
            local_rows[identity] = row
        rows = engine.backend._gather(local_rows)
        actual = {row["prompt_id"] for row in rows.values() if row["phase"] == "training"}
        if actual != selected:
            raise ValueError("saved training groups do not match the applied training batch")
        document = {"schema": SCHEMA, "dataset": plan.get("dataset"), "model": plan.get("model", "unrecorded"),
                    "seed": engine.config.seed, "arm": task, "step": engine.step,
                    "shared_prefix_updates": plan["shared_prefix_updates"], "total_updates": plan["total_updates"],
                    "switched_at": engine.switched_at,
                    # A retry can have different timing receipts for identical
                    # generated responses. Timing stays in the cost ledgers.
                    "history_record": {k: v for k, v in record.items()
                                       if not k.endswith(("_gpu_seconds", "_wall_seconds")) and k != 'timing_receipt_ids'},
                    "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    "identity": {k: protocol[k] for k in ("plan_sha256", "input_sha256", "implementation_sha256")},
                    "groups": [rows[key] for key in sorted(rows)]}
        if hasattr(engine, "recording_extension_sha256"):
            document["recording_extension_sha256"] = engine.recording_extension_sha256
        def write():
            from .reward_curves import recording_extension

            path = directory / "rollouts" / f"step-{engine.step}.json.gz"
            if path.exists():
                existing = read(path)
                recording_extension(directory, protocol, existing)
                # An interrupted update may already have published its identical
                # responses using the previous recorder. Keep those original bytes
                # and attribution; publish still rejects every scientific change.
                document.pop("recording_extension_sha256", None)
                if "recording_extension_sha256" in existing:
                    document["recording_extension_sha256"] = existing["recording_extension_sha256"]
            publish(path, document)

        primary(write)


def read(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def compact_groups(document):
    """Exports retain every reward/length; full token IDs stay in durable gzip."""
    rows = []
    for group in document["groups"]:
        from srgc_research.rollout_distribution import group_report
        group_report(group)  # Reject corrupt/empty/binary-invalid groups first.
        if any("token_ids" not in r for r in group["responses"]):
            raise ValueError("durable rollout record lacks response tokens")
        rows.append({**group, "responses": [
            {"reward": response["reward"], "completion_tokens": len(response["token_ids"])}
            for response in group["responses"]]})
    return rows
