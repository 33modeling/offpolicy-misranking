"""Read recorded intermediate rewards without importing a GPU runtime."""

import json

from srgc_rebuttal.plan import digest
from srgc_rebuttal.srgc import Engine

from .reward_curves import SCHEMA, INTERVAL, RESPONSES, finite_reward, fingerprint, training_points, windows


def collect(folder, arm, plan, expected, ids, read):
    directory = folder / "reward-curves" / arm
    points, sources, held_out, costs = [], [], {}, []
    progress_path = folder / f"{arm}-progress.json"
    progress = read(progress_path)
    if progress is not None:
        if not isinstance(progress, dict):
            raise ValueError("training progress must be an object")
        if progress.get("seed") != expected["seed"] or progress.get("arm") != arm or progress.get("sampling_protocol") != Engine.SAMPLING_PROTOCOL:
            raise ValueError("training progress seed/arm/sampling protocol mismatch")
        step = progress["step"]
        if type(step) is not int or not plan["shared_prefix_updates"] <= step <= plan["total_updates"]:
            raise ValueError("training progress step is outside the frozen plan")
        marker = read(folder / "run.json")
        if not marker or any(marker.get(k) != v for k, v in expected.items()):
            raise ValueError("training progress is not bound to the current run identity")
        points = training_points(progress["history"], start=plan["shared_prefix_updates"], stop=step)
        sources.append(str(progress_path))

    def protocol_for(name):
        path = folder / "reward-curves" / name / "protocol.json"
        protocol = read(path)
        if protocol is None:
            return None
        if not isinstance(protocol, dict):
            raise ValueError("measurement protocol must be an object")
        if (any(protocol.get(k) != v for k, v in expected.items()) or
                protocol.get("schema") != SCHEMA or protocol.get("arm") != name or
                protocol.get("interval_updates") != INTERVAL or
                protocol.get("responses_per_prompt") != RESPONSES or
                protocol.get("stream") != "reporting-evaluation" or
                protocol.get("training_batch_reward_is_held_out") is not False or
                protocol.get("question_count") != len(ids) or
                protocol.get("evaluation_ids_sha256") != fingerprint(ids) or
                not isinstance(protocol.get("extension_sha256"), str) or len(protocol["extension_sha256"]) != 64):
            raise ValueError("intermediate reward protocol differs from this experiment")
        return protocol

    protocol = protocol_for(arm)
    training_path = directory / "training.json"
    training = read(training_path)
    if training is not None:
        if protocol is None or any(training.get(k) != v for k, v in protocol.items()):
            raise ValueError("scalar training rewards lack their measurement protocol")
        proposed = training["points"]
        step = training["step"]
        if type(step) is not int or not plan["shared_prefix_updates"] <= step <= plan["total_updates"]:
            raise ValueError("scalar training reward step is outside the plan")
        if not isinstance(proposed, list) or any(not isinstance(p, dict) for p in proposed) or [p.get("step") for p in proposed] != list(range(plan["shared_prefix_updates"] + 1, step + 1)):
            raise ValueError("scalar training rewards have missing or duplicate updates")
        for point in proposed:
            reward = finite_reward(point["reward"])
            if point.get("kind") != "selected_training_batch" or point.get("reward_percent") != 100 * reward:
                raise ValueError("training reward kind/units differ from the recording protocol")
        proposed = [{k: point.get(k) for k in ("step", "reward", "reward_percent", "selector", "kind")} for point in proposed]
        overlap = {p["step"]: p for p in proposed}
        if any(overlap.get(p["step"]) != p for p in points if p["step"] in overlap):
            raise ValueError("progress history and scalar training rewards disagree")
        if len(proposed) >= len(points):
            points = proposed
            sources.append(str(training_path))

    prefix_points = []
    prefix_protocol = protocol_for("prefix")
    prefix_path = folder / "reward-curves/prefix/training.json"
    prefix_training = read(prefix_path)
    if prefix_training is not None:
        if prefix_protocol is None or any(prefix_training.get(k) != v for k, v in prefix_protocol.items()):
            raise ValueError("shared-prefix scalar rewards lack their protocol")
        step = prefix_training["step"]
        proposed = prefix_training["points"]
        if (type(step) is not int or not 0 <= step <= plan["shared_prefix_updates"] or
                not isinstance(proposed, list) or any(not isinstance(p, dict) for p in proposed) or
                [p.get("step") for p in proposed] != list(range(1, step + 1))):
            raise ValueError("shared-prefix training rewards have missing or duplicate updates")
        for point in proposed:
            reward = finite_reward(point["reward"])
            if (point.get("kind") != "selected_training_batch" or point.get("reward_percent") != 100 * reward or
                    point.get("selector") != "on_policy"):
                raise ValueError("shared-prefix training reward kind/units/selector differ")
            prefix_points.append({k: point[k] for k in ("step", "reward", "reward_percent", "selector", "kind")})
        sources.append(str(prefix_path))
    all_points = [{**p, "shared_prefix": True} for p in prefix_points] + points

    for name in ("prefix", arm):
        bound = protocol_for(name)
        if bound is None:
            continue
        for step in range(INTERVAL, plan["total_updates"], INTERVAL):
            if (name == "prefix") != (step == plan["shared_prefix_updates"]):
                continue
            path = folder / "reward-curves" / name / f"step-{step}.json"
            row = read(path)
            if row is None:
                continue
            from .reward_curves import validate_evaluation

            if (row.get("schema") != SCHEMA or row.get("protocol_sha256") != fingerprint(bound) or
                    row.get("arm") != name or row.get("seed") != expected["seed"] or row.get("step") != step or
                    row.get("kind") != "fixed_held_out_evaluation" or row.get("responses_per_prompt") != RESPONSES or
                    row.get("question_count") != len(ids)):
                raise ValueError("held-out curve point differs from its bound protocol")
            validate_evaluation(row, ids)
            checkpoint = path.with_suffix(".pt")
            if not checkpoint.is_file() or row.get("checkpoint_sha256") != digest(checkpoint):
                raise ValueError("held-out curve checkpoint is missing or changed")
            held_out[step] = {"step": step, "reward": row["reward"], "reward_percent": 100 * row["reward"],
                              "kind": "fixed_held_out_evaluation", "source": str(path),
                              "shared_prefix": name == "prefix", "responses_per_prompt": RESPONSES,
                              "question_count": len(ids), "extension_sha256": bound["extension_sha256"]}
            costs.append({"step": step, "arm": name, "scope": row.get("cost_scope"), "costs": row.get("cost_receipts")})
    return {"seed": expected["seed"], "arm": arm, "training_rewards": all_points,
            "shared_prefix_training_recorded_updates": len(prefix_points),
            "training_windows_25": windows(all_points), "training_sources": sources,
            "held_out_rewards": list(held_out.values()), "diagnostic_costs": costs}
