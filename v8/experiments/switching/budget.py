"""Budget endpoints (plan Code §7.4-7.6): selected from metadata only, never from rewards."""

from __future__ import annotations


def select_endpoint(snapshots, budget):
    """snapshots: [{completed_update, cumulative_cost, snapshot_hash, eligible?}] in any order.

    Selects the latest eligible snapshot whose cumulative algorithm cost at availability
    is <= budget, using metadata only. Costs are nondecreasing in completed_update, so the
    choice is DETERMINED only if every snapshot after the last known within-budget one and
    before the first known over-budget one has a known cost. Unknown cost is not over budget:
      selected      determined latest within-budget snapshot
      undetermined  an unknown cost could change the choice (or decide feasibility)
      infeasible_budget  known costs exceed the budget from the first snapshot on
    """
    if budget is None:
        return {"status": "budget_not_frozen", "endpoint": None}
    rows = sorted((s for s in snapshots if s.get("eligible", True)), key=lambda s: s["completed_update"])
    known = [s for s in rows if s.get("cumulative_cost") is not None]
    for a, b in zip(known, known[1:]):
        if b["cumulative_cost"] < a["cumulative_cost"]:
            raise ValueError("cumulative cost decreased between snapshots (event order violated)")
    unknown = [s["completed_update"] for s in rows if s.get("cumulative_cost") is None]
    within = [s for s in known if s["cumulative_cost"] <= budget]
    beyond = [s for s in known if s["cumulative_cost"] > budget]
    first_beyond = beyond[0]["completed_update"] if beyond else float("inf")
    last_within = within[-1]["completed_update"] if within else float("-inf")
    blocking = [u for u in unknown if last_within < u < first_beyond]
    base = {"unknown_cost_updates": unknown, "budget": budget, "overshoot": beyond[0] if beyond else None}
    if blocking:
        return {**base, "status": "undetermined", "endpoint": None, "blocking_unknown_updates": blocking,
                "last_known_within": within[-1] if within else None}
    if not within:
        if not rows:
            return {**base, "status": "undetermined", "endpoint": None, "reason": "no snapshots"}
        return {**base, "status": "infeasible_budget", "endpoint": None}
    chosen = within[-1]
    return {**base, "status": "selected", "endpoint": chosen, "slack": budget - chosen["cumulative_cost"]}


def common_target_cost(points, target):
    """points: [(completed_update, cumulative_cost, reward)] from evaluated checkpoints.

    reached: interval (cost at last evaluation below target, cost at first at/above);
    not_reached: no evaluated checkpoint reached it (never treated as zero cost).
    No interpolation or smoothing.
    """
    pts = sorted(points)
    if any(r is None for _, _, r in pts):
        return {"status": "unknown", "interval": None}
    previous = None
    for u, c, r in pts:
        if r >= target:
            return {"status": "reached", "first_update": u, "interval": [previous[1] if previous else None, c]}
        previous = (u, c)
    return {"status": "not_reached", "interval": None, "last_cost": pts[-1][1] if pts else None}


def should_continue(step, *, horizon, hard_cap, budget_cap, cumulative_cost, next_update_cost):
    """Training stop rule. Cheaper arms may pass the matched-update horizon inside the budget."""
    if step < horizon:
        return True
    if budget_cap is None or hard_cap is None or step >= hard_cap:
        return False
    if cumulative_cost is None or next_update_cost is None:
        return False
    return cumulative_cost + next_update_cost <= budget_cap
