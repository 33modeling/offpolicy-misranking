"""Transition policies (when to change selector), separate from selectors and caches.

Selectors (what data an update uses): on_policy, cached_sr, uniform.
Transition policies: never, fixed, sr_gc, reward_plateau, sampled_time,
and the optional bidirectional (E8). Cache policies live in cache.py.

Boundary convention: a decision at completed_update t takes effect for update t+1.
"""

from __future__ import annotations

import math

from .rules import SRGCController, canonical_rule


class TransitionPolicy:
    policy_id = "base"
    uses_signal = False

    def needs_signal(self, step: int) -> bool:
        return False

    def before_update(self, step: int, stats) -> dict | None:
        """Decide using only information available at boundary `step` (no evaluation access)."""
        return None

    def on_signal(self, step: int, value, *, valid=True) -> dict:
        raise RuntimeError(f"{self.policy_id} does not consume SR-GC signals")

    def state_dict(self) -> dict:
        return {"policy_id": self.policy_id}

    def load_state_dict(self, state: dict) -> None:
        if state.get("policy_id") != self.policy_id:
            raise ValueError("transition policy changed between save and resume")


class NeverPolicy(TransitionPolicy):
    policy_id = "never"


class FixedPolicy(TransitionPolicy):
    policy_id = "fixed"

    def __init__(self, tau: int):
        if tau < 0:
            raise ValueError("fixed tau must be a completed-update count")
        self.tau = int(tau)

    def before_update(self, step, stats):
        if step == self.tau:
            return {"completed_update": step, "action": "switch", "reason": f"fixed_tau_{self.tau}",
                    "rule_id": f"fixed_{self.tau}"}
        return None

    def state_dict(self):
        return {"policy_id": self.policy_id, "tau": self.tau}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if state["tau"] != self.tau:
            raise ValueError("fixed tau changed between save and resume")


class SampledTimePolicy(FixedPolicy):
    """E9: tau drawn ONCE before the run from a frozen development distribution.

    `tau=None` encodes the no-switch mass. Never re-drawn on resume; a tau past the
    budget means no switch inside the budget (it is not pulled earlier).
    """
    policy_id = "sampled_time"

    def __init__(self, tau, distribution_hash: str, schedule_seed: int):
        self.tau = None if tau is None else int(tau)
        self.distribution_hash, self.schedule_seed = distribution_hash, int(schedule_seed)

    def before_update(self, step, stats):
        if self.tau is not None and step == self.tau:
            return {"completed_update": step, "action": "switch", "reason": "sampled_time",
                    "rule_id": "sampled_time"}
        return None

    def state_dict(self):
        return {"policy_id": self.policy_id, "tau": self.tau, "distribution_hash": self.distribution_hash,
                "schedule_seed": self.schedule_seed}

    def load_state_dict(self, state):
        TransitionPolicy.load_state_dict(self, state)
        if (state["tau"], state["distribution_hash"], state["schedule_seed"]) != (
                self.tau, self.distribution_hash, self.schedule_seed):
            raise ValueError("sampled schedule changed between save and resume")


class SRGCPolicy(TransitionPolicy):
    policy_id = "sr_gc"
    uses_signal = True

    def __init__(self, rule="current", interval=25, first_check=25):
        self.controller = SRGCController(canonical_rule(rule), interval, first_check)

    def needs_signal(self, step):
        return not self.controller.transitioned and self.controller.scheduled(step)

    def on_signal(self, step, value, *, valid=True):
        return self.controller.observe(step, value, valid=valid)

    def state_dict(self):
        return {"policy_id": self.policy_id, "controller": self.controller.state_dict()}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        saved = SRGCController.from_state(state["controller"])
        if (saved.rule, saved.interval, saved.first_check) != (
                self.controller.rule, self.controller.interval, self.controller.first_check):
            raise ValueError("SR-GC rule/cadence changed between save and resume")
        self.controller = saved


class TrainingRewardStats:
    """Per-update training reward sums/counts from responses already generated for training."""

    def __init__(self):
        self.rows: dict[int, tuple[float, int]] = {}

    def add(self, completed_update: int, reward_sum, count):
        if completed_update in self.rows:
            return False
        if reward_sum is None or count is None or count <= 0 or not math.isfinite(reward_sum):
            self.rows[int(completed_update)] = (None, 0)
        else:
            self.rows[int(completed_update)] = (float(reward_sum), int(count))
        return True

    def window_mean(self, lo_exclusive: int, hi_inclusive: int):
        total, count = 0.0, 0
        for u in range(lo_exclusive + 1, hi_inclusive + 1):
            row = self.rows.get(u)
            if row is None or row[0] is None:
                return None
            total, count = total + row[0], count + row[1]
        return total / count if count else None

    def state_dict(self):
        return {str(k): list(v) for k, v in sorted(self.rows.items())}

    @classmethod
    def from_state(cls, state):
        s = cls()
        s.rows = {int(k): (v[0], v[1]) for k, v in state.items()}
        return s


class PlateauPolicy(TransitionPolicy):
    """E2: training-reward plateau, count-weighted windows, k consecutive checks.

    At check t: improvement = mean(updates (t-W, t]) - mean(updates (t-2W, t-W]).
    plateau_check = improvement <= epsilon (reward fraction; 0.005 = 0.5 pp).
    Missing/insufficient data resets the confirmation counter (warmup).
    """
    policy_id = "reward_plateau"

    def __init__(self, window=25, epsilon=0.005, consecutive=2, check_interval=25, first_check=None):
        self.window, self.epsilon, self.consecutive = int(window), float(epsilon), int(consecutive)
        self.check_interval = int(check_interval)
        self.first_check = int(first_check if first_check is not None else 2 * window)
        self.counter, self.last_check, self.transitioned = 0, None, False
        self.log: dict[int, dict] = {}

    def before_update(self, step, stats):
        if self.transitioned or step < self.first_check or (step - self.first_check) % self.check_interval:
            return None
        if step in self.log:
            entry = self.log[step]
            return entry if entry["action"] == "switch" else None
        if self.last_check is not None and step != self.last_check + self.check_interval:
            self.counter = 0
        self.last_check = step
        recent = stats.window_mean(step - self.window, step)
        previous = stats.window_mean(step - 2 * self.window, step - self.window)
        if recent is None or previous is None:
            self.counter = 0
            entry = {"completed_update": step, "action": "keep", "reason": "warmup_or_missing",
                     "improvement": None}
        else:
            improvement = recent - previous
            self.counter = self.counter + 1 if improvement <= self.epsilon else 0
            switch = self.counter >= self.consecutive
            self.transitioned = switch
            entry = {"completed_update": step, "action": "switch" if switch else "keep",
                     "reason": "plateau_confirmed" if switch else ("plateau_check" if self.counter else "improving"),
                     "improvement": improvement, "recent_mean": recent, "previous_mean": previous}
        entry.update(rule_id=f"plateau_w{self.window}_eps{self.epsilon:g}_k{self.consecutive}",
                     consecutive_plateau_checks=self.counter)
        self.log[step] = entry
        return entry if entry["action"] == "switch" else None

    def state_dict(self):
        return {"policy_id": self.policy_id, "window": self.window, "epsilon": self.epsilon,
                "consecutive": self.consecutive, "check_interval": self.check_interval,
                "first_check": self.first_check, "counter": self.counter, "last_check": self.last_check,
                "transitioned": self.transitioned, "log": {str(k): v for k, v in self.log.items()}}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if (state["window"], state["epsilon"], state["consecutive"], state["check_interval"]) != (
                self.window, self.epsilon, self.consecutive, self.check_interval):
            raise ValueError("plateau parameters changed between save and resume")
        self.counter, self.last_check, self.transitioned = state["counter"], state["last_check"], state["transitioned"]
        self.first_check = state["first_check"]
        self.log = {int(k): v for k, v in state["log"].items()}


def replay_plateau(stats: TrainingRewardStats, *, window=25, epsilon=0.005, consecutive=2,
                   check_interval=25, first_check=None, last_check=250, observed_through=None):
    """X3 replay on a logged On trajectory; censored if the log ends before last_check."""
    policy = PlateauPolicy(window, epsilon, consecutive, check_interval, first_check)
    end = observed_through if observed_through is not None else max(stats.rows, default=0)
    for step in range(policy.first_check, min(end, last_check) + 1, check_interval):
        decision = policy.before_update(step, stats)
        if decision:
            return {"rule_id": decision["rule_id"], "status": "triggered", "switch_update": step}
    rule_id = f"plateau_w{window}_eps{epsilon:g}_k{consecutive}"
    if end >= last_check:
        return {"rule_id": rule_id, "status": "not_triggered", "switch_update": None}
    return {"rule_id": rule_id, "status": "censored", "switch_update": None, "censored_after": end}


class BidirectionalPolicy(TransitionPolicy):
    """E8 scaffold (optional): On->SR on the forward rule, SR->On on its mirror.

    Not part of the default stages. Reverse checks require SR-phase monitoring
    (both 40-prompt sets scored), whose cost is charged to this policy. Each
    transition resets the window; `min_dwell` updates must pass before any
    reverse decision; `margin` is the hysteresis band (|D| <= margin is ignored).
    The code repo arm ``switch_repeat`` (scripts/srgc_switch_repeat.py) is the
    existing executable variant without dwell/margin.
    """
    policy_id = "bidirectional"
    uses_signal = True

    def __init__(self, rule="current", interval=25, first_check=25, min_dwell=50, margin=0.0, max_transitions=4):
        self.rule, self.interval, self.first_check = canonical_rule(rule), interval, first_check
        self.min_dwell, self.margin, self.max_transitions = int(min_dwell), float(margin), int(max_transitions)
        self.phase, self.entered, self.transitions = "on_policy", 0, []
        self._reset()

    def _reset(self):
        self.controller = SRGCController(self.rule, self.interval, self.first_check)
        self.controller.last_check_update = None

    def needs_signal(self, step):
        return step >= self.first_check and (step - self.first_check) % self.interval == 0 and \
            len(self.transitions) < self.max_transitions

    def on_signal(self, step, value, *, valid=True):
        if step - self.entered < self.min_dwell and self.transitions:
            return {"completed_update": step, "action": "keep", "reason": "min_dwell"}
        if valid and value is not None and abs(value) <= self.margin:
            value = None  # inside the hysteresis band: treated as no evidence (reset)
        signed = value if (value is None or self.phase == "on_policy") else -value
        decision = self.controller.observe(step, signed, valid=value is not None)
        if decision["action"] == "switch":
            target = "sr" if self.phase == "on_policy" else "on_policy"
            self.transitions.append({"completed_update": step, "from": self.phase, "to": target})
            self.phase, self.entered = target, step
            self._reset()
            return {"completed_update": step, "action": f"switch_to_{target}", "reason": decision["reason"]}
        return {"completed_update": step, "action": "keep", "reason": decision["reason"]}

    def state_dict(self):
        return {"policy_id": self.policy_id, "phase": self.phase, "entered": self.entered,
                "transitions": list(self.transitions), "controller": self.controller.state_dict(),
                "params": [self.rule, self.interval, self.first_check, self.min_dwell, self.margin,
                           self.max_transitions]}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if state["params"] != [self.rule, self.interval, self.first_check, self.min_dwell, self.margin,
                               self.max_transitions]:
            raise ValueError("bidirectional parameters changed")
        self.phase, self.entered, self.transitions = state["phase"], state["entered"], list(state["transitions"])
        self.controller = SRGCController.from_state(state["controller"])


def build_policy(spec: dict) -> TransitionPolicy:
    kind = spec["policy"]
    if kind == "never":
        return NeverPolicy()
    if kind == "fixed":
        return FixedPolicy(spec["fixed_update"])
    if kind == "sr_gc":
        return SRGCPolicy(spec.get("rule", "current"), spec.get("check_interval", 25), spec.get("first_check", 25))
    if kind == "reward_plateau":
        p = spec["plateau"]
        return PlateauPolicy(p["window_updates"], p["epsilon"], p["consecutive_checks"], p["check_interval"],
                             p.get("first_check"))
    if kind == "sampled_time":
        return SampledTimePolicy(spec["sampled_update"], spec["distribution_hash"], spec["schedule_seed"])
    if kind == "bidirectional":
        b = spec.get("bidirectional", {})
        return BidirectionalPolicy(spec.get("rule", "current"), spec.get("check_interval", 25),
                                   spec.get("first_check", 25), b.get("min_dwell", 50), b.get("margin", 0.0),
                                   b.get("max_transitions", 4))
    raise ValueError(f"unknown transition policy {kind!r}")
