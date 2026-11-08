"""SR-GC temporal rules, the one-way controller and censoring-aware replay.

D values arrive in time order at scheduled checks (completed-update counts).
A negative D favours the SR destination. Rule identifiers:

  one_negative    R1  latest valid check negative
  two_negative    R2  two consecutive negative checks
  three_negative  R4  three consecutive negative checks
  current         R3  the frozen rule of the code repository
                      (srgc_rebuttal.srgc.TemporalRule): two consecutive
                      negatives, OR negative / non-negative / negative whose
                      three-value sum is negative ("bridge").

The current rule can fire EARLIER than R2 (bridge), e.g. D=[-1,+0.5,-1,+0.5,-1,-1]
at a 25-update cadence from 25: current fires at 75, R2 at 150. Replaying a rule on
a log that stops (e.g. a Switch run that stopped checking after its own trigger)
therefore yields ``censored`` -- never ``not_triggered`` -- when the rule had not
fired by the last observed check and later scheduled checks were not observed.
"""

from __future__ import annotations

import math

RULES = ("one_negative", "two_negative", "three_negative", "current")
RULE_ALIASES = {"R1": "one_negative", "R2": "two_negative", "R3": "current", "R4": "three_negative"}
SIGNAL_DEFINITION = "srgc-40v40-validation-dot-v1"  # code repo gradient_contrast, unchanged


def canonical_rule(rule: str) -> str:
    rule = RULE_ALIASES.get(rule, rule)
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}")
    return rule


def _valid(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def should_switch(values, rule: str) -> bool:
    """Pure judgement on a contiguous history of VALID checks (plan Code §5.3)."""
    rule = canonical_rule(rule)
    values = [float(v) for v in values]
    if rule == "one_negative":
        return len(values) >= 1 and values[-1] < 0
    if rule == "two_negative":
        return len(values) >= 2 and all(v < 0 for v in values[-2:])
    if rule == "three_negative":
        return len(values) >= 3 and all(v < 0 for v in values[-3:])
    consecutive = len(values) >= 2 and values[-2] < 0 and values[-1] < 0
    bridge = (len(values) >= 3 and values[-3] < 0 and values[-2] >= 0 and values[-1] < 0
              and math.fsum(values[-3:]) < 0)
    return consecutive or bridge


class FrozenWindow:
    """Line-by-line port of the code repo TemporalRule window (parity-tested)."""

    def __init__(self):
        self.window = []

    def push(self, d: float) -> bool:
        if not self.window:
            if d < 0:
                self.window = [float(d)]
            return False
        if len(self.window) == 1:
            if d < 0:
                return True
            self.window.append(float(d))
            return False
        if d < 0 and math.fsum([*self.window, d]) < 0:
            return True
        self.window = [float(d)] if d < 0 else []
        return False


class SRGCController:
    """One-way controller: keep On until the rule confirms, then never re-evaluate.

    * scheduled checks are first_check, first_check+interval, ...
    * a skipped scheduled check (gap) or an invalid/NaN/missing value resets history
    * a repeated completed_update is applied once (idempotent re-delivery)
    * the A/B reference mean is NOT stored here; only the temporal sequence
    """

    def __init__(self, rule="current", interval=25, first_check=25):
        if interval < 1 or first_check < 1 or first_check % interval:
            raise ValueError("first_check must be a positive multiple of the interval")
        self.rule, self.interval, self.first_check = canonical_rule(rule), int(interval), int(first_check)
        self.history: list[float] = []
        self.last_check_update = None
        self.transitioned = False
        self.transition_update = None
        self.decisions: dict[int, dict] = {}

    def scheduled(self, step: int) -> bool:
        return step >= self.first_check and (step - self.first_check) % self.interval == 0

    def observe(self, step: int, value, *, valid: bool = True) -> dict:
        step = int(step)
        if step in self.decisions:
            return dict(self.decisions[step], duplicate=True)
        if self.transitioned:
            return {"completed_update": step, "action": "keep", "reason": "already_transitioned"}
        if not self.scheduled(step):
            raise ValueError(f"unscheduled check at completed_update {step}")
        if self.last_check_update is not None:
            if step <= self.last_check_update:
                raise ValueError("checks must strictly increase")
            if step != self.last_check_update + self.interval:
                self.history = []  # missing scheduled checks cannot confirm a transition
        self.last_check_update = step
        if not (valid and _valid(value)):
            self.history = []
            decision = {"completed_update": step, "action": "keep", "reason": "invalid_signal_reset"}
        else:
            self.history.append(float(value))
            if should_switch(self.history, self.rule):
                self.transitioned, self.transition_update = True, step
                decision = {"completed_update": step, "action": "switch", "reason": f"rule_{self.rule}"}
            else:
                decision = {"completed_update": step, "action": "keep", "reason": "not_confirmed"}
        decision.update(rule_id=self.rule, history=list(self.history))
        self.decisions[step] = decision
        return decision

    def state_dict(self) -> dict:
        return {"rule": self.rule, "interval": self.interval, "first_check": self.first_check,
                "history": list(self.history), "last_check_update": self.last_check_update,
                "transitioned": self.transitioned, "transition_update": self.transition_update,
                "decisions": {str(k): v for k, v in self.decisions.items()}}

    @classmethod
    def from_state(cls, state: dict) -> "SRGCController":
        c = cls(state["rule"], state["interval"], state["first_check"])
        c.history = list(state["history"])
        c.last_check_update = state["last_check_update"]
        c.transitioned, c.transition_update = state["transitioned"], state["transition_update"]
        c.decisions = {int(k): v for k, v in state["decisions"].items()}
        return c


def scheduled_checks(first_check: int, interval: int, last_check: int):
    return list(range(first_check, last_check + 1, interval))


def _as_step(step) -> int:
    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError(f"check step must be an integer completed-update count, got {step!r}")
    return step


def normalize_observations(observations):
    """Deduplicate deliveries; identical duplicates are dropped, conflicting ones raise."""
    obs = {}
    for step, value in observations:
        step = _as_step(step)
        value = None if not _valid(value) else float(value)
        if step in obs and obs[step] != value:
            raise ValueError(f"conflicting duplicate observations at completed_update {step}: "
                             f"{obs[step]!r} vs {value!r}")
        obs[step] = value
    return obs


def replay(observations, rule, *, first_check=25, interval=25, last_scheduled_check=250,
           observed_through=None, known_skipped=()) -> dict:
    """Replay a one-way rule on a logged D sequence (no inference beyond the log).

    observations: (completed_update, value) pairs; value None/NaN = measured but
    INVALID (known reset, as at runtime). A scheduled check that is absent from the
    log is UNKNOWN unless listed in `known_skipped` (runtime metadata proving the
    check was skipped -> reset). Replay walks the schedule in order and stops at the
    first unknown check: a trigger is reported only from a complete prefix of the
    schedule. Otherwise the result is `censored` with the last contiguous step in
    `observed_through` / `censored_after`. `not_triggered` requires every scheduled
    check through `last_scheduled_check` to be known. `observed_through`, if given,
    caps the usable log (e.g. a Switch run stops checking at its own trigger).
    """
    rule = canonical_rule(rule)
    obs = normalize_observations(observations)
    skipped = {_as_step(s) for s in known_skipped}
    schedule = scheduled_checks(first_check, interval, last_scheduled_check)
    for step in [*obs, *skipped]:
        if step not in schedule and step <= last_scheduled_check:
            raise ValueError(f"observation at unscheduled completed_update {step}")
    cap = last_scheduled_check if observed_through is None else min(_as_step(observed_through), last_scheduled_check)
    controller = SRGCController(rule, interval, first_check)
    used, contiguous = 0, first_check - interval
    for step in schedule:
        if step > cap:
            break
        if step in skipped:
            controller.history = []
            controller.last_check_update = step
            contiguous = step
            continue
        if step not in obs:
            break  # unknown from here on
        used += 1
        decision = controller.observe(step, obs[step], valid=obs[step] is not None)
        contiguous = step
        if decision["action"] == "switch":
            return {"rule_id": rule, "status": "triggered", "switch_update": step, "checks_used": used,
                    "observed_through": contiguous, "censored_after": None}
    if contiguous >= last_scheduled_check:
        return {"rule_id": rule, "status": "not_triggered", "switch_update": None, "checks_used": used,
                "observed_through": contiguous, "censored_after": None}
    return {"rule_id": rule, "status": "censored", "switch_update": None, "checks_used": used,
            "observed_through": contiguous, "censored_after": contiguous}


def subsample_cadence(observations, interval: int, first_check: int):
    """Interval-50 (etc.) cadence from a 25-cadence log: keep only its own checks."""
    return [(s, v) for s, v in observations if s >= first_check and (s - first_check) % interval == 0]
