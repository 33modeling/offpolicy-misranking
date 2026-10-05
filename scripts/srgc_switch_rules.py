"""Own-path SR-GC rule ablations; keep scoring and training unchanged."""

import math

from srgc_rebuttal.srgc import Engine, TemporalRule
from scripts.srgc_direction_records import DirectionRecordMixin

RULE_ARMS = ("switch_single", "switch_consecutive")
RULE_PROTOCOL = "single-reference-own-path-negative-rules-v1"


class ConsecutiveNegativeRule(TemporalRule):
    required = 2

    def observe(self, step, d):
        if self.switched:
            raise RuntimeError("checks stop after switching")
        if step < 0 or step % self.interval or not math.isfinite(d):
            raise ValueError("a check needs a scheduled step and a finite D")
        if self.last_step is not None:
            if step <= self.last_step:
                raise ValueError("check steps must strictly increase")
            if step != self.last_step + self.interval:
                self.window.clear()
        self.last_step = step
        self.window = [*self.window, float(d)][-self.required:] if d < 0 else []
        self.switched = len(self.window) == self.required
        return self.switched


class SingleNegativeRule(ConsecutiveNegativeRule):
    required = 1


def rule_type(arm):
    if arm not in RULE_ARMS:
        raise ValueError(f"unknown switching rule: {arm}")
    return SingleNegativeRule if arm == "switch_single" else ConsecutiveNegativeRule


class SwitchRuleEngine(DirectionRecordMixin, Engine):
    ARMS = Engine.ARMS | {"switch_rule"}

    def __init__(self, *args, rule_arm, **kwargs):
        self.rule_arm = rule_arm
        self.rule_class = rule_type(rule_arm)
        super().__init__(*args, **kwargs)
        self.rule = self.rule_class(self.config.check_interval)
        if self.arm == "switch_rule":
            self._charge_preparation()

    def update(self):
        if self.arm != "switch_rule":
            return super().update()
        self.arm = "switch"
        try:
            record = super().update()
        finally:
            self.arm = "switch_rule"
        record.update(rule_arm=self.rule_arm, rule_protocol=RULE_PROTOCOL)
        return record

    def state_dict(self):
        return {**super().state_dict(), "rule_arm": self.rule_arm, "rule_protocol": RULE_PROTOCOL}

    def load_state_dict(self, state, *, fork_arm=None):
        if fork_arm is None:
            if (state.get("arm") != "switch_rule" or state.get("rule_arm") != self.rule_arm
                    or state.get("rule_protocol") != RULE_PROTOCOL):
                raise ValueError("switch rule checkpoint protocol differs")
            # The base validator checks retained batches using its Switch label.
            super().load_state_dict({**state, "arm": "switch"})
            self.arm = "switch_rule"
            self.rule = self.rule_class(**state["rule"])
        else:
            if fork_arm != "switch_rule":
                raise ValueError("a rule ablation must fork as switch_rule")
            super().load_state_dict(state, fork_arm=fork_arm)
            self.rule = self.rule_class(self.config.check_interval)
            self._charge_preparation()
