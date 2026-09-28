"""Fixed-schedule switching control: On-policy until a preset update, then SR, no SR-GC rule.

The recorded Switch arm transitions when the SR-GC temporal rule fires (seed 3
at 125, seed 4 at 100). Reviewers can ask whether the rule adds anything over
simply switching at a fixed step. ``switch_fixed<N>`` answers that: it trains
exactly like On-policy through update N (same 25-update candidate refreshes,
same scoring of the 40-vs-40 sets, so its pre-transition cost equals Switch's),
records no decision, and from update N+1 trains with the fixed SR ranking
exactly like the recorded Switch does after its transition. ``N`` must be a
multiple of the selection interval so the transition falls on a refresh
boundary, as the rule's transitions do.

Forks from the seed's verified shared prefix in the existing run root and
writes ``switch_fixed<N>-{latest.pt,progress.json,endpoint.json}``.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srgc_rebuttal.srgc import Engine  # noqa: E402

try:
    from srgc_direction_records import DirectionRecordMixin
except ImportError:
    from scripts.srgc_direction_records import DirectionRecordMixin

ARM_PATTERN = re.compile(r"^switch_fixed(\d+)$")


def fixed_step_of(arm):
    match = ARM_PATTERN.match(arm)
    if not match:
        raise ValueError(f"not a fixed-schedule arm name: {arm!r}")
    return int(match.group(1))


class SwitchFixedEngine(DirectionRecordMixin, Engine):
    ARMS = Engine.ARMS | {"switch_fixed"}
    TRANSITION_PROTOCOL = "fixed-boundary-before-training-v2"

    def __init__(self, *args, fixed_step, **kwargs):
        super().__init__(*args, **kwargs)
        interval = self.config.selection_interval
        if fixed_step < 1 or fixed_step % interval:
            raise ValueError("the fixed transition step must be a positive multiple of the selection interval")
        self.fixed_step = fixed_step
        if self.arm == "switch_fixed":
            self._charge_preparation()

    def _end(self, phase, started):
        super()._end(phase, started)
        # Keep the boundary refresh cost, but switch before its training update.
        if (phase == "selection" and getattr(self, "_fixed_update", False)
                and self.switched_at is None and self.step == self.fixed_step):
            self.switched_at = self.step

    def update(self):
        if self.arm != "switch_fixed":
            return super().update()
        # Before the transition the base engine runs as On-policy (no check, no decision);
        # after it, a set switched_at makes the base engine train with the SR ranking.
        previous = self.switched_at
        self.arm = "on_policy"
        self._fixed_update = True
        try:
            record = super().update()
        finally:
            self.arm = "switch_fixed"
            self._fixed_update = False
        record["switched"] = previous is None and self.switched_at is not None
        record["fixed_step"] = self.fixed_step  # the base record already names the selector actually trained
        return record

    def state_dict(self):
        state = super().state_dict()
        state["fixed_step"] = self.fixed_step
        state["fixed_transition_protocol"] = self.TRANSITION_PROTOCOL
        return state

    def load_state_dict(self, state, *, fork_arm=None):
        if fork_arm is None and state.get("arm") == "switch_fixed":
            if state.get("fixed_transition_protocol") != self.TRANSITION_PROTOCOL:
                raise ValueError("fixed transition protocol changed; start a new fixed-control run")
            if state.get("fixed_step") != self.fixed_step:
                raise ValueError("checkpoint fixed transition step differs; use a new arm name for another step")
            # Validate the saved selection block as the base engine would for an On-policy phase.
            checked = {**state, "arm": "on_policy" if state.get("switched_at") is None else "switch_fixed"}
            super().load_state_dict(checked)
            self.arm = "switch_fixed"
            self.switched_at = state.get("switched_at")
            return
        super().load_state_dict(state, fork_arm=fork_arm)
        if fork_arm == "switch_fixed":
            self.switched_at = None
            self._charge_preparation()
