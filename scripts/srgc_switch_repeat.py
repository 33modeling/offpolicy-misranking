"""Switch with repeated transitions: the "repeated transitions" experiment named in the Limitations.

The recorded Switch arm starts On-policy, applies the SR-GC temporal rule at
every 25-update check and, once it transitions to SR, never checks again.
``switch_repeat`` keeps the same rule for the On-policy -> SR transition and
adds its mirror image for the way back: while training with SR it keeps
scoring the 40-vs-40 contrast at every check (On-policy's scoring budget) and
returns to On-policy when D is positive at two consecutive checks, or
positive / non-positive / positive with a positive three-check sum. Each
transition resets the temporal window, so a decision never mixes evidence
from before and after a transition. On return to On-policy the batch just
scored for the check is trained, so no scoring is repeated.

Forks from the seed's verified shared prefix in the existing run root and
writes ``switch_repeat-{latest.pt,progress.json,endpoint.json}``.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srgc_rebuttal.srgc import Engine, TemporalRule, cosine_scores, gradient_contrast, stream_seed, top_ids  # noqa: E402


class SwitchRepeatEngine(Engine):
    ARMS = Engine.ARMS | {"switch_repeat"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = "on"
        self.transitions = []
        self.back_rule = TemporalRule(self.config.check_interval)
        if self.arm == "switch_repeat":
            self._charge_preparation()

    def _first_transition(self):
        return self.transitions[0]["step"] if self.transitions else None

    def _on_policy_update(self):
        """One update as the recorded Switch arm before its transition (base engine, unchanged)."""
        self.arm, self.switched_at = "switch", None
        try:
            record = super().update()
        finally:
            self.arm = "switch_repeat"
        if self.switched_at is not None:
            self.transitions.append({"step": self.switched_at, "to": "sr"})
            self.mode = "sr"
            self.back_rule = TemporalRule(self.config.check_interval)
        self.switched_at = self._first_transition()
        return record

    def _sr_update(self):
        c = self.config
        before = dict(self.costs)
        check = self.step >= c.first_check and self.step % c.check_interval == 0
        record = {"checkpoint": self.step, "d": None, "switched": False, "selection_refreshed": check}
        if check:
            started = self._begin("selection")
            on_ids = self._draw_candidates()
            sr_ids = self._sr_comparison()
            union = tuple(dict.fromkeys((*on_ids, *sr_ids)))
            gradients = self._vectors(union, c.candidate_group_size, "selection")
            val = self._vectors(self.validation, c.responses, "validation")
            with self._timing_scope("cosine_ranking"):
                v = np.stack([val[i] for i in self.validation]).mean(axis=0)
                scores = cosine_scores(np.stack([gradients[i] for i in on_ids]), v)
                train_ids = top_ids(on_ids, scores, c.training_prompts,
                                    stream_seed(c.seed + 1000, self.step, "online-ties"))
            with self._timing_scope("sr_gc_check"):
                d = gradient_contrast(on_ids, sr_ids, gradients, v)
                on_dot = float(np.dot(v, np.stack([gradients[i] for i in on_ids]).mean(axis=0)))
                sr_dot = float(np.dot(v, np.stack([gradients[i] for i in sr_ids]).mean(axis=0)))
                # Mirror rule: feed -D, so "two consecutive negatives" means two consecutive positive D.
                back = self.back_rule.observe(self.step, -d)
            self._end("selection", started)
            record.update(d=d, switched=back, on_mean_validation_dot=on_dot, sr_mean_validation_dot=sr_dot,
                          on_ids=list(on_ids), sr_ids=list(sr_ids), ranking_scores=scores.tolist(),
                          selected_on_ids=list(train_ids), scored_distinct_prompts=len(union),
                          validation_ids=list(self.validation), scoring_responses_per_prompt=c.responses)
            if back:
                self.transitions.append({"step": self.step, "to": "on"})
                self.mode = "on"
                self.rule = TemporalRule(c.check_interval)
                self.active_selection = {"step": self.step, "on_ids": list(on_ids), "train_ids": list(train_ids)}
        used, cycle = set(self.used_training_ids), self.sampling_cycle
        started = self._begin("training")
        if self.mode == "on":
            train_ids = tuple(self.active_selection["train_ids"])
            record["selection_step"] = self.active_selection["step"]
        else:
            with self._timing_scope("candidate_sampling_and_ranking"):
                candidates = self._draw_candidates()
                train_ids = self._training_batch("sr", candidates)
            record.update(sampling_pool_size=len(self.candidates), training_candidate_ids=list(candidates))
        used.update(train_ids)
        if len(used) == len(self.candidates):
            used.clear()
            cycle += 1
        metrics = dict(self.backend.train(train_ids, responses=c.responses, objective=c.objective,
                                          seed=stream_seed(c.seed, self.step, "training")))
        self._end("training", started)
        self.used_training_ids, self.sampling_cycle = used, cycle
        self.step += 1
        record.update(completed_updates=self.step, train_ids=list(train_ids),
                      selection_gpu_seconds=self.costs["selection_gpu_seconds"] - before["selection_gpu_seconds"],
                      training_gpu_seconds=self.costs["training_gpu_seconds"] - before["training_gpu_seconds"],
                      selector="on_policy" if self.mode == "on" else "sr", metrics=metrics)
        self.history.append(record)
        return record

    def update(self):
        if self.arm != "switch_repeat":
            return super().update()
        record = self._on_policy_update() if self.mode == "on" else self._sr_update()
        record.update(mode=self.mode, transitions=list(self.transitions))
        return record

    def state_dict(self):
        state = super().state_dict()
        state.update(mode=self.mode, transitions=list(self.transitions),
                     back_rule={"interval": self.back_rule.interval, "last_step": self.back_rule.last_step,
                                "window": list(self.back_rule.window), "switched": self.back_rule.switched})
        return state

    def load_state_dict(self, state, *, fork_arm=None):
        if fork_arm is None and state.get("arm") == "switch_repeat":
            # The base validation treats a saved selection block as On-policy's; it is only required in "on" mode.
            base_arm = "switch" if state.get("mode", "on") == "on" else "switch_repeat"
            checked = {**state, "arm": base_arm, "switched_at": None if base_arm == "switch" else state.get("switched_at")}
            super().load_state_dict(checked)
            self.arm = "switch_repeat"
            self.mode = state.get("mode", "on")
            self.transitions = list(state.get("transitions", []))
            self.switched_at = self._first_transition()
            back = state.get("back_rule")
            self.back_rule = TemporalRule(**back) if back else TemporalRule(self.config.check_interval)
            return
        super().load_state_dict(state, fork_arm=fork_arm)
        if fork_arm == "switch_repeat":
            self.mode, self.transitions, self.switched_at = "on", [], None
            self.rule = TemporalRule(self.config.check_interval)
            self.back_rule = TemporalRule(self.config.check_interval)
            self._charge_preparation()
