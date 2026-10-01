"""Matched ablation of the gradient direction in On-policy selection (E09).

On-policy ranks its 40 candidates by the cosine between each candidate's
projected scoring gradient and the mean validation gradient, and trains the
top four for 25 updates. The reading "the early On-policy advantage comes from
that direction information" is only a correlation in the recorded runs. The
``direction_<mode>`` arms keep every other part of On-policy fixed - the same
25-update refreshes, the same 40 candidates drawn from the same stream,
the same scoring rollouts and validation gradients (so
the scoring cost and the recorded decomposition are matched), the same
four-prompt batch retained for 25 updates and the same training rollouts - and
change only what the ranking may use. "The same scoring" follows the engine
actually loaded: the archived ``contrast40-v2`` engine of the recorded P0
cohort scores the 40 SR comparison prompts at every On-policy refresh, so the
ablation scores them too; the current ``v3`` engine scores them only at Switch
checks, so the ablation scores the 40 candidates only. Either way the control
costs what that engine's On-policy costs.

  direction_removed    no direction and no magnitude: the four trained prompts
                       are drawn uniformly from the scored 40 (seeded tie-break
                       over equal scores), i.e. Random's choice at On-policy's cost;
  direction_magnitude  the projected gradient norm ranks the candidates, so the
                       magnitude survives and the direction is discarded;
  direction_replaced   the validation direction is replaced by a random unit
                       vector drawn at each refresh, so a direction is still used
                       but carries no validation information.

Each record keeps the true validation cosines (``ranking_scores``) next to
the scores the ablation ranked by (``ablation_scores``), so the direction
decomposition of ``srgc_direction_records`` applies unchanged. No SR-GC check
or transition is performed: the arms are On-policy controls, to be compared
with the recorded ``on_policy`` (and ``random``) arm of the same seed.

Forks from the seed's verified shared prefix in the existing run root and
writes ``direction_<mode>-{latest.pt,progress.json,endpoint.json}``.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srgc_rebuttal.srgc import Engine, cosine_scores, stream_seed, top_ids  # noqa: E402

try:
    from srgc_direction_records import DirectionRecordMixin
except ImportError:
    from scripts.srgc_direction_records import DirectionRecordMixin

MODES = ("removed", "magnitude", "replaced")
# Engines whose On-policy refresh scores the SR comparison preview as well as the candidates.
SR_PREVIEW_PROTOCOLS = {"random-candidate40-training4-contrast40-v2"}
ARMS = tuple(f"direction_{mode}" for mode in MODES)


def mode_of(arm):
    if arm not in ARMS:
        raise ValueError(f"not a direction ablation arm name: {arm!r}")
    return arm.removeprefix("direction_")


class DirectionAblationEngine(DirectionRecordMixin, Engine):
    ARMS = Engine.ARMS | {"direction_ablation"}
    ABLATION_PROTOCOL = "matched-on-policy-direction-ablation-v1"

    def __init__(self, *args, mode, **kwargs):
        if mode not in MODES:
            raise ValueError(f"ablation mode must be one of {MODES}")
        self.mode = mode
        super().__init__(*args, **kwargs)

    def _ablated_ranking(self, on_ids, gradients, v):
        """(true cosines, scores the ablation ranks by, the four trained prompts)."""
        c = self.config
        stack = np.stack([gradients[i] for i in on_ids])
        cosines = cosine_scores(stack, v)
        if self.mode == "magnitude":
            scores = np.linalg.norm(stack, axis=1)
        elif self.mode == "replaced":
            rng = np.random.default_rng(stream_seed(c.seed, self.step, "direction-replaced"))
            scores = cosine_scores(stack, rng.normal(size=v.shape))
        else:
            scores = np.zeros(len(on_ids))  # all tied: the seeded tie-break is a uniform draw of four
        train_ids = top_ids(on_ids, scores, c.training_prompts, stream_seed(c.seed + 1000, self.step, "online-ties"))
        return cosines, scores, train_ids

    def update(self):
        if self.arm != "direction_ablation":
            return super().update()
        c = self.config
        before = dict(self.costs)
        refresh = self.step % c.selection_interval == 0
        record = {"checkpoint": self.step, "d": None, "switched": False, "selection_refreshed": refresh,
                  "ablation": self.mode}
        if not refresh and self.active_selection is None:
            raise ValueError("mid-block continuation requires the saved selected prompts")
        if refresh:
            # Identical to the loaded engine's On-policy refresh (same candidates, same SR preview
            # when that engine scores one, same validation gradients) up to the ranking.
            started = self._begin("selection")
            on_ids = self._draw_candidates()
            sr_ids = self._sr_comparison() if self.SAMPLING_PROTOCOL in SR_PREVIEW_PROTOCOLS else ()
            union = tuple(dict.fromkeys((*on_ids, *sr_ids)))
            gradients = self._vectors(union, c.candidate_group_size, "selection")
            val = self._vectors(self.validation, c.responses, "validation")
            with self._timing_scope("cosine_ranking"):
                v = np.stack([val[i] for i in self.validation]).mean(axis=0)
                cosines, scores, train_ids = self._ablated_ranking(on_ids, gradients, v)
            self.active_selection = {"step": self.step, "on_ids": list(on_ids), "train_ids": list(train_ids)}
            self._end("selection", started)
            record.update(on_ids=list(on_ids), sr_ids=list(sr_ids), ranking_scores=cosines.tolist(),
                          ablation_scores=scores.tolist(), selected_on_ids=list(train_ids),
                          scored_distinct_prompts=len(union), validation_ids=list(self.validation),
                          scoring_responses_per_prompt=c.responses)
        train_ids = tuple(self.active_selection["train_ids"])
        record["selection_step"] = self.active_selection["step"]
        used, cycle = set(self.used_training_ids), self.sampling_cycle
        started = self._begin("training")
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
                      selector=f"direction_{self.mode}", metrics=metrics)
        self._annotate_direction(record)
        self.history.append(record)
        return record

    def state_dict(self):
        state = super().state_dict()
        state["ablation_mode"] = self.mode
        state["ablation_protocol"] = self.ABLATION_PROTOCOL
        return state

    def load_state_dict(self, state, *, fork_arm=None):
        if fork_arm is None and state.get("arm") == "direction_ablation":
            if state.get("ablation_protocol") != self.ABLATION_PROTOCOL:
                raise ValueError("ablation protocol changed; start a new direction ablation run")
            if state.get("ablation_mode") != self.mode:
                raise ValueError("checkpoint ablation mode differs; use the arm name of that mode")
            # Validate the saved selection block as the base engine does for an On-policy phase.
            super().load_state_dict({**state, "arm": "on_policy"})
            self.arm = "direction_ablation"
            return
        super().load_state_dict(state, fork_arm=fork_arm)
