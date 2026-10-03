"""Fresh SR rewards with the cached-SR control's fixed per-prompt tie order."""

from scripts.srgc_sr_refresh import SRRefreshEngine
from srgc_rebuttal.srgc import top_ids


class MatchedSRRefreshEngine(SRRefreshEngine):
    SR_TIE_PROTOCOL = "fresh-sr-global-cached-ties-v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.refresh_scope != "candidates":
            raise ValueError("matched SR requires the candidates scope")
        order = top_ids(self.candidates, [0.0] * len(self.candidates),
                        len(self.candidates), self.config.seed + 1000)
        self.tie_order = {prompt: index for index, prompt in enumerate(order)}

    def _rank_refreshed(self, ids, rewards):
        # Filtering the global keys matches sr_hold even when the draw order changes.
        return tuple(sorted(ids, key=lambda prompt: (
            abs(sum(rewards[prompt]) / self.config.responses - 0.5), self.tie_order[prompt])))

    def state_dict(self):
        return {**super().state_dict(), "sr_tie_protocol": self.SR_TIE_PROTOCOL}

    def load_state_dict(self, state, *, fork_arm=None):
        if fork_arm is None and state.get("sr_tie_protocol") != self.SR_TIE_PROTOCOL:
            raise ValueError("matched SR checkpoint has a different tie protocol")
        return super().load_state_dict(state, fork_arm=fork_arm)
