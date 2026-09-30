"""Independent training replicates from one verified shared prefix (E03).

Rerunning an arm with the same base seed resumes or reports its existing
checkpoint; changing the base seed changes the prefix, the data split and the
SR cache as well. Neither measures the run-to-run variation of one
continuation. ``replicate<k>-<arm>`` forks ``<arm>`` from the seed's verified
prefix exactly like the recorded arm (same model/optimizer state, candidate
pool, ranking-validation set, SR cache ranking, evaluation IDs and endpoint
update count) and changes only the random stream used after the fork: the
candidate draws, the scoring/training rollouts and the tie breaks. Every arm
of the same replicate shares that stream, so ``replicate1-sr`` and
``replicate1-switch`` are a paired comparison, and different replicates use
different streams. The endpoint evaluation keeps the recorded rule and seed.

The stream is derived from the base seed and the replicate id (``sampling_seed``)
and is recorded, with the replicate id and the prefix hash, in
``seed-N/replicate-<k>/replicate.json`` next to the replicate's own
``<arm>-{latest.pt,progress.json,endpoint.json}`` and cost receipts. Replicate
0 is the recorded run itself and cannot be requested.

The hashed ``srgc_rebuttal`` package is not modified: the engine reads all of
its stream seeds from ``config.seed``, so the mixin substitutes a config whose
seed is the sampling seed for the duration of each update and restores the
base config for checkpoints and identity checks.
"""

import dataclasses
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srgc_rebuttal.srgc import Engine, stream_seed  # noqa: E402

try:
    from srgc_direction_records import DirectionRecordMixin
    from srgc_switch_fixed import SwitchFixedEngine, fixed_step_of
except ImportError:
    from scripts.srgc_direction_records import DirectionRecordMixin
    from scripts.srgc_switch_fixed import SwitchFixedEngine, fixed_step_of

REPLICATE_PROTOCOL = "same-prefix-independent-sampling-stream-v1"
ARM_PATTERN = re.compile(r"^replicate(\d+)-(random|sr|on_policy|switch|switch_fixed\d+)$")


def parse_replicate_arm(name):
    """``replicate2-switch`` -> ``(2, "switch")``; ``None`` for any other arm name."""
    match = ARM_PATTERN.match(name)
    if not match:
        return None
    replicate = int(match.group(1))
    if replicate < 1:
        raise ValueError("replicate 0 is the recorded run; independent replicates start at 1")
    return replicate, match.group(2)


def sampling_seed(base_seed, replicate):
    """The post-fork stream seed of replicate k of a base seed; never the base seed's own stream."""
    if type(replicate) is not int or replicate < 1:
        raise ValueError("replicate ids are positive integers")
    value = stream_seed(base_seed, replicate, "independent-replicate-sampling-stream") % (2**31 - 1)
    if value == base_seed:
        value = (value + 1) % (2**31 - 1)
    return value


def replicate_folder(seed_folder, replicate):
    return Path(seed_folder) / f"replicate-{replicate}"


class ReplicateMixin:
    """Mix in before the engine: ``class E(ReplicateMixin, Engine)``.

    Only the stream seeds change. The SR cache ranking (built in ``Engine.__init__``
    from the base seed), the config the checkpoint records and every identity check
    keep the base seed.
    """

    def __init__(self, *args, replicate, **kwargs):
        super().__init__(*args, **kwargs)
        self.replicate = int(replicate)
        self.base_seed = self.config.seed
        self.sampling_seed = sampling_seed(self.base_seed, self.replicate)
        self._stream_config = dataclasses.replace(self.config, seed=self.sampling_seed)

    def replicate_record(self):
        return {"protocol": REPLICATE_PROTOCOL, "id": self.replicate, "base_seed": self.base_seed,
                "sampling_seed": self.sampling_seed}

    def update(self):
        base = self.config
        self.config = self._stream_config
        try:
            record = super().update()
        finally:
            self.config = base
        record["replicate"] = self.replicate
        return record

    def state_dict(self):
        state = super().state_dict()
        state["replicate"] = self.replicate_record()
        return state

    def load_state_dict(self, state, *, fork_arm=None):
        saved = state.get("replicate")
        if fork_arm is None:
            if saved is None:
                raise ValueError("checkpoint is a recorded arm, not an independent replicate")
            if saved != self.replicate_record():
                raise ValueError("checkpoint replicate id, sampling seed or protocol differs")
        elif saved is not None:
            raise ValueError("a replicate forks from the shared prefix, not from another replicate")
        super().load_state_dict(state, fork_arm=fork_arm)


class ReplicateEngine(ReplicateMixin, DirectionRecordMixin, Engine):
    """A recorded arm (random, sr, on_policy, switch) on a separate sampling stream."""


class ReplicateSwitchFixedEngine(ReplicateMixin, SwitchFixedEngine):
    """The fixed-schedule control on the same replicate stream as its paired arms."""


def make_engine(replicate, base_arm, backend, data, config):
    """(engine, engine label) for ``replicate<k>-<base_arm>``; the label is the recorded arm's."""
    if base_arm in Engine.ARMS:
        return ReplicateEngine(backend, data["candidate_ids"], data["ranking_validation_ids"], data["cached_rewards"],
                               arm=base_arm, config=config, replicate=replicate), base_arm
    return ReplicateSwitchFixedEngine(backend, data["candidate_ids"], data["ranking_validation_ids"],
                                      data["cached_rewards"], arm="switch_fixed", config=config,
                                      fixed_step=fixed_step_of(base_arm), replicate=replicate), "switch_fixed"
