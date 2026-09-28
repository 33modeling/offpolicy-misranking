"""Record the magnitude/direction decomposition of the SR-GC contrast at every refresh.

The contrast the switching rule observes is an inner product,
D = <v, g_on> - <v, g_sr> = ||v|| (||g_on|| cos_on - ||g_sr|| cos_sr),
where v is the mean validation gradient and g_on / g_sr the mean gradients of
the 40 On-policy candidates and the 40 SR prompts. The frozen engine records
D and the two inner products but not the norms and cosines, so the reading
"early On-policy wins on direction, later SR wins on magnitude" cannot be
checked from the recorded history alone. ``DirectionRecordMixin`` keeps the
vectors the engine already computed for the check and adds, to the same
history record:

  on_mean_norm, sr_mean_norm, validation_norm, on_mean_cos, sr_mean_cos,
  on_top4_dot (mean validation inner product of the four trained candidates),
  on_random4_expected_dot (the 40-candidate mean: the expectation of a random
  four), on_cos_top4_minus_mean, ranking_gap4 (fourth minus fifth cosine).

No training computation changes; the mixin only reads vectors that were
computed anyway. It is applied from the scripts layer (the training child
entry and the extra arms), so the hashed package is untouched.
"""

import numpy as np


def _mean(vectors, ids):
    return np.stack([vectors[i] for i in ids]).mean(axis=0)


def _cos(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


class DirectionRecordMixin:
    """Mix in before ``Engine``: ``class E(DirectionRecordMixin, Engine)``."""

    def _vectors(self, ids, group_size, purpose):
        received = super()._vectors(ids, group_size, purpose)
        stash = getattr(self, "_direction_stash", None)
        if stash is None:
            stash = self._direction_stash = {}
        stash[purpose] = received
        return received

    def _annotate_direction(self, record):
        stash = getattr(self, "_direction_stash", None) or {}
        selection, validation = stash.get("selection"), stash.get("validation")
        if not record.get("selection_refreshed") or selection is None or validation is None:
            return record
        on_ids, sr_ids = record.get("on_ids"), record.get("sr_ids")
        if not on_ids or not sr_ids or not all(i in selection for i in (*on_ids, *sr_ids)):
            self._direction_stash = {}
            return record
        v = _mean(validation, list(validation))
        g_on, g_sr = _mean(selection, on_ids), _mean(selection, sr_ids)
        scores = record.get("ranking_scores")
        selected = record.get("selected_on_ids") or record.get("train_ids") or []
        fields = {
            "validation_norm": float(np.linalg.norm(v)),
            "on_mean_norm": float(np.linalg.norm(g_on)), "sr_mean_norm": float(np.linalg.norm(g_sr)),
            "on_mean_cos": _cos(g_on, v), "sr_mean_cos": _cos(g_sr, v),
            "on_random4_expected_dot": float(np.dot(v, g_on)),
        }
        if selected and all(i in selection for i in selected):
            fields["on_top4_dot"] = float(np.dot(v, _mean(selection, selected)))
        if scores and len(scores) == len(on_ids):
            ordered = sorted(scores, reverse=True)
            k = min(4, len(ordered))
            fields.update(ranking_cos_mean=float(np.mean(scores)), ranking_cos_std=float(np.std(scores)),
                          ranking_cos_top4=float(np.mean(ordered[:k])),
                          on_cos_top4_minus_mean=float(np.mean(ordered[:k]) - np.mean(scores)),
                          ranking_gap4=float(ordered[k - 1] - ordered[k]) if len(ordered) > k else None)
        record.update(fields)
        self._direction_stash = {}
        return record

    def update(self):
        record = super().update()
        return self._annotate_direction(record)
