"""Scripts-layer tuning of the training child: progress counters and the attention kernel.

* ``count_progress`` wraps ``srgc_rebuttal.progress.record`` as bound in the torch backend so
  every ``rank-N.json`` also carries ``count``: how many events of that stage this rank has
  recorded since its last optimizer update. During a selection refresh that is the number of
  prompts already scored, which the NODE status line shows as ``scored 12,13,12,11 prompts``.
* ``SRGC_ATTENTION=sdpa|eager|flash_attention_2`` selects the attention implementation the
  training child loads the model with. The frozen runner loads eager attention; the cache
  builder already uses sdpa. Eager attention is several times slower for 2048-token
  generation and backward passes. The choice is recorded in every checkpoint's
  ``checkpoint_policy`` so a run cannot silently mix kernels.
"""

import functools
import os

ATTENTION_CHOICES = ("eager", "sdpa", "flash_attention_2")
RESET_STAGES = {"policy_update", "update", "model_ready"}


def counting_record(original):
    counts = {}

    def record(stage, **details):
        if stage in RESET_STAGES:
            counts.clear()
        counts[stage] = counts.get(stage, 0) + 1
        return original(stage, count=counts[stage], **details)
    record.counts = counts
    return record


def count_progress():
    """Patch the progress recorder where the torch backend and runner bound it; returns the wrapper."""
    from srgc_rebuttal import progress, run_experiment, torch_backend
    wrapper = counting_record(progress.record)
    torch_backend.progress = wrapper
    run_experiment.progress = wrapper
    return wrapper


def configured_attention(environment=None):
    value = (environment or os.environ).get("SRGC_ATTENTION")
    if value is None or value == "":
        return None
    if value not in ATTENTION_CHOICES:
        raise ValueError(f"SRGC_ATTENTION must be one of {ATTENTION_CHOICES}, not {value!r}")
    return value


def apply_attention(environment=None):
    """Bind the configured attention kernel into the runner's model loader; returns the kernel name in use."""
    from srgc_rebuttal import run_experiment
    attention = configured_attention(environment)
    if attention is None:
        return "eager"
    original = run_experiment.load_model
    run_experiment.load_model = functools.partial(original, attention=attention)
    return attention
