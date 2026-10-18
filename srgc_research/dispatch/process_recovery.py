"""Recognize recording ranks while preserving the frozen process guard code."""

from contextlib import contextmanager
from unittest.mock import patch


def recording_guard(original, guard, model):
    marker = ('srgc_qwen35_reward_curve_rank.py' if model == 'qwen35' else
              f'srgc_research.dispatch.reward_curve_rank --model {model}')

    @contextmanager
    def protected(plan_path):
        # Frozen Llama/Gemma CLIs replace TARGET_MARKERS immediately before
        # entering process_guard, so extending the outer run entry is too early.
        targets = tuple(dict.fromkeys((*guard.TARGET_MARKERS, marker)))
        with patch.object(guard, 'TARGET_MARKERS', targets), original(plan_path):
            print(f'GUARD recorded-rank recovery active model={model}', flush=True)
            yield

    return protected
