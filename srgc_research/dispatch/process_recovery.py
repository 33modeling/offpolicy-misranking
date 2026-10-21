"""Recognize recording ranks while preserving the frozen process guard code."""

from contextlib import contextmanager
from unittest.mock import patch

MODELS = ('qwen35', 'llama31', 'gemma4', 'gemma4_text')
RANK_MARKERS = (
    'srgc_qwen35_rank.py', 'srgc_qwen35_rank_runtime.py',
    'srgc_qwen35_reward_curve_rank.py', 'srgc_qwen35_smoke.py',
    'srgc_research.dispatch.llama31.rank', 'srgc_research.dispatch.gemma4.rank',
    'srgc_research.dispatch.gemma4_text.rank',
    *(f'srgc_research.dispatch.reward_curve_rank --model {name}' for name in MODELS),
    *(f'srgc_research.dispatch.rank_runtime --model {name}' for name in MODELS),
)
OWNER_MARKERS = (
    'srgc_research.dispatch.qwen_run',
    'srgc_research.dispatch.llama_run', 'srgc_research.dispatch.llama31.cli',
    'srgc_research.dispatch.gemma_run', 'srgc_research.dispatch.gemma4.entry',
    'srgc_research.dispatch.gemma4.cli',
    'srgc_research.dispatch.gemma4_text.entry', 'srgc_research.dispatch.gemma4_text.cli',
    'srgc_research.dispatch.gemma_text_run',
    'srgc_research.dispatch.information_queue', 'srgc_research.dispatch.information_run',
    'srgc_research.information_cli',
)


def recording_guard(original, guard, model):
    @contextmanager
    def protected(plan_path):
        # Frozen Llama/Gemma CLIs replace TARGET_MARKERS immediately before
        # entering process_guard, so extending the outer run entry is too early.
        # An orphan from any sibling model can occupy these GPUs. Register
        # their live owners too, before scanning across model families.
        targets = tuple(dict.fromkeys((*guard.TARGET_MARKERS, *RANK_MARKERS)))
        owners = tuple(dict.fromkeys((*guard.OWNER_MARKERS, *OWNER_MARKERS)))
        with patch.object(guard, 'TARGET_MARKERS', targets), \
                patch.object(guard, 'OWNER_MARKERS', owners), original(plan_path):
            print(f'GUARD recorded-rank recovery active model={model}', flush=True)
            yield

    return protected
