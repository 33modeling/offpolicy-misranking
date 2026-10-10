"""Shared failure footer after all backup/logging contexts close."""

from pathlib import Path

from srgc_research.dispatch.model_resume import failure_footer as print_footer


def failure_footer(root, dataset):
    names = ("math", "mbpp") if dataset == "all" else (dataset,)
    plans = [Path(root) / "experiments" / f"gemma4-12b-pt-{name}.json" for name in names]
    print_footer(root, plans, "GEMMA")
