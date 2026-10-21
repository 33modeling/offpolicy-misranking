"""Reuse containment and caches; never reuse an archived Gemma experiment root."""

import os
from pathlib import Path

from ..gemma4.storage import (
    group_work,
    inside,
    setup_storage,
    validate_plan_paths,
    validate_tree,
)
from .protocol import EXTENSION

__all__ = ["default_root", "group_work", "inside", "setup_storage",
           "validate_plan_paths", "validate_tree"]


def default_root(environment=None):
    environment = os.environ if environment is None else environment
    group = Path(environment.get("GROUP_VOLUME", "/group-volume")).resolve()
    work = Path(environment.get("OM_WORK", str(
        group / environment.get("OM_USER", "minsoo3.kim") / "offpolicy-misranking"))).resolve()
    if work == group or not work.is_relative_to(group):
        work = group / environment.get("OM_USER", "minsoo3.kim") / "offpolicy-misranking"
    return Path(environment.get("SRGC_GEMMA_TEXT_ROOT",
                                str(work / "srgc-rebuttal" / EXTENSION)))
