"""Expose runs displaced by the legacy automatic restart without changing them."""

from contextlib import contextmanager
import json
from pathlib import Path

from srgc_rebuttal.plan import load_plan


def previous_plans(plan_path):
    current = Path(plan_path).resolve()
    dataset = load_plan(current)["dataset"]
    seen = {current}
    while True:
        receipt = current.parent.parent / "automatic-restart.json"
        if not receipt.exists():
            return
        saved = json.loads(receipt.read_text())
        if not isinstance(saved, dict) or saved.get("reason") != "implementation changed":
            raise ValueError(f"invalid automatic restart receipt: {receipt}")
        if Path(saved["plan"]).resolve() != current:
            raise ValueError(f"restart receipt points to another run: {receipt}")
        previous = Path(saved["previous_plan"]).resolve()
        if previous in seen:
            raise ValueError(f"cycle in automatic restart history: {receipt}")
        if load_plan(previous)["dataset"] != dataset:
            raise ValueError(f"dataset differs in automatic restart history: {receipt}")
        seen.add(previous)
        yield previous
        current = previous


@contextmanager
def include_previous_runs():
    from srgc_rebuttal import reports
    original_snapshot, original_render = reports.snapshot, reports.render

    def snapshot(plan_path, **kwargs):
        report = original_snapshot(plan_path, **kwargs)
        report["previous_runs"] = []
        try:
            for previous in previous_plans(plan_path):
                prior = original_snapshot(previous, **kwargs)
                prior["plan_path"] = str(previous)
                report["previous_runs"].append(prior)
                report["errors"].extend(f"previous run {previous}: {error}" for error in prior["errors"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["errors"].append(f"previous run lookup failed: {exc}")
        if report["previous_runs"]:
            report["warnings"].append("An older launcher changed the active run. Previous runs are shown "
                                      "separately below; rewards and costs are NOT combined. No files moved.")
        if report["errors"]:
            report["complete"] = False
        return report

    def render(report, **kwargs):
        text = original_render(report, **kwargs)
        for prior in report.get("previous_runs", []):
            text += f"\nPREVIOUS RUN (read-only): {prior['plan_path']}\n"
            text += original_render(prior, **kwargs)
        return text

    reports.snapshot, reports.render = snapshot, render
    try:
        yield
    finally:
        reports.snapshot, reports.render = original_snapshot, original_render
