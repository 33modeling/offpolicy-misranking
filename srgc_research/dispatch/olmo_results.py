"""Render existing OLMo results as copyable tables without changing submitted data."""

import argparse
import json
from pathlib import Path

from scripts import srgc_live_status as live
from scripts.srgc_run_history import include_previous_runs
from srgc_rebuttal import reports
from srgc_rebuttal.plan import DEFAULT_PLAN, load_plan

from .model_result_text import render
from .submission_time import attach


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('results', 'costs'), nargs='?', default='results')
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    with include_previous_runs():
        report = live.snapshot(reports.snapshot, args.plan, include_costs=True)
    attach(report, args.plan, plan)
    dataset = 'math' if plan['dataset'] == 'math_train' else 'mbpp'
    report['plan'] = plan
    if args.action == 'results':
        destination = args.output or Path.home() / f'srgc-rebuttal-{dataset}-results.txt'
        reports.export(report, destination)
    print(json.dumps(report, indent=2, allow_nan=False) if args.json else render({
        'model': plan['model'], 'datasets': {dataset: report}, 'errors': report['errors']}), flush=True)
    if args.action == 'results' and not args.json:
        print(f'\nJSON: {destination.with_suffix(".json")}', flush=True)
    return int(bool(report['errors']))
