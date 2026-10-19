"""Small, copyable terminal tables for measured model outcomes and GPU times."""

import math
import statistics

from scripts.srgc_time_status import aligned_table

ARMS = ("on_policy", "switch", "sr", "random")
NAMES = ("On-policy", "Switch", "SR", "Random")


def cell(value):
    return "-" if value is None else f"{value:.3f}"


def excluded_time(report, seeds):
    """Show known excluded phases, never subtract overlapping session timers.

    A partial ledger supplies a lower bound, not a complete excluded duration.
    Missing phase receipts do not establish a zero. Diagnostic measurements
    remain in their separate ledgers and are never added to primary costs.
    """
    lookup = {(r['seed'], r['arm']): r for r in report['tasks']}

    def measured(seed, arm):
        row = lookup.get((seed, arm), {})
        timing = row.get('timing', {})
        if row.get('status') == 'invalid' or timing.get('record') not in {'COMPLETE', 'PARTIAL'}:
            return '-'
        hours = timing.get('gpu_hours', {})
        fields = ('phases',) if arm in {'cache', 'prefix'} else ('evaluation', 'checkpoint', 'startup')
        values = [hours.get(key) for key in fields]
        if not any(value is not None for value in values):
            return '-'
        value = math.fsum(v for v in values if v is not None)
        lower = timing['record'] == 'PARTIAL' or any(v is None for v in values)
        return ('>=' if lower else '') + cell(value)

    return [[seed, *[measured(seed, arm) for arm in ('cache', 'prefix', *ARMS)]] for seed in seeds]


def render(result):
    lines = [f"MODEL: {result['model']}"]
    for name, report in result['datasets'].items():
        label = name.upper()
        plan = report.get('plan', {})
        seeds = plan.get('seeds', sorted({r['seed'] for r in report['tasks']}))
        lookup = {(r['seed'], r['arm']): r for r in report['tasks'] if r.get('arm') in ARMS}

        def reward(seed, arm, lookup=lookup):
            row = lookup.get((seed, arm), {})
            return row.get('reward_percent') if row.get('status') == 'complete' else None

        def time(seed, arm, component, lookup=lookup):
            row = lookup.get((seed, arm), {})
            timing = row.get('submission_timing', {})
            if row.get('status') == 'invalid' or timing.get('record') == 'INVALID':
                return None
            if component == 'primary' and timing.get('record') != 'COMPLETE':
                return None
            field = {'selection': 'selection_gpu_seconds', 'training': 'training_gpu_seconds'}.get(component)
            # A partial component must not look like the full continuation time.
            if field and timing.get('missing_update_timer_counts', {}).get(field, 0):
                return None
            return timing.get('gpu_hours', {}).get(component)

        def mean_sd(values):
            if not values or any(v is None for v in values):
                return '-', '-'
            return cell(statistics.mean(values)), cell(statistics.stdev(values)) if len(values) > 1 else '-'

        def transition(seed, lookup=lookup):
            row = lookup.get((seed, 'switch'), {})
            value = row.get('switched_at')
            return value if row.get('status') != 'invalid' and value is not None else '-'

        rows = [[seed, *[cell(reward(seed, arm)) for arm in ARMS], transition(seed)] for seed in seeds]
        summaries = [mean_sd([reward(seed, arm) for seed in seeds]) for arm in ARMS]
        rows += [['Mean', *[v[0] for v in summaries], '-'], ['SD', *[v[1] for v in summaries], '-']]
        lines += ['', f'{label} FINAL REWARD (%)',
                  *aligned_table(('Seed', *NAMES, 'Switch after'), rows)]

        basis = report.get('submission_time_basis', {})
        columns = (('random', 'primary'), ('sr', 'primary'), ('on_policy', 'selection'),
                   ('on_policy', 'primary'), ('switch', 'selection'), ('switch', 'primary'))
        rows = [[seed, *[cell(time(seed, arm, component)) for arm, component in columns]] for seed in seeds]
        summaries = [mean_sd([time(seed, arm, component) for seed in seeds]) for arm, component in columns]
        rows += [['Mean', *[v[0] for v in summaries]], ['SD', *[v[1] for v in summaries]]]
        window = f"; updates {basis['first_update']}--{basis['last_update']}" if basis else ''
        lines += ['', f'{label} GPU TIME (GPU-hours; submitted OLMo basis{window})',
                  *aligned_table(('Seed', 'Random total', 'SR total', 'On select', 'On total', 'Switch select', 'Switch total'), rows)]
        lines += ['', f'{label} EXCLUDED GPU TIME (GPU-hours; cache/prefix shared once; other columns = evaluation + checkpoint I/O + startup)',
                  *aligned_table(('Seed', 'Cache', 'Prefix', 'On other', 'Switch other', 'SR other', 'Random other'),
                                 excluded_time(report, seeds))]
        accuracy = report.get('initial_candidate_accuracy', [])
        if accuracy:
            lines += [f'{label} initial candidate success: ' + ' | '.join(
                f"seed {r['seed']} {r['success_percent']:.1f}%" for r in accuracy)]
        for row in report['tasks']:
            error = row.get('submission_timing', {}).get('error')
            if error:
                lines.append(f"TIME ERROR {label} seed-{row['seed']}.{row['arm']}: {error}")
    lines += ['', 'Total = retained-update training + full selection + one common SR subset preparation.',
              'Cache, prefix, evaluation, checkpoints, startup and discarded attempts excluded; all-attempt costs retained separately.',
              'Separate reward-curve evaluation/recording is also excluded. Excluded values are not added to Total; >= is a recorded lower bound.',
              'SD = sample standard deviation across planned seeds; - = unavailable/incomplete.']
    lines.extend(f'ERROR: {error}' for error in result.get('errors', []))
    return '\n'.join(lines)
