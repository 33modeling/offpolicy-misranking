"""Plain terminal readouts of validated research and information measurements."""

import math

from scripts.srgc_time_status import aligned_table


def number(value, scale=1.):
    if value is None:
        return '-'
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('report values must be finite measured numbers or missing')
    return f'{value * scale:.4g}'


def phase_cost(cost, phase=None):
    if not cost:
        return '-'
    value = cost.get('total_gpu_seconds') if phase is None else cost.get('known_gpu_seconds', {}).get(phase + '_gpu_seconds')
    complete = cost.get('complete') is True
    if phase is None and value is None:
        known = cost.get('known_gpu_seconds', {})
        value = math.fsum(known.values()) if known and any(known.values()) else None
    return ('>=' if value is not None and not complete else '') + number(value, 1 / 3600)


def research(report):
    if report.get('scope') == 'n08':
        rows = [[r.get('dataset', '-'), r.get('study', '-'), r.get('left', '-'), r.get('right', '-'),
                 len(r.get('paired_seeds', [])), number(r.get('mean_difference')),
                 number(r.get('training_seed_sample_sd'))] for r in report.get('summary', [])]
        lines = ['N08 PAIRED ANALYSIS', *aligned_table(('Dataset', 'Study', 'Left', 'Right', 'Seeds', 'Mean difference', 'Sample SD'), rows)]
    else:
        rows, diagnostics = [], []
        for row in report.get('rows', []):
            cost, result = row.get('costs', {}), row.get('result', {})
            rows.append([row['dataset'].upper(), row['seed'], row['condition'], number(row.get('reward'), 100),
                         phase_cost(cost, 'selection'), phase_cost(cost, 'training'), phase_cost(cost),
                         result.get('switched_at') if result.get('switched_at') is not None else '-'])
            for point in result.get('rows', []):
                if not isinstance(point, dict):
                    continue
                metrics = [(k, v) for k, v in point.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
                diagnostics.extend([row['dataset'].upper(), row['seed'], row['condition'],
                                    point.get('method', point.get('mode', point.get('name', '-'))), k, number(v)] for k, v in metrics)
        lines = [f"{report.get('scope', 'research').upper()} RESULTS",
                 *aligned_table(('Dataset', 'Seed', 'Condition', 'Final reward %', 'Select GPU-h', 'Train GPU-h', 'Phases GPU-h', 'Switch after'), rows)]
        if diagnostics:
            lines += ['', 'DIAGNOSTIC MEASUREMENTS',
                      *aligned_table(('Dataset', 'Seed', 'Condition', 'Method', 'Metric', 'Value'), diagnostics)]
        for summary in report.get('performance_summary', []):
            count = len(summary['completed_seeds'])
            lines.append(f"{summary['dataset'].upper()} {summary['condition']}: "
                         f"observed mean={number(summary['mean_reward'], 100)}% seeds={count}/5 "
                         + ('COMPLETE' if summary['all_seeds_complete'] else 'PARTIAL'))
        lines += [f"Pending: {len(report.get('pending', []))}",
                  'GPU-hours are recorded costs; phases include their component timers and are not added to them.',
                  'Diagnostics, cache and trajectory costs remain separate; >= is a recorded lower bound.']
    lines.extend(f'ERROR: {error}' for error in report.get('errors', []))
    return '\n'.join(lines)


def information(report):
    coverage = report.get('coverage', {})
    lines = [f"INFORMATION {coverage.get('completed_initial_measurements', 0)}/{coverage.get('planned_initial_measurements', 0)} initial measurements",
             'SAME-STATE INFORMATION (B uses independent responses)']
    rows = [[r['dataset'].upper(), r['seed'], r['stage'], r['method'], r.get('selected_count', '-'),
             number(r.get('selected_success_B'), 100), number(r.get('selected_mixed_B'), 100),
             number(r.get('mixed_enrichment_B'), 100), number(r.get('selected_cosine_B'))]
            for r in report.get('information_summary', [])]
    lines += aligned_table(('Dataset', 'Seed', 'Stage', 'Method', 'Selected', 'Success B %', 'Mixed B %', 'Mixed gain pp', 'Cosine B'), rows)
    rows = [[r['dataset'].upper(), r['seed'], r['stage'], r['method'], r.get('updates', '-'),
             number(r.get('gradient_before_clip_norm', r.get('gradient_norm_mean_before_clip'))),
             number(r.get('update_norm')), number(r.get('probe_ascent_update_cosine'))]
            for r in report.get('batch_updates', [])]
    lines += ['', 'ACTUAL BATCH UPDATES',
              *aligned_table(('Dataset', 'Seed', 'Stage', 'Method', 'Updates', 'Gradient norm', 'Update norm', 'Probe cosine'), rows)]
    rows = [[r['dataset'].upper(), r['seed'], r['stage'], phase_cost(r.get('costs', {}))]
            for r in report.get('measurements', [])]
    lines += ['', 'MEASUREMENT GPU TIME (GPU-hours)', *aligned_table(('Dataset', 'Seed', 'Stage', 'Phases GPU-h'), rows),
              f"Pending: {len(report.get('pending', []))}",
              'These are same-state measurements, not final held-out rewards. - = unmeasured; >= = recorded lower bound.']
    lines.extend(f"ERROR {e['dataset']} seed-{e['seed']} t{e['stage']}: {e['error']}" for e in report.get('errors', []))
    return '\n'.join(lines)


def extra(report):
    rows, transitions = [], []
    for repeat, collection in ((0, report.get('rows', [])), (None, report.get('replicates', []))):
        for row in collection:
            for arm, value in row.items():
                if not isinstance(value, dict) or 'reward_percent' not in value:
                    continue
                lower = '' if value.get('cost_measurement_complete') is True else '>='
                def hours(field, value=value, lower=lower):
                    measured = value.get(field)
                    return '-' if measured is None else lower + number(measured, 1 / 3600)
                rows.append([row['seed'], repeat if repeat is not None else row['replicate'], arm,
                             number(value['reward_percent']), hours('selection_gpu_seconds'), hours('training_gpu_seconds'),
                             value.get('switched_at') if value.get('switched_at') is not None else '-'])
                for move in value.get('transitions') or []:
                    transitions.append([row['seed'], repeat if repeat is not None else row['replicate'],
                                        arm, move['step'], move['to']])
    lines = [f"{report.get('dataset', '').upper()} EXTRA RESULTS ({report.get('scope', 'all')})",
             *aligned_table(('Seed', 'Repeat', 'Arm', 'Reward %', 'Select GPU-h', 'Train GPU-h', 'Switch after'), rows),
             'Repeat 0 is the original stream; other repeats remain separate.',
             '- = unmeasured; >= = incomplete recorded cost. Selection and training are separate from cache/prefix/evaluation.']
    if not rows:
        lines.append('No validated endpoints yet.')
    if transitions:
        lines += ['', 'RECORDED SELECTOR TRANSITIONS',
                  *aligned_table(('Seed', 'Repeat', 'Arm', 'After update', 'Selector'), transitions)]
    pairs = []
    for row in [*report.get('rows', []), *report.get('replicates', [])]:
        if 'switch' in row and 'sr' in row:
            pairs.append([row['seed'], row.get('replicate', 0), number(row['switch']['reward_percent'] - row['sr']['reward_percent'])])
    if pairs:
        lines += ['', 'PAIRED REPEAT REWARD DIFFERENCE',
                  *aligned_table(('Seed', 'Repeat', 'Switch - SR (pp)'), pairs)]
    lines.extend(f'WARNING: {w}' for w in report.get('warnings', []))
    lines.extend(f'ERROR: {e}' for e in report.get('errors', []))
    return '\n'.join(lines)
