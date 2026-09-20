"""One-off, read-only MBPP plus Pair diagnostic without an aggregate upload cap."""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import mbpp_diagnostic_parts as mbpp
import selector_pair_diagnostic as pair


def sections(work, mbpp_roots, pair_root):
    yield ('MBPP + SELECTOR PAIR DIAGNOSTIC\n'
           'READ-ONLY. No workers started/stopped and no run artifacts changed.\n'
           'One TXT, no aggregate output-size cap. Per-record/read/log safety bounds remain.\n'
           'Model, optimizer and rollout payloads are not copied; this is not an atomic snapshot.\n')
    yield '\n========== MBPP: ORIGINAL AND REPAIR ==========\n'
    try:
        yield from mbpp.sections(work, mbpp_roots, single_file=True)
    except (OSError, ValueError, RuntimeError) as exc:
        yield f'INCOMPLETE MBPP DIAGNOSTIC: {type(exc).__name__}: {exc}\n'
    yield '\n========== SELECTOR PAIR ==========\n'
    for report in (pair.collect, pair.queue_report, pair.cost_report):
        try:
            yield report(pair_root, uncapped=True) + '\n'
        except (OSError, ValueError, RuntimeError) as exc:
            yield f'INCOMPLETE PAIR {report.__name__}: {type(exc).__name__}: {exc}\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=Path.home())
    args = parser.parse_args()
    work = Path(os.environ.get('OM_WORK',
                f'/group-volume/{os.environ.get("OM_USER", "minsoo3.kim")}/offpolicy-misranking'))
    source = Path(os.environ.get('MBPP_REPAIR_SOURCE') or os.environ.get('SWITCH_MBPP_QUALITY_ROOT')
                  or str(work / 'runs/selection-switch-mbpp-quality-v1'))
    repair = Path(os.environ.get('MBPP_REPAIR_ROOT',
                  str(work / 'runs/selection-switch-mbpp-quality-repair-v1')))
    pair_root = Path(os.environ.get('PAIR_ROOT', str(work / 'runs/selector-pair-v1')))
    roots = list(dict.fromkeys(root.resolve() for root in (source, repair)))
    try:
        paths = mbpp.write_single(sections(work, roots, pair_root), args.report_dir,
                                 prefix='mbpp-pair-why')
    except OSError as exc:
        print(f'[report-save-failed] {exc}', file=sys.stderr)
        return 2
    print('[single] MBPP + Pair combined in one TXT; no total output-size cap; source files unchanged')
    print(f'[saved] {paths[0]} ({paths[0].stat().st_size} bytes; send only this TXT)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
