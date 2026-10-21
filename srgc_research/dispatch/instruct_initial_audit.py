"""Inspect saved Qwen/Llama initial responses without modifying an experiment."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from .gemma_initial_audit import audit_seed, below, read_object, render

CONFIGS = {
    'qwen35': ('scripts.srgc_qwen35', 'scripts.srgc_qwen35_storage',
               'qwen35-9b', 'qwen35_tokenizer_chat_thinking_off'),
    'llama31': ('srgc_research.dispatch.llama31.adapter', 'srgc_research.dispatch.llama31.storage',
                'llama31-8b', 'llama31_tokenizer_chat'),
}


def audit(model, dataset='all', *, root=None, samples=1, rescore=0):
    if model not in CONFIGS or dataset not in {'all', 'math', 'mbpp'}:
        raise ValueError('unsupported model or dataset')
    if not 0 <= samples <= 20 or not 0 <= rescore <= 100:
        raise ValueError('samples must be 0..20 and rescore must be 0..100')
    adapter_name, storage_name, prefix, prompt_format = CONFIGS[model]
    adapter = importlib.import_module(adapter_name)
    storage = importlib.import_module(storage_name)
    group, _ = storage.group_work(os.environ)
    root = storage.inside(storage.default_root(os.environ) if root is None else root, group)
    storage.validate_tree(root)
    result = {'model': adapter.MODEL, 'revision': adapter.REVISION, 'root': str(root),
              'measurement': 'Saved responses only; format-only rescoring is a separate diagnostic, not a replacement reward.',
              'datasets': [], 'errors': []}
    for name in ('math', 'mbpp') if dataset == 'all' else (dataset,):
        try:
            plan_path = below(root / 'experiments' / f'{prefix}-{name}.json', root)
            plan = adapter.validate_extension(plan_path, read_only=True)
            for seed in plan['seeds']:
                try:
                    data = read_object(below(plan_path.parent / plan['input_pattern'].format(seed=seed), root))
                    adapter.validate_bundle_model(data, plan, seed)
                    row = audit_seed(root, plan_path, plan, name, seed, samples=samples, rescore=rescore,
                                     model=adapter.MODEL, revision=adapter.REVISION,
                                     prompt_formats=(prompt_format,), strict_receipts=True)
                    result['datasets'].append(row)
                    result['errors'].extend(f'{name} seed {seed}: {error}' for error in row['errors'])
                except (OSError, ValueError, KeyError, TypeError) as error:
                    result['errors'].append(f'{name} seed {seed}: {error}')
        except (OSError, ValueError, KeyError, TypeError) as error:
            result['errors'].append(f'{name}: {error}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=tuple(CONFIGS))
    parser.add_argument('dataset', choices=('all', 'math', 'mbpp'), nargs='?', default='all')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--rescore', type=int, default=0)
    parser.add_argument('--output', type=Path, help='also write a new JSON report outside the experiment')
    args = parser.parse_args(argv)
    if not 0 <= args.samples <= 20 or not 0 <= args.rescore <= 100:
        parser.error('samples must be 0..20 and rescore must be 0..100')
    storage = importlib.import_module(CONFIGS[args.model][1])
    root = storage.default_root(os.environ) if args.root is None else args.root
    if args.output and args.output.resolve().is_relative_to(root.resolve()):
        parser.error('the audit output must be outside the experiment directory')
    result = audit(args.model, args.dataset, root=root, samples=args.samples, rescore=args.rescore)
    print(render(result, title=f'{args.model.upper()} INITIAL-POLICY AUDIT'))
    if args.output:
        with args.output.open('x') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    return int(bool(result['errors']))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('INITIAL AUDIT unavailable: ' + str(error), file=sys.stderr)
        raise SystemExit(2) from None
