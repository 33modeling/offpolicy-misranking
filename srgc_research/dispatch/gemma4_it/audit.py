"""Read the IT cache's actual rewards and response receipts without GPU work."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from .. import gemma_initial_audit as shared
from . import preflight
from .adapter import MODEL, REVISION, validate_extension
from .protocol import EOS_TOKENS, PROMPT_FORMAT
from .storage import default_root
from .verifier import code_reward


def audit(root, dataset='all', *, samples=1, rescore=0):
    root = Path(root).resolve()
    result = {'model': MODEL, 'measurement': 'IT initial candidate cache; not held-out evaluation',
              'datasets': [], 'preflight': [], 'errors': []}
    for name in ('math', 'mbpp') if dataset == 'all' else (dataset,):
        path = root / 'experiments' / f'gemma4-12b-it-{name}.json'
        try:
            plan = validate_extension(path, read_only=True)
            original = shared._rescore
            with patch.object(shared, '_rescore', lambda record, response, dataset, **kwargs:
                              code_reward(record, response) if dataset == 'mbpp'
                              else original(record, response, dataset, **kwargs)):
                panel = shared.audit_preflight(root, path, plan, name,
                    samples=samples, rescore=rescore, preflight_module=preflight,
                    eos_tokens=EOS_TOKENS)
                result['preflight'].append(panel)
                result['errors'].extend(f"{name}: {error}" for error in panel['errors'])
                for seed in plan['seeds']:
                    result['datasets'].append(shared.audit_seed(root, path, plan, name, seed,
                        samples=samples, rescore=rescore, model=MODEL, revision=REVISION,
                        prompt_formats=(PROMPT_FORMAT,), strict_receipts=True))
        except (OSError, ValueError, KeyError, TypeError) as error:
            result['errors'].append(f'{name}: {error}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='?', choices=('all', 'math', 'mbpp'), default='all')
    parser.add_argument('--root', type=Path, default=default_root())
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--rescore', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.samples <= 20 or not 0 <= args.rescore <= 100:
        parser.error('samples must be 0..20 and rescore 0..100')
    result = audit(args.root, args.dataset, samples=args.samples, rescore=args.rescore)
    print(shared.render(result, title='GEMMA IT INITIAL-POLICY AUDIT'))
    if args.output:
        if args.output.resolve().is_relative_to(args.root.resolve()):
            parser.error('audit output must be outside the experiment directory')
        with args.output.open('x') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    return int(bool(result['errors'] or any(row['errors'] for row in result['datasets'])))


if __name__ == '__main__':
    raise SystemExit(main())
