"""Actual-weight IT candidate diagnosis using the production cache generator.

The official-sampling control changes only top-k/top-p. Neither arm trains;
this small candidate panel is not full initial accuracy or held-out evaluation.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace


def canary_bundle(dataset):
    """Basic instruction/verifier controls, explicitly separate from benchmark data."""
    record = ({'prompt': 'Compute 17 * 23. Reply with only Answer: NUMBER.', 'answer': '391'}
              if dataset == 'math' else
              {'prompt': 'Return only a Python code block defining add_one(x), which returns x + 1.',
               'answer': 'assert add_one(2) == 3\nassert add_one(-5) == -4\nassert add_one(0) == 1'})
    candidate = f'synthetic-{dataset}-instruction-canary-v1'
    return {'candidate_ids': [candidate], 'records': {candidate: record}}


def generation_options(profile, kwargs, native):
    options = dict(kwargs)
    if profile == 'official':
        for name in ('temperature', 'top_k', 'top_p'):
            options[name] = native[name]
    elif profile == 'greedy':
        options['do_sample'] = False
        for name in ('temperature', 'top_k', 'top_p'):
            options.pop(name, None)
    elif profile != 'comparison':
        raise ValueError('unknown diagnostic sampling profile')
    return options


def trim_response(ids):
    from .protocol import EOS_TOKENS
    stop = next((i for i, token in enumerate(ids) if token in EOS_TOKENS), None)
    return ids if stop is None else ids[:stop + 1]


def summarize(cases):
    from .protocol import EOS_TOKENS, SUPPRESS_TOKENS
    groups = {}
    for case in cases:
        key = f"{case['dataset']}:{case['profile']}"
        value = groups.setdefault(key, {'prompts': 0, 'responses': 0, 'correct': 0,
                                      'empty': 0, 'truncated': 0, 'suppressed_tokens': 0})
        value['prompts'] += 1
        value['responses'] += len(case['rewards'])
        value['correct'] += sum(case['rewards'])
        value['empty'] += sum(not text.strip() for text in case['responses'])
        value['truncated'] += sum(not ids or ids[-1] not in EOS_TOKENS for ids in case['token_ids'])
        value['suppressed_tokens'] += sum(token in SUPPRESS_TOKENS for ids in case['token_ids'] for token in ids)
    for value in groups.values():
        value['success_percent'] = 100 * value['correct'] / value['responses']
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=2048)
    parser.add_argument('--prompts', type=int, default=2)
    parser.add_argument('--responses', type=int, default=1)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--profiles', default='comparison,official')
    parser.add_argument('--dataset', choices=('all', 'math', 'mbpp'), default='all')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--gpu-offload', action='store_true')
    parser.add_argument('--canaries', action='store_true', help='synthetic instruction controls, not benchmark prompts')
    args = parser.parse_args(argv)
    profiles = args.profiles.split(',')
    if (not 1 <= args.prompts <= 8 or not 1 <= args.tokens <= 2048
            or not 1 <= args.responses <= 8 or args.threads < 1 or args.seed < 0
            or not profiles or len(set(profiles)) != len(profiles)
            or any(profile not in {'comparison', 'official', 'greedy'} for profile in profiles)):
        parser.error('invalid panel size, seed, threads or sampling profiles')
    if args.output.exists():
        parser.error('output already exists; preserve the earlier diagnostic and choose a new path')
    from .runtime import disable_optional_media
    disable_optional_media()
    import torch
    import transformers
    from .adapter import MODEL, REVISION, REPO, adapter_digest, engine_digest
    from .model import load_text_model, resolve_snapshot
    from .protocol import sampling
    from .verifier import code_reward
    from scripts.srgc_verifier_fallback import tolerant_math_reward
    from srgc_rebuttal.build_cache import candidate_seed, generate_rewards
    from srgc_rebuttal.runtime import atomic_json

    torch.set_num_threads(args.threads)
    report = {'schema': 'gemma4-it-candidate-diagnostic-v2', 'model': MODEL,
              'revision': REVISION, 'torch': torch.__version__, 'transformers': transformers.__version__,
              'dtype': 'bfloat16', 'attention': 'eager', 'thinking': 'off',
              'max_new_tokens': args.tokens, 'requested_responses': args.responses,
              'seed': args.seed, 'profiles': profiles, 'training_updates': 0,
              'cpu_threads': args.threads,
              'panel_kind': 'synthetic_canaries' if args.canaries else 'fixed_candidates',
              'scope': ('synthetic instruction/verifier controls; not benchmark accuracy' if args.canaries else
                        'fixed small candidate panel; not full initial-cache accuracy or held-out evaluation'),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
              'adapter_sha256': adapter_digest(), 'engine_sha256': engine_digest(),
              'cases': [], 'summary': {}, 'complete': False}
    atomic_json(args.output, report)
    try:
        print('Verifying actual IT checkpoint', flush=True)
        path, identity = resolve_snapshot({**os.environ, 'SRGC_GEMMA_IT_MODEL_PATH': str(args.snapshot)})
        report['snapshot_sha256'] = identity
        native = json.loads((path / 'generation_config.json').read_text())
        report['comparison_sampling'] = sampling()
        report['official_generation_config'] = native
        atomic_json(args.output, report)
        model, tokenizer = load_text_model(path, 'cpu', dtype=torch.bfloat16)
        device = 'cpu'
        if args.gpu_offload:
            from accelerate import cpu_offload
            device = 'cuda'
            cpu_offload(model, execution_device=torch.device(device), offload_buffers=False)
            report['device'] = torch.cuda.get_device_name()
            report['weight_placement'] = 'CPU offload; GPU execution'
        else:
            report['device'] = 'cpu'
        original = model.generate
        captured = []
        active = None

        def capture(*positional, **kwargs):
            kwargs = generation_options(active, kwargs, native)
            result = original(*positional, **kwargs)
            start = kwargs['input_ids'].shape[-1]
            captured.extend(trim_response(sequence[start:].tolist()) for sequence in result)
            return result

        # Accelerate parks parameters on meta; production's input routing must
        # use the actual execution device, never infer it from parked weights.
        generator = SimpleNamespace(device=torch.device(device), generate=capture)
        atomic_json(args.output, report)
        for dataset, filename in (('math', 'seed-5.json'), ('mbpp', 'mbpp-seed-5.json')):
            if args.dataset not in {'all', dataset}:
                continue
            bundle_path = REPO / 'srgc_rebuttal/inputs' / filename
            bundle_bytes = (json.dumps(canary_bundle(dataset), sort_keys=True).encode()
                            if args.canaries else bundle_path.read_bytes())
            bundle = json.loads(bundle_bytes)
            for candidate in bundle['candidate_ids'][:args.prompts]:
                record = dict(bundle['records'][candidate])
                source_prompt = record['prompt']
                record['prompt'] = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': source_prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                tokenizer(record['prompt'])
                for profile in profiles:
                    active = profile
                    captured.clear()
                    count = 1 if profile == 'greedy' else args.responses
                    print(f'Generating {dataset} {candidate} {profile} n={count}', flush=True)
                    started = time.perf_counter()
                    rewards, texts = generate_rewards(generator, tokenizer, record,
                        tolerant_math_reward if dataset == 'math' else code_reward,
                        responses=count, seed=candidate_seed(args.seed, candidate),
                        max_new_tokens=args.tokens, candidate=candidate)
                    if len(captured) != count:
                        raise ValueError('raw token receipts do not cover every generated response')
                    row = {'dataset': dataset, 'candidate': candidate, 'profile': profile,
                           'sampling_seed': candidate_seed(args.seed, candidate),
                           'source_bundle': None if args.canaries else str(bundle_path.relative_to(REPO)),
                           'source_definition': 'canary_bundle:v1' if args.canaries else None,
                           'source_bundle_sha256': hashlib.sha256(bundle_bytes).hexdigest(),
                           'source_prompt': source_prompt, 'prompt': record['prompt'],
                           'gold': record['answer'], 'responses': texts, 'rewards': rewards,
                           'token_ids': list(captured), 'seconds': time.perf_counter() - started}
                    report['cases'].append(row)
                    report['summary'] = summarize(report['cases'])
                    atomic_json(args.output, report)
                    print(json.dumps({'dataset': dataset, 'candidate': candidate, 'profile': profile,
                                      'rewards': rewards, 'tokens': list(map(len, captured)),
                                      'seconds': row['seconds']}), flush=True)
        report['complete'] = True
        atomic_json(args.output, report)
        print(json.dumps(report['summary'], sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        report['summary'] = summarize(report['cases'])
        atomic_json(args.output, report)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
