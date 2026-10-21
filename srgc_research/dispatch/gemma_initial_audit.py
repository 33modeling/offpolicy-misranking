"""Inspect saved initial-policy responses; never generate, train or rewrite a run."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

MODEL = "google/gemma-4-12B"
REVISION = "023679ed352de9bb66cc873c9009ce3482585c08"
FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
ANSWER = re.compile(r"(?im)^\s*Answer:\s*.+$")


def tokenizer_report(path=None):
    """Hash small metadata/tokenizer files; no weight loading or model downloads."""
    from .gemma4.model import OFFICIAL_FILES, find_snapshot
    try:
        path = Path(path) if path is not None else find_snapshot(os.environ)
        config = read_object(path / 'config.json')
        generation = read_object(path / 'generation_config.json')
        metadata = read_object(path / 'tokenizer_config.json')
        checks = {}
        for name in ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json'):
            size, expected, kind = OFFICIAL_FILES[name]
            actual_size = (path / name).stat().st_size
            digest = hashlib.sha1(f'blob {actual_size}\0'.encode()) if kind == 'blob' else hashlib.sha256()
            with (path / name).open('rb') as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            checks[name] = actual_size == size and digest.hexdigest() == expected
        text = config.get('text_config', {})
        ids = {key: text.get(key) for key in ('bos_token_id', 'eos_token_id', 'pad_token_id')}
        return {'available': True, 'path': str(path), 'pinned_file_matches': checks, 'text_token_ids': ids,
                'expected_token_ids_match': ids == {'bos_token_id': 2, 'eos_token_id': 1, 'pad_token_id': 0},
                'tokenizer_class': metadata.get('tokenizer_class'), 'native_generation_config': generation,
                'note': 'A tokenizer_config byte difference can be formatting only; it is not by itself a tokenizer fault.'}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {'available': False, 'reason': str(error)}


def read_object(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def below(path, root):
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"saved input path escapes the experiment: {path}")
    return path


def features(text, dataset):
    result = {"empty_text": not text.strip(), "answer_marker": bool(ANSWER.search(text)),
              "multiple_answer_markers": len(ANSWER.findall(text)) > 1,
              "boxed_answer": "\\boxed{" in text,
              "prompt_continuation_marker": any(marker in text for marker in
                  ("Solve the following problem", "Write a Python function for the task below", "\nProblem:", "\nQuestion:"))}
    if dataset == "mbpp":
        blocks = FENCE.findall(text)
        result["python_fence"] = bool(blocks)
        code = (blocks[-1] if blocks else text).strip()
        try:
            tree = ast.parse(code)
            result["python_syntax_error"] = False
            result["has_function"] = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree))
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            result.update(python_syntax_error=True, has_function=False)
    return result


def _rescore(record, response, dataset):
    if dataset == "math":
        from srgc_rebuttal.run_experiment import math_reward
        return math_reward(record, response)
    from srgc_rebuttal.verifiers import code_reward
    return code_reward(record, response)


def audit_seed(root, plan_path, plan, dataset, seed, *, samples=1, rescore=0):
    path = below(plan_path.parent / plan["input_pattern"].format(seed=seed), root)
    data = read_object(path)
    provenance = data.get('provenance', {})
    if (provenance.get('model'), provenance.get('model_revision'), provenance.get('prompt_format')) != (MODEL, REVISION, 'gemma4_pt_plain_text_single_bos'):
        raise ValueError('input bundle has a different model or prompt format')
    candidates = data["candidate_ids"]
    if not isinstance(candidates, list) or not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("invalid candidate IDs")
    cache = data.get("cached_rewards", {})
    if not isinstance(cache, dict) or set(cache) - set(candidates):
        raise ValueError("cached rewards include unknown candidate IDs")
    count = plan["responses"]
    if type(count) is not int or count != 8:
        raise ValueError("expected eight initial responses per candidate")
    for values in cache.values():
        if not isinstance(values, list) or len(values) != count or any(type(x) not in (int, float) or x not in (0, 1) for x in values):
            raise ValueError("invalid initial rewards")
    directory = below(path.with_suffix('.cache'), root)
    protocol_path = below(directory / 'protocol.json', root)
    protocol = read_object(protocol_path) if protocol_path.is_file() else None
    if protocol is not None and any(protocol.get(key) != plan.get(key) for key in ("model", "model_revision", "verifier", "responses", "max_new_tokens", "attention")):
        raise ValueError("saved cache protocol differs from the recorded plan")
    if protocol is not None and protocol.get('cache_seed') != seed:
        raise ValueError('saved cache has a different sampling seed')
    records = data["records"]
    metrics = Counter()
    evidence = []
    errors = []
    receipts = 0
    rescored = []
    raw_success = 0
    for candidate in candidates:
        record = records[candidate]
        receipt_path = below(directory / (hashlib.sha256(candidate.encode()).hexdigest() + '.json'), root)
        if not receipt_path.is_file():
            continue
        receipt = read_object(receipt_path)
        rewards, texts = receipt.get('rewards'), receipt.get('responses')
        if receipt.get('id') != candidate or not isinstance(rewards, list) or not isinstance(texts, list) or len(rewards) != count or len(texts) != count or any(type(r) not in (int, float) or r not in (0, 1) for r in rewards) or any(not isinstance(t, str) for t in texts):
            raise ValueError('invalid raw initial response receipt: ' + str(receipt_path))
        receipts += 1
        if candidate in cache and cache[candidate] != rewards:
            errors.append('bundle/receipt reward mismatch: ' + candidate)
        for index, (text, reward) in enumerate(zip(texts, rewards)):
            metrics['responses'] += 1
            raw_success += reward
            traits = features(text, dataset)
            metrics.update({key: int(value) for key, value in traits.items()})
            if len(evidence) < samples:
                evidence.append({'candidate': candidate, 'response_index': index, 'reward': reward,
                                 'prompt_tail': record['prompt'][-350:], 'answer': str(record.get('answer', ''))[:350],
                                 'text_head': text[:500], 'text_tail': text[-500:] if len(text) > 500 else '', 'features': traits})
            if len(rescored) < rescore:
                row = {'candidate': candidate, 'response_index': index, 'recorded_reward': reward}
                try:
                    value = float(_rescore(record, text, dataset))
                    if not math.isfinite(value) or value not in (0, 1):
                        raise ValueError('verifier returned a nonbinary reward')
                    row['current_reward'] = value
                    if value != reward:
                        errors.append(f'rescore differs: {candidate} response {index}')
                except Exception as error:
                    row['error'] = type(error).__name__ + ': ' + str(error)
                    errors.append(f'rescore error: {candidate} response {index}: ' + row['error'])
                rescored.append(row)
    rewards = [value for values in cache.values() for value in values]
    return {'dataset': dataset, 'seed': seed, 'candidate_prompts': len(candidates), 'cached_prompts': len(cache),
            'initial_success_percent': 100 * sum(rewards) / len(rewards) if len(cache) == len(candidates) else None,
            'raw_receipt_prompts': receipts, 'raw_success_percent': 100 * raw_success / metrics['responses'] if metrics['responses'] else None,
            'raw_coverage_complete': receipts == len(candidates), 'features': dict(metrics), 'samples': evidence,
            'rescore': rescored, 'errors': errors, 'cache_protocol': protocol,
            'recorded_sampling': plan.get('sampling'), 'prompt_format': data.get('provenance', {}).get('prompt_format')}


def audit(root, dataset='all', *, samples=1, rescore=0, tokenizer_dir=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Gemma experiment directory is unavailable: ' + str(root))
    result = {'model': MODEL, 'measurement': 'initial training-candidate cache; not held-out evaluation',
              'tokenizer': tokenizer_report(tokenizer_dir), 'datasets': [], 'errors': []}
    for name in ('math', 'mbpp') if dataset == 'all' else (dataset,):
        plan_path = below(root / 'experiments' / f'gemma4-12b-pt-{name}.json', root)
        try:
            plan = read_object(plan_path)
            if (plan.get('model'), plan.get('model_revision')) != (MODEL, REVISION):
                raise ValueError('not the recorded Gemma 4 12B PT plan')
            for seed in plan['seeds']:
                if type(seed) is not int:
                    raise ValueError('invalid seed')
                try:
                    result['datasets'].append(audit_seed(root, plan_path, plan, name, seed, samples=samples, rescore=rescore))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    result['errors'].append(f'{name} seed {seed}: {error}')
        except (OSError, ValueError, KeyError, TypeError) as error:
            result['errors'].append(f'{name}: {error}')
    return result


def render(result):
    lines = ['GEMMA INITIAL-POLICY AUDIT', result['measurement'],
             'Dataset  Seed  Cache prompts  Raw prompts  Bundle success  Raw success']
    if 'tokenizer' in result:
        lines.insert(2, 'TOKENIZER ' + json.dumps(result['tokenizer'], ensure_ascii=False))
    for row in result['datasets']:
        success = '-' if row['initial_success_percent'] is None else f"{row['initial_success_percent']:.3f}%"
        raw_success = '-' if row['raw_success_percent'] is None else f"{row['raw_success_percent']:.3f}%"
        lines.append(f"{row['dataset']:<8} {row['seed']:<5} {row['cached_prompts']}/{row['candidate_prompts']:<12} {row['raw_receipt_prompts']}/{row['candidate_prompts']:<10} {success:<15} {raw_success}")
        total = row['features'].get('responses', 0)
        lines.append('  Observed raw text: ' + (', '.join(f'{name}={value}/{total}' for name, value in row['features'].items() if name != 'responses') if total else 'unavailable; cause cannot be determined from scalar rewards alone'))
        lines.append('  Recorded sampling: ' + json.dumps(row['recorded_sampling'], ensure_ascii=False))
        for sample in row['samples']:
            lines += [f"  SAMPLE {sample['candidate']} response {sample['response_index']} reward={sample['reward']}",
                      '    PROMPT TAIL: ' + repr(sample['prompt_tail']), '    GOLD: ' + repr(sample['answer']),
                      '    RESPONSE HEAD: ' + repr(sample['text_head'])]
            if sample['text_tail']:
                lines.append('    RESPONSE TAIL: ' + repr(sample['text_tail']))
        for entry in row['rescore']:
            lines.append('  RESCORE ' + json.dumps(entry, ensure_ascii=False))
        lines += ['  ERROR ' + error for error in row['errors']]
    lines += ['ERROR ' + error for error in result['errors']]
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('all', 'math', 'mbpp'), nargs='?', default='all')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--tokenizer-dir', type=Path, help='inspect an explicit local tokenizer/model directory; no weights are loaded')
    parser.add_argument('--samples', type=int, default=1, help='raw response excerpts per seed')
    parser.add_argument('--rescore', type=int, default=0, help='recheck this many saved responses per seed with the existing CPU verifier')
    parser.add_argument('--output', type=Path, help='also write a new JSON report, without replacing an existing file')
    args = parser.parse_args(argv)
    if not 0 <= args.samples <= 20 or not 0 <= args.rescore <= 100:
        parser.error('samples must be 0..20 and rescore must be 0..100')
    if args.root is None:
        from .gemma4.storage import default_root
        args.root = default_root(os.environ)
    result = audit(args.root, args.dataset, samples=args.samples, rescore=args.rescore, tokenizer_dir=args.tokenizer_dir)
    print(render(result))
    if args.output:
        if args.output.resolve().is_relative_to(args.root.resolve()):
            parser.error('the audit output must be outside the experiment directory')
        with args.output.open('x') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    return int(bool(result['errors'] or any(row['errors'] for row in result['datasets'])))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('GEMMA AUDIT unavailable: ' + str(error), file=sys.stderr)
        raise SystemExit(2) from None
