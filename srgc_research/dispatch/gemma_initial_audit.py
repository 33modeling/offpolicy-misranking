"""Inspect saved initial-policy responses; never generate, train or rewrite a run."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

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
        raise TypeError(f"expected an object: {path}")
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
        result["literal_python_language_line"] = bool(code.splitlines() and code.splitlines()[0].strip() == "python")
        try:
            tree = ast.parse(code)
            result["python_syntax_error"] = False
            result["has_function"] = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree))
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            result.update(python_syntax_error=True, has_function=False)
    return result


def _rescore(record, response, dataset, *, verifier=None):
    if dataset == "math":
        from scripts.srgc_verifier_fallback import tolerant_math_reward
        return tolerant_math_reward(record, response)
    if verifier == 'srgc_research.dispatch.gemma4_text.verifier:code_reward':
        from .gemma4_text.verifier import code_reward
    else:
        from srgc_rebuttal.verifiers import code_reward
    return code_reward(record, response)


def alternate_code_score(record, text):
    """Separate format diagnostic; never replace a recorded experiment reward."""
    from srgc_rebuttal.verifiers import code_reward, extract_code
    lines = extract_code(text).splitlines()
    if not lines or lines[0].strip() != 'python':
        return None
    return code_reward(record, '\n'.join(lines[1:]))


def audit_seed(root, plan_path, plan, dataset, seed, *, samples=1, rescore=0,
               model=MODEL, revision=REVISION,
               prompt_formats=('gemma4_pt_plain_text_single_bos', 'gemma4_pt_solution_single_bos'),
               strict_receipts=False):
    path = below(plan_path.parent / plan["input_pattern"].format(seed=seed), root)
    data = read_object(path)
    provenance = data.get('provenance', {})
    if ((provenance.get('model'), provenance.get('model_revision')) != (model, revision)
            or provenance.get('prompt_format') not in prompt_formats):
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
    if strict_receipts and protocol is not None:
        expected_hash = hashlib.sha256(json.dumps({i: records[i] for i in candidates},
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if protocol.get('candidate_ids') != candidates or protocol.get('records_sha256') != expected_hash:
            raise ValueError('raw cache protocol does not match candidate records')
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
        if strict_receipts:
            from srgc_rebuttal.srgc import stream_seed

            if protocol is None or receipt.get('sampling_seed') != (stream_seed(seed, 0, f'cache:{candidate}') & 0x7FFFFFFFFFFFFFFF):
                raise ValueError('raw response has no matching cache protocol or sampling seed')
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
                    value = float(_rescore(record, text, dataset, verifier=plan.get('verifier')))
                    if not math.isfinite(value) or value not in (0, 1):
                        raise ValueError('verifier returned a nonbinary reward')
                    row['current_reward'] = value
                    if value != reward:
                        errors.append(f'rescore differs: {candidate} response {index}')
                    if dataset == 'mbpp' and value == 0 and traits['literal_python_language_line']:
                        row['diagnostic_score_without_literal_python_line'] = alternate_code_score(record, text)
                except Exception as error:  # noqa: BLE001 - report verifier errors without changing recorded rewards
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


def audit_preflight(root, plan_path, plan, dataset, *, samples=1, rescore=0):
    """Read the recorded small panel separately from full initial-cache accuracy."""
    from .gemma4_text import preflight

    data = read_object(below(plan_path.parent / plan['input_pattern'].format(seed=plan['seeds'][0]), root))
    protocol = preflight.identity(plan_path, plan, data)
    directory = below(preflight.folder(plan_path, plan), root)
    rows, examples, rescored, errors = [], [], [], []
    traits = Counter()
    for candidate in protocol['candidate_ids']:
        path = below(directory / (hashlib.sha256(candidate.encode()).hexdigest() + '.json'), root)
        row = preflight.read_receipt(path, protocol, candidate)
        if row is None:
            continue
        rows.append(row)
        record = data['records'][candidate]
        for index, (text, reward, tokens) in enumerate(zip(row['responses'], row['rewards'], row['token_ids'])):
            traits.update({key: int(value) for key, value in features(text, dataset).items()})
            traits['answer_label_only'] += bool(re.fullmatch(r'\s*Answer\s*:\s*', text, re.IGNORECASE))
            traits['eos_responses'] += 1 in tokens
            traits['length_limit_responses'] += len(tokens) == protocol['max_new_tokens'] and 1 not in tokens
            if len(examples) < samples:
                examples.append({'candidate': candidate, 'response_index': index, 'reward': reward,
                                 'text': text[:500], 'tokens': len(tokens), 'eos': 1 in tokens})
            if len(rescored) < rescore:
                item = {'candidate': candidate, 'response_index': index, 'recorded_reward': reward}
                try:
                    value = float(_rescore(record, text, dataset, verifier=plan.get('verifier')))
                    if not math.isfinite(value) or value not in (0, 1):
                        raise ValueError('verifier returned a nonbinary reward')
                    item['current_reward'] = value
                    if value != reward:
                        errors.append(f'preflight rescore differs: {candidate} response {index}')
                except Exception as error:  # noqa: BLE001 - report verifier errors without changing recorded rewards
                    item['error'] = type(error).__name__ + ': ' + str(error)
                    errors.append(f'preflight rescore error: {candidate} response {index}: ' + item['error'])
                rescored.append(item)
    complete = len(rows) == len(protocol['candidate_ids'])
    summary = preflight.summarize(protocol, rows) if complete else None
    saved = directory / 'summary.json'
    if saved.is_file() and (summary is None or read_object(saved) != summary):
        errors.append('preflight summary differs from raw receipts')
    return {'dataset': dataset, 'directory': str(directory), 'complete': complete,
            'measurement': 'fixed small preflight panel; not full initial-cache accuracy',
            'summary': summary, 'features': dict(traits), 'samples': examples,
            'rescore': rescored, 'errors': errors}


def audit(root, dataset='all', *, samples=1, rescore=0, tokenizer_dir=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Gemma experiment directory is unavailable: ' + str(root))
    result = {'model': MODEL, 'measurement': 'initial training-candidate cache; not held-out evaluation',
              'tokenizer': tokenizer_report(tokenizer_dir), 'datasets': [], 'preflight': [], 'errors': []}
    for name in ('math', 'mbpp') if dataset == 'all' else (dataset,):
        current = root / 'experiments' / f'gemma4-12b-text-{name}.json'
        archived = root / 'experiments' / f'gemma4-12b-pt-{name}.json'
        try:
            if current.is_file() and archived.is_file():
                raise ValueError('ambiguous Gemma protocols in one experiment root')
            plan_path = below(current if current.is_file() else archived, root)
            plan = read_object(plan_path)
            if (plan.get('model'), plan.get('model_revision')) != (MODEL, REVISION):
                raise ValueError('not the recorded Gemma 4 12B PT plan')
            if current.is_file():
                panel = audit_preflight(root, plan_path, plan, name, samples=samples, rescore=rescore)
                result['preflight'].append(panel)
                result['errors'].extend(f"{name}: {error}" for error in panel['errors'])
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


def render(result, *, title='GEMMA INITIAL-POLICY AUDIT'):
    lines = [title, result['measurement'],
             'Dataset  Seed  Cache prompts  Raw prompts  Bundle success  Raw success']
    if 'tokenizer' in result:
        lines.insert(2, 'TOKENIZER ' + json.dumps(result['tokenizer'], ensure_ascii=False))
    for panel in result.get('preflight', []):
        summary = panel['summary']
        detail = (f"{summary['correct']:g}/{summary['responses']} correct; "
                  f"{'PASS' if summary['passed'] else 'PAUSED'}" if summary else 'incomplete')
        lines.append(f"PREFLIGHT {panel['dataset']}: {detail}; not full initial-cache accuracy")
        lines.append('  Raw responses: ' + panel['directory'])
        lines.append('  Observed: ' + json.dumps(panel['features'], ensure_ascii=False))
        lines.extend('  SAMPLE ' + json.dumps(row, ensure_ascii=False) for row in panel['samples'])
        lines.extend('  RESCORE ' + json.dumps(row, ensure_ascii=False) for row in panel['rescore'])
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
        if os.environ.get('SRGC_GEMMA_LEGACY_PROTOCOL') == '1' or os.environ.get('SRGC_GEMMA_TEXT_PROTOCOL') == '0':
            from .gemma4.storage import default_root
        else:
            from .gemma4_text.storage import default_root
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
