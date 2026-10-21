"""Compare the actual pinned Gemma 4 12B checkpoint with its experiment loader.

Run native and adapter modes in separate processes to release the full model.
This is a loading/tokenization diagnostic, not an estimate of init success.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .gemma4.adapter import MODEL, REVISION


def probe_cases(repo):
    from srgc_rebuttal.build_inputs import MATH_PROMPT
    result = [
        {'id': 'plain-completion', 'prompt': '2 + 3 =', 'answer': '5', 'generate': True},
        {'id': 'rlzero-simple-math', 'prompt': MATH_PROMPT.format(question='What is 2 + 3?'), 'answer': '5', 'generate': True},
    ]
    for dataset, filename in [('math', 'seed-5.json'), ('mbpp', 'mbpp-seed-5.json')]:
        bundle = json.loads((repo / 'srgc_rebuttal/inputs' / filename).read_text())
        candidate = bundle['candidate_ids'][1]
        result.append({'id': candidate, **bundle['records'][candidate], 'dataset': dataset, 'generate': False})
        if dataset == 'math':
            question = bundle['records'][candidate]['question']
            result.extend([
                {'id': 'real-math-qa-completion', 'prompt': 'Question: ' + question + '\nAnswer:', 'answer': bundle['records'][candidate]['answer'], 'generate': True},
                {'id': 'real-math-two-shot-completion', 'prompt': 'Question: What is 2 + 3?\nAnswer: 5\n\nQuestion: Solve 3x = 12.\nAnswer: 4\n\nQuestion: ' + question + '\nAnswer:', 'answer': bundle['records'][candidate]['answer'], 'generate': True},
            ])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('native', 'adapter', 'corrected'), required=True)
    parser.add_argument('--sample', action='store_true', help='temperature=1, top-p=1, top-k=0 sampling; still only a short diagnostic')
    parser.add_argument('--responses', type=int, default=1)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--gpu-offload', action='store_true', help='stage BF16 weights in CPU memory and execute on CUDA with Accelerate')
    parser.add_argument('--case', action='append', help='restrict to exact diagnostic case IDs')
    parser.add_argument('--generate-real', action='store_true', help='also generate on selected real candidate prompts; still not an init accuracy estimate')
    parser.add_argument('--official-suppression', action='store_true', help='restore only the two official modality-token suppressions in this separate diagnostic')
    args = parser.parse_args(argv)
    if not 1 <= args.tokens <= 128 or not 1 <= args.threads <= 16:
        parser.error('tokens must be 1..128 and threads 1..16')
    if not 1 <= args.responses <= 8 or (args.responses != 1 and not args.sample):
        parser.error('responses must be 1..8; multiple responses require --sample')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from .gemma4.runtime import disable_optional_media
    disable_optional_media()
    import torch
    import transformers
    from transformers import (
        AutoTokenizer,
        Gemma4UnifiedForCausalLM,
        Gemma4UnifiedForConditionalGeneration,
    )

    from .gemma4.model import OFFICIAL_FILES, PromptTokenizer, load_text_model
    if args.mode == 'corrected':
        from .gemma4_text.model import load_text_model

    torch.set_num_threads(args.threads)
    provenance = json.loads((args.snapshot / 'download-provenance.json').read_text())
    if (provenance['model'], provenance['revision']) != (MODEL, REVISION):
        raise ValueError('not the experiment Gemma 4 12B checkpoint')
    weight = next(row for row in provenance['files'] if row['name'] == 'model.safetensors')
    if weight['sha256'] != OFFICIAL_FILES['model.safetensors'][1] or (args.snapshot / 'model.safetensors').stat().st_size != OFFICIAL_FILES['model.safetensors'][0]:
        raise ValueError('actual checkpoint identity differs')
    if args.gpu_offload and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for --gpu-offload')
    dtype = torch.bfloat16 if args.gpu_offload else torch.float32
    report = {'model': MODEL, 'revision': REVISION, 'mode': args.mode, 'device': 'cuda with CPU weight offload' if args.gpu_offload else 'cpu',
              'dtype': str(dtype), 'torch': torch.__version__, 'transformers': transformers.__version__,
              'measurement': 'actual pretrained checkpoint loading and short diagnostic output; not init accuracy',
              'sampled': args.sample, 'responses': args.responses, 'max_new_tokens': args.tokens,
              'cases': []}
    report['official_generation_config'] = json.loads((args.snapshot / 'generation_config.json').read_text())
    report['probe_generation_suppress_tokens'] = report['official_generation_config'].get('suppress_tokens') if args.official_suppression or args.mode == 'corrected' else None
    print('LOADING ' + json.dumps(report), flush=True)
    start = time.monotonic()
    if args.mode in {'adapter', 'corrected'}:
        if args.gpu_offload:
            from unittest.mock import patch
            original = Gemma4UnifiedForCausalLM.from_pretrained
            def bf16_cpu_stage(*positional, **keywords):
                return original(*positional, **{**keywords, 'dtype': torch.bfloat16})
            # The frozen CPU loader chooses FP32. Override only the staging
            # dtype to match its actual CUDA BF16 path, before attaching hooks.
            with patch.object(Gemma4UnifiedForCausalLM, 'from_pretrained', side_effect=bf16_cpu_stage):
                model, tokenizer = load_text_model(args.snapshot, 'cpu', attention='eager')
            report['probe_staging_override'] = 'CPU BF16 instead of CPU FP32; matches frozen CUDA weight dtype'
        else:
            model, tokenizer = load_text_model(args.snapshot, 'cpu', attention='eager')
    else:
        model, info = Gemma4UnifiedForConditionalGeneration.from_pretrained(args.snapshot, local_files_only=True,
            dtype=dtype, attn_implementation='eager', output_loading_info=True)
        if any(info.get(key) for key in ('missing_keys', 'mismatched_keys', 'unexpected_keys', 'error_msgs')):
            raise RuntimeError('native pretrained weights failed to load: ' + str(info))
        report['loading_info'] = {key: value for key, value in info.items() if value}
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    report['load_seconds'] = time.monotonic() - start
    report['parameters'] = sum(p.numel() for p in model.parameters())
    if args.gpu_offload:
        from accelerate import cpu_offload
        cpu_offload(model, execution_device=torch.device('cuda'), offload_buffers=False)
        report['gpu'] = torch.cuda.get_device_name()
    print('LOADED ' + str(report['parameters']), flush=True)
    native_tokenizer = tokenizer.tokenizer if isinstance(tokenizer, PromptTokenizer) else tokenizer
    wrapped = PromptTokenizer(native_tokenizer)
    repo = Path(__file__).resolve().parents[2]
    selected = probe_cases(repo)
    if args.case:
        if set(args.case) - {case['id'] for case in selected}:
            parser.error('unknown diagnostic case')
        selected = [case for case in selected if case['id'] in args.case]
    for case in selected:
        started = time.monotonic()
        native_ids = native_tokenizer.encode(case['prompt'], add_special_tokens=True)
        adapter_ids = wrapped.encode(case['prompt'], add_special_tokens=True)
        inputs = tokenizer(case['prompt'], return_tensors='pt', add_special_tokens=True)
        if args.gpu_offload:
            inputs = {key: value.to('cuda') for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
        if not torch.isfinite(logits).all():
            raise RuntimeError('nonfinite pretrained logits')
        torch.save(logits, args.output_dir / (case['id'] + '.pt'))
        values, ids = logits.topk(10)
        row = {'id': case['id'], 'prompt': case['prompt'], 'answer': case['answer'],
               'native_prompt_token_ids': native_ids, 'adapter_prompt_token_ids': adapter_ids,
               'prompt_tokens_equal': native_ids == adapter_ids,
               'top_tokens': [{'id': int(i), 'logit': float(v), 'text': native_tokenizer.decode([int(i)])} for i, v in zip(ids, values)]}
        probabilities = logits.softmax(-1)
        suppressed = report['official_generation_config'].get('suppress_tokens', [])
        row['official_suppressed_token_probabilities'] = {str(i): probabilities[i].item() for i in suppressed}
        row['official_suppressed_probability_mass'] = sum(row['official_suppressed_token_probabilities'].values())
        print('PREFILL ' + json.dumps(row, ensure_ascii=False), flush=True)
        if case['generate'] or args.generate_real:
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=args.sample, num_return_sequences=args.responses,
                    temperature=1.0, max_new_tokens=args.tokens,
                    pad_token_id=0, eos_token_id=1, logits_to_keep=1, use_cache=True,
                    suppress_tokens=report['probe_generation_suppress_tokens'], top_p=1.0, top_k=0)
            suffix = output[0, inputs['input_ids'].shape[1]:].tolist()
            row.update(generated_token_ids=suffix, text=native_tokenizer.decode(suffix, skip_special_tokens=True),
                       stopped_at_eos=1 in suffix, reached_short_probe_limit=len(suffix) == args.tokens and 1 not in suffix)
            if args.responses > 1:
                suffixes = []
                for sequence in output:
                    ids = sequence[inputs['input_ids'].shape[1]:].tolist()
                    if 1 in ids:
                        ids = ids[:ids.index(1) + 1]
                    suffixes.append(ids)
                row['generated_response_token_ids'] = suffixes
                row['response_texts'] = [native_tokenizer.decode(ids, skip_special_tokens=True) for ids in suffixes]
            if args.mode == 'native' and not args.official_suppression and any(token in suppressed for token in suffix):
                with torch.inference_mode():
                    filtered = model.generate(**inputs, do_sample=False, max_new_tokens=args.tokens,
                        pad_token_id=0, eos_token_id=1, logits_to_keep=1, use_cache=True,
                        suppress_tokens=suppressed, top_p=1.0, top_k=0)
                filtered_suffix = filtered[0, inputs['input_ids'].shape[1]:].tolist()
                row['diagnostic_official_suppression'] = {'token_ids': filtered_suffix,
                    'text': native_tokenizer.decode(filtered_suffix, skip_special_tokens=True),
                    'stopped_at_eos': 1 in filtered_suffix}
        row['seconds'] = time.monotonic() - started
        report['cases'].append(row)
        (args.output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print('CASE ' + json.dumps(row, ensure_ascii=False), flush=True)
    report['complete'] = True
    (args.output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
