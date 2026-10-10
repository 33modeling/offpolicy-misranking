"""Four real CPU ranks for Llama/Gemma LoRA; no claim of H100 measurement."""

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model

from srgc_rebuttal.distributed import primary


def forced_rollout(prompt_id, responses, seed):
    offset = int(prompt_id.removeprefix('p')) % 4
    sequences = [torch.tensor([2,4,5,8+offset,9]) if i % 2 else torch.tensor([2,4,5,12+offset,13,14]) for i in range(responses)]
    return sequences, np.array([0.0,1.0] * (responses//2)), 3


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def __call__(self, *args, **kwargs):
        return {'input_ids': torch.tensor([[2,4,5]]), 'attention_mask': torch.ones((1,3), dtype=torch.long)}
    def decode(self, tokens, **kwargs):
        return ' '.join(map(str,tokens.tolist()))


def backend_for(family, dtype):
    torch.manual_seed(7)
    if family == 'gemma':
        from transformers import Gemma4UnifiedForCausalLM

        from srgc_research.dispatch.gemma4 import adapter
        from srgc_research.dispatch.gemma4.memory import GemmaBackend as Backend
        from srgc_research.dispatch.gemma4.memory import bounded_generate
        from srgc_research.tests.test_gemma4_model import tiny_config
        model = Gemma4UnifiedForCausalLM(tiny_config()).to(dtype)
    else:
        from transformers import LlamaConfig, LlamaForCausalLM

        from srgc_research.dispatch.llama31 import adapter
        from srgc_research.dispatch.llama31.memory import LlamaBackend as Backend
        from srgc_research.dispatch.llama31.memory import bounded_generate
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=1,
            bos_token_id=2, eos_token_id=1, pad_token_id=0, attention_dropout=0.0)).to(dtype)
    policy = adapter.attach_adapter(model, LoraConfig(r=2,lora_alpha=4,lora_dropout=0.0,task_type='CAUSAL_LM'),get_peft_model)
    model.generate = bounded_generate(model.generate, torch.device('cpu'))
    return Backend(policy, Tokenizer(), {f'p{i}':{'prompt':f'problem {i}','answer':'9'} for i in range(4)},
                   lambda record,text:float('9' in text.split()),projection_dim=16,max_new_tokens=3,logit_chunk_tokens=2)


def digest(value):
    sha=hashlib.sha256()
    def walk(item):
        if isinstance(item,torch.Tensor):
            sha.update(str((item.dtype,tuple(item.shape))).encode())
            sha.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item,dict):
            for key in sorted(item,key=str):
                sha.update(str(key).encode()); walk(item[key])
        elif isinstance(item,(list,tuple)):
            for part in item: walk(part)
        else: sha.update(repr(item).encode())
    walk(value)
    return sha.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', choices=('gemma','llama'),required=True)
    parser.add_argument('--dtype',choices=('fp32','bf16'),default='fp32')
    args=parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    folder=Path(primary(lambda:tempfile.mkdtemp(prefix=f'srgc-{args.model}-four-rank-')))
    try:
        backend=backend_for(args.model,torch.bfloat16 if args.dtype=='bf16' else torch.float32)
        ids=list(backend.records)
        first,rewards,_=backend._rollout(f'p{dist.get_rank()}',8,17)
        second,other,_=backend._rollout(f'p{dist.get_rank()}',8,17)
        assert len(first)==8 and all(torch.equal(a,b) for a,b in zip(first,second))
        np.testing.assert_array_equal(rewards,other)
        backend._rollout=forced_rollout
        with backend.cost_meter.phase('selection'):
            gradients=backend.score_gradients(ids,responses=8,group_size=4,seed=17)
        assert set(gradients)==set(ids) and all(np.isfinite(g).all() and np.linalg.norm(g)>0 for g in gradients.values())
        with backend.cost_meter.phase('training'):
            backend.train(ids,responses=8,objective='grpo',seed=17)
        first_state=backend.state_dict()
        primary(lambda:torch.save(first_state,folder/'checkpoint.pt'))
        replicas=[None]*4
        dist.all_gather_object(replicas,digest(first_state))
        assert len(set(replicas))==1
        backend.train(ids,responses=8,objective='grpo',seed=19)
        endpoint=backend.state_dict()
        checkpoint=primary(lambda:torch.load(folder/'checkpoint.pt',weights_only=False))
        backend.load_state_dict(checkpoint)
        backend.train(ids,responses=8,objective='grpo',seed=19)
        assert digest(backend.state_dict())==digest(endpoint)
        serial=backend_for(args.model,torch.bfloat16 if args.dtype=='bf16' else torch.float32)
        serial.rank,serial.world=0,1
        serial._rollout=forced_rollout
        for seed in (17,19): serial.train(ids,responses=8,objective='grpo',seed=seed)
        for name,value in serial.state_dict()['trainable'].items():
            torch.testing.assert_close(value,endpoint['trainable'][name],rtol=1e-5,atol=1e-7)
        with backend.cost_meter.phase('evaluation'):
            evaluations=backend.evaluate(ids,responses=8,seed=23)
        assert all(value==0.5 for value in evaluations.values())
        primary(lambda:print(f'PASS: {args.model} {args.dtype}, 4 ranks, 8 responses, native scoring, GRPO, exact disk resume, replica and serial update agreement',flush=True))
    finally:
        primary(lambda:shutil.rmtree(folder))
        dist.destroy_process_group()


if __name__=='__main__':
    main()
