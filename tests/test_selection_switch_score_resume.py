"""CPU-only regression for saved candidate gradients and completed shard reuse."""
import sys
from types import SimpleNamespace

import torch

import grads
import net_gate_memory_worker as memory
import rollout
import selection_gate as core
import selection_gate_gpu as base
import selection_switch_score as scoring


def test_candidate_reuses_saved_prompt_and_completed_shard(tmp_path, monkeypatch):
    parent = tmp_path / 'parent'
    parent.mkdir()
    (parent / 'adapter_model.safetensors').write_bytes(b'fake adapter, never loaded')
    prompts = tmp_path / 'prompts.json'
    core.atomic_json(prompts, {'train': [{}] * 8, 'val': [{}] * 8})
    root = tmp_path / 'fresh-r'
    core.atomic_json(root / 'scoring.json', {
        'config': {'model': 'fake', 'fresh_k': 32, 'micro_group': 4, 'behavior_k': 8,
                   'val_k': 4, 'max_new_tokens': 4, 'temperature': .7, 'grad_layers': 1, 'proj_dim': 2},
        'parent': str(parent), 'adapter_sha256': base.digest(parent / 'adapter_model.safetensors'),
        'prompts': str(prompts), 'prompts_sha256': base.digest(prompts), 'sampling_seed': 123})
    rows = root / 'candidate-0.jsonl'
    rows.write_text('saved rollout bytes; fake reader below\n')
    core.atomic_json(root / 'direction.json', {'direction': [1., 0.]})
    binding = {'contract_sha256': base.digest(root / 'scoring.json'), 'stage': 'candidate', 'shard': 0}
    saved = root / 'candidate/prompt-0.json'
    core.atomic_json(saved, {'binding': {**binding, 'rollouts_sha256': base.digest(rows)}, 'value': .25})
    before = saved.read_bytes(), rows.read_bytes()
    groups = {i: [{'prompt_idx': i, 'rollout_idx': j, 'reward': float(j % 2),
                   'input_ids': torch.tensor([1, 2]), 'resp_start': 1} for j in range(8)] for i in (0, 1)}
    monkeypatch.setitem(sys.modules, 'experiment', SimpleNamespace(read_rollouts=lambda _: groups))
    monkeypatch.setitem(sys.modules, 'evidence_downstream', SimpleNamespace(reward_rows=lambda *a: None))
    loads, computed = [], []
    def load(*a):
        loads.append(1)
        return object(), object()
    def gradient(model, params, chunk, weights, spec, **kw):
        computed.append(chunk[0]['prompt_idx'])
        return torch.tensor([1., 0.])
    monkeypatch.setattr(rollout, 'load_policy', load)
    monkeypatch.setattr(rollout, 'collect_rollouts', lambda *a, **kw: None)
    monkeypatch.setattr(grads, 'grad_params', lambda *a: [])
    monkeypatch.setattr(grads, 'prompt_gradient', gradient)
    monkeypatch.setattr(memory, 'checkpoint_decoder_layers', lambda *a: None)
    scoring.worker(root, 'candidate', 0)
    assert computed == [1, 1], 'only the unsaved prompt gets its two LOO4 gradients'
    assert core.read(root / 'candidate-0.json')['0'] == .25
    assert (saved.read_bytes(), rows.read_bytes()) == before
    scoring.worker(root, 'candidate', 0)
    assert loads == [1] and computed == [1, 1], 'completed shard needs no model or gradient work'
