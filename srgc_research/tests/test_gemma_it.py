"""Official IT token semantics, immutable cohorts, cache-first and fresh reset."""

import copy
import fcntl
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import GenerationConfig, Gemma4UnifiedForCausalLM

from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.runtime import atomic_json
from srgc_research.dispatch import gemma_it_run as run
from srgc_research.dispatch.gemma4_it import adapter, cli, entry, model, preflight
from srgc_research.dispatch.gemma4_it.protocol import EOS_TOKENS, completion_cue, prompt_prefix, sampling
from srgc_research.dispatch.gemma4_it.reset import session
from srgc_research.tests.test_gemma4_model import tiny_config

REPO = Path(__file__).resolve().parents[2]


class ChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return prompt_prefix('math') + messages[0]['content'].strip() + '<turn|>\n' + completion_cue('math')


@pytest.mark.parametrize('dataset,source', [('math', 'additional_seeds.json'), ('mbpp', 'mbpp_seeds.json')])
def test_new_cohort_drops_source_rewards_and_uses_it_for_every_split(tmp_path, dataset, source):
    source_path = REPO / 'srgc_rebuttal/experiments' / source
    path = adapter.prepare(dataset, source_path, tmp_path, ChatTokenizer(), snapshot_sha256='a' * 64)
    plan = adapter.validate_extension(path)
    assert plan['model'] == 'google/gemma-4-12B-it'
    assert plan['model_initialization'] == 'instruction_tuned' and plan['thinking'] == 'off'
    assert plan['sampling']['eos_token_id'] == [1, 106, 50]
    assert plan['responses'] == 8 and plan['max_new_tokens'] == 2048
    source_plan = load_plan(source_path)
    for seed in plan['seeds']:
        data = json.loads(input_path(path, plan, seed).read_text())
        original = json.loads(input_path(source_path, source_plan, seed).read_text())
        assert data['cached_rewards'] == {}
        for split in ('candidate_ids', 'ranking_validation_ids', 'evaluation_ids'):
            assert data[split] == original[split]
        for key, record in data['records'].items():
            assert record['answer'] == original['records'][key]['answer']
        adapter.validate_bundle_model(data, plan, seed)
        broken = copy.deepcopy(data)
        broken['records'][broken['evaluation_ids'][0]]['prompt'] = '<bos>' + broken['records'][broken['evaluation_ids'][0]]['prompt']
        with pytest.raises(ValueError, match='chat template'):
            adapter.validate_bundle_model(broken, plan, seed)
        with pytest.raises(ValueError, match='pinned Gemma protocol'):
            broken = copy.deepcopy(data)
            broken['cached_rewards'] = {data['candidate_ids'][0]: [1] * 8}
            adapter.validate_bundle_model(broken, plan, seed)
    assert adapter.prepare(dataset, source_path, tmp_path, ChatTokenizer(), snapshot_sha256='a' * 64) == path


class Tokenizer:
    bos_token_id, eos_token_id, pad_token_id = 2, 1, 0
    def encode(self, text, **kwargs):
        assert kwargs['add_special_tokens'] is False
        return [2, 105, 9] if text == 'rendered' else [9]


def test_no_double_bos_and_all_it_stops_reach_cache_and_training():
    tokenizer = model.PromptTokenizer(Tokenizer())
    assert tokenizer.eos_token_id == [1, 106, 50]
    assert tokenizer('rendered')['input_ids'].tolist() == [[2, 105, 9]]
    with pytest.raises(ValueError, match='exactly one'):
        tokenizer('plain')


@pytest.mark.parametrize('stop', EOS_TOKENS)
def test_real_hf_generation_stops_on_each_official_it_end_token(stop):
    torch.set_num_threads(1)
    config = tiny_config()
    config.vocab_size = 128
    policy = Gemma4UnifiedForCausalLM(config).eval()
    policy.generation_config = GenerationConfig(bos_token_id=2, eos_token_id=list(EOS_TOKENS), pad_token_id=0)
    from transformers import LogitsProcessor, LogitsProcessorList
    class Force(LogitsProcessor):
        def __call__(self, ids, logits):
            logits.fill_(-torch.inf)
            logits[:, stop] = 0
            return logits
    ids = torch.tensor([[2, 9]])
    with torch.inference_mode():
        result = policy.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            do_sample=False, max_new_tokens=4, logits_processor=LogitsProcessorList([Force()]))
    assert result.tolist() == [[2, 9, stop]]


def test_reset_archives_cache_rollouts_checkpoints_and_leaves_clean_root(tmp_path, monkeypatch):
    monkeypatch.setenv('GROUP_VOLUME', str(tmp_path))
    monkeypatch.setenv('OM_WORK', str(tmp_path / 'work'))
    root = tmp_path / 'work/gemma-it'
    atomic_json(root / 'experiments/gemma4-12b-it-math.json', {'model': adapter.MODEL})
    for name in ('inputs/math-seed-5.json', 'inputs/math-seed-5.cache/responses.json',
                 'runs/math/seed-5/random-latest.pt', 'preflight/math/summary.json'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'previous experiment')
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}
    with session(root, reset=True):
        assert not (root / 'inputs').exists() and not (root / 'runs').exists()
        assert json.loads((root / 'initialization.json').read_text())['cached_rewards_reused'] is False
        with (root.parent / f'.{root.name}.session.lock').open('a+') as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    archives = list(root.parent.glob('gemma-it.archive-*'))
    assert len(archives) == 1
    assert all((archives[0] / name).read_bytes() == data for name, data in before.items())
    assert json.loads((archives[0] / '.startup-control/mbpp/stop.json').read_text())['immediate']


def test_reset_refuses_another_models_root_without_touching_it(tmp_path, monkeypatch):
    monkeypatch.setenv('GROUP_VOLUME', str(tmp_path))
    root = tmp_path / 'llama'
    path = root / 'experiments/llama.json'
    atomic_json(path, {'model': 'meta-llama/Llama-3.1-8B-Instruct'})
    before = path.read_bytes()
    with pytest.raises(ValueError, match='another experiment'), session(root, reset=True):
        pass
    assert path.read_bytes() == before
    assert not (root / '.startup-control').exists()


def test_reset_runs_both_caches_before_training_and_clears_chain_scope(tmp_path, monkeypatch):
    monkeypatch.setenv('GROUP_VOLUME', str(tmp_path))
    monkeypatch.setenv('OM_WORK', str(tmp_path / 'work'))
    monkeypatch.setenv('SRGC_DATASET_PHASE', 'mbpp')
    monkeypatch.setenv('SRGC_RESUME_ONLY', '1')
    stages = []
    def stage(args):
        stages.append((os.environ.get('SRGC_CACHE_ONLY'), os.environ.get('SRGC_DATASET_PHASE'), os.environ.get('SRGC_RESUME_ONLY')))
        return 0
    with patch.object(entry, 'main', side_effect=stage):
        assert run.main(['all', 'reset', '--root', str(tmp_path / 'fresh')]) == 0
    assert stages == [('1', None, '0'), ('0', None, '0')]
    assert os.environ['SRGC_DATASET_PHASE'] == 'mbpp' and os.environ['SRGC_RESUME_ONLY'] == '1'


def test_reset_requires_both_datasets():
    with pytest.raises(SystemExit):
        cli.parse_args(['math', 'reset'])


@pytest.mark.skipif(not Path('/home/kms/models/gemma-4-12B-it/tokenizer.json').is_file(), reason='local official IT tokenizer')
def test_real_official_tokenizer_matches_rendered_cohort(tmp_path):
    from transformers import AutoTokenizer
    path = Path('/home/kms/models/gemma-4-12B-it')
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model.validate_tokenizer_metadata(path, json.loads((path / 'config.json').read_text()))
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': 'What is 2+2?'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    wrapper = model.PromptTokenizer(tokenizer)
    assert wrapper.encode(rendered) == tokenizer.encode(rendered, add_special_tokens=False)
    assert wrapper.encode(rendered).count(2) == 1
    assert rendered.endswith(completion_cue('math'))


def test_it_training_records_all_25_update_boundaries_and_exports_separate_rewards(tmp_path):
    from srgc_rebuttal.srgc import Engine
    from srgc_research.dispatch import model_results, reward_curves
    from srgc_research.dispatch.reward_curve_results import collect
    from srgc_research.tests.test_model_results import finish, make_cohort, read_result
    from srgc_research.tests.test_reward_curves import history, make_engine

    cohort = make_cohort("gemma4_it", tmp_path / "profile")
    with adapter.runtime_adapter():
        q = __import__("srgc_rebuttal.cluster_queue", fromlist=["TaskQueue"]).TaskQueue(cohort.plans["math"])
        command = adapter.task_command(q, next(task for task in q.tasks if task.arm == "sr"))
    recorded = reward_curves.recording_command(command)
    assert recorded[recorded.index("--model") + 1] == "gemma4_it"
    engine, _, plan, data, binding = make_engine(tmp_path / "recording", True)
    engine.run_until(275)
    folder = tmp_path / "recording"
    assert all((folder / f"reward-curves/{'prefix' if step == 25 else 'sr'}/step-{step}.pt").is_file()
               for step in range(25, 276, 25))
    measured = collect(folder, "sr", plan, binding, data["evaluation_ids"],
                       lambda p: json.loads(p.read_text()) if p.exists() else None)
    assert len(measured["training_rewards"]) == 250
    assert [row["step"] for row in measured["held_out_rewards"]] == list(range(25, 275, 25))
    # The final held-out evaluation belongs to the final endpoint, not to an
    # invented recorder measurement at update 275.
    final_archive = torch.load(folder / "reward-curves/sr/step-275.pt", weights_only=False)
    assert final_archive["evaluation_only"] and "optimizer" not in final_archive["backend"]
    folder = finish(cohort, "math")
    atomic_json(folder / "on_policy-progress.json", {"seed": 5, "arm": "on_policy", "step": 50,
                "sampling_protocol": Engine.SAMPLING_PROTOCOL,
                "history": history()})
    assert model_results.export(cohort.model, "math", environment=cohort.env) == 0
    curve = next(row for row in read_result(cohort)["datasets"]["math"]["reward_curves"]
                 if row["seed"] == 5 and row["arm"] == "on_policy")
    assert len(curve["training_rewards"]) == 25
    assert [row["step"] for row in curve["held_out_rewards"]] == [275]


def test_public_it_status_and_results_do_not_need_a_gpu_or_model_python(tmp_path):
    import subprocess
    from srgc_research.tests.test_model_results import make_cohort
    cohort = make_cohort('gemma4_it', tmp_path)
    environment = {**cohort.env, 'GEMMA_PYTHON': '/missing', 'CUDA_VISIBLE_DEVICES': ''}
    before = {path: path.read_bytes() for path in cohort.root.rglob('*') if path.is_file()}
    for action in ('status', 'results'):
        result = subprocess.run(['sh', str(REPO / 'scripts/run_srgc_gemma4.sh'), action],
            cwd='/tmp', env=environment, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'MATH' in result.stdout and 'MBPP' in result.stdout
    assert all(path.read_bytes() == data for path, data in before.items())
    output = json.loads((cohort.work / 'results/gemma4_it-results.json').read_text())
    assert output['datasets']['math']['plan']['model'] == adapter.MODEL


def test_reset_stops_previous_pt_without_reusing_or_erasing_its_data(tmp_path, monkeypatch):
    monkeypatch.setenv('GROUP_VOLUME', str(tmp_path))
    work = tmp_path / 'work'
    monkeypatch.setenv('OM_WORK', str(work))
    old = work / 'srgc-rebuttal/gemma4-12b-text-v3'
    atomic_json(old / 'experiments/gemma4-12b-text-math.json', {'model': 'google/gemma-4-12B'})
    cache = old / 'inputs/math-seed-5.json'
    atomic_json(cache, {'cached_rewards': {'old': [0] * 8}})
    before = cache.read_bytes()
    with session(work / 'srgc-rebuttal/gemma4-12b-it-v1', reset=True):
        assert cache.read_bytes() == before
        assert json.loads((old / 'runs/math/.queue/stop.json').read_text())['immediate']
        assert json.loads((old / '.startup-control/mbpp/stop.json').read_text())['immediate']


def test_partial_it_snapshot_downloads_before_gpu_and_restores_offline_mode(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub import constants
    environment = {'GROUP_VOLUME': str(tmp_path), 'OM_WORK': str(tmp_path / 'work')}
    destination = tmp_path / 'models/gemma-4-12B-it'
    atomic_json(destination / 'generation_config.json', {'eos_token_id': [1, 106, 50]})
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setattr(constants, 'HF_HUB_OFFLINE', True)
    def download(**kwargs):
        assert kwargs['repo_id'] == adapter.MODEL and kwargs['revision'] == adapter.REVISION
        assert kwargs['token'] is False and kwargs['local_files_only'] is False
        assert os.environ['HF_HUB_OFFLINE'] == '0' and constants.HF_HUB_OFFLINE is False
        for name in model.OFFICIAL_FILES:
            path = destination / name
            if not path.exists():
                path.touch()
    with patch.object(huggingface_hub, 'snapshot_download', side_effect=download) as fetch:
        assert model.ensure_snapshot(environment) == destination
        assert model.ensure_snapshot(environment) == destination
        assert fetch.call_count == 1
    assert os.environ['HF_HUB_OFFLINE'] == '1' and constants.HF_HUB_OFFLINE is True


def test_old_explicit_pt_directory_is_never_overwritten_by_it_download(tmp_path):
    import huggingface_hub
    destination = tmp_path / 'models/gemma-4-12B'
    atomic_json(destination / 'generation_config.json', {'eos_token_id': 1})
    before = (destination / 'generation_config.json').read_bytes()
    with patch.object(huggingface_hub, 'snapshot_download') as fetch, pytest.raises(ValueError, match='PT or another model'):
        model.ensure_snapshot({'GROUP_VOLUME': str(tmp_path), 'OM_WORK': str(tmp_path / 'work'),
                               'SRGC_GEMMA_MODEL_PATH': str(destination)})
    assert fetch.call_count == 0 and (destination / 'generation_config.json').read_bytes() == before


def test_it_audit_reads_its_own_model_and_both_empty_caches(tmp_path):
    from srgc_research.dispatch.gemma4_it.audit import audit
    from srgc_research.tests.test_model_results import make_cohort
    cohort = make_cohort('gemma4_it', tmp_path)
    report = audit(cohort.root)
    assert not report['errors'], report['errors']
    assert len(report['datasets']) == 10
    assert all(row['initial_success_percent'] is None and not row['errors'] for row in report['datasets'])
    assert report['model'] == adapter.MODEL


@pytest.mark.parametrize('correct,empty,passed', [(1, False, False), (15, False, False),
                                                 (16, False, True), (64, True, False)])
def test_it_baseline_gate_rejects_one_correct_draw_and_empty_responses(correct, empty, passed):
    protocol = {'candidate_ids': [f'p{i}' for i in range(8)],
                'minimum_success_percent': preflight.MIN_SUCCESS_PERCENT}
    rows = [{'id': name, 'responses': ['Answer: 4'] * 8,
             'rewards': [int(index * 8 + j < correct) for j in range(8)],
             'token_ids': [[4, 106]] * 8}
            for index, name in enumerate(protocol['candidate_ids'])]
    if empty:
        rows[0]['responses'][0] = ''
    result = preflight.summarize(protocol, rows)
    assert result['passed'] is passed
    assert result['correct'] == correct and result['responses'] == 64
    assert result['success_percent'] == 100 * correct / 64
    assert 'not full initial-cache accuracy' in result['scope']


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_it_training_logps_and_gradients_match_native_model_with_both_turn_stops(dtype):
    from peft import LoraConfig, get_peft_model
    from srgc_research.dispatch.gemma4_it.memory import GemmaBackend
    from srgc_research.dispatch.gemma4_it.protocol import SUPPRESS_TOKENS, mask_logits
    from srgc_research.tests.test_gemma4_model import TinyTokenizer
    torch.set_num_threads(1)
    torch.manual_seed(19)
    config = tiny_config()
    config.vocab_size = 262144
    policy = Gemma4UnifiedForCausalLM(config).to(dtype)
    policy.generation_config.eos_token_id = list(EOS_TOKENS)
    policy.generation_config.suppress_tokens = list(SUPPRESS_TOKENS)
    policy = adapter.attach_adapter(policy, LoraConfig(r=2, lora_alpha=4,
        task_type='CAUSAL_LM', lora_dropout=0.0), get_peft_model)
    backend = GemmaBackend(policy, TinyTokenizer(), {'p': {'prompt': 'x'}},
                           lambda record, response: 0.0, projection_dim=8,
                           max_new_tokens=4, logit_chunk_tokens=2)
    sequences = [torch.tensor([2, 9, 17, 106]), torch.tensor([2, 9, 18, 19, 50])]
    actual = backend._logps_batch(sequences, 2)
    ids = torch.tensor([[2, 9, 17, 106, 0], [2, 9, 18, 19, 50]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    logits = policy(ids, attention_mask=mask, use_cache=False).logits[:, :-1]
    normalized = mask_logits(logits).log_softmax(-1)
    expected = [normalized[row, 1:len(sequence)-1].gather(-1, sequence[2:, None]).squeeze(-1)
                for row, sequence in enumerate(sequences)]
    for result, reference in zip(actual, expected):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, reference, rtol=0, atol=2e-5)
    parameters = [parameter for _, parameter in backend.train_parameters]
    gradients = torch.autograd.grad(sum(value.sum() for value in actual), parameters)
    references = torch.autograd.grad(sum(value.sum() for value in expected), parameters)
    for result, reference in zip(gradients, references):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, reference, rtol=0.03,
                                   atol=3e-3 if dtype == torch.bfloat16 else 2e-6)


def test_it_diagnostic_controls_retain_stops_and_suppression_and_preserve_raw_records(tmp_path):
    from srgc_research.dispatch.gemma4_it import probe
    native = {'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}
    options = {'do_sample': True, 'top_k': 0, 'top_p': 1.0, 'temperature': 1.0,
               'eos_token_id': [1, 106, 50], 'suppress_tokens': [258883, 258882]}
    control = probe.generation_options('official', options, native)
    assert control['top_k'] == 64 and control['top_p'] == 0.95
    assert control['eos_token_id'] == [1, 106, 50]
    assert control['suppress_tokens'] == [258883, 258882]
    assert options['top_k'] == 0 and options['top_p'] == 1.0
    assert probe.trim_response([9, 106, 0, 0]) == [9, 106]
    assert probe.trim_response([9, 50, 0]) == [9, 50]
    report = probe.summarize([{'dataset': 'math', 'profile': 'comparison',
        'rewards': [1, 0, 0], 'responses': ['Answer: 4', '', 'unfinished'],
        'token_ids': [[9, 106], [50], [9, 9]]}])
    assert report['math:comparison']['correct'] == 1
    assert report['math:comparison']['empty'] == 1
    assert report['math:comparison']['truncated'] == 1
    path = tmp_path / 'original.json'
    path.write_text('original experiment')
    with pytest.raises(SystemExit):
        probe.main(['--snapshot', str(tmp_path), '--output', str(path)])
    assert path.read_text() == 'original experiment'
