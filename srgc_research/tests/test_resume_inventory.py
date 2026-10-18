"""Cross-model resume ordering inspects receipts, never model weights or CUDA."""

import json
import sys
from unittest.mock import patch

from srgc_rebuttal import cluster_queue
from srgc_rebuttal.build_cache import write_cost_summary
from srgc_rebuttal.cluster_queue import Task
from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.runtime import atomic_json, lease, run_root
from srgc_research.dispatch.resume_inventory import failed_signature, signature
from srgc_research.tests.test_model_results import (
    cohort as cohort,  # noqa: PLC0414 -- Validate every model's prepared layout.
)
from srgc_research.tests.test_model_results import finish, saved


def test_saved_failure_is_found_without_writing_state_or_loading_model_packages(cohort):
    output = run_root(cohort.plans['math'], load_plan(cohort.plans['math']))
    atomic_json(output / '.queue/tasks/seed-5.sr.json', {'task': 'seed-5.sr', 'status': 'failed', 'attempt': 99})
    checkpoint = output / 'seed-5/sr-latest.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b'opaque checkpoint; never deserialize during scheduling')
    before = saved(cohort)
    with patch.dict(sys.modules, {'torch': None, 'transformers': None, 'peft': None}):
        first = signature(cohort.model, 'math', cohort.env)
        assert first
        assert signature(cohort.model, 'mbpp', cohort.env) is None
    assert saved(cohort) == before
    checkpoint.write_bytes(b'new fully committed checkpoint')
    assert signature(cohort.model, 'math', cohort.env) != first


def test_live_task_lease_excludes_running_work_from_abandoned_inventory(cohort):
    output = run_root(cohort.plans['math'], load_plan(cohort.plans['math']))
    atomic_json(output / '.queue/tasks/seed-5.sr.json', {'task': 'seed-5.sr', 'status': 'running'})
    with lease(output / 'seed-5/.sr.execution.lock'):
        assert signature(cohort.model, 'math', cohort.env) is None
    assert signature(cohort.model, 'math', cohort.env)


def test_custom_root_is_used_without_touching_the_default_model_root(cohort, tmp_path):
    empty = {**cohort.env, {'qwen35': 'SRGC_QWEN_ROOT', 'llama31': 'SRGC_LLAMA_ROOT',
                          'gemma4': 'SRGC_GEMMA_ROOT'}[cohort.model]: str(tmp_path / 'missing')}
    output = run_root(cohort.plans['math'], load_plan(cohort.plans['math']))
    atomic_json(output / '.queue/tasks/seed-5.prefix.json', {'task': 'seed-5.prefix', 'status': 'interrupted'})
    assert signature(cohort.model, 'math', empty) is None
    assert signature(cohort.model, 'math', empty, root=cohort.root)


def test_failed_inventory_distinguishes_error_from_interrupted_or_owned_work(cohort):
    output = run_root(cohort.plans['math'], load_plan(cohort.plans['math']))
    receipt = output / '.queue/tasks/seed-5.sr.json'
    atomic_json(receipt, {'task': 'seed-5.sr', 'status': 'interrupted', 'attempt': 99})
    assert signature(cohort.model, 'math', cohort.env)
    assert failed_signature(cohort.model, 'math', cohort.env) is None
    atomic_json(receipt, {'task': 'seed-5.sr', 'status': 'failed', 'attempt': 99})
    before = saved(cohort)
    with patch.dict(sys.modules, {'torch': None, 'transformers': None, 'peft': None}):
        assert failed_signature(cohort.model, 'math', cohort.env)
    assert saved(cohort) == before
    with lease(output / '.queue/leases/seed-5.sr.lock'):
        assert failed_signature(cohort.model, 'math', cohort.env) is None


def test_partial_cache_without_task_receipt_is_found_for_recovery(cohort):
    plan_path = cohort.plans['math']
    cache = input_path(plan_path, load_plan(plan_path), 5).with_suffix('.cache')
    atomic_json(cache / 'question-1.json', {'saved': 'one completed prompt'})
    before = saved(cohort)
    assert signature(cohort.model, 'math', cohort.env)
    assert failed_signature(cohort.model, 'math', cohort.env) is None
    assert saved(cohort) == before


def test_completed_cache_and_stale_error_do_not_create_false_recovery(cohort):
    finish(cohort, 'math')
    plan_path = cohort.plans['math']
    output = run_root(plan_path, load_plan(plan_path))
    atomic_json(output / '.queue/tasks/seed-5.cache.json',
                {'task': 'seed-5.cache', 'status': 'failed', 'attempt': 99})
    cache = input_path(plan_path, load_plan(plan_path), 5).with_suffix('.cache')
    atomic_json(cache / 'old-prompt.json', {'saved': True})
    bundle = input_path(plan_path, load_plan(plan_path), 5)
    write_cost_summary(bundle, json.loads(bundle.read_text()))
    before = saved(cohort)
    assert signature(cohort.model, 'math', cohort.env) is None
    assert failed_signature(cohort.model, 'math', cohort.env) is None
    assert saved(cohort) == before


def test_full_rewards_without_handoff_cost_remain_in_error_recovery(cohort):
    finish(cohort, 'math')
    plan_path = cohort.plans['math']
    with cohort.adapter.runtime_adapter():
        queue = cluster_queue.TaskQueue(plan_path)
        atomic_json(queue.receipt(Task(5, 'cache')),
                    {'task': 'seed-5.cache', 'status': 'failed', 'attempt': 99})
        assert not queue.complete(Task(5, 'cache'))
    assert failed_signature(cohort.model, 'math', cohort.env)
    assert signature(cohort.model, 'math', cohort.env)


def test_malformed_cache_handoff_cost_is_left_for_dispatch_validation(cohort):
    finish(cohort, 'math')
    plan_path = cohort.plans['math']
    output = run_root(plan_path, load_plan(plan_path))
    atomic_json(output / '.queue/protocol.json', {'inputs': {'5': {'pending_cache': True}}})
    cache = input_path(plan_path, load_plan(plan_path), 5).with_suffix('.cache')
    atomic_json(cache / 'cost-summary.json', [])
    before = saved(cohort)
    assert signature(cohort.model, 'math', cohort.env)
    assert saved(cohort) == before
