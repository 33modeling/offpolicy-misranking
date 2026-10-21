"""Actual generation, masked gradients, GRPO recovery and preserved v1 identities."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Gemma4UnifiedForCausalLM

from srgc_rebuttal import cluster_queue, existing_runtime, verifiers
from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.runtime import atomic_json
from srgc_research.dispatch.gemma4 import adapter as archived
from srgc_research.dispatch.gemma4_text import adapter, cli, model, preflight, storage
from srgc_research.dispatch.gemma4_text.memory import GemmaBackend
from srgc_research.dispatch.gemma4_text.protocol import (
    CODE_VERIFIER,
    SUPPRESS_TOKENS,
    mask_logits,
    sampling,
)
from srgc_research.dispatch.gemma4_text.verifier import code_reward, extract_code
from srgc_research.tests.test_gemma4_model import (
    TinyTokenizer,
    forced_rollout,
    snapshot,  # noqa: F401 - shared real HF fixture
    tiny_config,
)

REPO = Path(__file__).resolve().parents[2]


def prepare(root, dataset="math"):
    source = REPO / "srgc_rebuttal/experiments" / (
        "additional_seeds.json" if dataset == "math" else "mbpp_seeds.json")
    return adapter.prepare(dataset, source, root, None, snapshot_sha256="a" * 64)


@pytest.mark.parametrize("dataset", ["math", "mbpp"])
def test_corrected_cohort_identity_and_cache_cannot_mix_with_v1(tmp_path, dataset):
    old = archived.prepare(dataset, REPO / "srgc_rebuttal/experiments" / (
        "additional_seeds.json" if dataset == "math" else "mbpp_seeds.json"),
        tmp_path / "old", None, snapshot_sha256="a" * 64)
    before = {path: path.read_bytes() for path in (tmp_path / "old").rglob("*.json")}
    new = prepare(tmp_path / "new", dataset)
    plan = adapter.validate_extension(new)
    assert plan["sampling"] == sampling()
    assert plan["verifier"] == (CODE_VERIFIER if dataset == "mbpp" else load_plan(old)["verifier"])
    old_bundle = json.loads(input_path(old, load_plan(old), 5).read_text())
    bundle = json.loads(input_path(new, plan, 5).read_text())
    for name in ("records", "candidate_ids", "ranking_validation_ids", "evaluation_ids"):
        assert bundle[name] == old_bundle[name]
    assert bundle["cached_rewards"] == {}
    assert bundle["provenance"]["sampling"] == sampling()
    with pytest.raises(ValueError):
        adapter.validate_extension(old, read_only=True)
    with pytest.raises(ValueError):
        archived.validate_extension(new, read_only=True)
    with pytest.raises(ValueError):
        adapter.validate_bundle_model(old_bundle, plan, 5)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert prepare(tmp_path / "new", dataset) == new


def test_existing_v1_inputs_and_results_are_never_replaced(tmp_path):
    old = archived.prepare("math", REPO / "srgc_rebuttal/experiments/additional_seeds.json",
                           tmp_path, None, snapshot_sha256="a" * 64)
    checkpoint = tmp_path / "runs/math/seed-5/random-latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"original checkpoint")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="restore the original plan"):
        prepare(tmp_path)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert old.is_file()


@pytest.mark.parametrize("response", [
    "```\npython\ndef f(x):\n    return x + 1\n```",
    "```python\ndef f(x):\n    return x + 1\n```",
    "```Python\ndef f(x):\n    return x + 1\n```",
    "```py\ndef f(x):\n    return x + 1\n```",
    "```\r\npython\r\ndef f(x):\r\n    return x + 1\r\n```",
    "def f(x):\n    return x + 1",
])
def test_real_mbpp_verifier_accepts_language_fences_and_keeps_assertions(response):
    before = verifiers.extract_code
    assert code_reward({"answer": "assert f(2) == 3\nassert f(-1) == 0"}, response) == 1
    assert code_reward({"answer": "assert f(2) == 4"}, response) == 0
    assert verifiers.extract_code is before


def test_code_failure_is_reproduced_and_unfenced_code_is_not_rewritten():
    response = "```\npython\ndef f(x):\n    return x + 1\n```"
    record = {"answer": "assert f(2) == 3"}
    assert verifiers.code_reward(record, response) == 0
    assert code_reward(record, response) == 1
    assert extract_code("python\ndef f():\n    return 2") == "python\ndef f():\n    return 2"
    assert code_reward({"answer": "assert f() == 2"}, "python\ndef f():\n    return 2") == 0
    assert code_reward(record, "def f(x):\n    return 0") == 0


def backend(dtype=torch.float32):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = tiny_config()
    config.vocab_size = 262144
    policy = Gemma4UnifiedForCausalLM(config).to(dtype)
    policy.generation_config.suppress_tokens = list(SUPPRESS_TOKENS)
    policy = adapter.attach_adapter(policy, LoraConfig(r=2, lora_alpha=4,
        task_type="CAUSAL_LM", lora_dropout=0.0), get_peft_model)
    result = GemmaBackend(policy, TinyTokenizer(), {f"p{i}": {"prompt": "x"} for i in range(4)},
                         lambda record, text: float("9" in text), projection_dim=8,
                         max_new_tokens=4, logit_chunk_tokens=2)
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_masked_softcapped_logps_and_gradients_match_native(dtype):
    b = backend(dtype)
    sequences, _, start = forced_rollout("p0", 8, 17)
    selected = [sequences[0], sequences[1]]
    actual = b._logps_batch(selected, start)
    ids = torch.zeros((2, max(map(len, selected))), dtype=torch.long)
    attention = torch.zeros_like(ids)
    for row, sequence in enumerate(selected):
        ids[row, :len(sequence)], attention[row, :len(sequence)] = sequence, 1
    logits = b.model(ids, attention_mask=attention, use_cache=False).logits[:, :-1]
    native = mask_logits(logits).log_softmax(-1)
    expected = [native[row, start-1:len(sequence)-1].gather(-1, sequence[start:, None]).squeeze(-1)
                for row, sequence in enumerate(selected)]
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=2e-5)
    params = [p for _, p in b.train_parameters]
    ga = torch.autograd.grad(sum(v.sum() for v in actual), params)
    ge = torch.autograd.grad(sum(v.sum() for v in expected), params)
    for a, e in zip(ga, ge):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, e, rtol=0.03, atol=3e-3 if dtype == torch.bfloat16 else 2e-6)


def test_actual_generation_and_grpo_checkpoint_resume(tmp_path):
    b = backend()
    sequences, _, start = b._rollout("p0", 8, 17)
    assert len(sequences) == 8
    assert all(not any(token in SUPPRESS_TOKENS for token in seq[start:].tolist()) for seq in sequences)
    b._rollout = forced_rollout
    before = copy.deepcopy(b.state_dict())
    scores = b.score_gradients(["p0", "p1"], responses=8, group_size=4, seed=17)
    assert all(np.isfinite(v).all() and np.linalg.norm(v) > 0 for v in scores.values())
    b.train(["p0", "p1"], responses=8, objective="grpo", seed=17)
    after = copy.deepcopy(b.state_dict())
    assert any(not torch.equal(before["trainable"][name], value) for name, value in after["trainable"].items())
    checkpoint = tmp_path / "saved.pt"
    torch.save(before, checkpoint)
    b.load_state_dict(torch.load(checkpoint, weights_only=False))
    b.train(["p0", "p1"], responses=8, objective="grpo", seed=17)
    for name, value in b.state_dict()["trainable"].items():
        torch.testing.assert_close(value, after["trainable"][name], rtol=0, atol=0)
    invalid = torch.tensor([2, 4, SUPPRESS_TOKENS[0]])
    with pytest.raises(ValueError, match="do not reuse v1"):
        b._logps(invalid, 2)


def test_masked_distribution_has_zero_probability_and_zero_gradient_on_modality_tokens():
    logits = torch.tensor([[0., 1., 100., 100.]], requires_grad=True)
    masked = mask_logits(logits, suppressed=(2, 3))
    probabilities = masked.softmax(-1)
    assert probabilities[0, 2:].tolist() == [0., 0.]
    (-masked.log_softmax(-1)[0, 1]).backward()
    assert logits.grad[0, 2:].tolist() == [0., 0.]
    assert torch.isfinite(logits.grad).all()


def test_verified_loader_restores_official_suppression_and_rejects_changed_metadata(snapshot):  # noqa: F811
    path, _, _, _ = snapshot
    native = json.loads((path / "generation_config.json").read_text())
    native["suppress_tokens"] = list(SUPPRESS_TOKENS)
    (path / "generation_config.json").write_text(json.dumps(native))
    policy, _ = model.load_text_model(path, torch.device("cpu"))
    assert policy.generation_config.suppress_tokens == list(SUPPRESS_TOKENS)
    assert policy.generation_config.top_p == 1 and policy.generation_config.top_k == 0
    native["suppress_tokens"] = []
    (path / "generation_config.json").write_text(json.dumps(native))
    with pytest.raises(ValueError, match="official modality suppression"):
        model.load_text_model(path, torch.device("cpu"))


def test_absorbing_modality_loop_is_prevented_in_real_transformers_sampling():
    b = backend()
    base = b.model.get_base_model()
    base.config.final_logit_softcapping = None
    def dominant_modality(module, args, result):
        result = result.clone()
        result[..., SUPPRESS_TOKENS[0]] = 1000
        result[..., SUPPRESS_TOKENS[1]] = 999
        return result
    with torch.no_grad():
        hook = base.lm_head.register_forward_hook(dominant_modality)
        try:
            kwargs = {"input_ids": torch.tensor([[2, 4, 5]]), "do_sample": True,
                      "top_p": 1., "top_k": 0, "max_new_tokens": 4, "pad_token_id": 0,
                      "eos_token_id": 1, "num_return_sequences": 8}
            broken = base.generate(**kwargs, suppress_tokens=[])
            fixed = base.generate(**kwargs)
        finally:
            hook.remove()
    assert all(token in SUPPRESS_TOKENS for row in broken[:, 3:] for token in row.tolist())
    assert all(token not in SUPPRESS_TOKENS for row in fixed[:, 3:] for token in row.tolist())


def write_panel(path, *, success=True):
    plan = adapter.validate_extension(path)
    bundle = json.loads(input_path(path, plan, 5).read_text())
    protocol = preflight.identity(path, plan, bundle)
    root = preflight.folder(path, plan)
    rows = []
    for candidate in protocol["candidate_ids"]:
        row = {"protocol": protocol, "id": candidate, "rewards": [int(success)] + [0] * 7,
               "responses": ["Answer: 6"] * 8, "token_ids": [[4, 1]] * 8}
        atomic_json(root / f"{hashlib.sha256(candidate.encode()).hexdigest()}.json", row)
        rows.append(row)
    atomic_json(root / "summary.json", preflight.summarize(protocol, rows))
    return root


def test_preflight_requires_real_receipts_and_detects_corruption(tmp_path):
    path = prepare(tmp_path)
    with pytest.raises(ValueError, match="preflight is missing"):
        preflight.validate_gate(path)
    root = write_panel(path)
    assert preflight.validate_gate(path)["responses"] == 64
    receipt = next(p for p in root.glob("*.json") if p.name != "summary.json")
    row = json.loads(receipt.read_text())
    row["token_ids"][0] = [SUPPRESS_TOKENS[0]]
    atomic_json(receipt, row)
    with pytest.raises(ValueError, match="invalid preflight"):
        preflight.validate_gate(path)
    write_panel(path, success=False)
    with pytest.raises(ValueError, match="no verified correct response"):
        preflight.validate_gate(path)


def test_runtime_routes_cache_training_and_reports_and_restores_shared_code(tmp_path):
    path = prepare(tmp_path)
    loader, queue = existing_runtime.load_model, cluster_queue.TaskQueue
    with adapter.runtime_adapter():
        from srgc_rebuttal import build_cache

        assert existing_runtime.load_model is adapter.load_model
        assert build_cache.verifier_protocol(CODE_VERIFIER)["sampling"] == sampling()
        assert cluster_queue.TaskQueue is not queue
        q = cluster_queue.TaskQueue(path)
        command = adapter.task_command(q, next(t for t in q.tasks if t.arm == "cache"))
        assert "srgc_research.dispatch.gemma4_text.rank" in command
    assert existing_runtime.load_model is loader
    assert cluster_queue.TaskQueue is queue
    env = {"GROUP_VOLUME": str(tmp_path), "OM_WORK": str(tmp_path / "work"),
           "SRGC_GEMMA_ROOT": str(tmp_path / "old-v1")}
    assert storage.default_root(env).name == "gemma4-12b-text-v2"


def test_new_status_and_result_export_never_replace_archived_result(tmp_path, capsys):
    from srgc_research.dispatch import model_results, model_status

    work = tmp_path / "work"
    env = {**os.environ, "GROUP_VOLUME": str(tmp_path), "OM_WORK": str(work)}
    root = storage.default_root(env)
    prepare(root, "math")
    prepare(root, "mbpp")
    archived_result = work / "results/gemma4-results.json"
    archived_result.parent.mkdir(parents=True)
    archived_result.write_text("archived result")
    snapshots = model_status.load_reports("gemma4_text", "all", root, env)
    assert len(snapshots) == 2 and all(row["prepared"] and not row["errors"] for row in snapshots)
    assert model_results.export("gemma4_text", "all", root=root, environment=env, text=True) == 0
    exported = json.loads((work / "results/gemma4_text-results.json").read_text())
    assert exported["datasets"]["math"]["plan"]["sampling"] == sampling()
    assert exported["datasets"]["mbpp"]["plan"]["verifier"] == CODE_VERIFIER
    assert archived_result.read_text() == "archived result"
    assert "JSON:" in capsys.readouterr().out


def test_normal_public_command_selects_corrected_protocol_and_explicit_legacy_is_preserved(tmp_path):
    driver = tmp_path / "python"
    driver.write_text("#!" + sys.executable + "\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    driver.chmod(0o755)
    env = {**os.environ, "GEMMA_PYTHON": str(driver), "SRGC_MODEL_CHAIN": "0"}
    env.pop("SRGC_GEMMA_LEGACY_PROTOCOL", None)
    env.pop("SRGC_GEMMA_TEXT_PROTOCOL", None)
    command = ["sh", "scripts/run_srgc_gemma4.sh", "all"]
    fixed = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, check=True)
    assert json.loads(fixed.stdout)[1] == "srgc_research.dispatch.gemma_text_run"
    legacy = subprocess.run(command, cwd=REPO, env={**env, "SRGC_GEMMA_LEGACY_PROTOCOL": "1"},
                            capture_output=True, text=True, check=True)
    assert json.loads(legacy.stdout)[1] == "srgc_research.dispatch.gemma_run"


def test_corrected_public_results_needs_no_model_interpreter(tmp_path):
    from srgc_research.tests.test_model_results import make_cohort

    cohort = make_cohort("gemma4_text", tmp_path)
    env = {**cohort.env, "GEMMA_PYTHON": "/missing", "SRGC_GEMMA_LEGACY_PROTOCOL": "0"}
    for script in ("run_srgc_gemma4.sh", "run_srgc_gemma4_recovery.sh"):
        result = subprocess.run(["sh", str(REPO / "scripts" / script), "results"], cwd="/tmp",
                                env=env, capture_output=True, text=True, timeout=30, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "MATH FINAL REWARD (%)" in result.stdout and "MBPP FINAL REWARD (%)" in result.stdout
    assert (cohort.work / "results/gemma4_text-results.json").is_file()


def test_every_requested_dataset_has_a_real_preflight_before_training(tmp_path):
    commands = []
    environment = {"SRGC_GEMMA_PLAN": str(tmp_path / "math.json"),
                   "SRGC_GEMMA_PLANS": json.dumps([str(tmp_path / "math.json"), str(tmp_path / "mbpp.json")])}
    def child(command, log, env, **kwargs):
        commands.append(command)
        return 0
    cli.admit_with_smoke(lambda *args, **kw: {"allocated_gpu_seconds": 0},
                         tmp_path / "admission", environment, child)
    assert [c[c.index("--stage") + 1] for c in commands] == ["smoke", "preflight", "preflight"]
    assert [c[c.index("--plan") + 1] for c in commands[1:]] == [environment["SRGC_GEMMA_PLAN"], str(tmp_path / "mbpp.json")]


def test_real_candidate_failure_is_not_misreported_as_failed_backward_smoke(tmp_path):
    codes = iter([0, 1])
    with pytest.raises(RuntimeError, match="init preflight failed"):
        cli.admit_with_smoke(lambda *a, **kw: {"allocated_gpu_seconds": 0}, tmp_path,
                             {"SRGC_GEMMA_PLAN": str(tmp_path / "math.json")},
                             lambda *a, **kw: next(codes))
    report = json.loads((tmp_path / "gemma-admission.json").read_text())
    assert report["gemma_model_smoke"] == "passed"
    assert report["gemma_initial_preflight"] == "failed"


def test_corrected_training_records_all_25_update_boundaries_and_exports_separate_rewards(tmp_path):
    from srgc_rebuttal.srgc import Engine
    from srgc_research.dispatch import model_results, reward_curves
    from srgc_research.dispatch.reward_curve_results import collect
    from srgc_research.tests.test_model_results import finish, make_cohort, read_result
    from srgc_research.tests.test_reward_curves import history, make_engine

    cohort = make_cohort("gemma4_text", tmp_path / "profile")
    with adapter.runtime_adapter():
        q = cluster_queue.TaskQueue(cohort.plans["math"])
        command = adapter.task_command(q, next(task for task in q.tasks if task.arm == "sr"))
    recorded = reward_curves.recording_command(command)
    assert recorded[recorded.index("--model") + 1] == "gemma4_text"
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
