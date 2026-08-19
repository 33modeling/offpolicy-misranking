"""Generation manifest and exact prompt/K coverage regressions (CPU only)."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from artifact_contract import validate_generation_contract  # noqa: E402


def write_source(run: Path, prefix: str, n: int, k: int) -> None:
    manifest = {
        "explicit_kwargs": {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "max_new_tokens": 16,
        },
        "eos_token_ids": [99],
        "model_name_or_path": "/models/test-model",
        "k": k,
        "n_prompts": n,
        "idx_offset": 0,
    }
    (run / f"{prefix}.manifest.json").write_text(json.dumps(manifest))
    rows = []
    for prompt_idx in range(n):
        for rollout_idx in range(k):
            rows.append({
                "prompt_idx": prompt_idx,
                "rollout_idx": rollout_idx,
                "input_ids": [1, 2, 3, 99],
                "resp_start": 2,
                "resp_end": 4,
                "reward": 0.0,
            })
    (run / f"{prefix}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp)
    (run / "run_config.json").write_text(json.dumps({
        "behavior_k": 2,
        "fresh_k": 4,
        "val_k": 2,
        "max_new_tokens": 16,
        "model": "/models/test-model",
    }))
    (run / "prompts.json").write_text(json.dumps({
        "train": [{}, {}],
        "val": [{}],
    }))
    write_source(run, "rollouts_behavior_train", 2, 2)
    write_source(run, "rollouts_fresh_train", 2, 4)
    write_source(run, "rollouts_fresh_val", 1, 2)

    result = validate_generation_contract(run)
    assert result["validated_rows"] == 14
    assert validate_generation_contract(
        run, ("rollouts_behavior_train",)
    )["validated_rows"] == 4

    behavior = run / "rollouts_behavior_train.jsonl"
    original = behavior.read_text()
    behavior.write_text("\n".join(original.splitlines()[:-1]) + "\n")
    try:
        validate_generation_contract(run, ("rollouts_behavior_train",))
    except ValueError as exc:
        assert "prompt/K coverage mismatch" in str(exc)
    else:
        raise AssertionError("partial rollout coverage must be rejected")
    behavior.write_text(original)

    manifest_path = run / "rollouts_behavior_train.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["explicit_kwargs"]["top_k"] = 20
    manifest_path.write_text(json.dumps(manifest))
    try:
        validate_generation_contract(run, ("rollouts_behavior_train",))
    except ValueError as exc:
        assert "generation contract mismatch" in str(exc)
    else:
        raise AssertionError("sampling mismatch must be rejected")

print("PASS generation contract and exact coverage")
