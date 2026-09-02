"""Runtime-only rollout recovery keeps the immutable experiment config intact."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_recovery_rotates_gpus_and_forces_the_recovery_batch(tmp_path: Path) -> None:
    pipeline = tmp_path / "pipeline"
    (pipeline / "src").mkdir(parents=True)
    experiment = pipeline / "src/experiment.py"
    experiment.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "row={'batch':os.environ['OM_GEN_BATCH'],"
        "'gpu':os.environ['CUDA_VISIBLE_DEVICES'],"
        "'stage':args[args.index('--stage')+1],"
        "'shard':args[args.index('--shard')+1]}\n"
        "fd=os.open(os.environ['RECOVERY_CAPTURE'], os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0o600)\n"
        "os.write(fd, (json.dumps(row)+'\\n').encode()); os.close(fd)\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=pipeline, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.com"],
        cwd=pipeline,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "fixture"], cwd=pipeline, check=True
    )
    subprocess.run(["git", "add", "."], cwd=pipeline, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=pipeline, check=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=pipeline, text=True
    ).strip()

    run = tmp_path / "run"
    run.mkdir()
    config = {
        "git": head,
        "model_resolved": str(tmp_path / "model"),
        "dataset": "math500",
        "drift": 0,
        "behavior_k": 8,
        "fresh_k": 32,
        "val_k": 8,
        "micro_group": 4,
        "gradient_micro_batch": 1,
        "n_train": 4,
        "n_val": 4,
        "seed": 0,
        "max_new_tokens": 2048,
        "proj_dim": 64,
        "grad_layers": 1,
        "clip_cap": 10.0,
        "temperature": 1.0,
        "topk_frac": 0.1,
        "prompt_format": "olmo_rlzero_math",
        "attn": "eager",
        "gen_batch": "4",
    }
    (run / "run_config.json").write_text(json.dumps(config))
    capture = tmp_path / "capture.jsonl"
    result = subprocess.run(
        [
            "/bin/bash",
            str(REPO / "scripts/recover_rollout_stage.sh"),
            str(pipeline),
            str(run),
            "rollout-fresh",
            "1",
            "0,1",
        ],
        env={
            **os.environ,
            "OM_RECOVERY_PY": sys.executable,
            "OM_RECOVERY_INDEX": "2",
            "RECOVERY_CAPTURE": str(capture),
        },
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in capture.read_text().splitlines()]
    assert {row["batch"] for row in rows} == {"1"}
    assert {(row["shard"], row["gpu"]) for row in rows} == {
        ("0:2", "1"),
        ("1:2", "0"),
    }
    assert {row["stage"] for row in rows} == {"rollout-fresh"}
    assert json.loads((run / "run_config.json").read_text())["gen_batch"] == "4"
    recovery = [
        json.loads(line)
        for line in (run / "rollout_recovery.jsonl").read_text().splitlines()
    ]
    assert [row["status"] for row in recovery] == ["started", "completed"]
    assert {row["configured_generation_batch"] for row in recovery} == {"4"}
    assert {row["recovery_generation_batch"] for row in recovery} == {1}
    assert recovery[0]["gpu_order"] == ["0", "1"]
