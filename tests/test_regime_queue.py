"""Multiple workers claim each regime point once and recover a transient failure."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_shared_regime_queue_is_unique_and_retryable() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        checkout = root / "checkout"
        work = root / "work"
        fake_bin = root / "bin"
        (checkout / "scripts").mkdir(parents=True)
        (checkout / "src").mkdir()
        (work / "venv/bin").mkdir(parents=True)
        (work / "models/model").mkdir(parents=True)
        fake_bin.mkdir()
        shutil.copy2(REPO / ".gitignore", checkout / ".gitignore")
        shutil.copy2(REPO / "scripts/run_matrix.sh", checkout / "scripts/run_matrix.sh")
        shutil.copy2(
            REPO / "scripts/_report_cache.sh", checkout / "scripts/_report_cache.sh"
        )
        shutil.copy2(
            REPO / "src/regime_resume_commit.py",
            checkout / "src/regime_resume_commit.py",
        )
        shutil.copy2(
            REPO / "src/materialize_prompt_dataset.py",
            checkout / "src/materialize_prompt_dataset.py",
        )
        shutil.copy2(REPO / "src/reuse_behavior.py", checkout / "src/reuse_behavior.py")
        shutil.copy2(REPO / "src/artifact_contract.py", checkout / "src/artifact_contract.py")
        shutil.copy2(REPO / "src/compact_artifacts.py", checkout / "src/compact_artifacts.py")
        shutil.copy2(REPO / "src/rollout_contract.py", checkout / "src/rollout_contract.py")
        (work / "models/model/config.json").write_text("{}\n", encoding="utf-8")
        (work / "venv/bin/python").symlink_to(Path(sys.executable))
        (checkout / "scripts/setup_env.sh").write_text(
            'export OM_WORK="${OM_WORK:?}"\n'
            'export VENV_DIR="$OM_WORK/venv"\n'
            'export MODELS_DIR="$OM_WORK/models"\n',
            encoding="utf-8",
        )
        executable(
            checkout / "scripts/run_point.sh",
            "#!/usr/bin/env bash\n"
            'cd "$(dirname "$0")/.."\n'
            'point="$DATASET $SEED $DRIFT"\n'
            'printf "%s|%s\\n" "${OM_TEST_WORKER:-worker}" "$point" >> "$OM_WORK/claims"\n'
            "/bin/sleep 0.05\n"
            'if [ "$point" = "a 0 0" ] && mkdir "$OM_WORK/hang-once" 2>/dev/null; then '
            "while :; do /bin/sleep 1; done; fi\n"
            'mkdir -p "$OUT_ROOT"\n'
            'if [ "${TEST_PROMPT_MISMATCH_ONCE:-0}" = 1 ] && '
            '[ "$point" = "a 0 25" ] && mkdir "$OM_WORK/prompt-mismatch-once" 2>/dev/null; then '
            'printf "prompt mismatch\\n" > "$OUT_ROOT/prompts.json"; exit 42; fi\n'
            "python3 - \"$OUT_ROOT\" <<'PY'\n"
            "import json, os, subprocess, sys\n"
            "from pathlib import Path\n"
            "dataset=os.environ['DATASET']; drift=int(os.environ['DRIFT'])\n"
            "config={'model_resolved':str(Path(os.environ['MODEL_PATH']).resolve()),"
            "'dataset':dataset,'seed':int(os.environ['SEED']),'drift':drift,"
            "'n_train':int(os.environ.get('N_TRAIN','512')),'n_val':int(os.environ.get('N_VAL','100')),"
            "'behavior_k':int(os.environ.get('BEHAVIOR_K','8')),'fresh_k':int(os.environ.get('FRESH_K','32')),"
            "'val_k':int(os.environ.get('VAL_K','8')),'training_objective':'base_control' if drift==0 else 'grpo',"
            "'micro_group':int(os.environ.get('MICRO_GROUP','4')),"
            "'max_new_tokens':int(os.environ.get('MAX_NEW_TOKENS','512')),"
            "'proj_dim':int(os.environ.get('PROJ_DIM','4096')),'grad_layers':int(os.environ.get('GRAD_LAYERS','4')),"
            "'clip_cap':float(os.environ.get('CLIP_CAP','10')),'temperature':float(os.environ.get('TEMPERATURE','1.0')),"
            "'topk_frac':float(os.environ.get('TOPK_FRAC','0.10')),'top_p':float(os.environ.get('OM_TOP_P','1.0')),"
            "'thinking':os.environ.get('OM_THINKING','off'),'attn':os.environ.get('OM_ATTN','eager'),"
            "'lora_targets':os.environ.get('OM_LORA_TARGETS'),'skip_hybrid':os.environ.get('OM_SKIP_HYBRID','1'),"
            "'policy_update':'none' if drift==0 else 'clipped_policy_gradient',"
            "'reward_source':'none' if drift==0 else 'verifier','supervised_loss':False,"
            "'positive_only_filter':False,'grpo_world_size':int(os.environ.get('GRPO_WORLD_SIZE','4')),"
            "'grpo_group_size':int(os.environ.get('GRPO_GROUP_SIZE','8')),"
            "'grpo_clip_epsilon':float(os.environ.get('GRPO_CLIP_EPSILON','0.2')),"
            "'grpo_learning_rate':float(os.environ.get('GRPO_LEARNING_RATE','1e-5')),"
            "'grpo_reference_kl_beta':0.0,"
            "'grpo_epochs_per_batch':int(os.environ.get('GRPO_EPOCHS_PER_BATCH','2')),"
            "'grpo_max_grad_norm':float(os.environ.get('GRPO_MAX_GRAD_NORM','1.0')),"
            "'grpo_advantage_epsilon':float(os.environ.get('GRPO_ADVANTAGE_EPSILON','1e-4')),"
            "'grpo_lora_rank':int(os.environ.get('GRPO_LORA_RANK','16')),"
            "'grpo_lora_alpha':int(os.environ.get('GRPO_LORA_ALPHA','32')),"
            "'behavior_source':os.environ.get('OM_BEHAVIOR_SOURCE'),"
            "'git':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}\n"
            "out=Path(sys.argv[1]); (out/'run_config.json').write_text(json.dumps(config))\n"
            "prompts={'train':[{'question':f'q{i}','answer':str(i)} for i in range(config['n_train'])],"
            "'val':[{'question':f'v{i}','answer':str(i)} for i in range(config['n_val'])]}\n"
            "(out/'prompts.json').write_text(json.dumps(prompts))\n"
            "scores={str(i):{'score':float(i)} for i in range(config['n_train'])}\n"
            "off={name:scores for name in ('g00','g10','g01','g11')}\n"
            "halves={str(i):{'r':float(i),'a':float(i),'b':float(i)} "
            "for i in range(config['n_train'])}\n"
            "(out/'scores_oracle.json').write_text(json.dumps(scores))\n"
            "(out/'scores_offpolicy.json').write_text(json.dumps(off))\n"
            "(out/'scores_splithalf.json').write_text(json.dumps(halves))\n"
            "protocol={'generation_validation':{'validated_rows':1}}\n"
            "(out/'score_protocol.json').write_text(json.dumps({**protocol,'schema':"
            "'offpolicy-score-validation-split/v2'}))\n"
            "(out/'oracle_protocol.json').write_text(json.dumps({**protocol,'schema':"
            "'offpolicy-oracle-validation-split/v2'}))\n"
            "PY\n"
            "for name in DONE manifest.json report.json divergence_stats.json oracle_micro_groups.pt "
            "val_groups.pt; do "
            'printf "{}\\n" > "$OUT_ROOT/$name"; done\n'
            'if [ "$DRIFT" -gt 0 ]; then '
            'p="$OUT_ROOT/policy_step_$DRIFT"; mkdir -p "$p"; '
            "for name in policy_train.json adapter_config.json adapter_model.safetensors optimizer.pt grpo_stats.jsonl; do "
            'printf "x\\n" > "$p/$name"; done; fi\n',
        )
        executable(
            checkout / "scripts/diagnose_run_failure.sh",
            "#!/usr/bin/env bash\nexit 0\n",
        )
        (checkout / "src/regime_contract.py").write_text(
            "import pathlib,sys\n"
            "args=sys.argv[1:]\n"
            "if args[0] in ('check-collection','mark-collection'):\n"
            " results=pathlib.Path(args[args.index('--results')+1])\n"
            " marker=results/'.regime_collection.json'\n"
            " if args[0]=='check-collection': raise SystemExit(0 if marker.exists() else 1)\n"
            " marker.write_text('{}'); raise SystemExit(0)\n"
            "run=pathlib.Path(args[args.index('--run')+1])\n"
            "marker=run/'.regime_validated.json'\n"
            "if args[0]=='prepare-run': print('[fixture] prepare'); raise SystemExit(0)\n"
            "if '--deep' in args and '--mark' in args:\n"
            " run.mkdir(parents=True,exist_ok=True); marker.write_text('{}')\n"
            "raise SystemExit(0 if marker.exists() else 1)\n",
            encoding="utf-8",
        )
        (checkout / "src/train_policy_grpo.py").write_text(
            "def validate_policy_manifest(*args, **kwargs): return {}\n",
            encoding="utf-8",
        )
        shutil.copy2(REPO / "src/gate_rules.py", checkout / "src/gate_rules.py")
        shutil.copy2(
            REPO / "src/score_artifacts.py", checkout / "src/score_artifacts.py"
        )
        shutil.copy2(REPO / "src/select_rules.py", checkout / "src/select_rules.py")
        (checkout / "src/regime_map.py").write_text(
            "import os,pathlib,sys\n"
            "args=sys.argv[1:]; out=pathlib.Path(args[args.index('--output-dir')+1])\n"
            "out.mkdir(parents=True,exist_ok=True)\n"
            "if os.environ.get('FAIL_ANALYSIS'): raise SystemExit(9)\n"
            "with (out/'analysis-count').open('a') as f: f.write('1\\n')\n"
            "for name in ('REGIME.json','REGIME.csv','REGIME_SUMMARY.csv','FINAL_REPORT.md'):\n"
            " (out/name).write_text('# pass\\n')\n",
            encoding="utf-8",
        )
        executable(fake_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n")
        subprocess.run(
            ["git", "init", "-q"], cwd=checkout, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=checkout,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"], cwd=checkout, check=True
        )
        subprocess.run(["git", "add", "."], cwd=checkout, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "fixture"], cwd=checkout, check=True
        )
        checkout_git = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()

        env = os.environ.copy()
        env.update(
            {
                "OM_WORK": str(work),
                "PATH": f"{fake_bin}:{env['PATH']}",
                "MODEL_PATH": str(work / "models/model"),
                "REGIME_MODEL_TAG": "fixture",
                "REGIME_ROOT": str(work / "runs/regime-fixture"),
                "REGIME_RESULTS": str(work / "results/regime-fixture"),
                "REGIME_DATASETS": "a b",
                "REGIME_SEEDS": "0 1",
                "REGIME_DRIFTS": "0 25",
                "REGIME_MAX_RETRIES": "2",
                "REGIME_N_TRAIN": "8",
                "REGIME_N_VAL": "4",
                "REGIME_WATCH_INTERVAL_SECONDS": "1",
                "REGIME_STALL_SECONDS": "1",
                "REGIME_WATCH_KILL_GRACE_SECONDS": "0",
                "REGIME_WATCH_GPU_SAMPLES": "1",
            }
        )
        workers = []
        for index in range(3):
            workers.append(
                subprocess.Popen(
                    ["/bin/bash", "scripts/run_matrix.sh"],
                    cwd=checkout,
                    env={**env, "OM_TEST_WORKER": f"worker-{index}"},
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
            )
        outputs = []
        for worker in workers:
            output, _ = worker.communicate(timeout=30)
            outputs.append(output)
            assert worker.returncode == 0, output
        assert any("[regime-watchdog]" in output for output in outputs)
        assert (
            work / "runs/regime-fixture/.queue/generation.git"
        ).read_text(encoding="utf-8") == f"{checkout_git}\n"

        claim_rows = (work / "claims").read_text(encoding="utf-8").splitlines()
        claim_workers = {row.split("|", 1)[0] for row in claim_rows}
        claims = [row.split("|", 1)[1] for row in claim_rows]
        expected = {
            f"{dataset} {seed} {drift}"
            for dataset in ("a", "b")
            for seed in (0, 1)
            for drift in (0, 25)
        }
        assert set(claims) == expected
        assert len(claims) == len(expected) + 1
        assert claims.count("a 0 0") == 2
        assert claim_workers == {"worker-0", "worker-1", "worker-2"}
        assert (work / "results/regime-fixture/FINAL_REPORT.md").is_file()
        analyses = (
            (work / "results/regime-fixture/analysis-count").read_text().splitlines()
        )
        assert analyses == ["1"]

        damaged = work / "runs/regime-fixture/fixture-s0-a-d25"
        (damaged / "score_protocol.json").write_text(
            '{"schema":"offpolicy-score-validation-split/v1"}\n'
        )
        (damaged / "scores_oracle.json").write_text('{"0":{"score":0}}\n')
        repaired = subprocess.run(
            ["/bin/bash", "scripts/run_matrix.sh"],
            cwd=checkout,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert repaired.returncode == 0, repaired.stdout + repaired.stderr
        repaired_rows = (work / "claims").read_text(encoding="utf-8").splitlines()
        repaired_claims = [row.split("|", 1)[1] for row in repaired_rows]
        assert repaired_claims[len(claims) :] == ["a 0 25"]
        assert (
            work / "results/regime-fixture/analysis-count"
        ).read_text().splitlines() == [
            "1",
            "1",
        ]

        # A supervisor pulled after an interruption must transparently resume
        # generation with the commit recorded by the existing matrix.
        (checkout / "src/supervisor_revision.txt").write_text(
            "new supervisor\n", encoding="utf-8"
        )
        subprocess.run(["git", "add", "."], cwd=checkout, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "new supervisor"], cwd=checkout, check=True
        )
        (damaged / "score_protocol.json").write_text(
            '{"schema":"offpolicy-score-validation-split/v1"}\n'
        )
        (damaged / "scores_oracle.json").write_text('{"0":{"score":0}}\n')
        claims_before_upgrade_resume = len(repaired_rows)
        wrong_pipeline = subprocess.run(
            ["/bin/bash", "scripts/run_matrix.sh"],
            cwd=checkout,
            env={**env, "OM_PIPELINE_REPO": str(checkout)},
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert wrong_pipeline.returncode != 0
        assert "partial matrix requires generation commit" in wrong_pipeline.stdout
        assert len((work / "claims").read_text().splitlines()) == len(repaired_rows)

        upgraded = subprocess.run(
            ["/bin/bash", "scripts/run_matrix.sh"],
            cwd=checkout,
            env={
                **env,
                "OM_PIPELINE_CACHE": str(root / "pipeline-cache"),
                "TEST_PROMPT_MISMATCH_ONCE": "1",
            },
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert upgraded.returncode == 0, upgraded.stdout + upgraded.stderr
        assert "[queue] restored generation checkout=" in upgraded.stdout
        upgraded_rows = (work / "claims").read_text(encoding="utf-8").splitlines()
        upgraded_claims = [row.split("|", 1)[1] for row in upgraded_rows]
        assert upgraded_claims[claims_before_upgrade_resume:] == ["a 0 25", "a 0 25"]
        assert json.loads((damaged / "run_config.json").read_text())["git"] == checkout_git
        assert list((work / "quarantine/regime-fixture").glob("*-prompt-mismatch-*"))

        collection = work / "results/regime-fixture/.regime_analysis.key"
        assert collection.is_file()
        collection.unlink()
        failed_env = {**env, "FAIL_ANALYSIS": "1"}
        failed = subprocess.run(
            ["/bin/bash", "scripts/run_matrix.sh"],
            cwd=checkout,
            env=failed_env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert failed.returncode != 0, failed.stdout + failed.stderr
        assert not collection.exists()
