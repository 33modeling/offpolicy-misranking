"""Multiple workers claim each regime point once and recover a transient failure."""

from __future__ import annotations

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
        shutil.copy2(REPO / "scripts/go_regime.sh", checkout / "scripts/go_regime.sh")
        (work / "models/model/config.json").write_text("{}\n", encoding="utf-8")
        (work / "matrix.json").write_text("{}\n", encoding="utf-8")
        (work / "venv/bin/python").symlink_to(Path(sys.executable))
        (checkout / "scripts/setup_env.sh").write_text(
            'export OM_WORK="${OM_WORK:?}"\n'
            'export VENV_DIR="$OM_WORK/venv"\n'
            'export MODELS_DIR="$OM_WORK/models"\n',
            encoding="utf-8",
        )
        executable(
            checkout / "scripts/run_14b.sh",
            "#!/usr/bin/env bash\n"
            'point="$DATASET $SEED $DRIFT"\n'
            'printf "%s\\n" "$point" >> "$OM_WORK/claims"\n'
            'if [ "$point" = "a 0 0" ] && mkdir "$OM_WORK/fail-once" 2>/dev/null; then exit 9; fi\n'
            'mkdir -p "$OUT_ROOT"\n'
            'for name in DONE run_config.json manifest.json score_protocol.json '
            'oracle_protocol.json report.json scores_oracle.json scores_offpolicy.json '
            'scores_splithalf.json oracle_micro_groups.pt val_groups.pt; do '
            'printf "{}\\n" > "$OUT_ROOT/$name"; done\n',
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

        env = os.environ.copy()
        env.update(
            {
                "OM_WORK": str(work),
                "PATH": f"{fake_bin}:{env['PATH']}",
                "MODEL_14B": str(work / "models/model"),
                "REGIME_MODEL_TAG": "fixture",
                "REGIME_ROOT": str(work / "runs/regime-fixture"),
                "REGIME_RESULTS": str(work / "results/regime-fixture"),
                "REGIME_MATRIX": str(work / "matrix.json"),
                "REGIME_DATASETS": "a b",
                "REGIME_SEEDS": "0 1",
                "REGIME_DRIFTS": "0 25",
                "REGIME_MAX_RETRIES": "2",
                "REGIME_N_TRAIN": "8",
                "REGIME_N_VAL": "4",
            }
        )
        workers = [
            subprocess.Popen(
                ["/bin/bash", "scripts/go_regime.sh"],
                cwd=checkout,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            for _ in range(3)
        ]
        outputs = []
        for worker in workers:
            output, _ = worker.communicate(timeout=30)
            outputs.append(output)
            assert worker.returncode == 0, output

        claims = (work / "claims").read_text(encoding="utf-8").splitlines()
        expected = {
            f"{dataset} {seed} {drift}"
            for dataset in ("a", "b")
            for seed in (0, 1)
            for drift in (0, 25)
        }
        assert set(claims) == expected
        assert len(claims) == len(expected) + 1
        assert claims.count("a 0 0") == 2
        assert (work / "results/regime-fixture/FINAL_REPORT.md").is_file()
        analyses = (work / "results/regime-fixture/analysis-count").read_text().splitlines()
        assert analyses == ["1"]

        collection = work / "results/regime-fixture/.regime_collection.json"
        collection.unlink()
        failed_env = {**env, "FAIL_ANALYSIS": "1"}
        failed = subprocess.run(
            ["/bin/bash", "scripts/go_regime.sh"],
            cwd=checkout,
            env=failed_env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert failed.returncode != 0, failed.stdout + failed.stderr
        assert not collection.exists()
