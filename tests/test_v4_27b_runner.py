"""27B rerun assignment and sibling-shard failure isolation."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/go_v4_27b.sh"


def plan(worker: int, workers: int) -> list[str]:
    output = subprocess.check_output(
        ["/bin/bash", str(RUNNER), "--plan", str(worker), str(workers)],
        cwd=REPO,
        text=True,
    )
    return output.splitlines()


expected = {
    f"{seed} {dataset}"
    for seed in range(5)
    for dataset in ("gsm8k", "math500")
}
for workers in range(1, 11):
    assignments = [job for worker in range(1, workers + 1) for job in plan(worker, workers)]
    assert len(assignments) == 10
    assert len(set(assignments)) == 10
    assert set(assignments) == expected

invalid = subprocess.run(
    ["/bin/bash", str(RUNNER), "--plan", "1", "11"],
    cwd=REPO,
    text=True,
    capture_output=True,
    check=False,
)
assert invalid.returncode == 2
for worker, workers in (("0", "1"), ("1", "0"), ("01", "2"), ("1", "08")):
    invalid = subprocess.run(
        ["/bin/bash", str(RUNNER), "--plan", worker, workers],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert invalid.returncode == 2

source = (REPO / "scripts/run_14b.sh").read_text(encoding="utf-8")
match = re.search(r"^wait_all_stages\(\) \{.*?^\}", source, re.MULTILINE | re.DOTALL)
assert match is not None
assert source.count('wait_all_stages "${pids[@]}"') == 6

with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    script = f"""
log() {{ :; }}
{match.group(0)}
(sleep 0.02; exit 7) & p1=$!
(sleep 0.10; printf ok > {tmp / 'second'}) & p2=$!
(sleep 0.15; printf ok > {tmp / 'third'}) & p3=$!
wait_all_stages "$p1" "$p2" "$p3"
rc=$?
[ "$rc" -eq 1 ] && [ -s {tmp / 'second'} ] && [ -s {tmp / 'third'} ]
"""
    waited = subprocess.run(["/bin/bash", "-c", script], check=False)
    assert waited.returncode == 0


def executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    checkout = tmp / "checkout"
    work = tmp / "work"
    fake_bin = tmp / "bin"
    model = work / "models/Qwen3.8-27B-BF16"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "src").mkdir()
    fake_bin.mkdir()
    model.mkdir(parents=True)
    (work / "venv/bin").mkdir(parents=True)
    shutil.copy2(RUNNER, checkout / "scripts/go_v4_27b.sh")
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (checkout / "scripts/setup_env.sh").write_text(
        'export OM_WORK="${OM_WORK:?}"\n'
        'export VENV_DIR="$OM_WORK/venv"\n'
        'export MODELS_DIR="$OM_WORK/models"\n',
        encoding="utf-8",
    )
    executable(
        checkout / "scripts/go_v2.sh",
        "#!/usr/bin/env bash\n"
        'run="$RUN_BASE-s$SEEDS"\n'
        '[ "$DATASETS" = gsm8k ] || run="$run-$DATASETS"\n'
        'printf "%s %s\\n" "$SEEDS" "$DATASETS" >> "$OM_WORK/claims"\n'
        'if [ "$SEEDS $DATASETS" = "0 gsm8k" ] '
        '&& mkdir "$OM_WORK/fail-once" 2>/dev/null; then exit 9; fi\n'
        "/bin/sleep 0.08\n"
        'mkdir -p "$run"\n'
        "for artifact in DONE run_config.json manifest.json score_protocol.json "
        "oracle_protocol.json report.json scores_oracle.json scores_offpolicy.json "
        "scores_splithalf.json oracle_micro_groups.pt val_groups.pt; do "
        'printf "{}\\n" > "$run/$artifact"; done\n',
    )
    executable(
        checkout / "scripts/collect_v4.sh",
        "#!/usr/bin/env bash\n"
        'mkdir -p "$OM_WORK/results/v4-27b" "$OM_WORK/results/v4-7b"\n'
        'printf "# tables\\n" > "$OM_WORK/results/v4-27b/TABLES.md"\n'
        'printf "# tables\\n" > "$OM_WORK/results/v4-7b/TABLES.md"\n',
    )
    executable(
        fake_bin / "nvidia-smi",
        "#!/usr/bin/env bash\n"
        'if [ "${1:-}" = -L ]; then printf "GPU 0\\nGPU 1\\nGPU 2\\nGPU 3\\n"; exit 0; fi\n'
        'case "$*" in *index,memory.used*) printf "0, 0\\n1, 0\\n2, 0\\n3, 0\\n";; esac\n',
    )
    executable(fake_bin / "sleep", "#!/usr/bin/env bash\n/bin/sleep 0.01\n")
    executable(
        work / "venv/bin/python",
        "#!/usr/bin/env bash\n"
        '[ "${1:-}" = - ] && /bin/cat >/dev/null\n'
        "exit 0\n",
    )

    artifacts = (
        "DONE",
        "run_config.json",
        "manifest.json",
        "score_protocol.json",
        "oracle_protocol.json",
        "report.json",
        "scores_oracle.json",
        "scores_offpolicy.json",
        "scores_splithalf.json",
        "oracle_micro_groups.pt",
        "val_groups.pt",
    )
    for seed in range(5):
        for suffix in ("", "-math500"):
            run = work / f"runs/v4-7b-s{seed}{suffix}"
            run.mkdir(parents=True)
            for artifact in artifacts:
                (run / artifact).write_text("{}\n", encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=queue-test",
            "-c",
            "user.email=queue@test.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=checkout,
        check=True,
    )
    env = os.environ.copy()
    env.update({"OM_WORK": str(work), "PATH": f"{fake_bin}:{env['PATH']}"})
    processes = [
        subprocess.Popen(
            ["/bin/bash", "scripts/go_v4_27b.sh"],
            cwd=checkout,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for _ in range(3)
    ]
    for process in processes:
        output, _ = process.communicate(timeout=20)
        assert process.returncode == 0, output

    claims = (work / "claims").read_text(encoding="utf-8").splitlines()
    assert len(claims) == 11
    assert len(set(claims)) == 10
    assert set(claims) == expected
    assert claims.count("0 gsm8k") == 2

print("PASS 27B jobs are unique and sibling shards survive one failure")


def test_v4_27b_runner() -> None:
    pass
