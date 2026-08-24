"""27B rerun assignment and sibling-shard failure isolation."""

from __future__ import annotations

import re
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

print("PASS 27B jobs are unique and sibling shards survive one failure")


def test_v4_27b_runner() -> None:
    pass
