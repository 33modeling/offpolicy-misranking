"""pack_family.sh: one archive per hand-over with every log/score/report and no raw rollouts or weights."""

from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "olmo3-1025-7b-base-rlzero-grpo-h100-v2"


def _family(work: Path, dataset: str, seed: int, *, done: bool) -> Path:
    fam = work / "runs" / TAG / f"family-{dataset}-s{seed}"
    for drift in (0, 25, 100, 400):
        run = fam / f"{TAG}-s{seed}-{dataset}-d{drift}"
        (run / "logs").mkdir(parents=True)
        (run / "run_config.json").write_text('{"dataset": "%s"}' % dataset)
        (run / "scores_offpolicy.json").write_text("{}")
        (run / "report.json").write_text("{}")
        (run / "logs/main.log").write_text("progress\n")
        (run / "logs/keepalive.log").write_text("noise\n")
        (run / "rollouts_fresh_train.jsonl").write_bytes(b"x" * 200_000)
        (run / "rollouts_fresh_train.manifest.json").write_text("{}")
        (run / "rollouts_behavior_train.shard0.partial").write_bytes(b"y" * 1000)
        (run / "oracle_micro_groups.pt").write_bytes(b"t" * 5000)
        if drift:
            step = run / f"policy_step_{drift}"
            step.mkdir()
            (step / "adapter_model.safetensors").write_bytes(b"w" * 3000)
            (step / "optimizer.pt").write_bytes(b"o" * 3000)
            (step / "grpo_stats.jsonl").write_text('{"step": 1}\n')
            (step / "policy_train.json").write_text("{}")
        if done:
            (run / "DONE").write_text("done\n")
    return fam


def _env(work: Path) -> dict[str, str]:
    return {**os.environ, "OM_WORK": str(work), "GROUP_VOLUME": str(work / "no-volume"), "PACK_READOUT": "0"}


def test_packs_complete_families_without_rollouts_or_weights(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _family(work, "math500", 0, done=True)
    _family(work, "mbpp", 1, done=False)  # incomplete: not packed by default
    root = work / "runs" / TAG
    (root / ".queue").mkdir(parents=True)
    (root / ".queue/generation.git").write_text("0e4cd412ce2d09be482ee3dbba4fa1a5e969dff4\n")
    (root / "logs").mkdir()
    (root / "logs/ALERTS.log").write_text("alert\n")
    (root / ".families").mkdir()
    (root / ".families/math500-s0.owner.json").write_text("{}")
    result = subprocess.run(
        ["bash", "scripts/pack_family.sh"], cwd=ROOT, env=_env(work), text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    archives = list((work / "exports").glob("*.tar.gz"))
    assert len(archives) == 1, result.stdout
    assert "math500-s0" in archives[0].name and "mbpp" not in archives[0].name
    names = tarfile.open(archives[0]).getnames()
    joined = "\n".join(names)
    assert "MANIFEST.txt" in names
    assert f"runs/{TAG}/family-math500-s0/{TAG}-s0-math500-d25/policy_step_25/grpo_stats.jsonl" in names
    assert f"runs/{TAG}/family-math500-s0/{TAG}-s0-math500-d0/logs/main.log" in names
    assert f"runs/{TAG}/family-math500-s0/{TAG}-s0-math500-d0/rollouts_fresh_train.manifest.json" in names
    assert f"runs/{TAG}/logs/ALERTS.log" in names
    assert f"runs/{TAG}/.families/math500-s0.owner.json" in names
    assert f"runs/{TAG}/.queue/generation.git" in names
    for forbidden in ("rollouts_fresh_train.jsonl", ".partial", ".safetensors", "optimizer.pt", "oracle_micro_groups.pt", "keepalive.log"):
        assert forbidden not in joined, forbidden
    assert "family-mbpp-s1" not in joined
    manifest = tarfile.open(archives[0]).extractfile("MANIFEST.txt").read().decode()
    assert "generation_git=0e4cd412ce2d09be482ee3dbba4fa1a5e969dff4" in manifest
    assert "--- excluded files" in manifest and "rollouts_fresh_train.jsonl" in manifest
    assert "[pack] " in result.stdout and "to hand over" in result.stdout


def test_named_family_is_packed_even_when_incomplete_and_large_archives_are_split(tmp_path: Path) -> None:
    work = tmp_path / "work"
    fam = _family(work, "mbpp", 3, done=False)
    (fam / f"{TAG}-s3-mbpp-d0/logs/big.log").write_bytes(os.urandom(2_500_000))  # incompressible
    result = subprocess.run(
        ["bash", "scripts/pack_family.sh", "h100", "mbpp", "3"], cwd=ROOT,
        env={**_env(work), "PACK_PART_MB": "1"}, text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    parts = sorted((work / "exports").glob("*.tar.gz.part-*"))
    assert len(parts) >= 2, result.stdout
    assert not list((work / "exports").glob("*.tar.gz"))
    assert "split into parts" in result.stdout and "cat " in result.stdout
    joined = b"".join(p.read_bytes() for p in parts)
    rebuilt = tmp_path / "rebuilt.tar.gz"
    rebuilt.write_bytes(joined)
    names = tarfile.open(rebuilt).getnames()
    assert f"runs/{TAG}/family-mbpp-s3/{TAG}-s3-mbpp-d0/logs/big.log" in names


def test_usage_errors(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "runs" / TAG).mkdir(parents=True)
    bad = subprocess.run(["bash", "scripts/pack_family.sh", "math500"], cwd=ROOT, env=_env(work), text=True, capture_output=True)
    assert bad.returncode == 2 and "usage" in bad.stdout + bad.stderr
    none = subprocess.run(["bash", "scripts/pack_family.sh"], cwd=ROOT, env=_env(work), text=True, capture_output=True)
    assert none.returncode == 1 and "no family has all four points DONE" in none.stdout
