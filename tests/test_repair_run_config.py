"""repair_run_config must rewrite only runtime fields of unfinished points and
keep the digest exactly as scripts/run_point.sh recomputes it at re-entry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import repair_run_config as rrc


def _run_point_digest(config: dict) -> str:
    """The formula from scripts/run_point.sh (digest key excluded)."""
    body = {k: v for k, v in config.items() if k != "digest"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_point(run: Path, *, gen_batch: str = "8", micro: int = 4, done: bool = False) -> dict:
    run.mkdir(parents=True)
    config = {
        "dataset": "mbpp",
        "seed": 0,
        "drift": 25,
        "n_train": 512,
        "gen_batch": gen_batch,
        "gradient_micro_batch": micro,
        "grpo_logprob_micro_batch": 4,
        "git": "0" * 40,
        "git_status": "",
        "git_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "temperature": 1.0,
        "lora_targets": "q_proj,v_proj",
    }
    config["digest"] = _run_point_digest(config)
    (run / "run_config.json").write_text(json.dumps(config, indent=1))
    manifest = {"time": "2026-09-06 00:00:00", **config, "torch": "2.7.1", "cuda": "12.6"}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if done:
        (run / "DONE").write_text("ok\n")
    return config


def test_dry_run_changes_nothing_and_reports(tmp_path: Path) -> None:
    run = tmp_path / "family-mbpp-s0" / "tag-s0-mbpp-d25"
    before = _write_point(run)
    lines = rrc.repair_run(run, {"gen_batch": 16, "gradient_micro_batch": 1}, apply=False)
    assert [line for line in lines if "gen_batch '8' -> '16' (dry run)" in line]
    assert [line for line in lines if "gradient_micro_batch 4 -> 1 (dry run)" in line]
    assert json.loads((run / "run_config.json").read_text()) == before


def test_apply_rewrites_fields_digest_and_manifest(tmp_path: Path) -> None:
    run = tmp_path / "family-mbpp-s0" / "tag-s0-mbpp-d25"
    before = _write_point(run)
    lines = rrc.repair_run(run, {"gen_batch": 16, "gradient_micro_batch": 1}, apply=True)
    assert len(lines) == 2
    after = json.loads((run / "run_config.json").read_text())
    # gen_batch stays a string (run_point reads OM_GEN_BATCH as text), the micro-batch an int
    assert after["gen_batch"] == "16" and isinstance(after["gen_batch"], str)
    assert after["gradient_micro_batch"] == 1 and isinstance(after["gradient_micro_batch"], int)
    # nothing else moved
    untouched = {k: v for k, v in after.items() if k not in {"gen_batch", "gradient_micro_batch", "digest"}}
    assert untouched == {k: v for k, v in before.items() if k not in {"gen_batch", "gradient_micro_batch", "digest"}}
    # digest is what run_point would compute for this exact record
    assert after["digest"] == _run_point_digest(after)
    assert after["digest"] != before["digest"]
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["gen_batch"] == "16" and manifest["gradient_micro_batch"] == 1
    assert manifest["digest"] == after["digest"]
    assert manifest["torch"] == "2.7.1"


def test_finished_points_are_never_touched(tmp_path: Path) -> None:
    run = tmp_path / "family-mbpp-s0" / "tag-s0-mbpp-d0"
    before = _write_point(run, done=True)
    assert rrc.repair_run(run, {"gen_batch": 16, "gradient_micro_batch": 1}, apply=True) == []
    assert json.loads((run / "run_config.json").read_text()) == before


def test_matching_values_are_a_no_op(tmp_path: Path) -> None:
    run = tmp_path / "family-mbpp-s0" / "tag-s0-mbpp-d25"
    before = _write_point(run, gen_batch="16", micro=1)
    assert rrc.repair_run(run, {"gen_batch": 16, "gradient_micro_batch": 1}, apply=True) == []
    assert json.loads((run / "run_config.json").read_text()) == before


def test_family_root_walks_every_point_and_skips_done(tmp_path: Path) -> None:
    family = tmp_path / "family-math500-s1"
    _write_point(family / "tag-s1-math500-d25", done=True)
    _write_point(family / "tag-s1-math500-d400")
    (family / "logs").mkdir()
    rc = rrc.main(["--family-root", str(family), "--gen-batch", "32", "--apply"])
    assert rc == 0
    done_cfg = json.loads((family / "tag-s1-math500-d25" / "run_config.json").read_text())
    live_cfg = json.loads((family / "tag-s1-math500-d400" / "run_config.json").read_text())
    assert done_cfg["gen_batch"] == "8"
    assert live_cfg["gen_batch"] == "32" and live_cfg["gradient_micro_batch"] == 4
    assert live_cfg["digest"] == _run_point_digest(live_cfg)


def test_cli_rejects_nothing_to_do_and_bad_values(tmp_path: Path) -> None:
    run = tmp_path / "p"
    _write_point(run)
    with pytest.raises(SystemExit):
        rrc.main(["--run", str(run)])
    with pytest.raises(SystemExit):
        rrc.main(["--run", str(run), "--gen-batch", "0"])
