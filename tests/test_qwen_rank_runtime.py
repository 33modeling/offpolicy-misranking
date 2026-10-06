import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.srgc_qwen35_rank_runtime import isolated_storage, main


CACHE_KEYS = ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH", "TMPDIR")


def test_operational_rank_entries_keep_live_owners_and_identify_orphans():
    from scripts import srgc_process_guard as guard
    uid = os.getuid()
    for entry in ("srgc_qwen35_smoke.py", "srgc_qwen35_rank_runtime.py"):
        table = {
            900001: (1, uid, "python scripts/srgc_qwen35_diagnostics.py all run"),
            900002: (900001, uid, f"python -m torch.distributed.run scripts/{entry}"),
            900003: (900002, uid, f"python scripts/{entry}"),
            900004: (1, uid, f"python scripts/{entry}"),
            900005: (1, uid, "python -m torch.distributed.run unrelated.py"),
        }
        assert guard.orphan_pids(table=table) == [900004]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    storage = importlib.import_module("srgc_qwen35_storage")
    group = tmp_path / "group"
    group.mkdir()
    environment = {"GROUP_VOLUME": str(group), "OM_WORK": str(group / "work"),
                   "LOCAL_RANK": "0", "WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4"}
    return storage, group / "experiment", environment


def test_ranks_have_distinct_stable_caches_and_preserve_shared_hf_and_old_cache(setup):
    storage, root, environment = setup
    storage.setup_storage(root, environment)
    old = Path(environment["TRITON_CACHE_DIR"])
    marker = old / "existing-artifact"
    marker.write_text("preserved")
    hf = environment["HF_HOME"]
    with isolated_storage():
        result = storage.setup_storage(root, environment)
        first = {key: environment[key] for key in CACHE_KEYS}
        storage.setup_storage(root, environment)
        assert first == {key: environment[key] for key in CACHE_KEYS}
        other = {**environment, "LOCAL_RANK": "1"}
        storage.setup_storage(root, other)
    assert environment["HF_HOME"] == other["HF_HOME"] == hf
    assert result[0] == Path(environment["GROUP_VOLUME"])
    for key in CACHE_KEYS:
        assert first[key] != other[key]
        assert "rank-runtime-v1/rank-0/" in first[key]
        assert "rank-runtime-v1/rank-1/" in other[key]
        assert Path(first[key]).is_dir() and Path(other[key]).is_dir()
    assert Path(first["TRITON_CACHE_DIR"]) != old
    assert marker.read_text() == "preserved"


def test_missing_group_has_no_home_fallback(setup, tmp_path):
    storage, root, environment = setup
    environment["GROUP_VOLUME"] = str(tmp_path / "missing")
    with isolated_storage(), pytest.raises(ValueError, match="group volume is unavailable"):
        storage.setup_storage(root, environment)
    assert "TRITON_CACHE_DIR" not in environment
    assert not Path(environment["GROUP_VOLUME"]).exists()


def test_rank_cache_symlink_escape_fails_closed(setup, tmp_path):
    storage, root, environment = setup
    storage.setup_storage(root, environment)
    runtime = Path(environment["TRITON_CACHE_DIR"]).parent / "rank-runtime-v1/rank-0"
    runtime.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (runtime / "triton").symlink_to(outside, target_is_directory=True)
    with isolated_storage(), pytest.raises(ValueError, match="path must be below group storage"):
        storage.setup_storage(root, environment)
    assert list(outside.iterdir()) == []


def test_original_storage_validation_and_return_value_are_preserved(setup, tmp_path, monkeypatch):
    storage, root, environment = setup
    original = storage.setup_storage
    results = []

    def recorded(*args, **kwargs):
        assert kwargs == {"scan_tree": False}
        result = original(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(storage, "setup_storage", recorded)
    with isolated_storage():
        assert storage.setup_storage(root, environment, scan_tree=False) is results[-1]
        with pytest.raises(ValueError, match="path must be below group storage"):
            storage.setup_storage(tmp_path / "outside-group", environment, scan_tree=False)
    assert storage.setup_storage is recorded


@pytest.mark.parametrize("override", [{"LOCAL_RANK": "4"}, {"LOCAL_RANK": "-1"},
    {"LOCAL_RANK": "bad"}, {"WORLD_SIZE": "8"}, {"LOCAL_WORLD_SIZE": "2"}, {"LOCAL_RANK": None}])
def test_invalid_rank_environment_fails_closed(setup, override):
    storage, root, environment = setup
    environment.update(override)
    with isolated_storage(), pytest.raises(ValueError, match="rank"):
        storage.setup_storage(root, environment)


@pytest.mark.parametrize("stage", ["cache", "train"])
def test_entry_delegates_with_exact_module_patch_and_restores(setup, monkeypatch, stage):
    storage, root, environment = setup
    original = storage.setup_storage
    arguments = ["rank-runtime", "--stage", stage, "--plan", "/unchanged/plan.json"]
    monkeypatch.setattr(sys, "argv", arguments)

    def rank_main():
        assert sys.argv is arguments
        assert storage.setup_storage is not original
        assert importlib.import_module("srgc_qwen35_storage") is storage
        storage.setup_storage(root, environment, scan_tree=False)
        return "delegated"

    monkeypatch.setitem(sys.modules, "srgc_qwen35_rank", SimpleNamespace(main=rank_main))
    assert main() == "delegated"
    assert storage.setup_storage is original
    assert "rank-runtime-v1/rank-0/triton" in environment["TRITON_CACHE_DIR"]


def test_entry_restores_storage_and_propagates_rank_error(setup, monkeypatch):
    storage, _, _ = setup
    original = storage.setup_storage
    failure = RuntimeError("rank failed")

    def rank_main():
        raise failure

    monkeypatch.setitem(sys.modules, "srgc_qwen35_rank", SimpleNamespace(main=rank_main))
    with pytest.raises(RuntimeError) as caught:
        main()
    assert caught.value is failure
    assert storage.setup_storage is original
