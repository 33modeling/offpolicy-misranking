"""Llama shim: identity rule and the exact blocked reasons (no Llama weights on CPU)."""

import hashlib
import json

import pytest

from v9lib import llama
from v9lib.common import Blocked


def test_expected_identity_is_engine_plus_adapter_digest():
    from switching.coderepo import implementation_digest
    from srgc_research.dispatch.llama31.adapter import adapter_digest
    from v9lib import env
    identity, adapter = llama.expected_identity(env.code_root())
    assert adapter == adapter_digest()
    assert identity == hashlib.sha256((implementation_digest(env.code_root()) + adapter).encode()).hexdigest()


def test_missing_plan_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("GROUP_VOLUME", str(tmp_path))
    monkeypatch.setenv("OM_USER", "u")
    monkeypatch.delenv("OM_WORK", raising=False)
    monkeypatch.delenv("V9_ALLOW_FAKE", raising=False)
    monkeypatch.setenv("V9_ROOT", str(tmp_path / "v9"))
    with pytest.raises(Blocked, match="Llama plan not found"):
        llama.locate("llama_mbpp")


def test_verify_child_refuses_other_identity():
    with pytest.raises(SystemExit):
        llama.verify_child({"run_implementation_sha256": "0" * 64})


def test_fake_locator_requires_explicit_permission(tmp_path, monkeypatch):
    monkeypatch.setenv("GROUP_VOLUME", str(tmp_path))
    monkeypatch.setenv("OM_USER", "u")
    monkeypatch.delenv("OM_WORK", raising=False)
    monkeypatch.setenv("V9_ROOT", str(tmp_path / "v9"))
    fake = tmp_path / "u" / "offpolicy-misranking" / "v9-fake-inputs" / "llama_math-locator.json"
    fake.parent.mkdir(parents=True)
    fake.write_text(json.dumps({"seeds": {}}))
    monkeypatch.delenv("V9_ALLOW_FAKE", raising=False)
    with pytest.raises(Blocked):
        llama.locate("llama_math")
    monkeypatch.setenv("V9_ALLOW_FAKE", "1")
    assert llama.locate("llama_math") == {"seeds": {}}
