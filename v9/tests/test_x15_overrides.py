"""X15: the plan override changes only prompts per update and candidates per draw, after code-repo activation."""

import sys
import types

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import child  # noqa: E402


def test_plan_override_applies_after_activation(monkeypatch):
    from switching import coderepo
    plan_mod = types.ModuleType("srgc_rebuttal.plan")
    plan_mod.load_plan = lambda path: {"training_prompts": 4, "scoring_prompts_per_set": 40, "responses": 8}
    pkg = types.ModuleType("srgc_rebuttal")
    pkg.plan = plan_mod
    monkeypatch.setitem(sys.modules, "srgc_rebuttal", pkg)
    monkeypatch.setitem(sys.modules, "srgc_rebuttal.plan", plan_mod)
    monkeypatch.setattr(coderepo, "activate", lambda *a, **k: "root")
    monkeypatch.setattr(coderepo, "activate_source_runtime", lambda *a, **k: None)
    child.install_plan_overrides({"training_prompts": 16, "scoring_prompts_per_set": 160})
    assert plan_mod.load_plan("p")["training_prompts"] == 4  # not before activation
    assert coderepo.activate() == "root"
    plan = sys.modules["srgc_rebuttal.plan"].load_plan("p")
    assert plan == {"training_prompts": 16, "scoring_prompts_per_set": 160, "responses": 8}
    coderepo.activate_source_runtime()  # second activation keeps one wrapper
    assert sys.modules["srgc_rebuttal.plan"].load_plan("p")["training_prompts"] == 16


def test_plan_override_rejects_other_keys():
    with pytest.raises(SystemExit):
        child.install_plan_overrides({"responses": 4})
