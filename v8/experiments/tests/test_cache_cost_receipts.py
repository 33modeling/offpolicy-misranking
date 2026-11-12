"""cyHH Q3: cold-start must include the initial SR-cache generation; unknown stays unknown, never zero."""
import json

from switching import cluster


def _setup(tmp_path, summary):
    bundle = tmp_path / "inputs" / "seed-5.json"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("{}")
    (tmp_path / "locator.json").write_text(json.dumps({"seeds": {"5": {"input_bundle": str(bundle)}}}))
    if summary is not None:
        cache = bundle.with_suffix(".cache")
        cache.mkdir()
        (cache / "cost-summary.json").write_text(json.dumps(summary))


def test_complete_cache_receipts_give_the_inclusive_total(tmp_path):
    _setup(tmp_path, {"complete": True, "invocations": {"total_gpu_seconds": 7200.0, "known_gpu_seconds": {"a": 7200.0}}})
    assert cluster.cache_cost_from_receipts(tmp_path, 5) == 7200.0


def test_incomplete_or_missing_receipts_are_unknown_with_reason(tmp_path):
    _setup(tmp_path, {"complete": False, "invocations": {"total_gpu_seconds": None, "known_gpu_seconds": {"a": 100.0}}})
    detail = {}
    assert cluster.cache_cost_from_receipts(tmp_path, 5, detail) is None
    assert detail["status"] == "incomplete receipts" and detail["known_lower_bound_gpu_seconds"] == 100.0
    detail = {}
    assert cluster.cache_cost_from_receipts(tmp_path, 6, detail) is None and "locator" in detail["status"]


def test_no_summary_is_unknown(tmp_path):
    _setup(tmp_path, None)
    detail = {}
    assert cluster.cache_cost_from_receipts(tmp_path, 5, detail) is None and "no cost-summary" in detail["status"]
