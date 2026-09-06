import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diagnose_launch_failure import diagnose, last_exception

CASES = {
    'RuntimeError: math-verify is required for verifier-reward math experiments': "math-verify",
    '[model-abort] qwen3.5-9b-posttrained: safetensors shard set incomplete': "*.safetensors",
    '[model-abort] no *.safetensors in /group-volume/models/Qwen3.5-9B (and no usable model.safetensors.index.json)': "*.safetensors",
    "ModuleNotFoundError: No module named 'transformers.models.qwen3_5'": "transformers",
    "RuntimeError: expected FLA 0.5.2, got 0.4.1": "fla-core",
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB": "memory",
    "[abort] exactly four H100 GPUs required (GPUs=2 H100=2)": "4x H100",
    "[additional] worker=x queued behind local primary at git=abc": "primary",
    "[additional] waiting for shared snapshot preparation lock": "prepare is blocked",
}


def test_known_signatures_get_a_diagnosis():
    for line, keyword in CASES.items():
        found = diagnose(f"noise\n{line}\n[exit] utc=x rc=1\n")
        assert found is not None, line
        assert keyword in found[0], (line, found[0])


def test_unknown_failure_reports_last_exception():
    text = "Traceback (most recent call last):\n  File x\nZeroDivisionError: division by zero\n[exit] rc=1\n"
    assert diagnose(text) is None
    assert last_exception(text).startswith("ZeroDivisionError")


def test_latest_signature_wins():
    text = "RuntimeError: math-verify is required\n... later ...\n[abort] exactly four H100 GPUs required\n"
    assert "4x H100" in diagnose(text)[0]
