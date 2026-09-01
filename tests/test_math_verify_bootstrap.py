"""The RL-Zero math verifier must bootstrap without pip or network access."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_math_verify_bootstraps_from_an_empty_cache(tmp_path: Path) -> None:
    cache = tmp_path / "empty-cache"
    interpreter = Path("/usr/bin/python3")
    assert interpreter.is_file()
    result = subprocess.run(
        [
            str(interpreter),
            str(ROOT / "src/bootstrap_math_verify.py"),
            "--cache-root",
            str(cache),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PIP_NO_INDEX": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    runtime = Path(result.stdout.strip())
    stamp = json.loads((runtime / ".bundle.json").read_text(encoding="utf-8"))

    assert runtime.is_dir()
    assert set(stamp["wheels"]) == {
        "antlr4_python3_runtime-4.13.2-py3-none-any.whl",
        "latex2sympy2_extended-1.11.0-py3-none-any.whl",
        "math_verify-0.9.0-py3-none-any.whl",
        "mpmath-1.3.0-py3-none-any.whl",
        "sympy-1.14.0-py3-none-any.whl",
    }
    smoke = subprocess.run(
        [
            str(interpreter),
            "-c",
            (
                "from pathlib import Path; import math_verify; "
                "from math_verify import parse, verify; "
                "root=Path(r'" + str(runtime) + "').resolve(); "
                "assert root in Path(math_verify.__file__).resolve().parents; "
                "assert verify(parse(r'\\frac{1}{2}'), parse('0.5'))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(runtime),
            "PYTHONNOUSERSITE": "1",
            "PIP_NO_INDEX": "1",
        },
    )
    assert smoke.returncode == 0
