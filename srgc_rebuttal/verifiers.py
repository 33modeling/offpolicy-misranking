"""Binary reward verifiers selectable from a plan's ``verifier`` field.

``math_reward`` is the manuscript's MATH verifier (math-verify). ``code_reward``
executes the record's test assertions against the response's Python code block
in a separate, time-limited interpreter, as in the MBPP experiments. Both return
exactly 0.0 or 1.0 and never raise on a bad response; they raise only when the
*record* itself is unusable, so bad inputs stop before training.
"""
from __future__ import annotations

import os
import math
import re
import resource
import signal
import subprocess
import sys
import tempfile

from .run_experiment import math_reward  # noqa: F401  (re-exported for plans)

CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
CODE_TIMEOUT_SECONDS = float(os.environ.get("SRGC_CODE_TIMEOUT", "10"))
CODE_MEMORY_BYTES = int(os.environ.get("SRGC_CODE_MEMORY_MB", "1024")) * 1024 * 1024
if not math.isfinite(CODE_TIMEOUT_SECONDS) or CODE_TIMEOUT_SECONDS <= 0 or CODE_MEMORY_BYTES <= 0:
    raise ValueError("code verifier time and memory limits must be positive and finite")


def extract_code(response: str) -> str:
    """The last fenced Python block, or the whole response when there is none."""
    blocks = CODE_BLOCK.findall(response)
    return (blocks[-1] if blocks else response).strip()


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (CODE_MEMORY_BYTES, CODE_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (int(CODE_TIMEOUT_SECONDS) + 1, int(CODE_TIMEOUT_SECONDS) + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))


def code_reward(record: dict, response: str) -> float:
    tests = str(record["answer"]).strip()
    if not tests.startswith("assert"):
        raise ValueError("code records need test assertions in 'answer'; fix input before training")
    code = extract_code(response)
    if not code:
        return 0.0
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "candidate.py")
        with open(path, "w") as handle:
            handle.write(code + "\n\n" + tests + "\n")
        process = subprocess.Popen([sys.executable, "-I", "-B", path], cwd=folder,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
            preexec_fn=_limits, env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"})
        try:
            return 1.0 if process.wait(timeout=CODE_TIMEOUT_SECONDS) == 0 else 0.0
        except subprocess.TimeoutExpired:
            return 0.0
        finally:
            # A timed-out generated program must not leave descendants behind.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
