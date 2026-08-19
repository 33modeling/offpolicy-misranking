"""Generated-code reward must fail closed outside a bubblewrap sandbox."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import _apps_reward, _code_reward, _run_untrusted_python  # noqa: E402


available = shutil.which("bwrap") is not None and Path("/usr/bin/python3").is_file()
assert _code_reward("```python\ndef add(a, b): return a + b\n```", "assert add(2, 3) == 5") == (
    1.0 if available else 0.0
)
assert _code_reward("```python\ndef add(a, b): return 0\n```", "assert add(2, 3) == 5") == 0.0

if available:
    assert _run_untrusted_python("open('/home/kms/.zshrc').read()") is not None
    assert _run_untrusted_python("open('/home/kms/.zshrc').read()")[0] != 0
    gold = 'APPS:{"inputs":["2 3\\n"],"outputs":["5\\n"]}'
    code = "```python\na,b=map(int,input().split()); print(a+b)\n```"
    assert _apps_reward(code, gold) == 1.0

print("PASS")
