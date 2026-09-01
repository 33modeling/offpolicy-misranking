"""Generated-code reward must fail closed in every supported sandbox."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import (
    _apps_reward,
    _code_reward,
    _code_sandbox_backend,
    _run_untrusted_python,
)

available = shutil.which("bwrap") is not None and Path("/usr/bin/python3").is_file()
assert _code_reward("```python\ndef add(a, b): return a + b\n```", "assert add(2, 3) == 5") == 1.0
assert _code_reward("```python\ndef add(a, b): return 0\n```", "assert add(2, 3) == 5") == 0.0
assert _code_reward("```python\nraise SystemExit(0)\n```", "assert False") == 0.0
assert _code_reward(
    """```python
import os
for fd in range(3, 64):
    try:
        os.write(fd, b"VERIFIED")
    except OSError:
        pass
os._exit(0)
```""",
    "assert False",
) == 0.0
assert _code_reward(
    """```python
import sys
sys.modules["builtins"].len = lambda _: 0
def add(a, b): return 0
```""",
    "assert add(2, 3) == 5",
) == 0.0

if available and _code_sandbox_backend() == "bubblewrap":
    assert _run_untrusted_python("open('/home/kms/.zshrc').read()") is not None
    assert _run_untrusted_python("open('/home/kms/.zshrc').read()")[0] != 0
    gold = 'APPS:{"inputs":["2 3\\n"],"outputs":["5\\n"]}'
    code = "```python\na,b=map(int,input().split()); print(a+b)\n```"
    assert _apps_reward(code, gold) == 1.0

old_backend = os.environ.get("OM_CODE_SANDBOX")
os.environ["OM_CODE_SANDBOX"] = "restricted-subprocess"
try:
    assert _code_sandbox_backend() == "restricted-subprocess"
    assert _code_reward(
        "```python\ndef add(a, b): return a + b\n```", "assert add(2, 3) == 5"
    ) == 1.0
    assert _code_reward(
        "```python\ndef steal(): return open('/etc/passwd').read()\n```",
        "assert steal()",
    ) == 0.0
    assert _code_reward(
        "```python\nimport sys\ndef steal(): return sys.modules['os'].getcwd()\n```",
        "assert steal()",
    ) == 0.0
    assert _code_reward(
        "```python\nfrom sys import modules\ndef steal(): return modules['os'].getcwd()\n```",
        "assert steal()",
    ) == 0.0
finally:
    if old_backend is None:
        os.environ.pop("OM_CODE_SANDBOX", None)
    else:
        os.environ["OM_CODE_SANDBOX"] = old_backend

print("PASS")
