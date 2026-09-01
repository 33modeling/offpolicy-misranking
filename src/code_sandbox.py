"""Trusted in-sandbox launcher that applies limits before generated code."""

from __future__ import annotations

import ast
import base64
import builtins
import os
import resource
import sys

_ALLOWED_IMPORTS = {
    "array",
    "bisect",
    "calendar",
    "cmath",
    "collections",
    "copy",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
    "sys",
}
_BLOCKED_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}
_BLOCKED_ATTRIBUTES = {
    "__base__",
    "__bases__",
    "__builtins__",
    "__class__",
    "__closure__",
    "__code__",
    "__dict__",
    "__getattribute__",
    "__globals__",
    "__loader__",
    "__mro__",
    "__spec__",
    "__subclasses__",
    "modules",
    "open",
    "path_hooks",
    "meta_path",
    "setprofile",
    "settrace",
    "_exit",
    "abort",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "fork",
    "forkpty",
    "kill",
    "killpg",
    "popen",
    "posix_spawn",
    "posix_spawnp",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "system",
}


def _validated_candidate(source: str):
    tree = ast.parse(source, filename="<candidate>", mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".", 1)[0] for alias in node.names]
            if any(module not in _ALLOWED_IMPORTS for module in modules):
                raise ValueError("candidate imports a module outside the verifier allowlist")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if node.level or module not in _ALLOWED_IMPORTS:
                raise ValueError("candidate imports a module outside the verifier allowlist")
            if any(alias.name.startswith("_") for alias in node.names):
                raise ValueError("candidate imports a private module attribute")
            if module == "sys" and any(alias.name != "maxsize" for alias in node.names):
                raise ValueError("candidate imports an unsafe sys attribute")
        elif isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            raise ValueError(f"candidate uses blocked name {node.id}")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError(f"candidate uses blocked private name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in _BLOCKED_ATTRIBUTES:
            raise ValueError(f"candidate uses blocked attribute {node.attr}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"candidate uses blocked private attribute {node.attr}")
    return compile(tree, "<candidate>", "exec")


def _restore_builtins(snapshot: dict[str, object]) -> None:
    namespace = builtins.__dict__
    for name in set(namespace) - set(snapshot):
        del namespace[name]
    namespace.update(snapshot)


def main() -> None:
    timeout = int(sys.argv[1])
    code = base64.b64decode(sys.argv[2], validate=True).decode()
    tests = None if sys.argv[3] == "-" else base64.b64decode(
        sys.argv[3], validate=True
    ).decode()
    mib = 1024 * 1024
    resource.setrlimit(resource.RLIMIT_CPU, (timeout + 1, timeout + 1))
    resource.setrlimit(resource.RLIMIT_AS, (1024 * mib, 1024 * mib))
    resource.setrlimit(resource.RLIMIT_FSIZE, (mib, mib))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    if tests is None:
        os.execv("/usr/bin/python3", ["/usr/bin/python3", "-I", "-c", code])

    # Candidate and checks stay in one resource-limited process. Static checks
    # reject APIs that can terminate or replace the interpreter before the
    # trusted assertions run; SystemExit and all other BaseException paths fail.
    try:
        candidate = _validated_candidate(code)
        checks = compile(tests, "<verifier-tests>", "exec")
    except (SyntaxError, ValueError):
        raise SystemExit(1)
    namespace = {"__name__": "__candidate__"}
    pristine_builtins = builtins.__dict__.copy()
    try:
        exec(candidate, namespace)  # noqa: S102 - bubblewrap-isolated candidate
        _restore_builtins(pristine_builtins)
        namespace["__builtins__"] = pristine_builtins
        exec(checks, namespace)  # noqa: S102 - trusted checks, same namespace
    except BaseException:  # noqa: BLE001 - SystemExit must fail verification
        raise SystemExit(1)


if __name__ == "__main__":
    main()
