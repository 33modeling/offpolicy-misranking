"""Untrusted candidate interpreter; only the parent decides whether tests pass.

This worker receives expressions, never expected answers or an authority to
award reward. Resource limits and disposable compute are still required.
"""

import builtins
import json
import os
import sys


def literal_result(value, depth=0):
    if depth > 64:
        raise ValueError("candidate result is too deeply nested")
    kind = type(value)
    if kind in (type(None), bool, int, float, complex, str, bytes):
        return repr(value)
    if kind in (list, tuple, set):
        for item in value:
            literal_result(item, depth + 1)
    elif kind is dict:
        for key, item in value.items():
            literal_result(key, depth + 1)
            literal_result(item, depth + 1)
    else:
        raise ValueError("candidate result must be literal data, not executable objects")
    return repr(value)


def main():
    payload_path, result_fd = sys.argv[1:]
    with open(payload_path) as handle:
        payload = json.load(handle)
    execute, evaluate, write, encode = exec, eval, os.write, json.dumps
    pristine = builtins.__dict__.copy()
    namespace = {"__name__": "__main__"}
    try:
        candidate = compile(payload["code"], "<candidate>", "exec")
        expressions = [compile(expression, "<test-input>", "eval") for expression in payload["expressions"]]
        execute(candidate, namespace)
        builtins.__dict__.clear()
        builtins.__dict__.update(pristine)
        namespace["__builtins__"] = pristine
        values = []
        for expression in expressions:
            value = evaluate(expression, namespace)
            builtins.__dict__.clear()
            builtins.__dict__.update(pristine)
            values.append(literal_result(value))
        report = encode({"values": values}).encode("utf-8")
        offset = 0
        while offset < len(report):
            offset += write(int(result_fd), report[offset:])
    except BaseException:
        # SystemExit(0) must fail just like a failed assertion.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
