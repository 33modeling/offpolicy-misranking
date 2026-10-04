"""Trusted assertion runner; resource limits and process isolation belong to the parent.

This completion protocol prevents early exits from passing. It is not an
OS security sandbox; generated-code deployments still require disposable compute.
"""

import builtins
import json
import os
import sys


def main():
    payload_path, completion_fd, token = sys.argv[1:]
    with open(payload_path) as handle:
        payload = json.load(handle)
    execute, write = exec, os.write
    pristine = builtins.__dict__.copy()
    namespace = {"__name__": "__main__"}
    try:
        candidate = compile(payload["code"], "<candidate>", "exec")
        assertions = compile(payload["tests"], "<tests>", "exec")
        execute(candidate, namespace)
        builtins.__dict__.clear()
        builtins.__dict__.update(pristine)
        namespace["__builtins__"] = pristine
        execute(assertions, namespace)
    except BaseException:
        # SystemExit(0) must fail just like a failed assertion.
        return 1
    write(int(completion_fd), token.encode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
