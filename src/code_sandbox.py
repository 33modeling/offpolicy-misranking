"""Trusted in-sandbox launcher that applies limits before generated code."""

from __future__ import annotations

import os
import resource
import sys


def main() -> None:
    timeout, code = int(sys.argv[1]), sys.argv[2]
    mib = 1024 * 1024
    resource.setrlimit(resource.RLIMIT_CPU, (timeout + 1, timeout + 1))
    resource.setrlimit(resource.RLIMIT_AS, (1024 * mib, 1024 * mib))
    resource.setrlimit(resource.RLIMIT_FSIZE, (mib, mib))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    os.execv("/usr/bin/python3", ["/usr/bin/python3", "-I", "-c", code])


if __name__ == "__main__":
    main()
