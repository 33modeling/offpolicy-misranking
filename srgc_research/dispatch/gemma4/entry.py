"""One command for Gemma: downloaded weights, isolated packages, shared workers."""

import os
import subprocess
import sys


def main(argv=None):
    from . import cli

    args = list(sys.argv[1:] if argv is None else argv) or ["all"]
    if args[0] == "status":
        args = ["all", "status", *args[1:]]
    options = cli.parse_args(args)
    # argparse help/status stay GPU- and dependency-free.
    if options.action not in {"status", "results", "stop"}:
        from .model import find_snapshot
        from .runtime import activate
        from .storage import setup_storage

        setup_storage(options.root, os.environ)
        find_snapshot(os.environ)
        activate(os.environ)
    try:
        return cli.main(args)
    except (subprocess.CalledProcessError, OSError, ValueError, TypeError, RuntimeError, ImportError):
        from .diagnostics import failure_footer

        failure_footer(options.root, options.dataset)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode if error.returncode > 0 else 128 - error.returncode) from None
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as error:
        print(f"GEMMA refused: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from None
