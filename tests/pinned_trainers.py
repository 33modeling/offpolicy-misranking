"""Hash Pair trainers as the deployed runtime does.

The deployed Pair executes the pinned commit c0c38d6, and its operations overlay
never replaces a trainer. ba2f4d8 later changed the working-tree trainers for
checkpoint retention, which the frozen Pair contract correctly refuses. Pair
tests therefore hash the released trainers with their exact `git show c0c38d6:path`
digests. Run a Pair child process the same way with:

    python tests/pinned_trainers.py <script> [args...]
    python tests/pinned_trainers.py -c <code> [args...]
"""
import runpy
import sys
import subprocess
from pathlib import Path

import selection_gate_gpu as base


RELEASED_TRAINER_HASHES = {
    "src/train_mopps_grpo.py": "da79c48c2c41679322e61481f613f772d8744927a23cfbd9b5c7d30d5e12af9a",
    "src/selection_switch_curve_train.py": "49a36f999ae0e65ee1335d14af39d61c5a6467906534341af4ea593a70e456f5",
    "src/train_policy_grpo.py": "e53b8eb2b2135ed246ace8f68c9011f8fb8fa442690902575121c680ed45b070",
    "src/train_selection_gate_grpo.py": "c84f4a63cdeb40ee63feedffec4f3491089db35fb9fd9bbe243e1a2f09efbc9f",
}
COMMAND = str(Path(__file__).resolve())
RELEASED_COMMIT = "c0c38d62e893fdcf25920d5a70d3149a06b5450f"


def legacy_pair_command(mode):
    """Exercise the deployed legacy launcher; current dispatch has separate tests."""
    path = "scripts/run_selector_pair.sh"
    source = subprocess.check_output(["git", "show", f"{RELEASED_COMMIT}:{path}"], cwd=base.ROOT, text=True)
    return ["bash", "-c", source, str(base.ROOT / path), mode]


def write_python_wrapper(path):
    """Use the same reviewed trainer identities in shell-launched test children."""
    import shlex
    path.write_text("#!/bin/sh\nexec " + shlex.join([sys.executable, COMMAND]) + ' "$@"\n')
    path.chmod(0o755)
    return str(path)


def released_digest(digest):
    released = {(base.ROOT / name).resolve(): value for name, value in RELEASED_TRAINER_HASHES.items()}

    def pinned(path):
        return released.get(Path(path).resolve()) or digest(path)
    return pinned


def main(argv):
    base.digest = released_digest(base.digest)
    if argv[0] == "-c":
        sys.argv = ["-c", *argv[2:]]
        sys.path[0] = ""
        exec(compile(argv[1], "<string>", "exec"), {"__name__": "__main__"})
        return
    sys.argv = argv
    sys.path[0] = str(Path(argv[0]).resolve().parent)
    runpy.run_path(argv[0], run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
