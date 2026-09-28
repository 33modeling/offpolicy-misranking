"""Historical test identities must be real release blobs, not relaxed guards."""

import hashlib
import subprocess

import pinned_trainers as pins


def test_released_trainer_hashes_match_archived_git_blobs():
    for path, expected in pins.RELEASED_TRAINER_HASHES.items():
        content = subprocess.check_output(["git", "show", f"{pins.RELEASED_COMMIT}:{path}"], cwd=pins.base.ROOT)
        assert hashlib.sha256(content).hexdigest() == expected


def test_pin_wrapper_does_not_hide_unrelated_or_output_file_changes(tmp_path):
    target = tmp_path / "src/train_policy_grpo.py"
    target.parent.mkdir()
    target.write_text("first")
    digest = pins.released_digest(pins.base.digest)
    before = digest(target)
    target.write_text("second")
    assert digest(target) != before
    assert digest(pins.base.ROOT / "src/grads.py") == pins.base.digest(pins.base.ROOT / "src/grads.py")
