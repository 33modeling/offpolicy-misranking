import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.srgc_prefix_check import verify_prefix


class PrefixCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.expected = {"seed": 5, "implementation_sha256": "frozen"}
        self.content = b"prefix fixture" * 100
        self.checkpoint = self.folder / "prefix.pt"
        self.checkpoint.write_bytes(self.content)
        self.sha = hashlib.sha256(self.content).hexdigest()
        (self.folder / "prefix-ready.json").write_text(json.dumps({
            **self.expected, "completed_updates": 25, "checkpoint_sha256": self.sha}))

    def test_hash_matches_without_reading_whole_file_into_memory(self):
        output = io.StringIO()
        with patch.object(Path, "read_bytes", side_effect=AssertionError("unbounded read")), \
                contextlib.redirect_stderr(output):
            self.assertEqual(verify_prefix(self.folder, self.expected, 25), self.sha)
        self.assertIn("PREFIX checking", output.getvalue())
        self.assertIn("PREFIX verified", output.getvalue())

    def test_nonprimary_preparation_does_not_hash_checkpoint(self):
        with patch("scripts.srgc_prefix_check.hashlib.sha256", side_effect=AssertionError("duplicate hash")):
            self.assertEqual(verify_prefix(self.folder, self.expected, 25, verify_checkpoint=False), self.sha)

    def test_changed_bytes_and_wrong_identity_are_rejected(self):
        self.checkpoint.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "checkpoint differs"):
            verify_prefix(self.folder, self.expected, 25)
        with self.assertRaisesRegex(ValueError, "different experiment"):
            verify_prefix(self.folder, {**self.expected, "seed": 6}, 25, verify_checkpoint=False)

    def test_large_file_reads_are_bounded_and_report_progress(self):
        payload = b"x" * (9 * 1024 * 1024)
        self.checkpoint.write_bytes(payload)
        marker = self.folder / "prefix-ready.json"
        receipt = json.loads(marker.read_text())
        receipt["checkpoint_sha256"] = hashlib.sha256(payload).hexdigest()
        marker.write_text(json.dumps(receipt))
        reads = []
        original = Path.open

        class Reader:
            def __init__(self, handle):
                self.handle = handle
            def read(self, size=-1):
                reads.append(size)
                if not 0 < size <= 8 * 1024 * 1024:
                    raise AssertionError("unbounded checkpoint read")
                return self.handle.read(size)
            def fileno(self):
                return self.handle.fileno()
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.handle.close()

        def open_path(path, mode="r", *args, **kwargs):
            handle = original(path, mode, *args, **kwargs)
            return Reader(handle) if path == self.checkpoint and mode == "rb" else handle

        output = io.StringIO()
        with patch.object(Path, "open", open_path), contextlib.redirect_stderr(output), \
                patch("scripts.srgc_prefix_check.time.monotonic", side_effect=[0, 6, 12, 13]):
            self.assertEqual(verify_prefix(self.folder, self.expected, 25), receipt["checkpoint_sha256"])
        self.assertEqual(len(reads), 3)
        self.assertIn("PREFIX checked 8388608/9437184", output.getvalue())


if __name__ == "__main__":
    unittest.main()
