"""Long-running jobs must not load their whole log just to explain a failure."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.srgc_live_status import last_error_line
from scripts.srgc_process_guard import failure_tail
from scripts.srgc_log_tail import tail_lines


class LogTailRegressionTests(unittest.TestCase):
    def test_read_volume_is_bounded_even_when_lines_are_very_long(self):
        reads = []

        class Tracked(io.BytesIO):
            def read(self, size=-1):
                reads.append(size)
                self.assert_bounded(size)
                return super().read(size)

        handle = Tracked(b"x" * 1000000 + b"\nRuntimeError: failed\n")
        handle.assert_bounded = lambda size: self.assertTrue(0 < size <= 4096)
        with patch.object(Path, "open", return_value=handle):
            self.assertEqual(tail_lines("large.log", 20, max_bytes=4096), ["RuntimeError: failed"])
        self.assertLessEqual(sum(reads), 4096)

    def test_empty_missing_invalid_utf8_and_unterminated_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.log"
            self.assertIsNone(last_error_line(path))
            self.assertEqual(failure_tail(path), ["(log file not readable)"])
            for raw, expected in ((b"", []), (b"first\nsecond", ["second"]),
                                  (b"\xff\nValueError: bad\n", ["ValueError: bad"])):
                path.write_bytes(raw)
                self.assertEqual(tail_lines(path, 1), expected)
            self.assertEqual(tail_lines(path, 0), [])

    def test_failure_and_status_do_not_read_the_whole_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.log"
            path.write_bytes(b"old progress\n" * 100000 +
                             "Traceback (most recent call last):\nValueError: 검증 실패\n".encode())
            with patch.object(Path, "read_text", side_effect=AssertionError("unbounded full-log read")):
                for reader in (failure_tail, last_error_line):
                    with self.subTest(reader=reader.__name__):
                        self.assertIn("ValueError: 검증 실패", reader(path))


if __name__ == "__main__":
    unittest.main()
