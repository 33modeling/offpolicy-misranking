import io
import time
import unittest

from scripts.srgc_log_format import LineFormatter, format_line, uniform_log

CLOCK = lambda: time.localtime(0)  # noqa: E731
STAMP = time.strftime("%H:%M:%S", time.localtime(0))


class LogFormatTest(unittest.TestCase):
    def render(self, line):
        return format_line(line, clock=CLOCK)

    def test_every_known_shape_becomes_one_format(self):
        cases = {
            "RUN seed-5.cache": "TASK    seed-5.cache start",
            "DONE seed-5.cache exit=0": "TASK    seed-5.cache done exit=0",
            "TRAIN seed=8 phase=shared-prefix arm=prefix step=3/25 status=completed completed=3/25": "TRAIN   seed-8.prefix update=3/25",
            "CHECKPOINT saved seed=8 task=prefix step=3 file=/x/prefix-latest.pt": "CKPT    seed-8.prefix update=3 saved",
            "BACKUP CHECK 2026-09-28T05:00:00Z state=copied found=1 saved=1 unchanged=0 errors=0 scan_interval=30s; unchanged_scans_silent=true":
                "BACKUP  state=copied found=1 saved=1 errors=0",
            "BACKUP prefix-latest.pt -> /x/checkpoint.pt": "BACKUP  copied prefix-latest.pt",
            "BACKUP ERROR seed 5: boom": "BACKUP  error: seed 5: boom",
            "WORKER RUNNING seed-8.prefix attempt=1 update=0/25": "WORKER  seed-8.prefix running attempt=1 update=0/25",
            "WORKER IDLE complete=3 | pending: seed-5.prefix:running": "WORKER  idle complete=3 | pending: seed-5.prefix:running",
            "[cache] rank=0 prompts=12/100 last=41.2s remaining_estimate=60.1min": "CACHE   rank=0 prompts=12/100 last=41.2s eta=60.1min",
            "[cache] rank=0 loading model attention=sdpa": "CACHE   rank=0 loading model attention=sdpa",
            "PASS: cached 100 candidates into /x.json; timings in /x.cache": "CACHE   cached 100 candidates into /x.json; timings in /x.cache",
            "GUARD terminating orphan pid=7 cmd=python": "GUARD   terminating orphan pid=7 cmd=python",
            "VERIFY fallback exact-match gold='$\\boxed{7}$'": "VERIFY  fallback exact-match gold='$\\boxed{7}$'",
            "ADMISSION PASSED; task log: /x/task.log": "ADMIT   passed; task log: /x/task.log",
            "Traceback (most recent call last):": "LOG     Traceback (most recent call last):",
        }
        for line, expected in cases.items():
            self.assertEqual(self.render(line), f"{STAMP} {expected}", line)

    def test_pure_repetition_is_dropped(self):
        self.assertIsNone(self.render("TRAIN seed=8 phase=shared-prefix arm=prefix step=4/25 status=running completed=3/25"))
        self.assertIsNone(self.render("[cache] rank=2 generating prompt 5/100"))
        self.assertIsNone(self.render("   "))

    def test_stream_buffers_partial_chunks_and_flushes_the_tail(self):
        sink = io.StringIO()
        stream = LineFormatter(sink, clock=CLOCK)
        stream.write("RUN seed-5.ca")
        self.assertEqual(sink.getvalue(), "")
        stream.write("che\nDONE seed-5.cache exit=0\nPASS: done")
        self.assertEqual(sink.getvalue().splitlines(), [f"{STAMP} TASK    seed-5.cache start", f"{STAMP} TASK    seed-5.cache done exit=0"])
        stream.flush()
        self.assertEqual(sink.getvalue().splitlines()[-1], f"{STAMP} CACHE   done")

    def test_uniform_log_replaces_and_restores_stdout(self):
        import sys
        original = sys.stdout
        with uniform_log() as formatter:
            self.assertIs(sys.stdout, formatter)
        self.assertIs(sys.stdout, original)


if __name__ == "__main__":
    unittest.main()
