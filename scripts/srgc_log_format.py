"""One line format for everything a worker prints: ``HH:MM:SS TAG      task key=value ...``.

The hashed ``srgc_rebuttal`` package and the checkpoint/backup/guard scripts
each print their own shapes (``RUN seed-5.cache``, ``TRAIN seed=8 ...``,
``[cache] rank=0 ...``, ``BACKUP CHECK ...``). The worker wraps its stdout in
``LineFormatter`` so every line, including relayed child output, is rewritten
to the same shape. Lines that are pure repetition (a step's "running" echo
that is followed by its "completed" line) are dropped.

Tags: TASK (start/finish of a queue task), CACHE, TRAIN, CKPT, BACKUP, NODE,
GUARD, ADMIT, LOG (anything else, unchanged).
"""

from contextlib import contextmanager
import io
import re
import sys
import time

WIDTH = 7
RULES = [
    (re.compile(r"^RUN (\S+)$"), lambda m: ("TASK", f"{m[1]} start")),
    (re.compile(r"^DONE (\S+) exit=(\d+)$"), lambda m: ("TASK", f"{m[1]} done exit={m[2]}")),
    (re.compile(r"^TRAIN seed=(\d+) phase=\S+ arm=(\S+) step=\S+ status=running"), lambda m: None),
    (re.compile(r"^TRAIN seed=(\d+) phase=\S+ arm=(\S+) step=\S+ status=\w+ completed=(\d+)/(\d+)"),
     lambda m: ("TRAIN", f"seed-{m[1]}.{m[2]} update={m[3]}/{m[4]}")),
    (re.compile(r"^CHECKPOINT saved seed=(\d+) task=(\S+) step=(\d+) file=\S+"),
     lambda m: ("CKPT", f"seed-{m[1]}.{m[2]} update={m[3]} saved")),
    (re.compile(r"^BACKUP CHECK \S+ state=(\S+) found=(\d+) saved=(\d+) unchanged=(\d+) errors=(\d+)"),
     lambda m: ("BACKUP", f"state={m[1]} found={m[2]} saved={m[3]} errors={m[5]}")),
    (re.compile(r"^BACKUP (\S+) -> (\S+)$"), lambda m: ("BACKUP", f"copied {m[1]}")),
    (re.compile(r"^BACKUP (ERROR|WARNING) (.*)$"), lambda m: ("BACKUP", f"{m[1].lower()}: {m[2]}")),
    (re.compile(r"^NODE (.*)$"), lambda m: ("NODE", m[1])),
    (re.compile(r"^WORKER (.*)$"), lambda m: ("NODE", m[1])),
    (re.compile(r"^\[cache\] rank=(\d+) prompts=(\d+)/(\d+) last=(\S+) remaining_estimate=(\S+)"),
     lambda m: ("CACHE", f"rank={m[1]} prompts={m[2]}/{m[3]} last={m[4]} eta={m[5]}")),
    (re.compile(r"^\[cache\] rank=(\d+) generating prompt \d+/\d+$"), lambda m: None),
    (re.compile(r"^\[cache\] (.*)$"), lambda m: ("CACHE", m[1])),
    (re.compile(r"^PASS: (.*)$"), lambda m: ("CACHE", m[1])),
    (re.compile(r"^GUARD (.*)$"), lambda m: ("GUARD", m[1])),
    (re.compile(r"^VERIFY (.*)$"), lambda m: ("VERIFY", m[1])),
    (re.compile(r"^BLOCKED (.*)$"), lambda m: ("BLOCKED", m[1])),
    (re.compile(r"^FAILED (.*)$"), lambda m: ("FAILED", m[1])),
    (re.compile(r"^ADMISSION (.*)$"), lambda m: ("ADMIT", m[1].lower())),
]


def format_line(line, *, clock=time.localtime):
    """The uniform line for ``line`` (without newline), or None when it is dropped."""
    text = line.rstrip("\r\n")
    if not text.strip():
        return None
    tag, body = "LOG", text
    for pattern, render in RULES:
        match = pattern.match(text)
        if match:
            rendered = render(match)
            if rendered is None:
                return None
            tag, body = rendered
            break
    return f"{time.strftime('%H:%M:%S', clock())} {tag:<{WIDTH}} {body}"


class LineFormatter(io.TextIOBase):
    """A text stream that rewrites complete lines; partial chunks are buffered."""

    def __init__(self, stream, *, clock=time.localtime):
        self.stream, self.clock, self.pending = stream, clock, ""

    def writable(self):
        return True

    def write(self, text):
        self.pending += text
        *lines, self.pending = self.pending.split("\n")
        for line in lines:
            rendered = format_line(line, clock=self.clock)
            if rendered is not None:
                self.stream.write(rendered + "\n")
        if lines:
            self.stream.flush()
        return len(text)

    def flush(self):
        if self.pending:
            rendered = format_line(self.pending, clock=self.clock)
            self.pending = ""
            if rendered is not None:
                self.stream.write(rendered + "\n")
        self.stream.flush()

    def fileno(self):
        return self.stream.fileno()

    def isatty(self):
        return self.stream.isatty()


@contextmanager
def uniform_log():
    original = sys.stdout
    formatter = LineFormatter(original)
    sys.stdout = formatter
    try:
        yield formatter
    finally:
        formatter.flush()
        sys.stdout = original
