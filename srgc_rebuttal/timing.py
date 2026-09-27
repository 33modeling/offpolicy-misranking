"""Exclusive rank-local stages inside synchronized, durable allocated-GPU phases."""

from contextlib import contextmanager
from functools import wraps
import math
import time
import uuid


class StageTimer:
    def __init__(self, synchronize=lambda: None, clock=time.perf_counter):
        self.synchronize, self.clock = synchronize, clock
        self.active = False
        self.sections = []
        self.stack = []

    def begin(self):
        if self.active:
            raise ValueError("timing phases cannot overlap")
        self.stages, self.counts = {}, {}
        self.started = self.clock()
        self.active = True

    @contextmanager
    def section(self, name):
        self.sections.append(name)
        try:
            yield
        finally:
            self.sections.pop()

    @contextmanager
    def stage(self, name):
        if not self.active:
            yield
            return
        self.synchronize()
        frame = [".".join([*self.sections, name]), self.clock(), 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            self.synchronize()
            elapsed = self.clock() - frame[1]
            self.stack.pop()
            if self.stack:
                self.stack[-1][2] += elapsed
            row = self.stages.setdefault(frame[0], {"wall_seconds": 0.0, "calls": 0})
            row["wall_seconds"] += max(0.0, elapsed - frame[2])
            row["calls"] += 1

    def count(self, name, value=1):
        if self.active:
            key = ".".join([*self.sections, name])
            self.counts[key] = self.counts.get(key, 0) + value

    def finish(self, local_gpu_count):
        if not self.active or self.stack:
            raise ValueError("cannot close an inactive phase or an open stage")
        result = {"wall_seconds": self.clock() - self.started,
                  "local_gpu_count": local_gpu_count, "stages": self.stages,
                  "counts": self.counts}
        self.active = False
        return result


def aggregate_ranks(parts):
    """Sum exclusive device-rank seconds, not sums of per-stage maxima."""
    if not parts:
        raise ValueError("missing rank timings")
    wall = max(p["wall_seconds"] for p in parts)
    gpu_count = sum(p["local_gpu_count"] for p in parts)
    stages, counts = {}, {}
    for p in parts:
        if not math.isfinite(p["wall_seconds"]) or p["wall_seconds"] < 0:
            raise ValueError("invalid phase duration")
        covered = 0.0
        for key, row in p["stages"].items():
            duration = row["wall_seconds"]
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("invalid stage duration")
            covered += duration
            total = stages.setdefault(key, {"rank_wall_seconds": 0.0, "gpu_seconds": 0.0, "calls": 0})
            total["rank_wall_seconds"] += duration
            total["gpu_seconds"] += duration * p["local_gpu_count"]
            total["calls"] += row["calls"]
        if covered > p["wall_seconds"] + 1e-6:
            raise ValueError("stage durations exceed their containing phase")
        for key, value in p["counts"].items():
            counts[key] = counts.get(key, 0) + value
    gpu_seconds = wall * gpu_count
    remainder = gpu_seconds - sum(s["gpu_seconds"] for s in stages.values())
    stages["unattributed_and_wait"] = {
        "gpu_seconds": max(0.0, remainder), "calls": len(parts),
        "rank_wall_seconds": max(0.0, wall * len(parts) -
                                  sum(s["rank_wall_seconds"] for s in stages.values()))}
    return {"wall_seconds": wall, "gpu_seconds": gpu_seconds,
            "gpu_count": gpu_count, "stages": stages, "counts": counts,
            "rank_timings": parts}


class CostMeter(StageTimer):
    def __init__(self, *, rank=0, local_gpu_count=0, synchronize=lambda: None,
                 synchronize_all=lambda: None, gather=lambda p: [p], record=lambda e: None,
                 clock=time.perf_counter):
        super().__init__(synchronize, clock)
        self.rank, self.local_gpu_count = rank, local_gpu_count
        self.synchronize_all, self.gather, self.record = synchronize_all, gather, record

    def begin_phase(self, name, checkpoint=None, gpu_count=0):
        self.synchronize_all()
        self.event = {"id": uuid.uuid4().hex, "phase": name, "checkpoint": checkpoint,
                      "gpu_count": gpu_count, "state": "started"}
        if self.rank == 0:
            self.record(self.event)
        self.synchronize_all()
        self.begin()

    def end_phase(self):
        self.synchronize_all()
        parts = self.gather(self.finish(self.local_gpu_count))
        report = aggregate_ranks(parts)
        if report["gpu_count"] != self.event["gpu_count"]:
            raise ValueError("rank allocation differs from phase allocation")
        event = {**self.event, **report, "state": "finished"}
        if self.rank == 0:
            self.record(event)
        return event

    @contextmanager
    def phase(self, name, checkpoint=None, gpu_count=0):
        self.begin_phase(name, checkpoint, gpu_count)
        try:
            yield
        except BaseException:
            # The durable start stays unfinished; unknown interrupted time is not zero.
            self.active = False
            raise
        else:
            self.end_phase()


def timed(name):
    """Instrument an adapter method without changing its public signature."""
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            with self.cost_meter.stage(name):
                return method(self, *args, **kwargs)
        return wrapped
    return decorate


def torch_meter(record=lambda e: None, *, cuda=None):
    import torch
    import torch.distributed as dist
    distributed = dist.is_initialized()
    world = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    cuda = torch.cuda.is_available() if cuda is None else cuda

    def local_sync():
        if cuda:
            torch.cuda.synchronize()

    def sync_all():
        local_sync()
        if distributed:
            dist.barrier()

    def gather(part):
        if not distributed:
            return [part]
        parts = [None] * world
        dist.all_gather_object(parts, part)
        return parts

    return CostMeter(rank=rank, local_gpu_count=int(cuda), synchronize=local_sync,
                     synchronize_all=sync_all, gather=gather, record=record)


@contextmanager
def invocation(ledger, meter, gpu_count, started=None):
    """Inclusive process receipt. Never add this to its nested phase receipts."""
    started = time.perf_counter() if started is None else started
    event = {"id": uuid.uuid4().hex, "phase": "session", "checkpoint": None,
             "gpu_count": gpu_count, "state": "started"}
    if meter.rank == 0:
        ledger.record(event)
    yield
    meter.synchronize_all()
    if meter.rank == 0:
        elapsed = time.perf_counter() - started
        ledger.record({**event, "state": "finished", "wall_seconds": elapsed,
                       "gpu_seconds": elapsed * gpu_count})
