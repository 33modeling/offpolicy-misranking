"""GPU 유휴 킬 회피 — 모든 GPU에 제한된 duty의 저강도 연산.

GPU마다 짧은 연산과 휴식을 반복하되 실제 경과 시간을 기준으로 기본 duty를 15%로
제한한다. 본 작업(생성·backward) 때문에 keepalive 커널이 늦어지면 그만큼 휴식도
길어져 실제 workload를 방해하지 않는다. 메모리 사용량은 GPU당 1MB 미만이다.

2026-09-07: the keepalive only has to defeat the cluster's idle-GPU reaper, and
a GPU that is already busy with a rollout or a backward pass is not idle. Each
worker therefore samples its GPU's utilization (nvidia-smi, every few seconds)
and skips its bursts while another process keeps the GPU above
OM_GPU_KEEPALIVE_BUSY_PERCENT (default 40). When the sampler is unavailable the
old fixed-duty behaviour applies unchanged.

    python3 scripts/gpu_keepalive.py [duty_percent]   # 기본 15
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

BUSY_PERCENT = float(os.environ.get("OM_GPU_KEEPALIVE_BUSY_PERCENT", "40"))
SAMPLE_SECONDS = float(os.environ.get("OM_GPU_KEEPALIVE_SAMPLE_SECONDS", "5"))


def _visible_gpu_ids() -> list[str]:
    """nvidia-smi indices behind CUDA_VISIBLE_DEVICES (torch index i -> real id)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        ids = [item.strip() for item in visible.split(",") if item.strip()]
        if ids and all(item.isdigit() for item in ids):
            return ids
        return []
    return [str(i) for i in range(torch.cuda.device_count())]


def _sample_utilization(gpu_ids: list[str]) -> list[float] | None:
    """Percent of the last sampling period each listed GPU spent running kernels."""
    if not gpu_ids:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu",
             "--format=csv,noheader,nounits", "-i", ",".join(gpu_ids)],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    by_id: dict[str, float] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            try:
                by_id[parts[0]] = float(parts[1])
            except ValueError:
                continue
    if any(gpu_id not in by_id for gpu_id in gpu_ids):
        return None
    return [by_id[gpu_id] for gpu_id in gpu_ids]


class BusyMonitor:
    """One sampling thread; `busy(i)` says whether torch device i is already loaded."""

    def __init__(self, gpu_ids: list[str]) -> None:
        self.gpu_ids = gpu_ids
        self._busy = [False] * len(gpu_ids)
        self.available = False
        self._lock = threading.Lock()

    def start(self) -> None:
        sample = _sample_utilization(self.gpu_ids)
        if sample is None:
            print("keepalive: utilization sampler unavailable; fixed duty on every GPU", flush=True)
            return
        self.available = True
        self._store(sample)
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"keepalive: bursts pause while a GPU is above {BUSY_PERCENT:.0f}% utilization", flush=True)

    def _store(self, sample: list[float]) -> None:
        with self._lock:
            self._busy = [value >= BUSY_PERCENT for value in sample]

    def _loop(self) -> None:
        while True:
            time.sleep(SAMPLE_SECONDS)
            sample = _sample_utilization(self.gpu_ids)
            if sample is not None:
                self._store(sample)

    def busy(self, index: int) -> bool:
        if not self.available:
            return False
        with self._lock:
            return self._busy[index] if index < len(self._busy) else False


def worker(gpu: int, duty: float, ready: threading.Event, monitor: BusyMonitor | None = None) -> None:
    """짧은 소형 커널 burst 뒤 실측 시간에 비례해 휴식한다.

    다른 프로세스의 큰 커널 뒤에서 대기한 시간도 active 구간에 포함되므로, 실제
    rollout 부하가 높을수록 다음 sleep이 길어지는 협조적 backoff가 된다. A GPU the
    monitor reports as busy gets no burst at all until it is idle again.
    """
    torch.cuda.set_device(gpu)
    a = torch.randn(256, 256, device="cuda")
    a = a @ a
    torch.cuda.synchronize()
    print(f"keepalive GPU{gpu}: continuous tiny-kernel mode", flush=True)
    ready.set()
    while True:
        if monitor is not None and monitor.busy(gpu):
            time.sleep(SAMPLE_SECONDS)
            continue
        started = time.monotonic()
        for _ in range(100):
            a = a @ a
            a = a / (a.norm() + 1e-6)
        torch.cuda.synchronize()
        active = max(time.monotonic() - started, 0.001)
        time.sleep(active * (1.0 - duty) / duty)


def main() -> None:
    duty = (float(sys.argv[1]) if len(sys.argv) > 1 else 15.0) / 100.0
    if not 0.01 <= duty <= 0.5:
        raise ValueError("duty_percent must be between 1 and 50")
    n = torch.cuda.device_count()
    if n == 0:
        print("keepalive: GPU 없음 — 종료")
        return
    print(f"keepalive: GPU {n}개 상시 가동, duty {duty:.0%}", flush=True)
    # Utilization gating is on by default (OM_GPU_KEEPALIVE_GATE=0 restores the
    # constant duty). The operator confirmed on 2026-09-07 that the cluster's
    # job kills are unrelated to GPU utilization, so a busy GPU gets no bursts.
    monitor = None
    if os.environ.get("OM_GPU_KEEPALIVE_GATE", "1") == "1":
        monitor = BusyMonitor(_visible_gpu_ids()[:n])
        if len(monitor.gpu_ids) == n:
            monitor.start()
        else:
            print("keepalive: cannot map torch devices to nvidia-smi ids; fixed duty on every GPU", flush=True)
            monitor = None
    events = [threading.Event() for _ in range(n)]
    threads = [
        threading.Thread(target=worker, args=(i, duty, events[i], monitor), daemon=True)
        for i in range(n)
    ]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 50
    for event in events:
        event.wait(max(0.0, deadline - time.monotonic()))
    if not all(event.is_set() for event in events):
        raise RuntimeError("one or more GPU keepalive workers failed to initialize")
    if ready_path := os.environ.get("OM_GPU_KEEPALIVE_READY_FILE"):
        path = Path(ready_path)
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        temporary.write_text(f"pid={os.getpid()} gpus={n}\n")
        temporary.replace(path)
    print(f"keepalive: all {n} GPU workers ready", flush=True)
    while True:
        time.sleep(5)
        for i, t in enumerate(threads):
            if not t.is_alive():
                raise RuntimeError(
                    f"GPU{i} keepalive worker exited; CUDA context restart required"
                )


if __name__ == "__main__":
    main()
