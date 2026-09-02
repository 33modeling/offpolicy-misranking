"""GPU 유휴 킬 회피 — 모든 GPU에 제한된 duty의 저강도 연산.

GPU마다 짧은 연산과 휴식을 반복하되 실제 경과 시간을 기준으로 기본 duty를 15%로
제한한다. 본 작업(생성·backward) 때문에 keepalive 커널이 늦어지면 그만큼 휴식도
길어져 실제 workload를 방해하지 않는다. 메모리 사용량은 GPU당 1MB 미만이다.

    python3 scripts/gpu_keepalive.py [duty_percent]   # 기본 15
"""

import os
import sys
import threading
import time
from pathlib import Path

import torch


def worker(gpu: int, duty: float, ready: threading.Event) -> None:
    """짧은 소형 커널 burst 뒤 실측 시간에 비례해 휴식한다.

    다른 프로세스의 큰 커널 뒤에서 대기한 시간도 active 구간에 포함되므로, 실제
    rollout 부하가 높을수록 다음 sleep이 길어지는 협조적 backoff가 된다.
    """
    torch.cuda.set_device(gpu)
    a = torch.randn(256, 256, device="cuda")
    a = a @ a
    torch.cuda.synchronize()
    print(f"keepalive GPU{gpu}: continuous tiny-kernel mode", flush=True)
    ready.set()
    while True:
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
    events = [threading.Event() for _ in range(n)]
    threads = [
        threading.Thread(target=worker, args=(i, duty, events[i]), daemon=True)
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
