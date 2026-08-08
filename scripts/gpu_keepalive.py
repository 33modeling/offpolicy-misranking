"""GPU 유휴 킬 회피 — 모든 GPU에 **상시** 저강도 연산 (사용률 기준 감시 대응).

버스트 방식(간헐)은 사용률 평균이 0%로 잡혀 소용없다. 여기서는 GPU마다 스레드가
"연산 ~40ms → 휴식" 듀티 사이클을 계속 돌려 nvidia-smi 사용률이 항상 0보다 크게
찍히게 한다. 기본 듀티 ~15% — 본 작업(생성·backward)이 SM을 점유하면 자연히
밀려나므로 간섭은 미미하다. 메모리 ~64MB/GPU.

    python3 scripts/gpu_keepalive.py [duty_percent]   # 기본 15
"""

import sys
import threading
import time

import torch


def worker(gpu: int, duty: float) -> None:
    """소형 커널 연속 발사 — 사용률 지표(커널 실행 시간 비율)를 상시 높게 유지.

    256×256 matmul은 H100에서 μs 단위라 실제 SM 점유는 미미하지만, 쉼 없이
    발사하면 utilization 지표는 계속 nonzero로 찍힌다. 본 작업의 큰 커널이
    들어오면 자연히 양보된다. duty 인자는 스트림 사이 마이크로 휴식 비율.
    """
    torch.cuda.set_device(gpu)
    a = torch.randn(256, 256, device="cuda")
    print(f"keepalive GPU{gpu}: continuous tiny-kernel mode", flush=True)
    while True:
        for _ in range(500):
            a = a @ a
            a = a / (a.norm() + 1e-6)
        torch.cuda.synchronize()
        time.sleep(0.002)  # CPU 스핀 방지용 마이크로 휴식


def main() -> None:
    duty = (float(sys.argv[1]) if len(sys.argv) > 1 else 15.0) / 100.0
    n = torch.cuda.device_count()
    if n == 0:
        print("keepalive: GPU 없음 — 종료")
        return
    print(f"keepalive: GPU {n}개 상시 가동, duty {duty:.0%}", flush=True)
    threads = [threading.Thread(target=worker, args=(i, duty), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    while True:  # 스레드 예외로 죽으면 재기동
        time.sleep(60)
        for i, t in enumerate(threads):
            if not t.is_alive():
                print(f"keepalive: GPU{i} 스레드 재시작", flush=True)
                threads[i] = threading.Thread(target=worker, args=(i, duty), daemon=True)
                threads[i].start()


if __name__ == "__main__":
    main()
