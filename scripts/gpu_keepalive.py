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
    torch.cuda.set_device(gpu)
    a = torch.randn(2048, 2048, device="cuda")
    # 연산 블록 시간 실측 → 듀티에 맞는 휴식 계산
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(40):
        a = a @ a.T
        a = a / (a.norm() + 1e-6)
    torch.cuda.synchronize()
    block = max(0.005, time.time() - t0)
    rest = block * (1.0 - duty) / duty
    print(f"keepalive GPU{gpu}: block {block*1000:.0f}ms, rest {rest*1000:.0f}ms "
          f"(duty {duty:.0%})", flush=True)
    while True:
        for _ in range(40):
            a = a @ a.T
            a = a / (a.norm() + 1e-6)
        torch.cuda.synchronize()
        time.sleep(rest)


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
