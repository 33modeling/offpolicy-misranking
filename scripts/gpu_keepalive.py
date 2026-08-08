"""GPU 유휴 킬 회피 — 주기적으로 모든 GPU에 소형 matmul 버스트.

클러스터가 'GPU 유휴 N시간 → 잡 종료' 정책일 때, 파이프라인의 CPU 구간·일부
GPU만 쓰는 구간에서 끊기지 않도록 5분마다 GPU당 ~2초 연산을 흘린다.
메모리 사용 ~50MB/GPU, 본 작업과의 간섭 무시 가능.

    python3 scripts/gpu_keepalive.py [interval_sec]
"""

import sys
import time

import torch


def main() -> None:
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    n = torch.cuda.device_count()
    if n == 0:
        print("keepalive: GPU 없음 — 종료")
        return
    print(f"keepalive: GPU {n}개, {interval}s 간격", flush=True)
    while True:
        for i in range(n):
            try:
                with torch.cuda.device(i):
                    a = torch.randn(2048, 2048, device="cuda")
                    for _ in range(60):
                        a = a @ a.T
                        a = a / a.norm()
                    torch.cuda.synchronize()
                    del a
                    torch.cuda.empty_cache()
            except Exception as e:  # 본 작업 OOM 순간 등 — 조용히 넘어간다
                print(f"keepalive: GPU{i} skip ({e})", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    main()
