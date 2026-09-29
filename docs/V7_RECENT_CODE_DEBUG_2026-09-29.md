# 최근 SR-GC 실행 코드 디버깅 — 2026-09-29

점검 시작 커밋은 `6a83d3ecd405d6946f2c9e717f71f1e376ad6481`입니다.
최근 추가한 고정 시점 전환, gradient 방향 기록, 자동 실행 폴더 분리,
prefix 공개 후 병렬 배정, MATH/MBPP 통합 큐와 실패 로그 출력을 확인했습니다.
기존 278개 테스트는 통과했지만 아래 실행·보고 경로의 회귀 테스트가 부족했습니다.

## 수정한 오류

| 오류 | 영향 | 수정 |
|---|---|---|
| 실패 로그의 `read_text().splitlines()` | 로그 끝부분만 보여주면서 전체 누적 로그를 메모리에 올림 | 공통 `srgc_log_tail.py`에서 마지막 256 KiB만 읽음. 실패 출력 40줄, 상태의 오류 검색 200줄 제한을 유지 |
| 통합 워커의 상태 파일이 주 큐에만 저장됨 | MBPP의 `seed-5.prefix`가 MATH 작업처럼 보이고 MBPP 워커 heartbeat가 빠짐 | 두 큐에 heartbeat를 기록하고 실제 작업 중인 큐에만 task를 연결. 다른 큐에는 `serving mbpp:seed-5.prefix`처럼 표시 |
| 주 큐의 정지가 통합 워커 전체 시작을 막음 | MATH만 정지해도 실행 가능한 MBPP 작업을 받지 못함 | 큐별 graceful stop을 적용하고 다른 큐는 계속 처리. 두 큐 모두 완료·정지 상태면 GPU admission 없이 종료 |
| 보조 큐의 실패 원인이 누락됨 | MBPP 실패로 전체 큐가 막혀도 MATH 로그만 조사해 원인을 출력하지 못함 | 모든 참여 큐의 실패 receipt와 로그 끝부분을 출력 |
| 데이터셋마다 process guard를 중첩 설치함 | 동일 child의 종료 처리와 실패 로그가 두 번 실행됨 | 노드 전체 SR-GC 작업을 담당하는 guard를 한 번만 설치. 백업은 데이터셋별 유지 |

처음 네 증상은 수정 전 테스트로 재현했습니다. 중복 guard는 진입 코드에서
확인하고, 양방향 데이터셋·일반/명시적 새 실행 조합에서 한 번만 설치되는지 검증했습니다.
`stop --now`와 Ctrl-C는 기존처럼 현재 child를 중단하고 워커를 종료합니다.
다른 데이터셋의 stop 표시를 새로 만들지는 않습니다.

## 확인 결과

- 전체 SR-GC/Qwen 테스트 **288개 통과**, 실패·skip 없음, 54.393초.
- 독립 CPU 프로세스 2개가 실제 로컬 파일 잠금을 사용해 두 큐의
  cache → prefix → continuation **60개 작업을 각각 한 번** 처리했습니다.
  동시 실행 2개, 의존성·큐별 heartbeat·종료 상태를 검증했습니다.
- child 오류 시 task lease와 GPU UUID lease가 해제되고 두 큐의 실패 상태가
  남는 것을 확인했습니다. GPU admission과 학습 자체는 테스트 대역입니다.
- 로그 읽기 크기, 긴 한 줄, 한글·잘못된 UTF-8, 마지막 개행이 없는 로그,
  빈 로그·없는 로그를 확인했습니다.
- 자동 실행 폴더 분리·명시적 새 실행·prefix 병렬 공개·캐시 복구·체크포인트
  재개·고정/반복 전환·방향 기록·Qwen 메모리 관련 기존 검사도 전체 suite에 포함됩니다.
- 변경한 Python 파일의 문법 검사와 `git diff --check` 통과.

실행 명령:

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/nsh/.venvs/proto-ml/bin/python -B -m unittest discover -s srgc_rebuttal/tests -v
```

이번 수정은 scripts의 실행·상태 관리 계층입니다. 학습 패키지의 SHA-256은
`f581eb043e89409e0e68d8ed77201fa030e34bd93d4d5babbab65304c24a5d6b`,
Qwen adapter SHA-256은
`f3be15e7ddd528027b5dc169895e0c4ad938f4c14dbc5d1726403fd21abb3330`으로 유지됩니다.
기존 캐시·체크포인트·실험 결과·원고는 수정하지 않았습니다.

실제 H100 학습이나 원격 그룹볼륨에서 새 실험을 실행한 검증은 아닙니다.
진행 중인 워커는 그대로 두고, 대기 또는 종료된 워커를 다음에 시작할 때 수정본을 적용합니다.
