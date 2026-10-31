# Gemma IT 배정·재시작·OOM 수정 — 2026-10-31 저녁

사용자 보고: MBPP prefix가 끝났는데 다음 단계가 시작되지 않음, 모든 노드가 LAUNCH/PREPARING을
반복함, seed5 switch가 CUDA out of memory로 계속 실패함, 나머지 작업이 배정되지 않음.
클러스터 로그는 받지 못했다. 아래는 코드에서 찾은 경로와 CPU 검사 결과이고, 원격 GPU에서
확인한 결과가 아니다.

## 수정

| 커밋 | 내용 |
| --- | --- |
| d375b93 | MATH가 남아 있어도 잡을 MATH 작업이 없는 worker는 MBPP arm을 잡는다(`late` 범위, MATH 작업 다음 순서). prefix 의존성과 MBPP 패널 PASS 조건은 그대로. |
| dff4471 | 학습 rank는 `rank_runtime → reward_curve_rank`로 감싸져 명령줄에 `gemma4_it.rank`가 없어서, 고정된 고아 정리기가 죽은 시도의 rank를 정리하지 못했다. Gemma IT 실행기가 감싼 rank도 정리 대상으로 본다. 주인이 살아 있는 rank는 남긴다. |
| 1e44c8b | 실패한 작업이 가장 먼저 재시도되므로, OOM으로 끝나는 switch를 모든 빈 GPU가 60초마다 다시 잡았다. 마지막 시도가 CUDA OOM으로 끝난 작업은 30분 동안 배정하지 않았다. 재시도가 끝난 문제 뒤에서 이어지게 된 뒤(f201ff9) 사용자 요청으로 이 대기를 없앴다. |
| f201ff9 | 검사 묶음을 GPU마다 한 문제씩 계산하고 끝난 문제의 gradient를 저장한다. 같은 가중치의 재시도는 저장된 문제 뒤에서 이어진다. rank별 문제·순서·seed는 그대로. |
| 97e258c | status의 ERROR·직전 시도 줄에 OOM 요청량, GPU 총량·남은 양, 이 프로세스 사용량, 다른 프로세스 수, 코드 위치를 한 줄로 보인다. |
| 829abc5 | Gemma만 남아 실행 스크립트 기본을 체인 없는 직접 실행으로 바꿨다(`SRGC_MODEL_CHAIN=1`이면 체인). 직접 실행 worker는 할 일이 없을 때 끝나지 않고 프로세스 안에서 기다리며, 작업이 실패해도 다음 작업을 잡는다. |
| b855283 | 직접 실행 worker가 GPU 점검 등 오류로 끝나면 스크립트가 60초 뒤 다시 실행한다. 0(완료·정지), 130/143(중단)은 끝낸다. `resume`은 첫 실행에서만 정지 표시를 지운다. |
| 03eaf66 | Gemma rank에서 CUDA 할당자의 expandable segments를 항상 켠다. 할당 방식만 바뀐다. |
| 7bb0a93, e37ec43, f61bf09 | status에서 STALE 줄·표시, 체인 순서, MODEL QUEUE 목록, 설명문을 뺐고, LAUNCH 줄에 직전 child의 종료 이유를 붙였다. |
| 0c02a07 | 옛 코드로 뜬 child가 시작 잠금에서 멈춘 경우 그 child만 끝내는 노드 명령(`sh scripts/unstick_gemma_launch.sh`). |

입력, plan, seed, sampling, 보상, checkpoint, 학습 코드와 adapter·engine digest 파일은 바꾸지 않았다.

## 검사

- 실제 코드 경로 CPU 재현: MATH가 seed별 switch만 남고(다른 노드 소유 또는 OOM 실패), MBPP
  prefix와 패널이 끝난 상태에서 체인 child와 직접 실행 worker 모두 MBPP arm 20개를 배정받았다.
  OOM으로 실패한 switch는 30분 동안 배정되지 않았고, 다른 이유로 실패한 경우는 바로 재시도됐다.
- 실제 프로세스: 감싼 고아 rank는 옛 정리기가 찾지 못하고 새 정리기가 찾아 끝낸다. 주인이 있는 rank는 남는다.
- `scripts/run_srgc_gemma4.sh`: 기본은 직접 실행, 오류 종료 뒤 재실행, 0이면 종료.
- `srgc_rebuttal/tests/test_process_guard.py`의 두 검사는 이 PC GPU를 데스크톱이 3,750MiB 쓰고
  있어 정리기가 GPU 비기를 600초 기다리느라 멈췄다. 이 파일과 정리기 코드는 오늘 바뀌지 않았다.

## 남은 것

seed5 switch의 OOM 원인은 확인하지 못했다. 시작 전 검사는 75,000MiB 이상 H100 네 장과
GPU별 사용량 4,000MiB 이하를 요구하므로, 시도 안에서 메모리가 모자란 것이다. status의
`CUDA OOM: …` 줄(요청량, 남은 양, 다른 프로세스 수, 코드 위치)이 있어야 원인을 정할 수 있다.

## 적용

RUN이 아닌 노드에서 coordinator를 Ctrl+C로 끄고 `git pull --ff-only` 뒤
`bash scripts/run_srgc_gemma4.sh`. RUN 중인 노드는 작업이 끝난 뒤 같은 방법으로 바꾼다.
