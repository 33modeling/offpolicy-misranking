# Gemma MBPP cache·prefix 선행 배정 — 2026-10-29

사용자 요청(2026-10-29): Gemma4 실험에서 MATH 작업이 남아 있어도 MBPP prefix 단계까지는
바로 배정한다. prefix 이후 단계의 의존성과 사전 점검은 유지한다.

## 바뀐 배정

- MATH 학습 단계 worker도 MBPP cache와 prefix를 맡는다. MBPP arm(on_policy, switch,
  sr, random)은 이전처럼 MATH 전체가 끝난 뒤 MBPP 단계에서만 배정한다.
- worker가 일을 고르는 순서는 MATH ERROR·재개 작업, MATH cache·prefix, 새 MBPP
  cache·prefix, MATH arm, 실패했던 MBPP cache·prefix 재시도다. MBPP ERROR는 그
  worker에 MATH 작업이 없을 때만 다시 시도한다. MATH가 끝나기 전에는 MBPP 작업이
  resume backlog로 잡히지 않아 MATH 작업을 붙잡지 않는다.
- 의존성은 그대로다. MBPP prefix는 같은 seed의 MBPP cache가 끝난 뒤, arm은 prefix가
  끝난 뒤에만 배정한다.
- MBPP cache·prefix는 저장된 MBPP 고정 패널(8문제×8응답, 25% 하한, 빈 응답 0)이
  PASS일 때만 배정한다. 패널을 아직 돌리지 않았으면 MATH 학습 단계 worker가 시작할 때
  GPU admission에서 함께 돌린다. 패널이 하한에 못 미치거나 중간에 실패하면 MBPP는 계속
  닫혀 있고, 그 worker는 MATH만으로 admission을 다시 받아 MATH를 이어 간다. 각 rank가
  cache·train 시작 전에 패널을 다시 확인하는 `validate_gate`는 바꾸지 않았다.
- MATH cache 단계(`SRGC_CACHE_ONLY=1`)는 이전처럼 MBPP를 보지 않는다.
- 체인 실행에서 MBPP cache·prefix가 실패해도 MATH 단계 child는 끝나지 않고 MATH를
  이어 간다. 실패 기록은 ERROR로 남는다.
- Gemma IT 실행 경로만 해당한다. Qwen, Llama, Gemma PT·text 배정은 그대로다.
  `SRGC_GEMMA_MBPP_EARLY=0`으로 시작하면 이전 장벽으로 돌아간다.

## 적용

```sh
git pull --ff-only
```

이미 실행 중인 Python 프로세스에는 새 코드가 들어가지 않는다. 체인 coordinator가 다음에
띄우는 Gemma child부터 적용된다. MATH 작업이 없어 대기하던 노드는 다음 재시작(10초
간격)에 적용된다. MATH 작업을 계속 잡는 child는 맡을 작업이 없어 다시 뜰 때까지 이전
코드로 돈다. `reset`으로 시작해 체인 없이 도는 노드는 명령을 다시 시작해야 적용된다.
가장 빠른 방법은 빈 노드나 새 노드에서 평소 명령을 실행하는 것이다.

```sh
bash scripts/run_srgc_gemma4.sh all
```

## 보존

입력, plan, sampling, seed, 보상, checkpoint와 학습 adapter·engine digest에 들어가는
파일(`gemma4_it/`, `gemma4/`, `srgc_rebuttal/` 등)은 바꾸지 않았다. 바뀐 코드는 배정
코드(`ready_dispatch.py`, `resume_drain.py`, `gemma_it_run.py`)뿐이다.

## 검사

- 새 검사 [test_gemma_mbpp_early.py](../srgc_research/tests/test_gemma_mbpp_early.py)
  23건이 통과했다. 변경 전 코드(7b98fee)에 같은 파일을 넣으면 19건이 실패하고, 장벽이
  유지돼야 하는 4건만 통과한다.
- 회귀: `srgc_research/tests`와 `srgc_rebuttal/tests` 전체에서 2,236건 통과, 37건 실패,
  2건 건너뜀. 실패 37건은 변경 전 코드(7b98fee)의 git 사본에서도 같은 37건이 실패한다
  (이 PC의 GPU 배치 검사, GPU 없는 Triton 커널 등). 이번 변경으로 새로 생긴 실패는 없다.
- 첫 회귀 실행에서 체인 worker가 다른 데이터셋의 밀린 작업 실패 때 coordinator로 돌아가는
  기존 동작(`test_model_chain_keeps_retrying_error_tasks_without_an_attempt_limit`)을 깨뜨린
  것을 찾아, 실패해도 계속 가는 예외를 MATH 단계의 MBPP cache·prefix로만 좁혔다.

CPU 검사이며 원격 H100에서 MBPP cache·prefix가 실제로 시작됐다는 기록은 아니다.

## 2026-10-31 수정: 남는 GPU는 MBPP cache·prefix부터

사용자 보고(2026-10-31): Gemma switch가 SELECT 25/275에서 24시간 넘게 멈춘 것처럼 보이고,
남는 GPU가 MBPP prefix를 먼저 진행하지 않는다.

코드에서 남는 GPU가 MBPP prefix를 잡지 못하는 경로 세 개를 확인했다.

1. 지금 잡을 수 없는 MATH 작업이 하나라도 있으면 새 MBPP cache·prefix가 막혔다.
   `ReadyClaims.claim`은 주인 없는 MATH 미완료 작업(재시도 대기 중인 ERROR, 다른 노드
   실행 잠금이 남은 작업 포함)이 있으면 새 작업 우선순위를 닫았다. 체인 child의
   `runnable`도 그런 MATH 작업만 실행 후보로 보았다. 둘 다 그 작업을 바로 잡을 수 없으면
   아무것도 잡지 않고 양보했고, GPU는 놀았다.
2. 실패한 MBPP prefix는 그 worker에 MATH 작업이 하나도 없을 때만 재시도했다. MATH arm은
   275 업데이트라 사실상 재시도되지 않았다.
3. 체인이 재개 전용(`SRGC_RESUME_ONLY=1`)으로 child를 띄우면 MBPP cache·prefix를 보지 않았다.

바뀐 순서: 새 MBPP cache·prefix → 실패한 MBPP cache·prefix 재시도(재시도 대기 시간은 그대로)
→ MATH ERROR·재개 작업 → MATH cache·prefix → MATH arm. MBPP arm은 여전히 MATH가 모두 끝난
뒤에만 배정하고, 의존성(cache → prefix)과 MBPP 고정 패널 PASS 조건도 그대로다. MATH 쪽
작업 사이의 순서도 그대로다. 실험 plan, 입력, seed, sampling, 보상, checkpoint, 학습 코드는
바꾸지 않았다. 배정 코드(`ready_dispatch.py`, `resume_drain.py`)만 바꿨다.

새 검사 3건(잡을 수 없는 MATH ERROR가 있을 때 MBPP prefix 배정, 체인 child 일반·재개 전용
모드)은 이전 코드에서 실패하고 수정 후 통과한다. 이전 순서를 고정하던 검사 4건은 새 순서로
바꿨다. CPU 검사이며 원격 H100에서 MBPP prefix가 실제로 시작됐다는 기록은 아니다.

같은 날 status에 선택 단계 진행률을 추가했다(908e50e). RUNNING 표의 Select 칸과 진행 줄에
`cand 저장/후보 수 val 저장/50`이 나온다. switch 검사 시점(25, 50, …)에는 후보가 최대 80개
(새 후보 40 + SR 비교 40)라 `<=80`으로 표시한다. 문제마다 응답 8개이므로 이 단계는 학습
업데이트 약 30번 분량이다. 숫자가 늘면 도는 중이고, 오래 그대로면 멈춘 것이다.

적용: `git pull --ff-only` 뒤 체인이 다음에 띄우는 Gemma child부터 새 코드가 들어간다.
놀고 있는 노드의 child는 10초마다 다시 뜨므로 곧바로 적용된다. 지금 돌고 있는 학습
프로세스는 끊지 않는다.
