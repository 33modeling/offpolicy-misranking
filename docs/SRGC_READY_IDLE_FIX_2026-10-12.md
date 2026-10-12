# Qwen·Llama·Gemma READY/IDLE 배정 수정

기준 코드: `63cbfcf554f97a863b9bcf3a34fa4c79870fb829`, `master`.
대상: `run_srgc_qwen35.sh`, `run_srgc_llama31.sh`, `run_srgc_gemma4.sh`의 현재 실행 entry.

## 원인과 동작

기존 `ResumeFirst.claim()`은 공유 resume backlog가 비워질 때까지 backlog 밖의
모든 작업을 거부했다. 다른 노드가 재개 작업을 실행하는 동안에도 독립적인
READY 작업을 배정하지 않아 빈 노드가 계속 IDLE로 남았다.
재시도 대기, 다른 데이터셋의 stop 기록도 같은 대기를 만들었다.

현재 entry는 `srgc_research/dispatch/ready_dispatch.py`를 사용한다.

- 지금 실행할 수 있는 재개 작업을 먼저 배정한다.
- 재개 작업이 이미 다른 노드에 배정됐거나 재시도 대기·한도에 걸려 있으면
  요청한 데이터셋의 독립적인 READY 작업을 배정한다.
- cache→prefix→continuation 의존성과 task/execution lease를 계속 검사한다.
- 배정을 건너뛴 작업의 lease는 바로 해제한다. 다음 작업이 학습하는 동안
  재시도 대기 작업의 lease까지 점유하지 않는다.
- 실패 한도에 도달한 작업은 더 실행하지 않는다. 남은 독립 작업을 처리한 뒤
  실패 작업 이름을 출력하고 오류로 종료한다. 미완료 작업을 완료로 바꾸지 않는다.
- `math`/`mbpp`는 다른 데이터셋의 기존 backlog만 복구한다. 그쪽의 새 작업은
  시작하지 않는다. `all`은 두 데이터셋의 READY 작업을 모두 처리한다.
- 중지된 다른 데이터셋의 기록은 보존하며 요청한 데이터셋을 막지 않는다.

## 기존 실행 보존

`qwen_resume.py`, `model_resume.py`, `llama31/`, `gemma4/`, 학습 engine은
수정하지 않았다. 고정된 파일의 내용으로 기준 commit의 digest를 재계산해
현재 값과 일치하는지 확인했다.

| 대상 | 변경 전과 동일한 SHA-256 |
| --- | --- |
| Qwen adapter | `20e90f784b19afa6602b839ff2f7934eb207f03868a4052896edc9355b7ffd8b` |
| Llama adapter | `1a7ae6ecf3b7fd24643797c97ce56750d32a9e49678125d8ddd2eabb5b61b53a` |
| Gemma adapter | `ba0a57d8ffa8802bb52c04b307ccad38a50f232132f0a58345176e3a33a552b1` |
| 공통 engine | `12cf5ef830ebfd92fa8a87ea62dc7df734cd9ceab57fbce18fc4b2548385f960` |

Plan, 입력, 결과, optimizer checkpoint, 실패 이력과 retry budget revision은
변경하지 않는다. 적용 방법은 [현재 실행 안내](CURRENT_EXPERIMENTS.md)에 있다.
이미 실행 중인 Python worker에는 새 배정기가 자동으로 로드되지 않는다.

## 검증

첫 READY/IDLE 수정(`462ef62`)에서 아래 직접 관련 검사는 **144개 모두 통과**했다.

```sh
python3 -m pytest -q \
  srgc_research/tests/test_resume_drain.py \
  srgc_research/tests/test_qwen_resume.py \
  srgc_research/tests/test_model_resume.py \
  srgc_research/tests/test_qwen_status.py \
  srgc_research/tests/test_llama_run.py \
  srgc_research/tests/test_gemma_run.py
```

세 모델 각각에 대해 같은 데이터셋/다른 데이터셋의 실행 중 작업, 실제 두
프로세스의 서로 다른 READY 작업 배정, prefix 의존성, checkpoint 보존,
재시도 한도·대기, stop, 선택하지 않은 task lease의 즉시 해제를 검사했다.
수정 전에는 READY/IDLE 관련 12개 재현 검사가 실패했다.
변경 Python 파일의 Ruff 검사와 `git diff --check`도 통과했다.

추가로 실행한 research 전체 검사에서는 **488개 통과, 9개 실패**를 확인했다.
이후 최종 lease 보강은 위 144개 검사로 다시 검증했다. 전체 검사 실패 9개는
별도 detached checkout의 수정 전 기준 코드에서도 같은 환경으로 점검했다.

| 실패 종류 | 전체 검사 | 수정 전 코드에서 해당 검사만 재실행 |
| --- | ---: | --- |
| Llama tokenizer metadata/테스트 fixture 호환성 | 6개 실패 | 6개 모두 실패 재현 |
| Gemma FP32 gradient 일치 허용 오차 | 1개 실패 | 실패 재현 |
| Information mixed precision 오류 재현 기대 | 1개 실패 | 실패 재현 |
| Information 동시 retry budget 검사 | 1개 실패 | 통과, 간헐성 원인 미확정 |

위 표는 첫 수정 당시의 기록이다. 아래 추가 점검에서 원인을 확인하고 해결했다.

## 후속 전체 점검 — 2026-10-12

사용자가 다른 버그도 분석하고 수정하도록 요청한 뒤, 배정기뿐 아니라 실제
worker 시작·완료 판단, admission, 자식 종료, 재시도 기록, 결과 반환까지 점검했다.

| 발견한 문제 | 수정한 동작 |
| --- | --- |
| 요청한 데이터셋이 완료되면 다른 데이터셋의 재개 backlog를 확인하기 전에 종료 | 시작 시 두 데이터셋의 backlog를 확인하고 기존 미완료 작업을 복구한다. 다른 데이터셋의 새 작업은 시작하지 않는다. |
| 다른 model snapshot의 sibling에 protocol/runtime을 먼저 기록 | snapshot 일치를 먼저 검사한다. 서로 다른 snapshot에는 protocol/runtime을 새로 기록하지 않는다. |
| sibling runtime 불일치를 GPU admission 뒤에 발견 | 저장된 runtime을 GPU 요청 전 검사하고, 전체 cohort를 lock 아래 다시 확인한 뒤 admission을 수행한다. |
| Qwen 작업 사이 CUDA 메모리 해제가 늦으면 worker 종료 | 기존 Llama/Gemma와 같이 최대 30초 기다린다. 임의로 다른 작업을 종료하지 않는다. |
| 음수 SIGINT/SIGTERM 종료 코드를 실제 실패로 계산 | 130/143으로 정규화하고 interruption을 기록하며 실패 재시도 횟수를 차감하지 않는다. 실제 worker도 중단 코드를 shell로 전달한다. |
| Qwen Python entry가 결과 추출 실패를 shell에 0으로 반환 | 결과 함수의 반환 코드를 `SystemExit`으로 전달한다. |
| information collector의 일반 예외가 running으로 남아 다음 시작 때 무제한 환급 | 예외 종류와 원인을 failed receipt에 기록하고 공유된 3회 한도를 적용한다. KeyboardInterrupt는 interruption으로 기록한다. |
| information 자식이 SIGTERM을 무시하면 무한 대기; leader 종료 후 ranks가 생존 | 종료 요청 후 최대 30초 뒤 SIGKILL로 전환한다. leader 종료 후에도 해당 process group을 정리하고 signal handler를 복원한 뒤 claim을 해제한다. |
| Gemma의 한 seed에서 캐시 검사 오류가 나면 데이터셋 작업 목록 전체가 사라짐 | 세 모델의 status/results를 공통 loader로 연결한다. 잘못된 seed만 INVALID로 표시하고 나머지 seed와 데이터셋은 계속 표시·추출한다. |
| 초기 캐시 성공률과 status/results 입력 검사 경로가 모델마다 다름 | 같은 snapshot loader와 renderer를 사용한다. 초기 캐시 성공률은 세 모델 모두 표시하고 held-out 최종 reward와 구분한다. 미완성 캐시에는 성공률을 표시하지 않는다. |
| 운영 Python entry에서 결과 추출 자체의 예외 처리·종료 코드가 모델마다 다름 | Qwen·Llama·Gemma 모두 공통 results CLI를 호출해 추출 오류를 같은 코드로 반환한다. `results` 축약형도 동일하게 처리한다. |

모델별 frozen worker 파일은 그대로 두고 외부 `resume_drain.py`에서 공통 운영
worker를 설치했다. Llama/Gemma CLI가 worker를 지역 변수로 import하기 전에
hook을 설치하며, context 종료 후 원래 함수로 복원한다.
진행 중인 실험의 plan, 입력, 응답, checkpoint, 결과, model snapshot과 adapter/
engine SHA-256은 첫 수정과 동일하다. Retry revision도 올리지 않았다.

이전 전체 검사의 9개 실패는 다음과 같이 정리했다. 학습 계산을 바꾸거나
scientific runtime identity를 다시 발급해 검사를 통과시키지 않았다.

| 이전 실패 | 원인과 보완 |
| --- | --- |
| Llama tokenizer 6개 | 설치된 Transformers 5.14.1의 `save_pretrained`가 generic backend 이름과 decoder table 형식을 바꿨다. 테스트 fixture를 고정된 Llama metadata 형식으로 생성한다. 실제 offline loader, vocabulary/BOS/EOS/chat 검증, 잘못된 metadata의 weights 접근 전 거부는 유지한다. |
| Gemma FP32 gradient 1개 | chunked head와 dense log-softmax backward의 FP32 누적 순서 차이다. 측정한 전체 gradient 상대 norm 차이는 약 `3.85e-7`이었다. 텐서별 오차를 `32 × FP32 eps × reference norm + 1e-7`로 검사한다. BF16 기준과 soft-capping 계산은 그대로다. |
| Information BF16 1개 | 과거 특정 kernel의 rounding 오류가 이 환경에서는 재현되지 않았다. 해당 테스트는 실제 backward capture, 재구성 오차, 같은 checkpoint의 정확한 replay/복원을 검사한다. 잘못된 정규화와 실행별 derivative 차이는 별도의 결정적 fault 주입 검사로 거부한다. |
| Information 동시 retry 1개 | 각 프로세스가 고정 5회만 polling하고 종료해 세 번째 실행 전에 모두 끝날 수 있었다. 공유 budget 소진 상태까지 deadline 안에서 기다리고 정확히 3회 실행·단일 history를 검사한다. |

### Qwen과 Llama/Gemma 비교

| 경로 | 대조한 동작과 확인 범위 |
| --- | --- |
| 모델 로딩 | Qwen의 고정 revision과 Llama/Gemma의 로컬 snapshot 검증을 유지한다. 다른 모델·snapshot·tokenizer metadata·dataset 입력을 거부하는 검사를 통과했다. |
| 생성/OOM | 세 모델 모두 micro-batch 2, OOM 시 RNG를 복원하고 전체 draw를 micro-batch 1로 재시도한다. 응답 수·padding·진행 기록을 확인했다. |
| LoRA/backward | Llama의 q/v와 Gemma의 sliding q/v·global q/k는 모델 구조에 맞는 기존 설정이다. Gemma의 native soft-capping과 FP32/BF16 gradient를 작은 실제 모델로 대조했다. |
| 메모리/checkpoint | logprob batch 1, logit chunk 64, nonreentrant checkpointing, CPU optimizer snapshot, fsync 저장·정확한 재개를 검사했다. |
| 저장소/자식 | 세 모델 모두 rank별 compiler cache를 분리한다. task/execution/GPU lease, 자식 종료, SIGINT/SIGTERM 처리와 재시도 한도를 검사했다. |
| status/results | 공개 shell/운영 Python entry를 공통 loader로 연결했다. 동일한 상태 판정·seed 표·JSON 필드·초기 캐시 성공률·최종 reward·비용·오류 처리와 한 파일 원자적 추출을 사용한다. |

Gemma의 모델 구조에 필요한 soft-capping/LoRA 차이와 각 모델의 고정된 package
설정은 유지한다. 이 점검은 실제 H100의 VRAM 사용량이나 원격 노드의 마지막
오류를 관측한 기록은 아니다.

### 후속 검증

- 상태/결과 공통화 전 research+engine 전체: **1,250 passed, 325 subtests passed**.
- 상태/결과 관련 중간 검사: **107 passed**. 새 재현 검사 6개는 수정 전 **5 failed, 1 passed**였다.
- 최종 상태/결과·모델 entry 관련 검사: **116 passed**. 실제 공개 shell 명령,
  dataset 필터, 입력 무변경, 초기 캐시 50%와 held-out reward 30%의 구분,
  한 seed 오류 격리와 결과 추출 종료 코드를 포함한 새 검사 **15개**도 재검증했다.
- 기존 launcher·rollout 복구·checkpoint·NCCL 관련 별도 검사: **310 passed**.
- 6개 프로세스의 공유 retry-budget 검사를 별도 실행으로 12회 반복해 모두 통과했다.
- multiprocessing Manager의 socket 권한에 의존하던 5개 검사는 실제 spawn
  프로세스와 공유 메모리 Barrier로 동기화하도록 바꿨다. 중복 실행 방지,
  cache/prefix 순서, 동시 실행 수와 GPU lease 검증은 그대로 유지한다.
- 수정한 운영/연구 Python 파일 Ruff, `git diff --check`, 세 모델 adapter/engine
  hash 및 information frozen runtime 제외 여부를 확인했다. 기존 engine 테스트
  4개 파일의 lint 위반 수는 증가하지 않았다.

최종 공통화와 테스트 경로 갱신을 포함한 research+engine 전체 검사:
**1,265 passed, 325 subtests passed**, 실패 없음(281.99초).

```sh
.work/.venv-cu126/bin/python -m pytest -v \
  srgc_research/tests srgc_rebuttal/tests
```

로컬 전체 실행 로그는 `/tmp/v7-final-verified-20261012.log`, 최종 상태/결과
관련 116개 검사 로그는 `/tmp/v7-common-status-results-final-20261012.log`에 있다.

로컬 검증은 `.work/.venv-cu126/bin/python`(Torch 2.7.1+cu126,
Transformers 5.14.1)을 사용했다. 작은 실제 Gemma 모델의 forward/backward
검사는 이 환경 기준이며, 원격 배포의 고정 Gemma overlay(Transformers 5.19.0)를
H100에서 실행한 검증은 아니다.

로컬 NVIDIA driver에 연결할 수 없고 `/group-volume`도 마운트되어 있지 않다.
실제 H100 노드에서 새 worker가 작업을 배정받고 학습을 진행하는지는 미확인이다.
