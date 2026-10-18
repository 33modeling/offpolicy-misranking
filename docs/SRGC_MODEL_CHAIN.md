# Qwen → Llama → Gemma 자동 연결

공개 학습 시작 명령은 **선택한 모든 모델의 MATH 완료 → 선택한 모든 모델의 MBPP** 순서로 실행한다. Qwen 시작의 기본 순서는 `Qwen MATH → Llama MATH → Gemma MATH → Qwen MBPP → Llama MBPP → Gemma MBPP`다. Llama 시작은 각 데이터셋 안에서 `llama31 → gemma4 → qwen35`, Gemma 시작은 `gemma4 → qwen35 → llama31` 순서를 따른다.

**2026-10-19 현재 정책: 현재 모델을 끝낸 뒤 다음 모델로 간다.** Llama 시작이면
Llama의 해당 데이터셋 ERROR·중단 작업·남은 학습·최종 평가가 모두 끝나기 전에는
Gemma를 시작하지 않는다. 다른 모델의 기존 ERROR도 이 순서를 앞지르지 않는다.

```sh
git pull --ff-only origin master
sh scripts/run_srgc_qwen35.sh all
# 이미 Llama를 진행하던 노드라면 다음 정상 시작 때:
sh scripts/run_srgc_llama31.sh all
```

노드당 위 명령 하나를 실행한다. `all`은 두 plan을 먼저 준비하되 MATH 단계에서 MBPP 학습·생성·평가는 실행하지 않는다. `math`는 MATH만 수행한다. `mbpp`도 준비돼 있는 MATH sibling plan을 먼저 완료한 뒤 MBPP를 실행한다. MATH plan이 없으면 오류를 내며, MATH를 건너뛰거나 빈 큐를 완료로 간주하지 않는다. `--root`, `--source-plan`, `--max-attempts` 등 모델별 인자와 Python 선택은 시작 모델에서 그대로 보존하며, 재방문과 데이터셋 전환에도 유지한다. 다른 모델은 각자의 기존 root·Python·고정 모델 snapshot을 사용한다.

각 모델의 학습·최종 평가는 기존 실행기에 맡긴다. 학습 update 275 도달만으로 완료로 표시하지 않는다. 남은 최종 평가도 기존 작업의 일부다. 현재 모델의 작업을 다른 노드가 소유하거나 실패 재시도 대기 중이면 GPU 잠금을 해제한 뒤 같은 모델을 다시 확인한다. 이때 다음 모델로 넘어가지 않는다. 선택한 모델의 MATH가 하나라도 남으면 MBPP로 넘어가지 않는다. endpoint가 저장돼도 실행 잠금이 남아 있으면 완료 경계를 넘지 않는다.

현재 모델 안에서 **ERROR → 다른 중단 작업 → 새 작업 → 남은 최종 평가 완료**를 처리한다. 복구 전용 worker가 반환해도 바로 다음 모델로 넘어가지 않고, 같은 모델의 전체 worker로 남은 작업의 완료를 확인한다. 뒤 모델의 오류 때문에 현재 모델의 새 작업을 건너뛰지 않는다. 다른 worker가 실행 중인 작업은 실행 잠금으로 구분하여 중복 실행하지 않고 같은 모델의 완료를 기다린다.

전환은 새 자식 프로세스로 실행하므로 모델별 Python·Gemma 패키지 overlay·CUDA 상태가 섞이지 않는다. 자동 Llama 전환에서는 수동 시작의 GPU 프로세스 정리를 호출하지 않는다. 기존 다른 작업의 GPU 소유권은 공유 lease로 보호한다. **누적 실패 3회 등의 재개 제한은 없다.** 서버가 GPU를 강제로 종료해 ERROR가 쌓여도 횟수만으로 복구를 막지 않는다. 과거 예산 파일도 실행 조건으로 사용하지 않는다. 기본 재시도 간격 60초 뒤 미완료 작업을 계속 재시도하며 실패 이력은 보존한다. 입력·모델 identity 오류나 명시적인 일시 중지는 별도로 보고하고, 완료로 간주하지 않는다. Ctrl-C/SIGTERM은 소유한 자식에게 전달하고 전체 연결 실행을 멈춘다.

`status`·`results`·`stop`·`doctor`·`prepare` 등은 기존처럼 해당 모델에만 적용된다. 각 모델의 text/JSON status에 공통 `MODEL QUEUE`/`model_chain` 상태와 현재 `dataset=math/mbpp`를 표시한다. heartbeat가 오래된 기록은 STALE로 구분한다. 단계의 완료 여부는 검증된 task·endpoint 및 실행 잠금으로 판단한다. 중지된 단계는 별도 종료 코드로 기록하여 성공/완료와 구분한다. `all resume`는 한 coordinator 실행에서 모델마다 한 번만 중지를 해제한다. 재방문하거나 MATH에서 MBPP로 넘어갈 때 다시 해제하지 않아, MATH 실행 중 새로 건 MBPP 중지 요청을 보존한다. 잘못된 coordinator JSON은 해당 기록만 조회에서 제외하고 다른 정상 세션을 계속 표시한다.

운영 상태는 `$SRGC_STORAGE_ROOT/.model-chain/workers/`에 별도로 기록한다. 기본 경로는 `$OM_WORK/srgc-rebuttal/.model-chain/workers/`다. 고정 engine·모델 adapter·plan·입력·기존 결과·체크포인트는 변경하지 않는다. 기존 중간 reward 기록과 25-update 고정 held-out 평가 확장도 같은 실행기를 통해 유지한다. 학습 배치 reward와 held-out reward는 기존처럼 분리해서 export한다.

필요할 때 `SRGC_MODEL_ORDER=qwen35,llama31,gemma4`로 등록된 모델의 순서를 정하거나 일부만 선택할 수 있다. 지정한 목록에는 시작 모델이 포함돼야 한다. `SRGC_MODEL_POLL_SECONDS`는 1~60초이며 기본 10초다. 의도적으로 한 모델만 실행하려면 `SRGC_MODEL_CHAIN=0`을 지정한다.

이미 떠 있는 Python 프로세스에는 새 연결 로직이 자동 주입되지 않는다. 정상 학습 중인 작업을 중단하지 말고 다음 정상 시작 때 현재 공개 명령을 사용한다. 이전 코드로 IDLE인 worker는 종료 후 같은 명령으로 다시 실행해야 한다. 이 로컬 PC에는 원격 학습 노드의 `/group-volume`이 없으므로 원격 전환이 실제 적용됐다고 간주하지 않는다.

2026-10-17 검증: 세 모델의 단계 전환·오류 재시도·peer 실행 잠금·중지·원래 명령 인자 보존에 대한 최종 검사 179개를 통과했다. 공통 status·중간 보상·실제 SIGKILL 복구·셸 진입점도 회귀 검사했다. 수정 Python 파일의 Ruff/공백 검사를 통과했고, 세 모델 adapter·공통 engine·보상 기록 확장·동결 runtime 파일은 수정 전과 동일함을 확인했다.

2026-10-18 수정: 실패 횟수 제한을 제거했다. 살아 있는 coordinator는 SIGKILL/137로 종료된 worker를 다시 실행한다. 일시적인 `nvidia-smi` 실패도 재확인한다. 이미 coordinator 자체가 종료된 경우에는 같은 공개 명령을 다시 실행해 저장된 작업부터 복구한다. 관련 회귀 316개 통과 후 실제 SIGKILL child 재시작·GPU 조회 복구를 추가해 최종 운영 검사 201개를 통과했다. 과거 실패 이력과 예산 파일은 삭제하지 않았고, 학습·보상 기록 hash를 유지했다.

같은 날 [전체 코드 재점검](REPOSITORY_AUDIT_2026-10-06.md)에서 데이터셋 전환 시 resume 재적용과 잘못된 coordinator 기록의 오류 격리를 추가로 수정했다. 손상된 중간 rollout의 results 오류도 해당 arm에 격리하며 정상 endpoint와 다른 arm의 결과를 유지한다.

## 2026-10-18 ERROR 우선 재개 재수정

아래는 당시 수정 기록이다. 모델 간 ERROR 우선 순회는 2026-10-19에 폐기했다.
현재는 위의 모델 완료 순서를 지키며, ERROR 우선순위는 현재 모델 안에서 적용한다.

실패 횟수 제한은 내부 task 배정에서만 제거돼 있었다. 바깥 model coordinator는
SIGKILL/137 외의 오류 코드 1·2 등으로 worker가 종료되면 그 모델을 `failed`로
고정하여 같은 실행에서 다시 시도하지 않았다. 저장된 작업이 있어도 다른 모델의
새 작업으로 넘어가거나, 복구 전용 실행 전체를 종료하는 경로를 재현했다.

- 현재 데이터셋에서 실행 잠금이 없는 **ERROR → 다른 중단 작업 → 새 작업** 순으로
  모델을 선택한다. Llama 시작도 Gemma·Qwen의 기존 ERROR를 함께 확인한다.
- worker가 오류로 종료되고 기존 미완료 receipt/checkpoint가 있거나 이미 복구 전용
  실행이면, 그 모델을 복구 대기 상태로 남겨 횟수 제한 없이 다시 실행한다.
  다른 노드가 해당 작업을 가져가도 복구 전용 모드를 유지하여 새 seed를 열지 않는다.
- 복구 worker의 종료 코드가 0이어도 실행 잠금 없는 ERROR receipt가 남아 있으면
  완료로 간주하지 않는다. 재시도 대상 모델은 `MODEL QUEUE`의 `WAITING ... RESUME`에도
  유지한다. 기존 실패 이력은 보존하고, 복구 성공 후 coordinator의 활성 오류만 해제한다.
- 오류 작업의 cache/prefix가 미완료이면 해당 선행 작업도 ERROR 우선순위를 상속한다.
  task 시작 중 `OSError`·subprocess 오류는 실패 receipt와 로그를 저장하고 같은 배정기에서
  다시 시도한다. receipt 없이 저장된 일부 cache prompt도 복구 대상으로 찾는다.
  반대로 이미 완료된 cache의 옛 ERROR receipt는 새 복구 작업으로 세지 않는다.

worker 재시작 확인 간격은 기본 10초이며, 실제 실패 task의 기본 재시도 간격은 60초다.
명시적인 stop·Ctrl-C/SIGTERM, MATH 완료 전 MBPP 금지, 다른 노드의 실행 잠금은 유지한다.
입력·모델을 바꿔 오류를 숨기거나 결과·checkpoint·rollout을 초기화하지 않는다.
공통 학습 engine, 세 모델 adapter와 고정 실행 hash, 25-update 보상 기록 확장은 수정하지 않았다.

공개 `sh scripts/run_srgc_llama31.sh all` 진입점에서 실제 자식 프로세스를 사용한
CPU 통합 검사는 `Llama 실패 작업(실행 이력 4회) → 같은 작업 반복 재시도 → Gemma 실패 작업(실행 이력 99회) 복구
→ 남은 MATH → MBPP` 순서와 checkpoint 바이트 보존을 검사한다. GPU worker는
시험 대역이며 H100 학습을 실행한 검사가 아니다. 이 환경에는 원격 `/group-volume`과
실행 콘솔이 없으므로 서버에서 실제로 발생한 traceback 및 재개 적용은 별도 확인이 필요하다.

최종 CPU 회귀 검사 400개와 subtest 10개를 통과했다. 이후 종료 코드 124·134·139도
운영자 중지와 구분하여 복구하도록 보강하고, 관련 coordinator·공개 셸 검사 70개를
재실행해 통과했다(앞선 검사와 중복 포함). Ruff·공백 검사도 통과했다.

원격의 이전 worker에는 코드가 자동 주입되지 않는다. 정상 학습 중인 worker를 건드리지
않고, 종료됐거나 IDLE인 이전 worker를 종료한 뒤 갱신된 checkout에서 같은 시작 명령을 쓴다.

```sh
git pull --ff-only origin master
sh scripts/run_srgc_llama31.sh all
```

## 2026-10-18 실제 MATH plan 이름과 완료 경계 수정

사용자가 보고한 `LLAMA refused: no math plan in requested queues`를 실제 plan 이름으로
재현했다. Qwen·Llama·Gemma adapter가 만드는 plan은 `dataset: math_train`인데,
운영 배정기는 CLI 단계 이름 `math`와 직접 비교했다. 앞선 합성 queue 시험이 plan의
데이터셋 이름을 `math`로 바꿔 이 문제를 놓쳤다. 이제 기존 시험도 실제 `math_train`을
사용하고, 실제 세 adapter가 준비한 plan으로 별도 통합 검사를 수행한다.

`datasets.py`는 배정 판단에만 `math_train → math`를 적용한다. plan JSON·입력·체크포인트의
원래 이름과 hash는 유지한다. MATH queue 선택, 복구 backlog 필터, 현재 단계 배정,
MBPP를 막는 MATH 완료 장벽 네 지점을 함께 수정했다. 이 수정은 `04ce845`로 우선 push했다.

추가 점검에서 다음 두 경계도 수정했다.

- 캐시 보상 배열만 있고 `cost-summary.json` 인계가 없거나 bundle hash가 다르면,
  실제 모델 queue와 동일하게 미완료로 판단한다. coordinator의 ERROR 우선 복구에서
  해당 캐시를 누락하지 않는다. 비용 합계의 측정 완료 여부와 캐시 인계 여부는 구분한다.
- endpoint가 저장됐어도 해당 작업의 queue/execution lease가 유지되면 worker 시작 검사와
  drain 종료 검사 모두 완료로 반환하지 않는다. 기존 ERROR backlog가 없는 경우도 검사한다.

실제 adapter 준비 plan 검사는 세 모델 각각 MATH 선택·실행 이력 99회인 실패 작업 배정·MBPP 선행 금지와
실제 cache 자식 명령 생성, SIGKILL 뒤 두 번 재개, 과거 실패 기록·plan 바이트 보존을 확인한다.
자식 GPU 계산은 CPU 시험 대역으로 교체하므로 원격 H100 실행을 확인한 검사는 아니다.

Git에 추적된 관련 코드와 변경 이력을 다음 실행 순서로 대조했다.

| 구간 | 점검한 코드·계약 |
| --- | --- |
| 공개 명령 → 모델 연결 | 세 `run_srgc_*.sh`, `model_chain.py`, 모델별 run entry의 Python·인자·root·중지 전달 |
| 기존 plan → 공유 배정 | 모델 adapter·storage의 실제 plan 이름, `ready_dispatch.py`, `resume_drain.py`, `resume_inventory.py`의 phase·복구·완료 조건 |
| task → GPU 자식 | 모델별 `task_command`·rank entry의 plan·seed·cache/train 명령과 학습 `--resume` 전달 |
| 중단 → 저장 복구 | `run_experiment.py`, `srgc_step_checkpoints.py`, rollout 저장·checkpoint backup의 원자적 저장·기존 optimizer 상태 복구 |
| 기록 → 조회·export | `reward_curve_rank.py`, 보상 기록·조회 모듈, 공통 model status/results의 원래 dataset·실험 identity 보존 |

공통 engine 30개 파일, Qwen adapter 구성 12개, Llama 구성 23개, Gemma 구성 27개와
학습 checkpoint·보상 기록 3개 파일의 바이트를 수정 전 Git과 비교했다(그룹 간 중복 포함).
이번 운영 수정으로 고정 학습/adapter/기록 hash가 달라지지 않았음을 확인했다.

관련 실행·재개·모델 entry·status/results·보상 기록·rollout·checkpoint backup 회귀
526개와 subtest 18개를 통과했다. 마지막 캐시 인계·완료 잠금 수정 후 영향받는 실제
plan/배정 검사를 200개 다시 실행해 통과했다(앞선 검사와 중복 포함).
Ruff와 공백 검사도 통과했다. 원격 학습 노드 실행 및 H100/NCCL 실측은 미확인이다.

## 2026-10-18 한 번 실패한 arm의 실행 잠금 복구

사용자가 seed 5 On-policy 재개 실패와 반복 ERROR를 보고했다. 실제 서버 traceback은
아직 받지 못했으며, 다음 차단 경로는 로컬의 실제 자식 프로세스와 실행 lease로 재현했다.

Llama/Gemma의 학습 명령은 보상 기록을 위해 `reward_curve_rank --model ...`로 감싸지만,
동결 CLI의 orphan 정리 대상은 원래 `llama31.rank`/`gemma4.rank`뿐이었다. 실패한 rank가
독립 세션으로 남으면 실행 잠금과 GPU 메모리를 계속 소유하여 같은 arm을 다시 잡지 못한다.
Qwen은 top-level `srgc_process_guard` import와 `scripts.srgc_process_guard`가 서로 다른
module 객체가 될 수 있어, 새 target/owner 보호가 실제 guard에 적용되지 않는 경로가 있었다.

운영 `resume_worker`에서 guard에 들어가는 시점에 현재 모델의 기록 rank를 등록한다.
두 import 이름도 같은 guard를 사용하게 묶는다. 원래 guard 파일·adapter·학습 hash는
보존한다. 살아 있는 launcher의 rank 및 다른 사용자의 프로세스는 기존 보호를 유지한다.
시작 로그의 `GUARD recorded-rank recovery active model=...`로 새 경로 적용을 확인한다.

추가로 GPU 점유 검사 또는 guard 대기 코드 75로 자식 실행이 시작되지 않은 경우는
재개 가능한 중단으로 남긴다. 학습 ERROR로 누적하지 않는다. 모델 연결 실행은 GPU 잠금을
반환하여 재확인하고, 단일 모델 실행은 폴링 간격 뒤 같은 저장 작업부터 재개한다.
입력 검증 실패나 실제 GPU 자식의 일반 오류는 ERROR로 유지하며 실패 이력을 삭제하지 않는다.

세 모델 × On-policy/Switch/SR/Random/Prefix의 실제 프로세스 정리·실행 lease 해제·
동일 arm 40→41번째 배정과 기존 checkpoint 바이트 보존을 검사했다. 공개 entry hook,
live-owner 보호, import 복원, GPU 미시작 재시도 관련 검사 38개를 통과했다.
이는 원격 GPU에서 optimizer update가 실제 전진한 것을 확인한 결과는 아니다.

이하 모델 간 순회는 이전 정책이며 2026-10-19 모델 완료 순서로 대체했다.
모델 연결의 복구 전용 순회도 점검했다. 한 모델이 일반 오류로 재실패하면 같은 순회의
다른 모델에 남아 있는 ERROR도 복구 기회를 받는다. Llama 하나만 반복하며 Gemma의
기존 오류까지 막던 경우를 재현·수정했다. 모든 기존 복구 작업이 끝날 때까지 새 seed와
MBPP 선행은 계속 금지한다. 공개 Llama shell의 실제 자식 프로세스 검사도 두 모델의
오류 재시도·원래 checkpoint 보존·MATH 완료 뒤 MBPP 전환을 확인한다.

프로세스 정리·GPU 대기·배정·체크포인트·중간 보상·rollout 기록 관련 회귀 434개와
subtest 18개를 통과했다. 마지막 모델 간 복구 배정 수정 후 coordinator·공개 shell
검사 72개도 다시 통과했다(앞선 검사와 중복 포함). 동결 코드와 저장 기록은 초기화하지 않았다.

## 2026-10-18 ChildFailedError의 원인 기록과 학습 실패 후 복구 순회

사용자가 세 모델에서 `torch.distributed.elastic.multiprocessing.errors.ChildFailedError`가
반복된다고 보고했다. 이 예외는 torchrun의 자식 종료 요약이며, 그 문구만으로는 실제
rank의 Python 예외 또는 종료 신호를 특정할 수 없다. 서버의 안쪽 traceback은 아직
확인하지 못했다. 아래 변경을 모든 서버 오류의 원인 해결로 해석하면 안 된다.

기존 공통 drain은 학습 자식의 일반 실패를 내부에서 계속 재시도했다. 모델 연결의
복구 순회까지 제어가 돌아오지 않아 다른 모델의 ERROR가 막힐 수 있었다. 당시 연결 실행은
실패 receipt를 보존한 뒤 실제 실패 코드를 coordinator로 돌려준다. task 및 GPU lease는
context를 빠져나오며 반환한다. coordinator가 다른 모델의 기존 오류에도 재개 차례를
준 뒤 저장 작업을 다시 시도했다. 이 모델 간 순회는 2026-10-19에 폐기했다. 현재는
자식 실패 후 같은 모델을 계속 재개한다. 운영상 횟수 제한이나 오류 이력 초기화는 없다.

같은 모델 안에서도 같은 우선순위의 ERROR는 마지막 실패 시각이 오래된 순서로 잡는다.
한 arm이 재실패해도 다른 기존 ERROR보다 계속 먼저 선택되지 않는다. ERROR와 그
선행 cache/prefix를 먼저 복구하는 조건, 기존 작업이 남으면 새 작업을 막는 조건,
MATH 완료 후 MBPP를 시작하는 조건은 유지한다. 수정 전 배정 코드로 세 모델의
arm 기아 현상을 각각 재현했다.

`rank_runtime.py`는 동결 rank 및 보상 기록 entry 바깥에 PyTorch elastic의 `record`
처리를 설치한다. cache/smoke/train의 원래 인자와 모델 코드는 그대로 실행한다.
`SRGC RANK START`에 모델, rank, seed/task를 포함한 실제 인자와 error file 경로가
출력된다. 학습 attempt의 `progress/<attempt_id>/elastic-errors/` 아래에 rank별
`error.json`을 보관한다. admission 로그도 해당 로그 폴더의 `elastic-errors/` 아래에
보관한다. 종료된 새 wrapper도 orphan 정리 대상에 등록하여 실행 잠금을 남기지 않는다.

실패 후 마지막 25줄만 재출력하던 대신 해당 attempt의 제한된 로그 구간에서 안쪽 예외와
elastic root cause를 함께 출력한다. 세 모델 공통 status도 `ChildFailedError`보다 실제
rank 예외를 우선 표시한다. SIGKILL처럼 Python 예외가 없는 종료는 신호 기록을 보여주며
없는 Python 원인을 만들어내지 않는다. 실제 실패를 성공으로 처리하지 않는다.

서로 다른 모델의 orphan도 점검했다. 이전 모델의 죽은 rank가 GPU를 점유한 상태에서
다른 모델로 재개하면, 모델 전용 target 목록에는 그 rank가 없어 GPU BUSY가 계속될
수 있었다. 운영 guard가 세 모델의 native/기록/오류 기록 rank를 함께 인식하게 하고,
세 모델의 살아 있는 launcher도 함께 보호한다. 다른 UID와 살아 있는 owner 아래의
rank는 정리하지 않는다. 이 교차 모델 누락을 세 모델에서 재현하고 실제 프로세스의
실행 lease 해제 및 checkpoint 보존도 검사한다.

검증에서는 각 모델 경로로 실제 torchrun 4개 프로세스와 Gloo를 실행했다. SRGC Engine과
작은 CPU AdamW backend로 Prefix/On-policy/Switch/SR/Random checkpoint를 저장한 후
rank 2를 실패시켰다. 원래 예외가 공유 attempt의 error.json에 남는 것을 확인했다.
다음 실행에서 Prefix는 1→2, 네 continuation은 26→27 업데이트로 전진했으며, 가중치,
optimizer, selection 상태가 중단 없는 기준 실행과 정확히 일치했다. 이 검사는 CPU
통합 검사이고 Qwen/Llama/Gemma 모델의 H100/NCCL 학습 재개를 실측한 결과는 아니다.

고정 engine·모델 adapter·체크포인트·보상 기록 파일의 바이트는 변경하지 않았다.
학습 batch reward와 고정 held-out reward 구분, 25-update 간격 기록, 기존 checkpoint와
rollout trace 검증도 유지한다. 이미 실행 중인 Python worker는 메모리에 예전 코드를
가지고 있으므로 pull만으로 그 내부 무한 재시도 경로가 교체되지는 않는다. launcher를
Ctrl+C로 종료한 뒤 `git pull --ff-only origin master`를 하고 기존 실행 명령으로 다시
시작해야 한다. checkpoint를 지우거나 seed를 처음부터 다시 만들 필요는 없다.

이번 공통 실행·재개·status/results·보상·rollout·checkpoint 회귀 561개와 subtest 18개를
통과했다. 마지막 교차 모델 orphan 수정 후 관련 검사 90개와 Qwen/오류 기록 검사
41개도 다시 통과했다(앞선 검사와 중복 포함). 동결 파일 78개의 바이트와 세 모델의
adapter digest가 수정 전 `d2b4cd2`와 같은 것을 확인했고 Ruff/공백 검사도 통과했다.

## 2026-10-19 Llama 완료 후 Gemma 실행

사용자가 Llama ERROR를 남긴 채 Gemma로 넘어가는 동작을 지적했다. 이전의 모델 간
오류 복구 순회를 제거하고, 각 데이터셋에서 현재 모델의 전체 완료를 전환 조건으로
삼는다. 일반 실패·강제 종료·GPU BUSY·peer 소유·재시도 cooldown 모두 같은 모델에서
대기/재개한다. 저장 작업이 없는 준비/검증 실패는 연결 실행을 오류로 끝내며 다음
모델을 실행하지 않는다. 명시적인 stop·Ctrl-C/SIGTERM도 다음 모델을 막는다.

복구 전용 실행의 성공은 해당 모델의 전체 완료가 아니다. 복구 후 같은 모델을 전체
모드로 다시 실행해 남은 seed·arm·평가까지 끝내야 다음 모델로 간다. 종료 코드 0이어도
현재 모델에 미해결 ERROR receipt가 남으면 같은 모델을 다시 재개한다. 내부 task의
실패 이력·checkpoint·원래 재시도 간격과 모델별 Python/인자는 유지한다.

Llama 시작의 `all` 순서는 Llama MATH 전체 완료 → Gemma MATH 전체 완료 → Qwen MATH
전체 완료 → Llama MBPP 전체 완료 → Gemma MBPP 전체 완료 → Qwen MBPP 전체 완료다.
모든 모델의 MATH를 MBPP보다 먼저 마치는 조건을 유지하며, 정상 완료 전 모델 전환은
허용하지 않는다. status의 현재 모델도 실패/대기 중에 그대로 표시한다.

이전 코드에서 오류/강제 종료/대기 9종과 뒤 모델 ERROR의 선행, 정상 종료로 감춰진
ERROR 등 11개 순서 위반을 재현한 뒤 수정했다. 공개 Llama shell의 실제 자식 프로세스
검사도 Llama 오류 재시도 → Llama 전체 실행 완료 → Gemma 오류 재시도 → Gemma 전체
완료를 검증한다. 저장 checkpoint를 초기화하지 않고 원래 바이트를 보존한다.

공통 실행·배정·dataset 순서·status·보상/rollout 기록 회귀 375개를 통과했다. 마지막
공개 shell의 MATH/MBPP 순서 확인을 보강한 뒤 coordinator·shell 83개도 다시 통과했다
(중복 포함). Ruff/공백 검사를 통과했고 동결 학습·모델 adapter·checkpoint·보상 기록
모듈은 수정하지 않았다. 원격 GPU의 실제 학습 실행을 확인한 검사는 아니다.
