# Qwen → Llama → Gemma 자동 연결

공개 학습 시작 명령은 **선택한 모든 모델의 MATH 완료 → 선택한 모든 모델의 MBPP** 순서로 실행한다. Qwen 시작의 기본 순서는 `Qwen MATH → Llama MATH → Gemma MATH → Qwen MBPP → Llama MBPP → Gemma MBPP`다. Llama 시작은 각 데이터셋 안에서 `llama31 → gemma4 → qwen35`, Gemma 시작은 `gemma4 → qwen35 → llama31` 순서를 따른다.

```sh
git pull --ff-only origin master
sh scripts/run_srgc_qwen35.sh all
# 이미 Llama를 진행하던 노드라면 다음 정상 시작 때:
sh scripts/run_srgc_llama31.sh all
```

노드당 위 명령 하나를 실행한다. `all`은 두 plan을 먼저 준비하되 MATH 단계에서 MBPP 학습·생성·평가는 실행하지 않는다. `math`는 MATH만 수행한다. `mbpp`도 준비돼 있는 MATH sibling plan을 먼저 완료한 뒤 MBPP를 실행한다. MATH plan이 없으면 오류를 내며, MATH를 건너뛰거나 빈 큐를 완료로 간주하지 않는다. `--root`, `--source-plan`, `--max-attempts` 등 모델별 인자와 Python 선택은 시작 모델에서 그대로 보존하며, 재방문과 데이터셋 전환에도 유지한다. 다른 모델은 각자의 기존 root·Python·고정 모델 snapshot을 사용한다.

각 모델의 학습·최종 평가는 기존 실행기에 맡긴다. 학습 update 275 도달만으로 완료로 표시하지 않는다. 남은 최종 평가도 기존 작업의 일부다. MATH 작업을 다른 노드가 소유하거나 실패 재시도 대기 중이면 현재 worker의 GPU 잠금을 해제한 뒤 다른 모델의 MATH를 확인한다. 선택한 모델의 MATH가 하나라도 남으면 MBPP로 넘어가지 않고 주기적으로 다시 확인한다. endpoint가 저장돼도 실행 잠금이 남아 있으면 완료 경계를 넘지 않는다.

현재 데이터셋에서 선택된 모델들의 **하다 멈춘 작업을 먼저 복구한 뒤 새 작업을 시작**한다. 복구 전용 실행은 체크포인트부터 기존 작업만 마치고 반환한다. 다른 모델에 멈춘 작업이 남아 있으면 새 seed로 넘어가지 않는다. 다른 worker가 실행 중인 작업은 실행 잠금으로 구분하여 중복 실행하지 않는다.

전환은 새 자식 프로세스로 실행하므로 모델별 Python·Gemma 패키지 overlay·CUDA 상태가 섞이지 않는다. 자동 Llama 전환에서는 수동 시작의 GPU 프로세스 정리를 호출하지 않는다. 기존 다른 작업의 GPU 소유권은 공유 lease로 보호한다. **누적 실패 3회 등의 재개 제한은 없다.** 서버가 GPU를 강제로 종료해 ERROR가 쌓여도 횟수만으로 복구를 막지 않는다. 과거 예산 파일도 실행 조건으로 사용하지 않는다. 기본 재시도 간격 60초 뒤 미완료 작업을 계속 재시도하며 실패 이력은 보존한다. 입력·모델 identity 오류나 명시적인 일시 중지는 별도로 보고하고, 완료로 간주하지 않는다. Ctrl-C/SIGTERM은 소유한 자식에게 전달하고 전체 연결 실행을 멈춘다.

`status`·`results`·`stop`·`doctor`·`prepare` 등은 기존처럼 해당 모델에만 적용된다. 각 모델의 text/JSON status에 공통 `MODEL QUEUE`/`model_chain` 상태와 현재 `dataset=math/mbpp`를 표시한다. heartbeat가 오래된 기록은 STALE로 구분한다. 단계의 완료 여부는 검증된 task·endpoint 및 실행 잠금으로 판단한다. 중지된 단계는 별도 종료 코드로 기록하여 성공/완료와 구분한다. `all resume`는 각 모델의 해당 단계 중지를 해제하되, 재방문할 때 중지 해제를 반복하지 않는다.

운영 상태는 `$SRGC_STORAGE_ROOT/.model-chain/workers/`에 별도로 기록한다. 기본 경로는 `$OM_WORK/srgc-rebuttal/.model-chain/workers/`다. 고정 engine·모델 adapter·plan·입력·기존 결과·체크포인트는 변경하지 않는다. 기존 중간 reward 기록과 25-update 고정 held-out 평가 확장도 같은 실행기를 통해 유지한다. 학습 배치 reward와 held-out reward는 기존처럼 분리해서 export한다.

필요할 때 `SRGC_MODEL_ORDER=qwen35,llama31,gemma4`로 등록된 모델의 순서를 정하거나 일부만 선택할 수 있다. 지정한 목록에는 시작 모델이 포함돼야 한다. `SRGC_MODEL_POLL_SECONDS`는 1~60초이며 기본 10초다. 의도적으로 한 모델만 실행하려면 `SRGC_MODEL_CHAIN=0`을 지정한다.

이미 떠 있는 Python 프로세스에는 새 연결 로직이 자동 주입되지 않는다. 정상 학습 중인 작업을 중단하지 말고 다음 정상 시작 때 현재 공개 명령을 사용한다. 이전 코드로 IDLE인 worker는 종료 후 같은 명령으로 다시 실행해야 한다. 이 로컬 PC에는 원격 학습 노드의 `/group-volume`이 없으므로 원격 전환이 실제 적용됐다고 간주하지 않는다.

2026-10-17 검증: 세 모델의 단계 전환·오류 재시도·peer 실행 잠금·중지·원래 명령 인자 보존에 대한 최종 검사 179개를 통과했다. 공통 status·중간 보상·실제 SIGKILL 복구·셸 진입점도 회귀 검사했다. 수정 Python 파일의 Ruff/공백 검사를 통과했고, 세 모델 adapter·공통 engine·보상 기록 확장·동결 runtime 파일은 수정 전과 동일함을 확인했다.

2026-10-18 수정: 실패 횟수 제한을 제거했다. 살아 있는 coordinator는 SIGKILL/137로 종료된 worker를 다시 실행한다. 일시적인 `nvidia-smi` 실패도 재확인한다. 이미 coordinator 자체가 종료된 경우에는 같은 공개 명령을 다시 실행해 저장된 작업부터 복구한다. 관련 회귀 316개 통과 후 실제 SIGKILL child 재시작·GPU 조회 복구를 추가해 최종 운영 검사 201개를 통과했다. 과거 실패 이력과 예산 파일은 삭제하지 않았고, 학습·보상 기록 hash를 유지했다.
