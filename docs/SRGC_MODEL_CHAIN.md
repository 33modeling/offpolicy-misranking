# Qwen → Llama → Gemma 자동 연결

공개 학습 시작 명령은 이제 한 모델의 큐가 완료되거나 당장 가져올 작업이 없으면 다음 모델을 확인한다. Qwen 시작은 `qwen35 → llama31 → gemma4`, Llama 시작은 `llama31 → gemma4 → qwen35`, Gemma 시작은 `gemma4 → qwen35 → llama31` 순서다.

```sh
git pull --ff-only origin master
sh scripts/run_srgc_qwen35.sh all
# 이미 Llama를 진행하던 노드라면 다음 정상 시작 때:
sh scripts/run_srgc_llama31.sh all
```

노드당 위 명령 하나를 실행한다. `math` 또는 `mbpp`를 지정하면 새로 시작하는 작업의 데이터셋 범위는 모든 모델에서 유지된다. 기존 다른 데이터셋의 미완료 작업을 우선 복구하는 정책도 유지한다. `--root`, `--source-plan`, `--max-attempts` 등 모델별 인자는 시작 모델에만 전달하며, 재방문할 때도 보존한다. 다른 모델은 각자의 기존 root·Python·고정 모델 snapshot을 사용한다.

각 모델의 학습·최종 평가는 기존 실행기에 맡긴다. 학습 update 275 도달만으로 다음 모델로 넘기거나 완료로 표시하지 않는다. 남은 최종 평가도 기존 작업의 일부로 처리한다. 대기 중인 작업을 다른 노드가 소유하거나 실패 재시도 대기 중이면, 작업 잠금을 유지한 채 다른 모델을 실행하지 않고 현재 worker의 GPU 잠금을 해제한 뒤 다음 모델로 넘어간다. 당장 실행할 작업이 세 모델 모두 없을 때만 재확인을 기다린다.

전환은 새 자식 프로세스로 실행하므로 모델별 Python·Gemma 패키지 overlay·CUDA 상태가 섞이지 않는다. 자동 Llama 전환에서는 수동 시작에 있던 GPU 프로세스 정리를 호출하지 않는다. 기존 다른 작업의 GPU 소유권은 공유 lease로 보호한다. 작업 실패가 재시도 한도에 도달하면 해당 실패를 보존하고 다른 모델을 확인한다. 실제 실패를 IDLE로 숨기거나 실패 예산을 초기화하지 않는다. Ctrl-C/SIGTERM은 현재 소유한 자식에게 전달하고 전체 연결 실행을 멈춘다.

`status`·`results`·`stop`·`doctor`·`prepare` 등은 기존처럼 해당 모델에만 적용된다. 각 모델의 text/JSON status에는 공통 `MODEL QUEUE`/`model_chain` 상태가 추가된다. 이는 어느 노드가 현재 Gemma 등 다른 모델을 실행 중인지 보여 준다. heartbeat가 오래된 기록은 STALE로 구분한다. 실험 결과의 완료 여부는 기존의 검증된 task·endpoint 기록을 사용한다. 연결 실행기의 `finished`는 명령이 끝났다는 뜻이며, 개별 실험의 중지·완료 판정을 대신하지 않는다.

운영 상태는 `$SRGC_STORAGE_ROOT/.model-chain/workers/`에 별도로 기록한다. 기본 경로는 `$OM_WORK/srgc-rebuttal/.model-chain/workers/`다. 고정 engine·모델 adapter·plan·입력·기존 결과·체크포인트는 변경하지 않는다. 기존 중간 reward 기록과 25-update 고정 held-out 평가 확장도 같은 실행기를 통해 유지한다. 학습 배치 reward와 held-out reward는 기존처럼 분리해서 export한다.

필요할 때 `SRGC_MODEL_ORDER=qwen35,llama31,gemma4`로 등록된 모델의 순서를 정하거나 일부만 선택할 수 있다. 지정한 목록에는 시작 모델이 포함돼야 한다. `SRGC_MODEL_POLL_SECONDS`는 1~60초이며 기본 10초다. 의도적으로 한 모델만 실행하려면 `SRGC_MODEL_CHAIN=0`을 지정한다.

이미 떠 있는 Python 프로세스에는 새 연결 로직이 자동 주입되지 않는다. 정상 학습 중인 작업을 중단하지 말고 다음 정상 시작 때 현재 공개 명령을 사용한다. 이전 코드로 IDLE인 worker는 종료 후 같은 명령으로 다시 실행해야 한다. 이 로컬 PC에는 원격 학습 노드의 `/group-volume`이 없으므로 원격 전환이 실제 적용됐다고 간주하지 않는다.
