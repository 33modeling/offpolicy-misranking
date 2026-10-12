# Qwen SRGC가 275/275에서 오래 보이는 이유 — 2026-10-12

## 코드에서 확인한 실행 순서

대상은 `scripts/run_srgc_qwen35.sh`로 시작하는 SRGC 실험이다.
`275`는 마지막 optimizer update다. 해당 숫자가 출력돼도 최종 평가와
결과 저장은 남아 있다.

1. 마지막 update 275를 수행하고 update progress를 기록한다.
2. 마지막 optimizer checkpoint를 저장한다. CPU 복사와 디스크 flush도 포함된다.
3. held-out 평가 300문항에서 문항당 응답 8개, 총 **2,400개 응답**을 생성하고 채점한다.
4. 4개 rank가 문항을 75개씩 나누어 처리하고 평가 결과를 모은다.
5. `<arm>-endpoint.json`을 원자적으로 저장하고 worker가 작업을 마친다.
6. status는 유효한 endpoint와 해제된 작업 lease를 확인한 뒤 `DONE`으로 표시한다.

Qwen generation micro-batch는 2다. 한 문항의 응답 8개를 네 번으로 나누어
생성하며, 각 응답은 최대 2,048 token이다. OOM 발생 시 micro-batch 1로
같은 draw를 재시도한다. 이 경우 평가 시간도 늘어날 수 있다.
평가에는 optimizer update나 gradient 계산이 없다.

관련 구현:

- `srgc_rebuttal/run_experiment.py`: 마지막 update, checkpoint, 평가, endpoint 순서.
- `srgc_rebuttal/torch_backend.py`: rank별 문항 분할과 평가 결과 gather.
- `scripts/srgc_qwen35_memory.py`: generation micro-batch와 OOM 재시도.
- `scripts/srgc_step_checkpoints.py`: checkpoint 저장과 rollout cache 설치.
- `scripts/srgc_resumable_rollouts.py`: 문항별 평가 응답 저장과 재사용.

## 확인한 표시 문제와 수정

기존 표는 평가 중에도 `RUN 275/275`로 표시했다. 상세 정보에 evaluation이
있더라도 학습이 같은 step에 멈춘 것으로 보일 수 있었다.
Qwen·Llama·Gemma가 사용하는 공통 status를 다음과 같이 수정했다.

| 표시 | 의미 |
| --- | --- |
| `RUN 274/275` | 기록상 학습 update 진행 중 |
| `EVAL 120/300 saved` | 마지막 기록이 평가이며, 해당 평가 cache에서 120문항의 저장 파일을 관찰함 |
| `EVAL` | 마지막 기록이 평가이지만 저장 파일 수를 확인할 수 없음 |
| `DONE 50.0%` | 작업 lease가 해제되고 유효한 최종 endpoint reward를 확인함 |

평가 cache 경로는 아래와 같다.

```text
<output_root>/seed-<seed>/rollout-cache/<arm>/eval-<evaluation_seed>/
```

`evaluation_seed`는 학습 seed와 마지막 update 번호로
`stream_seed(seed, total_updates, "reporting-evaluation")`를 계산한 값이다.
각 파일명은 평가 문항 ID의 SHA-256 앞 24자리와 응답 수로 결정된다.
해당 stream·arm·seed의 평가 문항에 해당하는 `*-8.npz` 일반 파일만 센다.
다른 문항, 다른 응답 수, 다른 stream, 임시 파일, 같은 이름의 디렉터리는 제외한다.

`saved`는 원자적으로 저장된 파일의 존재를 관찰한 수다. status가 token 배열을
읽거나 평가 reward를 재검증한 수가 아니다. **300/300 saved도 DONE과 다르다.**
평가가 끝나면 cache 디렉터리를 지우고 endpoint를 저장하므로, 디렉터리가 없는
순간에는 저장 수를 알 수 없다. 이 경우 0/300이나 300/300을 추측하지 않는다.
rank의 마지막 progress 정보도 함께 표시한다. 이 정보는 실제 GPU 사용률이나
예상 종료 시각을 보장하지 않는다.

## 현재 노드에서 구분할 항목

```sh
sh scripts/run_srgc_qwen35.sh all status
sh scripts/run_srgc_qwen35.sh all status --json
```

- `checkpoint_save`: 마지막 update 후 checkpoint 저장 중일 수 있다.
- `evaluation`: 최종 평가 단계. 저장 문항 수와 rank progress가 변하는지 확인한다.
- 로그에서 같은 task의 attempt 증가, CUDA OOM, NCCL 오류가 반복되는지 확인한다.
- cache 수가 그대로여도 한 문항의 응답 생성·채점 또는 rank 동기화 중일 수 있으므로,
  숫자 하나만으로 정지나 정상 진행을 단정하지 않는다.

중단된 평가를 재개하면 저장된 rollout을 읽고 나머지를 생성하도록 구현되어 있다.
평가 완료 후 cache 정리와 endpoint 저장 사이에 중단되면 cache를 다시 생성할
여지는 있지만, 현재 노드에서 이 상황이 발생했다는 증거는 확보하지 못했다.

이번 확인에서는 현재 원격 노드의 로그와 GPU 상태를 확보하지 못했다.
따라서 **오래 걸리는 노드가 정상 평가 중인지, 반복 실패 중인지는 미확인**이다.
표시 수정만으로 평가가 빨라지거나 실행 중인 정지가 해결됐다고 판단하지 않는다.

## 변경 범위와 검증

변경은 read-only status loader·renderer, 회귀 테스트, 이 설명 문서다.
frozen engine·모델 adapter와 runtime pin, 입력·결과·optimizer checkpoint를 보존한다.
최신 checkout에서 status를 별도로 실행하면 조회할 수 있으며, 이 표시를 보기 위해
정상 실행 중인 GPU 작업을 재시작할 필요는 없다.

공개 runner를 세 모델에 대해 검사한다. cache 없음 → 2/300 → 300/300 →
cache 정리 → endpoint 저장 → lease 해제 순서에서 조기 완료가 발생하지 않는지,
잘못된 stream·문항·응답 수와 임시 파일을 제외하는지 확인한다.
조회 전후 기존 실험 파일이 동일하고 GPU 패키지 없이 동작하는지도 검사한다.

검증 결과: 아래 테스트 **77개 통과**. 세 모델의 adapter/engine digest가 유지되고,
변경한 status 모듈이 frozen runtime 파일 목록에 포함되지 않음을 확인했다.

```sh
python3 -m pytest -q srgc_research/tests/test_model_status.py srgc_research/tests/test_qwen_status.py srgc_research/tests/test_model_results.py
```
