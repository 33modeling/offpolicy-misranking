# Gemma / Llama 중간 reward 기록

공개 실행 진입점의 Qwen·Gemma·Llama **train 작업**은 별도 기록 확장을 설치한다. 기존 frozen learner·adapter 파일과 구현 식별 해시는 바꾸지 않는다. 최초 기록 확장의 SHA-256은 `protocol.json`에 보존하며, 이후 기록 코드의 버전은 별도 receipt로 구분한다. 기존에 이미 떠 있는 프로세스에는 소스 파일을 갱신해도 자동 적용되지 않는다. 실행 중인 정상 작업을 임의로 중단하지 말고, 다음 정상 재시작에서 현재 공개 진입점을 사용한다.

## `reward curve protocol changed` 재개 오류 수정

이전 검사는 기록용 Python 파일의 해시까지 `protocol.json`과 완전히 같아야 한다고 요구했다.
따라서 응답 저장 기능을 추가하는 등 기록 코드만 업데이트해도 기존 Llama/Gemma 실행이
`reward curve protocol changed`에서 멈출 수 있었다. 현재는 다음처럼 구분한다.

- `seed`, arm, plan·input·frozen learner 해시, 평가 ID·문항 수, 응답 수, 기록 간격,
  평가 stream과 reward 종류는 그대로 일치해야 한다. 실제 차이는 GPU update 전에
  차이 난 필드와 함께 오류로 보고한다.
- 기록 코드의 해시만 달라졌다면 기존 `protocol.json`을 덮어쓰지 않고
  `reward-curves/ARM/recording-extensions/SHA256.json`에 현재 기록 코드와 원본
  protocol의 연결을 남긴다. 새로운 point·응답 기록은 작성한 코드 버전을 별도로 표시한다.
- 기존 평가 point·checkpoint·응답 gzip은 해시와 실제 weights를 검증한 뒤 그대로 재사용한다.
  응답 파일을 쓴 직후 중단된 update도 동일 응답으로 재시도하면 원본 바이트와 작성 버전을
  보존한다. 과거 학습 batch reward와 새 held-out 평가를 서로 대체하지 않는다.

오류가 난 노드에서 현재 코드로 업데이트한 다음 기존 root를 그대로 재개한다.
`prepare`, root 초기화, `protocol.json` 삭제, checkpoint 삭제는 하지 않는다.

```sh
git pull --ff-only origin master
sh scripts/run_srgc_llama31.sh math resume --root /group-volume/minsoo3.kim/offpolicy-misranking/srgc-rebuttal/llama31-8b-v1
```

이 명령은 해당 노드의 기존 실행 checkout에서 사용한다. 공개 launcher는 기존 모델·데이터셋
완료 순서와 실패 작업 재시도 정책을 유지하며, learner를 처음부터 다시 학습하지 않는다.
수정 코드를 push한 것만으로 원격 노드의 프로세스가 업데이트되거나 재개된 것은 아니다.

## 무엇을 언제 저장하는가

| 기록 | 시점 | 계산 대상 | 저장 위치 |
|---|---|---|---|
| 학습 batch reward | optimizer update 완료마다 | 그 update에서 실제 학습한 4개 prompt × 새 응답 8개, 32개 이진 reward 평균 | `seed-N/reward-curves/ARM/training.json` |
| 25 update 구간 평균 | 위 파일을 쓸 때마다 | 26–50, 51–75 등 실제 기록된 학습 batch reward 평균. 미완료 구간은 `complete: false` | 같은 파일의 `windows_25` |
| 평가용 weight archive | 완료 update 25, 50, …, 275 | 그 시점의 실제 learner weights·selector state·history. optimizer는 제외 | `seed-N/reward-curves/ARM/step-T.pt` |
| 고정 held-out reward | 완료 update 25, 50, …, 250 | 해당 seed bundle의 고정 `evaluation_ids` 300개 × 응답 8개. prompt별 평균과 전체 평균 모두 보존 | `seed-N/reward-curves/ARM/step-T.json` |
| 최종 held-out reward | 완료 update 275의 기존 최종 평가 | 기존 최종 평가를 그대로 사용하며 두 번 평가하지 않음 | 기존 `ARM-endpoint.json`을 results에 합침 |

25 update는 **학습 완료 횟수**다. 예를 들어 50의 기록은 50번째 update를 반영한 weights를 평가하고, 이후 51번째 update가 시작된다. shared prefix 1–25는 on-policy이며, 각 arm의 continuation history는 26부터 시작한다. step 25 archive·평가는 `reward-curves/prefix/`에 공통으로 저장한다. 서로 다른 arm이 동시에 시작하면 파일 lease로 중복 측정을 막는다.

학습 batch reward는 선택된 학습 데이터의 관측값이다. held-out 성능과 동일하게 해석하거나 표에서 같은 값으로 대체하면 안 된다. 최종 평가만 있던 과거 실행의 중간 held-out 성능을 학습 reward로 채우지 않는다.

## 재시작과 비용

- 매 update의 기존 `ARM-latest.pt`는 model·optimizer를 포함하는 **재개용 checkpoint**로 계속 사용한다. `step-T.pt`는 optimizer를 제외한 평가용 archive이므로 학습 재개에 사용하지 않는다.
- 경계에서 평가가 중단됐으면 실제 재개된 weights를 먼저 측정한 뒤 다음 update로 넘어간다. 완료된 point는 protocol·checkpoint 해시·평가 ID와 현재 실제 backend weights가 같은지 검증한 후 재사용한다. point 없이 남은 archive는 실제 weights에서 다시 쓴다.
- Python·NumPy·Torch RNG와 model train/eval 상태는 진단 평가 전후에 복원한다. 선택·학습 sampler나 optimizer에는 평가 결과를 전달하지 않는다.
- 추가 평가는 300 × 8개 응답을 생성하므로 학습 실행 시간이 크게 늘어난다. 진단 비용은 `reward-curves/ARM/cost-receipts/step-T/`에 분리한다. 각 point의 비용은 **그 step에서 발생한 모든 시도**의 합계이며, 다른 step의 누적 비용을 반복해서 더하지 않는다. 기존 learner의 선택·학습 비용과 합치지 않는다. 프로세스 전체의 포괄적 session 시간에는 진단 시간이 포함될 수 있으므로 session과 phase 비용을 더해서 사용하지 않는다.
- 이미 지나간 경계의 archive가 없으면 그 시점의 held-out reward를 복원할 수 없다. `missing_held_out_steps`로 누락을 그대로 표시한다.

## 결과 파일

`sh scripts/run_srgc_gemma4.sh results`, `sh scripts/run_srgc_llama31.sh results`의 공개 results 명령은 기존 단일 compact JSON에 `datasets.DATASET.reward_curves`를 추가한다. Qwen의 **이미 저장된 progress history**도 같은 export 경로에서 보존한다. 현재 Qwen·Gemma·Llama의 기록 진입점은 모두 이후 25-update 경계의 평가를 저장한다. 과거에 측정하지 않은 held-out 값은 채워 넣지 않는다.

shared prefix의 scalar 기록이 있으면 1–25도 각 arm의 `training_rewards`와 25-step 구간 평균에 포함한다. `shared_prefix: true`로 공통 학습 구간임을 표시하고, `shared_prefix_training_recorded_updates`로 기록된 개수를 알린다.

각 curve에 `training_rewards`, `training_windows_25`, `held_out_rewards`, `missing_held_out_steps`, `diagnostic_costs`가 있다. reward는 0–1, `reward_percent`는 0–100이다. 원시 응답·token/logprob 배열은 compact 결과에 넣지 않는다. 파일이 검사 도중 바뀌거나 run identity가 맞지 않으면 오류를 표시하고 results 재실행이 필요하다.
