# Gemma / Llama 중간 reward 기록

공개 실행 진입점의 Gemma·Llama **train 작업**은 이제 별도 기록 확장을 설치한다. 기존 frozen learner·adapter 파일과 구현 식별 해시는 바꾸지 않는다. 기록 확장은 자체 SHA-256을 `protocol.json`에 저장한다. 기존에 이미 떠 있는 프로세스에는 소스 파일을 갱신해도 자동 적용되지 않는다. 실행 중인 작업을 임의로 중단하지 말고, 다음 정상 재시작에서 현재 공개 진입점을 사용한다.

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

`sh scripts/run_srgc_gemma4.sh results`, `sh scripts/run_srgc_llama31.sh results`의 공개 results 명령은 기존 단일 compact JSON에 `datasets.DATASET.reward_curves`를 추가한다. Qwen의 **이미 저장된 progress history**도 같은 export 경로에서 보존한다. Qwen에 새로운 중간 held-out 평가를 설치하는 변경은 아니다.

shared prefix의 scalar 기록이 있으면 1–25도 각 arm의 `training_rewards`와 25-step 구간 평균에 포함한다. `shared_prefix: true`로 공통 학습 구간임을 표시하고, `shared_prefix_training_recorded_updates`로 기록된 개수를 알린다.

각 curve에 `training_rewards`, `training_windows_25`, `held_out_rewards`, `missing_held_out_steps`, `diagnostic_costs`가 있다. reward는 0–1, `reward_percent`는 0–100이다. 원시 응답·token/logprob 배열은 compact 결과에 넣지 않는다. 파일이 검사 도중 바뀌거나 run identity가 맞지 않으면 오류를 표시하고 results 재실행이 필요하다.
