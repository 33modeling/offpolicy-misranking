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

MATH뿐 아니라 MBPP의 `seed-8.random`, `seed-9.on_policy`에도 같은 수정이 적용된다.
위 두 MBPP 작업에서 기존 오류가 남아 있다면 해당 노드 checkout을 업데이트한 뒤
같은 root에서 MBPP를 재개한다. 다른 노드의 정상 학습을 중단할 필요는 없다.

```sh
git pull --ff-only origin master
git log -1 --oneline
sh scripts/run_srgc_llama31.sh mbpp resume --root /group-volume/minsoo3.kim/offpolicy-misranking/srgc-rebuttal/llama31-8b-v1
```

현재 검사는 실제 설정 차이가 있으면 `changed fields:` 뒤에 해당 필드를 명시한다.
기록 코드 해시만 바뀐 경우는 허용하며, 기존 protocol·reward·checkpoint는 보존한다.
이 동작은 MBPP seed 8 Random·seed 9 On-policy의 저장 경로에서도 검증한다.

## 제출한 OLMo에 맞춘 시간 집계

OLMo 제출본 `3d144ee666e480766e253d06221f6e29feaef25e`의 수치와 집계는 고정한다.
Qwen·Llama·Gemma의 공개 `status`와 `results`에 같은 비교 기준의
`submission_timing`을 별도로 제공한다. 기존 `GPU TIMES`와 원시 비용 기록은 유지한다.

| 항목 | 제출본과 맞추는 기준 |
| --- | --- |
| 비교 구간 | 공통 25 updates 이후의 26–275, 총 250개 완료 update |
| 학습 | 재개된 최종 학습 경로의 `history[].training_gpu_seconds`를 update별 한 번씩 합산 |
| 선별 | 같은 경로의 `history[].selection_gpu_seconds` 전체 합계. 후보 생성·gradient·validation·연산·부대 비용을 포함한 선별 타이머 |
| SR 준비 | 초기 보상 캐시에서 부분집합을 준비한 SR의 기록값을 SR·Switch에 동일하게 한 번 부과 |
| 합계 | 학습 + 누적 선별 + 공통 SR 부분집합 준비 |
| 제외 | 캐시 생성, 공통 prefix, 평가, checkpoint 저장·불러오기, 시작 비용, 별도 진단 측정, 최종 경로에 남지 않은 재시도 update |
| 단위 | 이미 기록된 GPU-seconds를 3,600으로 한 번 나눔. GPU 4개를 다시 곱하거나 나누지 않음 |
| 누락 | update 타이머 또는 공통 SR 준비가 누락되면 최종 합계는 `null`. 미완료 구간을 전체 구간으로 확대하지 않음 |

원래 phase 합계는 **완료된 재시도를 포함한 실제 소비 비용**이다. 제출 기준의 최종 경로
시간과 구분해 그대로 보존한다. 두 합계의 차이를 전부 재시도 시간이라고 단정하지 않는다.
중간 held-out 평가와 응답 저장도 별도 진단·기록 비용으로 남긴다.

OLMo seed 5–9의 후보 쪽 시간 추정치는 validation·대기 등을 제외하므로 이 합계의
대체값으로 사용하지 않는다. 시간 집계를 통일하는 것이 응답 길이, 선별 주기,
모델별 실행 설정까지 동일하게 만드는 것은 아니다.

```sh
sh scripts/run_srgc_qwen35.sh all status
sh scripts/run_srgc_qwen35.sh all results
sh scripts/run_srgc_llama31.sh all results
sh scripts/run_srgc_gemma4.sh all results
```

`status`는 `SUBMITTED OLMO TIME BASIS` 표에 구간·기록한 update 수·SR 준비·선별·학습·합계를
표시한다. 기존 단일 결과 JSON의 `datasets.DATASET.submission_time_basis`와
`tasks[].submission_timing`에도 동일한 기준, update별 원기록, 결측 목록, 출처를 남긴다.
기존 `timing`과 실행 설정도 compact JSON에서 보존한다.

`results`에는 `EXCLUDED GPU TIME`도 함께 나온다. 공통 캐시·prefix를 seed별 한 번 표시하고,
각 arm의 평가·checkpoint 입출력·시작 비용을 별도 열로 표시한다. 이 값들은 비교 합계에
들어가지 않는다. 일부 phase만 기록됐으면 `>=`, 기록이 없으면 `-`다. 별도 reward-curve
평가와 응답 기록 비용 역시 비교 합계 밖에 있다. 포괄적 session 시간을 update 타이머에
더하거나, 원래부터 제외된 평가 비용을 비교 합계에서 다시 빼지 않는다.

제출본의 고정 update 비교와 고정 allocation 비교는 집계 범위가 다르다. 이 표의 기준은
`v6/overleaf/sections/pair_common_step_costs.tex`의 250-update continuation이다.
실제 선별에 쓰는 후보·reference gradient, 응답 생성과 선별 단계 부대 비용은 포함한다.
측정된 학습·선별 시간을 불필요한 비용으로 간주해 수치를 임의로 줄이지 않는다.

**전달할 때는 파일을 열거나 JSON을 복사하지 않는다.** 공개 `results` 명령은 터미널에
MATH·MBPP별 seed 보상, 전환 시점, 제출 OLMo 기준 GPU 시간, 평균·SD를 짧은 표로 바로 출력한다.
화면 표와 JSON 파일을 동시에 제공하며, 끝의 `JSON: /절대/경로/...json`에서 파일 위치도
항상 확인할 수 있다. 표를 복사하거나 해당 JSON을 전달하면 된다. `--json`을 붙여도
모델별 공개 결과 명령의 화면 표를 숨기지 않는다. 자동 결과 저장·학습 worker의 기존 동작은 유지한다.

기존 Qwen MATH·MBPP의 사용자 제공 GPU 시간 요약표만으로는 최종 경로의 update별 비용을
복원할 수 없다. GPU 노드에서 위 `results`를 다시 내보내 실제 `*-progress.json`을 읽어야
새 기준의 값을 얻는다. 요약표만 보고 134.35 또는 142.88 GPU-hours를 임의 조정하지 않는다.

## 중단·재시작 비용 차감

기존 endpoint의 `costs`는 learner 누적 타이머에 전체 phase ledger를 합쳐 쓰므로,
재시작 전후에 완료된 재수행 phase도 포함한다. 이 endpoint 합계를 최종 경로의 시간으로
쓰지 않는다. Random·SR·On-policy·Switch, MATH·MBPP, Qwen·Llama·Gemma 및 OLMo
추가 seed 보고서 모두 같은 `submission_timing` reader를 사용한다.

- 최종 `*-progress.json`의 완료 update마다 학습·선별 타이머를 한 번만 합산한다.
  내용이 완전히 같은 update 사본은 제거한다. 같은 update의 내용이 충돌하거나 구간이
  빠졌으면 시간이나 보상을 임의 선택하지 않는다. 보상 내보내기도 동일한 사본 제거를 쓴다.
- 전체 완료 selection/training receipt를 phase·checkpoint·시간으로 최종 update와 대조한다.
  현재 기록 확장으로 실행한 update는 실제 receipt ID도 history/checkpoint에 저장한다.
  같은 ID의 같은 완료 receipt 사본은 한 번만 센다. ID가 같고 내용이 다르면 오류다.
- `RESTART TIME CORRECTION`은 `Before`, `Removed`, `Retained`를 표시한다.
  대응이 모두 확인되면 `Before - Removed = Retained`가 일치해야 한다.
  이 표는 학습+선별만 비교하며 SR 준비는 최종 합계에 한 번 더한다.
  history 사본의 중복 시간과 receipt 재수행 시간을 서로 더해 두 번 빼지 않는다.
- 시작만 남은 phase는 지속 시간을 모른다. 종료되지 않은 phase가 있으면 알려진 비용만
  하한으로 표시한다. 최종 update와 매칭되지 않거나 원시 receipt가 없으면 차감액은
  `-`이며, 최종 합계에서 추정 차감하지 않는다. GPU-seconds/3600 이외에 GPU 수나
  재시작 횟수로 다시 나누지 않는다.
- 실패·재시작의 실제 소비 비용, 원래 endpoint, receipt, checkpoint는 그대로 보존한다.
  자동 기록 변경은 다음 정상 시작에 적용되며 실행 중인 학습을 중단하지 않는다.

JSON의 `tasks[].submission_timing.restart_accounting.source_receipts`에는 재계산에 필요한
phase·checkpoint·ID·상태·wall-seconds·GPU 수·GPU-seconds와 기록된 단계별 비용을 보존한다.
`update_timers`의 최종 update 기록과 함께 전달하므로 합계 표만 있는 경우와 구분한다.

과거 합계 표만으로는 어느 update를 몇 번 다시 수행했는지 알 수 없다. 공개 `results`가
GPU 노드의 원시 progress/receipt를 읽어 차감과 대조를 계산한다. 코드 push는 원격
노드의 새 계산 결과를 수신·검증한 것과 구분한다.

## 다른 실험도 터미널에서 바로 복사

기존 공개 `results` 명령을 그대로 사용한다. 화면에 측정값 표를 출력하고 JSON 파일도
함께 저장하며, 마지막에 JSON 경로를 표시한다. 원본·자체 완결 결과 묶음은 유지한다.
N01–N08 및 추가 대조군의 명시적 JSON stdout 모드는 기계 판독용으로 유지한다.

| 대상 | 명령 예시 |
| --- | --- |
| OLMo 추가 seed | `sh scripts/run_srgc.sh all results` |
| OLMo 시간 조회만 | `sh scripts/run_srgc.sh all costs` |
| Qwen / Llama / Gemma | 위 모델별 `all results` |
| N01–N07 | `sh scripts/run_srgc_research.sh all n01 results` (번호 선택) |
| N08 기존 결과 분석 | `sh scripts/run_srgc_research.sh math n08 results /path/results.json` |
| 정보량·실제 update | `sh scripts/run_srgc_information.sh all results` |
| 개별 추가 실험 | `sh scripts/run_srgc_experiments.sh all pool results` (`direction`, `replicate`, `timing`, `rules` 등 선택) |

`pool`·`direction`·`timing`·`rules`·`replicate` 결과는 선택한 대조군과 공통 baseline만
읽고, 해당 scope의 별도 결과 폴더에 저장한다. 관계없는 실험의 손상된 endpoint나 replicate가
선택한 결과를 막지 않는다. `all results`는 MATH 오류가 있어도 MBPP 조회를 계속하며 최종
실패 코드를 유지한다. 보고서 조회는 학습용 Python 경로가 없어도 CPU Python으로 실행한다.
실제 학습의 interpreter 검사와 실험 identity 검사는 유지한다.

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
