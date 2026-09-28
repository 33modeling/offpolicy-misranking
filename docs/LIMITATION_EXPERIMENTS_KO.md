# Limitation 후속 실험: 구현 목록과 실행 기록

점검일: 2026-09-28. 코드 기준: `master`, 점검 시작 commit `67c9a7c`.
대응 원고: V7 `sections/discussion.tex`의 Discussion and Limitations.
실행 코드는 이 저장소에만 유지한다. 논문 쪽 목록은 `v7/EXPERIMENTS.md`다.

**상태:** 저자는 아직 실행하지 않은 실험이 많다고 알렸다. 아래는 구현과 실행
명령을 확인한 목록이지 GPU 실험 완료 목록이 아니다. 이번 작업에서 새 7B 학습은
시작하지 않았다. 로컬 `nvidia-smi`는 드라이버 통신 실패를 반환했고, 실제 사용할
H100 노드의 접속 정보와 빈 allocation은 확인되지 않았다. 원격 실행을 완료 또는
진행 중으로 기록하지 않는다. 기존 작업을 중단하거나 기존 결과를 덮어쓰지 않았다.

## 우선순위

2026-09-28 지정. 기준은 **핵심 결과의 재현성, SR-GC 시점 선택의 추가 가치,
비용 비교의 신뢰성**이다. 코드가 이미 있다는 이유만으로 더 중요한 미구현 대조보다
앞세우지 않는다. 아래 순서는 신규 자원 배정과 구현의 우선순위이며, 진행 중인
작업의 중단이나 낮은 순위 실험의 취소를 뜻하지 않는다.

| 순서 | 우선순위 | 실험/작업 | 먼저 하는 이유 | 준비 상태 |
| --- | --- | --- | --- | --- |
| 1 | P0 | 추가 seeds 5-9의 MATH/MBPP 네 arm 완성 및 전체 비용 수집 | 관측 이득의 재현성과 실제 계산 비용을 함께 검증하는 기본 증거 | 구현됨; 기존 진행 유지, 누락 결과 확인 |
| 2 | P1 | 동일 prefix에서 SR/Switch의 독립 학습 반복 | 같은 조건의 실행 변동과 Switch 이득을 직접 구분 | 전용 replicate runner 미구현, 구현 우선 |
| 3 | P1 | 사전 고정 total-step-200 전환 대조 | 전환 자체의 효과와 SR-GC timing rule의 추가 가치를 구분 | 전용 runner 미구현, 2번 다음 구현/실행 |
| 4 | P2 | 후보 40개의 SR 성공률 갱신 `sr_refresh` | 오래된 캐시를 유지하는 전략과 갱신 전략의 성능·비용 비교 | 구현됨; 순수 갱신 효과 주장에는 배치 유지 간격 통제 추가 필요 |
| 5 | P2 | 반복 전환 `switch_repeat` | 한 번만 전환하고 점검을 끝내는 선택의 성능·비용 trade-off 확인 | 구현됨; 전환 후 scoring 비용 포함 |
| 6 | P3 | 전체 400개 갱신 `sr_refresh-pool` | 후보 범위를 넓힌 갱신의 추가 이득과 비용 확인 | 구현됨; 4번 다음 확장 |
| 7 | P3 | 다른 backbone에서 동일 온라인 Switch | 모델 의존성과 일반화 검증 | 기존 selection 매트릭스만 있음; 온라인 adapter/plan 검증 필요 |
| 8 | P3 | 초기 gradient 방향의 matched ablation | 초기 이점에 대한 인과적 설명 보강 | 전용 대조 미구현; 현재 핵심 결과 검증 이후 |

### 자원 배정과 완료 기준

- **기존 실행은 유지:** MATH/MBPP의 정상 작업을 끄거나 처음부터 다시 시작하지 않는다. 새로 배정할 자원이 경쟁하면 주 결과인 MATH의 누락 paired 결과를 먼저 완성하고 MBPP를 완성한다. 이는 MBPP의 기존 진행 중단이나 계획 seed 제외를 뜻하지 않는다.
- **비용은 1번부터 동시 수집:** 모든 실험에서 cache 생성/재사용, prefix, selection, training, 평가·저장 비용과 불완전 계측을 함께 기록한다. 비용만 뒤로 미루거나 unknown을 0으로 채우지 않는다.
- **구현은 GPU 작업과 병행:** P0가 도는 동안 2번, 3번의 runner와 회귀 테스트를 우선 준비한다. 새로 할당 가능한 GPU는 준비된 상위 순위 작업에 먼저 배정한다. P1 구현 전 남는 별도 노드는 P2에 쓸 수 있지만 상위 작업을 밀어내지는 않는다.
- **이미 구현된 것만의 실행 순서:** 기본 네 arm 및 비용, `sr_refresh`, `switch_repeat`, `sr_refresh-pool` 순서다. 기존 목록의 30개 추가 continuation 전부를 P1 대조보다 먼저 완료해야 하는 것은 아니다.
- **seed·비교 조건은 결과와 무관하게 고정:** 계획된 seeds 5-9를 유지하고 모든 결과를 수집한다. 좋은 seed만 골라 다음 실험을 하거나 유리한 결과가 나온 시점에 반복을 종료하지 않는다. P1의 반복 수와 sampling stream은 실행 전에 고정한다.
- **P0 완료:** 예정된 네 arm/seed의 같은 total step 결과, paired 차이, 자기 경로의 전환 이력과 비용 receipt를 검증한다. 일부 arm만 끝난 평균을 최종 결과로 쓰지 않는다.
- **P1 완료:** 2번은 동일 prefix·캐시·평가 조건의 SR/Switch 반복을 짝지어 보고하고, 3번은 같은 조건의 고정 전환과 Switch를 직접 비교한다. 결과가 무차이 또는 불리해도 함께 보고하며, 두 실험의 완료를 효과 입증과 동일시하지 않는다.

6번은 refresh 한 번당 후보 응답 수가 320개에서 3,200개로 늘어난다. 이는
전체 wall-time이 정확히 10배라는 추정이 아니다. 1-5번이 답하는 핵심 질문을
먼저 다루고, 실측 자원 상황에 맞춰 나머지를 순서대로 실행한다.

## 1. Limitation과 실험 대응

| 항목 | 이미 있는 구현 | 이번 실행 대상 / 남은 일 | 상태 |
| --- | --- | --- | --- |
| 학습 seed 간 변동 | MATH/MBPP seeds 5-9, Random/SR/On-policy/Switch | 이미 돌고 있는 작업은 유지하고 미완료 arm만 이어서 실행 | 구현됨, 원격 완료 확인 필요 |
| 다른 task로의 확장 | 동일 네 arm의 MBPP plan과 입력 | MATH와 분리된 MBPP 결과 수집 | 구현됨, 원격 완료 확인 필요 |
| 자신의 경로에서 주기적 판단 | 현재 `Engine`, 25-update refresh, single-reference D | 각 Switch가 자기 checkpoint에서 판단한 기록 수집 | 구현됨, 과거 A/B replay와 별도 결과 |
| 공통 refresh 절차의 end-to-end 비용 | phase/stage ledger, cache, prefix, continuation, 평가·저장 비용 | 완료/미완료 계측을 구분한 cost export | 구현됨, 실측 결과 미수신 |
| SR cache refresh | `sr_refresh`, `sr_refresh-pool` | 두 scope 모두 MATH/MBPP seeds 5-9에서 기존 SR와 비교 | 구현됨, 이번 점검에서 신규 GPU 실행 안 함 |
| 반복 전환 | `switch_repeat` | 동일 seed의 일회 전환 `switch`와 비교 | 구현됨, 이번 점검에서 신규 GPU 실행 안 함 |
| 고정 schedule 대비 SR-GC timing 가치 | 현재 네 arm 및 두 변형은 이를 직접 검증하지 않음 | 사전 고정 total-step-200 대조, 동일 prefix/평가/길이 | 전용 runner 미구현 |
| 같은 시작 상태의 독립 training replicate | 추가 base seed 실험과 다른 질문 | prefix/캐시를 고정하고 분기 이후 sampling stream만 바꾸는 paired SR/Switch 반복 | 전용 replicate ID/runner 미구현 |
| 초기 gradient 방향의 인과적 효과 | retrospective 진단 및 objective 비교는 있음 | 방향 정보만 제거/대체하고 나머지를 맞추는 ablation | 전용 matched ablation 미구현 |
| 다른 backbone에서도 온라인 Switch가 유효한가 | Qwen/도메인 확장의 기존 selection 매트릭스는 있음 | 같은 온라인 Switch 절차의 모델별 adapter/plan/검증 | 기존 매트릭스를 이 실험으로 대체할 수 없음 |

모든 행을 한꺼번에 실행할 수 있는 단일 runner가 있는 것은 아니다. 특히
`switch_repeat`는 SR과 On-policy 사이를 여러 번 **전환**하는 방법이며, 같은
실험을 다른 난수로 **반복 재현**하는 runner가 아니다. CPU 테스트 성공도
논문 limitation이 실증적으로 해결됐다는 뜻이 아니다.

## 2. 구현된 온라인 실험의 정확한 차이

공통 plan은 OLMo-3-1025-7B, GRPO, LoRA rank 16, seed별 25-update 공유 prefix,
total 275 updates, 4-GPU 실행이다. 각 update의 학습은 4문제 x 새 응답 8개다.
후보 pool 400개와 validation/evaluation 분할은 plan/input hash로 고정한다.

| Arm | 후보/선별 | 전환 이후 | 주 비교 |
| --- | --- | --- | --- |
| `random` | 매 update마다 중복 없는 무작위 40개에서 무작위 4개 | 해당 없음 | 유지 대조군 |
| `sr` | 매 update마다 무작위 40개에서 캐시 SR 점수 상위 4개 | 해당 없음 | 캐시를 갱신하지 않는 유지 대조군 |
| `on_policy` | 25 updates마다 무작위 40개를 현재 gradient로 선별, 상위 4개를 다음 refresh까지 유지 | 계속 gradient scoring | 지속 선별 대조군 |
| `switch` | On-policy와 같이 시작, 40-vs-40 D의 기존 temporal rule | SR로 한 번 전환한 뒤 gradient scoring/check 중단 | 추가 seed의 핵심 비교 |
| `sr_refresh` | 25 updates마다 무작위 40개 x 8응답으로 현재 성공률을 다시 구하고 SR 상위 4개 유지 | 계속 성공률 갱신 | 기존 `sr`, `on_policy`와 비교 |
| `sr_refresh-pool` | 25 updates마다 전체 400개 x 8응답으로 성공률 갱신, 상위 4개 유지 | 계속 전체 pool 갱신 | `sr_refresh`와 범위/비용 비교 |
| `switch_repeat` | 기존 Switch와 같은 전환 규칙으로 시작 | SR 상태에서도 check를 계속하고 양의 D 확인 시 On-policy로 복귀 가능 | 일회 전환 `switch`와 비교 |

샘플 중복 금지는 **한 번의 40개 추출 내부**에 적용된다. 이전 update/refresh에서
쓴 문제가 이후에 다시 나올 수 있다. SR-GC의 SR 비교 40개는 별도의 미사용 문제
우선 preview이며 실제 SR 학습용 무작위 40개와 구분한다.

`sr_refresh`와 기존 `sr`는 캐시 갱신 여부뿐 아니라 선택 배치의 유지 간격도
다르다. 따라서 이 비교를 "갱신 여부만 바꾼 순수 ablation"으로 쓰면 안 된다.
그 효과만 분리하려면 동일한 배치 유지 간격의 cached-SR 대조가 추가로 필요하다.

현재 On-policy/Switch scoring은 On 후보 40개와 SR 비교 40개의 합집합 및 단일
validation 50개를 계산한다. 반면 `sr_refresh`는 후보 성공률만 얻으며 scoring
backward/validation 생성은 하지 않는다. 후보 320응답이라는 이유로 총 scoring
예산이 On-policy와 정확히 같다고 쓰지 않는다. 반복 전환은 SR 상태에서도 이
scoring 비용이 발생하므로 기존 일회 전환의 미미한 산술 비용과 혼동하지 않는다.

## 3. 실행 순서와 명령

아래 명령은 **코드 레포 루트, 해당 실험에 할당된 빈 4 x H100 노드**에서 실행한다.
기존 Pair/MBPP Python 환경을 그대로 쓰며 패키지를 재설치하지 않는다.
노드마다 하나의 GPU worker만 실행한다. 기본 `run_srgc.sh ... run`에는 시작 시
동일 사용자 GPU 프로세스를 정리하는 코드가 있으므로, 다른 학습이 도는 노드에서
추가 실험을 시작하는 용도로 호출하지 않는다.

### A. 기존 네 arm 및 prefix

기존 실행 노드에서는 상태만 확인한다. 아직 worker가 없는 빈 노드에서만 run을
추가한다. dataset별 명령을 구분하면 해당 활성 cohort의 경로를 확인하기 쉽다.

```sh
sh scripts/run_srgc.sh math status
sh scripts/run_srgc.sh mbpp status

# 각각 빈 노드에서 실행. 이미 실행 중인 worker를 교체하는 명령이 아니다.
sh scripts/run_srgc.sh math run
sh scripts/run_srgc.sh mbpp run
```

주 queue는 seed별 prefix 완료 후 네 continuation을 노드에 배정한다. 동일 명령은
기존 활성 cohort와 checkpoint를 사용하며 새 replicate를 만들지 않는다.
Pair seed-3/4 캐시 재사용 plan과 별도 준비 입력 plan을 섞지 않는다. Pair 입력이
사용되면 seeds 5/7/9는 source seed 3, seeds 6/8은 source seed 4의 데이터/캐시를
재사용하므로 다섯 독립 데이터 분할이라고 부르지 않는다.

### B. Limitation의 추가 세 arm

해당 seed의 **검증된 prefix가 완료된 다음** 실행한다. 같은 seed의 기본 네 arm이
모두 끝날 때까지 기다릴 필요는 없지만, 별도의 빈 노드를 사용해야 한다.
아래 한 줄은 한 seed의 한 arm만 실행한다. `5`를 `6`, `7`, `8`, `9`로 바꾸어
각 seed를 실행한다. 같은 노드에서는 앞 명령이 끝난 다음 다음 명령을 실행한다.

```sh
sh scripts/run_srgc_sr_refresh.sh math 5 candidates
sh scripts/run_srgc_sr_refresh.sh math 5 switch_repeat
sh scripts/run_srgc_sr_refresh.sh math 5 pool

sh scripts/run_srgc_sr_refresh.sh mbpp 5 candidates
sh scripts/run_srgc_sr_refresh.sh mbpp 5 switch_repeat
sh scripts/run_srgc_sr_refresh.sh mbpp 5 pool
```

추가 실행 목록은 2 datasets x 5 seeds x 3 arms = **30 continuations**다.
기본 네 arm이나 prefix를 여기에 다시 더해 "30회 새 seed"로 세지 않는다.
각 continuation은 prefix 이후 250 updates이며, 각 dataset의 prefix/기본 arm
진행 상태에 따라 실행 가능한 작업 수가 달라진다. 총 GPU 시간은 실측 없이
110/40 GPU-h 등으로 확정하지 않는다.

추가 세 arm은 현재 기본 네-arm 자동 queue에 등록되어 있지 않다. 따라서
`run_srgc.sh all run` 하나로 위 30개가 실행되지는 않는다. 추가 arm별 lease는
중복 실행을 막지만, 여러 노드가 자동으로 다른 seed/arm을 골라주는 기능은 아니다.
노드별로 다른 `(dataset, seed, arm)`을 배정한다.

### C. 재시작·상태·결과

추가 arm은 동일 명령을 다시 실행하면 같은 arm의 `latest.pt`에서 이어서 진행하고,
이미 완료된 동일 실험은 재학습하지 않는다. 모델·optimizer·선별 상태를 복구한다.
**추가 runner의 저장 간격은 25 updates**다. 기본 queue의 매-update 저장 wrapper가
여기에 자동 적용되는 것은 아니다. 중단 시 마지막 checkpoint 이후의 작업은
다시 수행될 수 있고, 기존 비용 receipt는 남는다.

```sh
# 기본 네 arm의 결과와 비용
sh scripts/run_srgc.sh math results
sh scripts/run_srgc.sh math costs
sh scripts/run_srgc.sh mbpp results
sh scripts/run_srgc.sh mbpp costs

# 추가 세 arm 및 기본 대조군의 결과, selection 비용, 전환 이력
sh scripts/run_srgc_sr_refresh.sh math results
sh scripts/run_srgc_sr_refresh.sh mbpp results
```

추가 runner에는 별도 `status` 하위 명령이 없다. 기본 queue의 status/results가
추가 arm을 자동 집계한다고 안내하지 않는다. 학습 콘솔의 `TRAIN ... step=N/275`,
활성 root 아래 `seed-N/<arm>-run.json`, `<arm>-progress.json`을 확인한다.
최종 산출물은 `<arm>-endpoint.json`이며, 상세 비용은 같은 seed 아래
`cost-receipts/<arm>/`, `invocations/<arm>/`에 남는다. 위 추가 results는
selection 비용 요약이지 전체 cold-start 비용 합계 보고서가 아니다.

## 4. 저장·비용·완료 기준

- 기본 저장 위치는 `${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking/srgc-rebuttal`이며, 실제 cohort는 활성 plan pointer를 따른다. 사용자 볼륨으로 fallback하지 않는다.
- 실행 중인 cohort의 frozen core/input/plan hash를 고치거나 mismatch 검사를 풀지 않는다. 다른 hash면 원 실행 버전과 source를 확인한다.
- 초기 cache 생성 또는 기존 cache의 재사용 출처, prefix, continuation, model/setup, checkpoint 저장·복구, evaluation을 분리한다. 새 실행에서 cache를 재사용했다고 과거 생성 비용이 0이 되는 것은 아니다.
- D 산술은 selection 비용 안의 부분 항목이다. 합계에 두 번 더하지 않는다. shared prefix/cache는 arm별 표시와 실험 전체 합계에서 중복 회계하지 않는다.
- 최종 reward/total step, plan/input/core hash, 공통 prefix hash, per-question reward와 비용 계측 완결 여부가 필요하다. 누락 cost를 0으로 채우지 않는다.
- `results` 파일이 생겼다는 사실만으로 paired 비교 완료를 선언하지 않는다. 실제로 필요한 모든 arm과 seed가 있는지 확인하고 불리한 결과도 유지한다.
- 추가 변형의 script 파일 자체는 frozen core hash 대상이 아니다. 실행 코드 commit과 launcher/script hash도 작업 기록에 남기고, 실행 도중 다른 변형 코드로 교체하지 않는다.

## 5. 이번 코드 점검과 수정

1. 추가 launcher가 `additional_seeds.json`/`mbpp_seeds.json`을 고정 선택하던 문제를 수정했다. 기본 launcher와 같은 `default_plan()`을 사용하여 활성 Pair 재사용 cohort를 따른다.
2. shell의 plan 조회 subprocess에서 설정한 환경은 torchrun에 전달되지 않는다. 실제 GPU rank에서 group-volume runtime cache 환경을 설정하도록 했다. prefix가 없거나 변경되었으면 모델 로드 전에 중단한다.
3. 사용자가 명시한 Python 경로가 없을 때 system Python으로 조용히 바꾸지 않고 실패하도록 했다.
4. 추가 results에서 실험 identity/공통 prefix/최종 update 수를 확인하고, 누락 비용은 `unknown`, 불완전 계측은 별도 표시한다.
5. `switch_repeat`를 서로 다른 module 이름으로 import하여 전체 테스트에서 class identity가 달라지던 문제를 canonical import로 수정했다.

`srgc_rebuttal/*.py`의 frozen 학습 패키지와 실험 JSON은 바꾸지 않았다.
점검 전 core SHA-256은
`1869fe1cf898d4ff3a6d5e9054790836442b5e0b81b485fb04bc27de4ebab20a`다.
수정 후에도 같은 hash임을 확인했다.
그룹 경로 및 변형 engine의 새 회귀 테스트는
[`test_extra_arm_launch.py`](../srgc_rebuttal/tests/test_extra_arm_launch.py),
[`test_sr_refresh.py`](../srgc_rebuttal/tests/test_sr_refresh.py),
[`test_switch_repeat.py`](../srgc_rebuttal/tests/test_switch_repeat.py)에 있다.

전체 테스트의 첫 실행에서는 로컬 `torch` 부재로 cache 복구 및 매-step checkpoint
테스트를 실행하지 못했다. GPU 환경을 바꾸거나 테스트를 통과했다고 처리하지 않는다.
검증 결과는 아래 실행 기록에 남긴다. H100 admission 및 7B 실행 검증은 별도로 필요하다.

## 6. 다른 작성된 실험 코드

전체 과거 실험은 [전체 실험 명세](EXPERIMENTS_COMPLETE_GUIDE_KO.md)에 이미 있다.
아래 코드도 존재하지만 위 온라인 Limitation 실험과 같은 결과로 합치지 않는다.

| 기존 코드 | 용도 | 구분할 점 |
| --- | --- | --- |
| `run_srgc_newseeds.sh` | 과거 seed 3/4의 d0/100/400에서 사전 고정 D와 후속 학습 | 현재 seeds 5-9 온라인 queue가 아님 |
| `run_selector_pair_srgc_repeat.sh` | 보존 checkpoint의 반복 진단 | 독립 training replicate나 `switch_repeat`가 아님 |
| `run_selector_pair.sh` | 42개 Pair continuation과 비용/목표 비교 | 과거 frozen 실행의 후속 처리와 새 periodic-refresh 실행 구분 |
| `run_rloo.sh` | 학습 objective를 바꾼 대조 | direction의 인과적 ablation이 아님 |
| `run_additional_experiments.sh` | Qwen 크기별/다른 domain 매트릭스 | 같은 온라인 Switch를 새 backbone에서 검증한 결과가 아님 |
| `run_reference_axes.sh`, `run_reliability_budget.sh` | reference 예산과 진단 신뢰도 | 진단 횟수를 실제 온라인 비용에 무조건 가산하지 않음 |
| `run_mopps_comparison.sh`, `run_low_order.sh`, `run_method_choice.sh` | 다른 선택·재사용 방법의 후속 비교 | 각 frozen protocol의 조건과 완료 기준 필요 |

이 목록은 모든 역사적 실험을 지금 재시작하라는 의미가 아니다. 현재 요청의
Limitation 대응과 코드/실험의 이름이 비슷하다는 이유로 서로 대체하지 않는다.

## 7. 실행 기록

| 날짜 | 항목 | 실제 수행 | 결과/다음 조건 |
| --- | --- | --- | --- |
| 2026-09-28 | 구현 inventory | Limitation, 현재 plan/engine/launcher/test 확인 | 구현됨/미구현 분리, 신규 과학적 결과 없음 |
| 2026-09-28 | 변형 engine + launcher 회귀 | 로컬 CPU 테스트 | 19개 통과, 실제 GPU 성능 검증 아님 |
| 2026-09-28 | 선별 규칙·캐시 재사용·비용·storage·shell 회귀 | 관련 9개 test module 실행 | 88개 통과, core hash 불변 확인 |
| 2026-09-28 | 전체 suite 첫 점검 | 210 tests 수집·실행 시도 | torch 부재 2 errors, import identity 1 failure 발견 후 수정; 16 skipped |
| 2026-09-28 | 전체 CPU 범위 재점검 | torch 필수인 두 항목을 명시적으로 제외하고 나머지 208 tests 실행 | 192개 통과, 선택 의존성 관련 16개 skipped; 나머지 오류 없음 |
| 2026-09-28 | H100 실행 | 실행하지 않음 | 로컬 드라이버 사용 불가, 실제 대상 노드/빈 allocation 확인 필요 |
| 2026-09-28 | 우선순위 지정 | P0-P3와 신규 자원/구현 순서 기록 | 문서 변경만 수행; 작업 실행·중단·계획 변경 없음 |

두 제외 항목은 `test_build_cache.CacheTests.test_resume_after_last_receipt_exports_without_loading_model_or_regenerating`와
torch를 import하는 `test_step_checkpoints` 모듈이다. 이 범위를 통과했다고
기록하지 않는다. H100의 기존 환경에서는 `python -m unittest discover -s srgc_rebuttal/tests`
전체를 다시 실행해 확인한다. shell 문법, CLI help, 문서의 로컬 링크/스크립트 경로도 확인했다.

새 GPU 작업을 실행할 때 이 표에 `(dataset, seed, arm)`, node/allocation,
시작 시각, code commit, active plan 경로와 hash, 로그 경로, prefix hash를 남긴다.
완료 시 endpoint/cost receipt 경로, 총 updates, reward, 비용 완결 여부와 실패/재시도
내역을 붙인다. 노드에서 학습 프로세스와 로그가 확인되기 전에는 `실행 중`으로 쓰지 않는다.
