# 추가 실험: 우선순위·목적·실행

**실험마다 필요한 이유, 비교 조건, 명령을 함께 정리했다.**
코드는 `master`에 있으며, 아래 명령은 코드 저장소 루트에서 실행한다.

## 우선순위

| 순서 | 실험 | 필요한 이유 | 전체 작업 수 |
| --- | --- | --- | ---: |
| 먼저 확인 | [P1 독립 반복·fixed200](#1-기존-p1-독립-반복과-고정-전환) | 이미 돌린 결과를 확보하고 Switch 이득의 재현성 확인 | 40 + 10 |
| 신규 1 | [E14-E16 초반·후반 원인 진단](#2-신규-1순위-초반-on-policy후반-sr-원인-진단) | 왜 초반에는 On-policy, 후반에는 SR이 유리한지 직접 분리 | 10 |
| 2 | [support 대조](#3-2순위-선별-기준배치-유지보상-갱신) | 선별 기준과 배치 유지·보상 갱신의 효과를 구분 | 30 |
| 3 | [E11·E13 전환 시점](#4-3순위-고정-전환-시점과-sr-gc-비교) | 단순히 바꾸기만 해도 되는지, SR-GC 시점 선택이 기여하는지 확인 | 50 |
| 4 | [E08 다른 모델](#5-4순위-다른-모델에서-재현) | 관측이 OLMo에만 해당하는지 확인 | 40 |
| 후순위 | [기존 SR 갱신·재전환·방향 대조](#6-후순위-기존-추가-대조) | 캐시의 오래됨, 재전환, gradient 구성요소를 추가 분리 | 조건별 아래 표 |
| 신규 배정 보류 | [E12 규칙 대조](#7-신규-배정-보류-전환-확인-규칙) | 단일 음수와 반복 확인 규칙의 차이 확인 | 20 |
| 완료 | [P0 MATH·MBPP](#8-완료한-p0) | 기본 네 방법의 추가 seed 재현 | 40 |

**기존 P1 수집 후, 신규 GPU는 원인 진단의 MATH부터 배정하고 MBPP를 이어간다.**
추가 RLOO는 배정하지 않는다. 실행 중인 작업과 기존 queue는 변경하지 않는다.
표의 수는 MATH·MBPP seeds 5-9의 설계상 작업 수다. 중복 조건이 있으므로 합산하지 않는다.
현재 남은 수는 status로 확인한다. 구현 완료를 실제 GPU 실험 완료로 표시하지 않는다.

## 공통 실행 방법

- 빈 **4-H100 노드마다 선택한 실행 명령 하나**를 실행한다. seed·조건은 자동 배정된다.
- 아래 실행 예시는 MATH다. `math`를 `mbpp`로 바꾸면 MBPP, `all`로 바꾸면 양쪽을 배정한다.
- 단, 기존 P1/추가 arm의 `results`는 `all`을 받지 않는다. MATH·MBPP를 각각 조회한다.
- 같은 명령을 다시 실행하면 중단 지점에서 자동 재개한다. 실행 중인 checkout에 pull하지 않는다.
- 환경·캐시·저장은 기존 그룹 볼륨을 유지한다. 잠금·해시 검증을 임의로 해제하지 않는다.

## 1. 기존 P1: 독립 반복과 고정 전환

**왜 필요한가:** Switch의 보상 이득이 학습 응답의 우연한 차이인지 확인하려면 같은 출발점에서
SR과 Switch를 다시 비교해야 한다. fixed200은 SR-GC 없이 정한 시점에 전환해도 이득이 나는지
확인하는 대조다. 이미 사용한 계산을 먼저 회수하기 위해 새 실험보다 결과 수집을 앞에 둔다.

**비교:** E03은 동일 step-25 prefix에서 SR/Switch를 독립 sampling stream k=1,2로 반복한다.
E04는 fixed200과 기존 Switch를 비교한다. 두 데이터셋 합계 40 + 10개다.
단일 fixed200만으로 시점 선택의 기여를 결론내리지 않고 4절의 전체 grid로 보강한다.

**먼저 결과 수집**

```sh
sh scripts/run_srgc_sr_refresh.sh math results
sh scripts/run_srgc_sr_refresh.sh mbpp results
```

통합 JSON은 **`<run root>/results/results.json`**이다. `source_results`에 원본 보상·비용·출처까지
들어 있으며, 개별 사본은 `results/exports/<수집 시각-ID>/raw/seed-N/`에 모인다.
출력의 **COLLECTED JSON / COLLECTED FILES**가 실제 경로다. 별도 옵션이나 리다이렉션은 필요 없다.

**상태 확인**

```sh
sh scripts/run_srgc_sr_refresh.sh all status replicate
sh scripts/run_srgc_sr_refresh.sh all status switch_fixed200
```

**미완료분 실행:** 빈 노드마다 아래 중 한 명령만 선택한다.

```sh
sh scripts/run_srgc_sr_refresh.sh math replicate
sh scripts/run_srgc_sr_refresh.sh math switch_fixed200
```

**볼 결과:** seed별 SR/Switch 보상 차이, 반복별 차이의 일관성, 실제 전환 시점과 비용.
노드 상한은 반복 20개/데이터셋, fixed200 5개/데이터셋이다.
[실행 코드](../scripts/run_srgc_sr_refresh.sh) · [자동 배정](../scripts/srgc_replicate_worker.py).

## 2. 신규 1순위: 초반 On-policy·후반 SR 원인 진단

**왜 필요한가:** 기존 그래프는 학습 단계에 따라 유리한 방법이 달라진다는 결과를 보여준다.
하지만 그 차이가 gradient 방향 정보 때문인지, SR이 학습 신호가 있는 문제를 고르기 때문인지는
분리하지 못한다. 같은 모델·optimizer 상태에서 선별 정보만 바꾸고 실제 보상 변화를 비교한다.

| 분석 | 비교 | 확인할 질문 |
| --- | --- | --- |
| E14 초반 방향 정보 | `on_policy` vs `direction_shuffle`, `random` | 문제와 방향 점수의 연결을 끊으면 초반 이득이 줄어드는가 |
| E15 후반 SR 정보 | `sr` vs `sr_shuffle`, `random`, `sr_fresh`, `on_policy` | SR 이득이 현재 난이도·유효 GRPO 보상 group과 연결되는가 |
| E16 점수와 학습 | 독립 A/B 점수 측정과 branch 학습 전후 평가 | 점수가 반복 가능하다는 사실이 실제 학습 이득으로 이어지는가 |

**조건:** GRPO, 별도 On-policy 경로를 t0부터 생성, t0/100/400에서 모델과 AdamW 상태 복제.
시점마다 후보 40개를 중복 없이 뽑아 4개를 선택하고 25 updates 유지한다.
학습은 매 update 문제당 새 응답 8개다. 여섯 branch 종료 후 원래 공통 경로 상태를 복구한다.
A만 선별에 쓰고 B는 진단이다. 기존 SR-GC 40 대 40은 변경하지 않는다.

**실행·상태·결과**

```sh
sh scripts/run_srgc_mechanism.sh math
sh scripts/run_srgc_mechanism.sh all status
sh scripts/run_srgc_mechanism.sh all results
```

**볼 결과:** 단계별 실제 평가 보상 증가, 방향/SR 점수 shuffle 효과, 혼합 보상 group 비율,
캐시-현재 성공률 차이, A/B 반복성과 학습 이득의 관계. 진단 비용과 운영 비용은 분리한다.
JSON 화면 출력은 `sh scripts/run_srgc_mechanism.sh all json`이다.
원본은 `<run root>/seed-N/stage_mechanism-endpoint.json`, results의 `output=`으로 위치를 확인한다.

**규모:** 5개 seed 작업/데이터셋, 양쪽 최대 10노드. seed당 물리적 update는
공통 경로 400 + 3시점 × 6branch × 25 = 850회, 평가 호출 21회다.
SR도 배치를 25 updates 유지하는 국소 진단이며 기존 매-update SR이나 100-update 실험의 재현과 구분한다.
[상세 설계](STAGE_MECHANISM_EXPERIMENTS_2026-10-06.md) ·
[학습·계측 코드](../scripts/srgc_stage_mechanism.py) · [분석 코드](../scripts/srgc_stage_report.py).

## 3. 2순위: 선별 기준·배치 유지·보상 갱신

**왜 필요한가:** On-policy와 SR의 결과 차이에는 점수뿐 아니라 같은 문제를 얼마나 오래 쓰는지,
SR 보상을 언제 갱신하는지도 영향을 줄 수 있다. 이를 맞추지 않으면 점수의 효과로 잘못 해석할 수 있다.
원인 진단의 짧은 개입과 별도로 긴 학습 경로에서 세 요인을 분리한다.

| 조건 | 비교 대상 | 분리하는 요인 |
| --- | --- | --- |
| `direction_removed` | `on_policy` | gradient 계산은 같고 선택 순위만 무작위 |
| `sr_hold` | 기존 `sr`, `on_policy` | SR 배치 유지 간격; 같은 유지 간격에서 SR과 gradient 선별 |
| `sr_refresh_matched` | `sr_hold` | 후보·유지 간격·동점 순서를 맞춘 현재 보상 갱신 |

**조건:** 기존 step-25 prefix에서 total step 275까지 학습한다.
후보 40개, 선택 4개, 문제당 8응답, 배치 유지 25 updates다.
`direction_removed`도 gradient 비용을 지출한다. 과거 `sr_refresh`와 새 matched 조건은 섞지 않는다.

**실행·상태·결과**

```sh
sh scripts/run_srgc_support.sh math
sh scripts/run_srgc_support.sh all status
sh scripts/run_srgc_support.sh all results
```

**볼 결과:** 같은 seed의 최종 평가 차이와 실측 selection/training 비용.
총 30개, 최대 15노드/데이터셋이다. 기존 유효한 방향 제거·SR 유지 결과는 재사용한다.
[방향 대조 코드](../scripts/srgc_direction_ablation.py) ·
[SR 대조 코드](../scripts/srgc_sr_matched.py) · [분석 코드](../scripts/srgc_support_report.py).

## 4. 3순위: 고정 전환 시점과 SR-GC 비교

**왜 필요한가:** Switch가 계속 On-policy를 쓰는 것보다 좋다는 결과만으로는 SR-GC가 필요한지
알 수 없다. 아무 고정 시점에 SR로 바꿔도 비슷한지, 상태를 보고 고른 시점이 더 유리한지 비교해야 한다.

**비교:** E11은 fixed50/100/150/200/250와 기존 Switch를 비교한다.
fixedN은 update N까지 On-policy, N+1부터 SR이다. step-25 prefix에서 total step 275까지 이어가며,
기존 fixed200 결과는 검증 후 재사용한다. 전체 50개, 최대 25노드/데이터셋이다.

**실행·상태·결과**

```sh
sh scripts/run_srgc_switch_validation.sh math timing
sh scripts/run_srgc_sr_refresh.sh all status timing
sh scripts/run_srgc_switch_validation.sh all results
```

`timing`을 생략하면 보류 중인 규칙 대조까지 실행한다. 위 status는 고정 시점만,
results는 규칙 대조를 포함한 전체 validation 결과를 보고한다.

**볼 결과:** 각 고정 시점 대비 Switch의 seed별 보상·비용·실제 선별 횟수.
E13은 전체 grid로 균등 시점 기대값과 LOSO를 함께 계산한다. LOSO는 나머지 네 seed에서
시점을 정해 제외한 seed에 적용하므로 평가할 seed에 맞춘 사후 최적 시점 선택을 피한다.
추가 GPU 작업은 없으며 전체 grid가 검증되어야 집계한다.
[상세 설계](SWITCH_ADDITIONAL_EXPERIMENTS_2026-10-06.md) ·
[고정 전환 코드](../scripts/srgc_switch_fixed.py) · [분석 코드](../scripts/srgc_switch_validation_report.py).

## 5. 4순위: 다른 모델에서 재현

**왜 필요한가:** 초기 gradient 선별과 이후 전환의 효과가 특정 모델의 학습 상태나
SR 캐시에만 의존하는지 확인한다. 같은 모델에서 seed만 늘리는 실험과 다른 검증이다.

**비교:** E08은 Qwen3.5-9B의 Random/SR/On-policy/Switch다.
OLMo의 가중치·cache/prefix를 가져오지 않고 자체 준비 단계를 사용한다.
post-trained 모델이며 Base로 표기하지 않는다. 기존 설치 환경을 임의로 바꾸지 않는다.

**실행·상태·결과**

```sh
sh scripts/run_srgc_qwen35.sh math
sh scripts/run_srgc_qwen35.sh all status
sh scripts/run_srgc_qwen35.sh all results
```

**볼 결과:** 모델 내 방법별 보상·전환 시점·비용과 OLMo에서의 관측 방향이 일치하는지.
총 40개다. 데이터셋당 준비 단계 최대 5노드, 네 arm 학습 최대 20노드다.
결과 보고서 위치는 `Saved:`에 표시한다.
[조건·환경 안내](QWEN35_SRGC_KO.md) · [실행 코드](../scripts/run_srgc_qwen35.sh).

## 6. 후순위: 기존 추가 대조

현재 핵심 원인 진단 이후에 배정한다. 기존 코드와 결과는 보존하며 실행 중인 작업은 중단하지 않는다.

| ID | 실험이 필요한 이유 | 실행 명령 | 양쪽 작업 수 |
| --- | --- | --- | ---: |
| E05 후보 SR 갱신 | 초기 SR cache가 오래된 것이 성능 차이의 원인인지 확인 | `sh scripts/run_srgc_sr_refresh.sh math candidates` | 10 |
| E06 재전환 | 한 번 전환한 뒤 다시 gradient 선별로 돌아올 필요가 있는지 확인 | `sh scripts/run_srgc_sr_refresh.sh math switch_repeat` | 10 |
| E07 전체 pool 갱신 | 후보 일부만 보상을 갱신하는 범위 제한이 영향을 주는지 확인 | `sh scripts/run_srgc_sr_refresh.sh math pool` | 10 |
| E09 방향 대조 | gradient 방향·크기·reference 교체의 영향을 분리 | `sh scripts/run_srgc_sr_refresh.sh math direction` | 30 |
| E10 SR 유지 | 매-update 재선별과 배치 유지의 차이를 확인 | `sh scripts/run_srgc_sr_refresh.sh math sr_hold` | 10 |

E10과 E09의 방향 제거는 support에서 재사용한다. E05는 동점 처리까지 맞춘 대조가 아니므로
현재 보상 갱신의 주 비교는 `sr_refresh_matched - sr_hold`로 둔다.

**상태·결과:** 아래 `candidates` 자리에 표의 scope를 넣어 해당 실험을 조회한다.

```sh
sh scripts/run_srgc_sr_refresh.sh all status candidates
sh scripts/run_srgc_sr_refresh.sh math results
sh scripts/run_srgc_sr_refresh.sh mbpp results
```

results는 해당 데이터셋의 검증된 추가 arm JSON을 함께 모으며 1절과 같은 위치에 저장한다.
기존 전체 120개 queue를 바꾸지 않는다.

## 7. 신규 배정 보류: 전환 확인 규칙

**왜 필요한가:** SR-GC가 한 번 음수가 되는 것만으로 충분한지, 반복 확인과 현재 규칙의
예외 분기가 실제로 기여하는지 분리한다. 초반·후반 학습 원인 설명보다 후순위다.

**비교:** E12는 `switch_single`, `switch_consecutive`와 기존 Switch다.
각 방법의 자기 학습 경로에서 단일 reference와 기존 40 대 40으로 점검한다.
총 20개다. 아래 실행은 규칙 대조를 배정할 때만 사용한다.

```sh
sh scripts/run_srgc_switch_validation.sh math rules
sh scripts/run_srgc_sr_refresh.sh all status rules
sh scripts/run_srgc_switch_validation.sh all results
```

**볼 결과:** 전환 시점, 최종 보상, 전환 전 점검 비용의 차이.
[규칙 정의](SWITCH_ADDITIONAL_EXPERIMENTS_2026-10-06.md#e12-sr-gc-규칙-대조) ·
[규칙 코드](../scripts/srgc_switch_rules.py).

## 8. 완료한 P0

E01·E02는 MATH·MBPP seeds 5-9 × Random/SR/On-policy/Switch, 총 40개다.
**2026-10-02 저자 완료 확인.** 추가 seed에서 기본 성능·비용을 비교하는 근거이며 재실행하지 않는다.

```sh
sh scripts/run_srgc.sh all status
sh scripts/run_srgc.sh all results
sh scripts/run_srgc.sh all costs
```

[결과와 완료 근거](EXPERIMENT_RESULTS_LEDGER_KO.md)를 유지한다.

## 비용·결과 확인 기준

SR cache 생성은 재사용 가능하더라도 **최초 1회 비용을 따로 기록**한다.
이를 cache 읽기·정렬 시간과 혼동하지 않는다. 선별용 현재 rollout·reward·gradient와
학습 rollout·backward를 구분하고, 전환 후 중단된 선별 비용을 계속 누적하지 않는다.

A/B 진단의 B 비용은 운영 선별 비용에 넣지 않는다. 평가·startup·checkpoint도 분리하며,
inclusive 시간과 그 안의 phase 시간을 중복 합산하지 않는다. 미계측은 `null`이며 0초가 아니다.
미종료 timer가 있으면 완전한 총비용으로 표시하지 않는다.

실제 독립 평가셋 보상, 같은 seed·runtime·attention의 paired 차이, 평균·표본 SD와 유효 쌍 수를 본다.
상관이나 학습 로그만으로 성능 향상을 판정하지 않는다. 새 결과 검증 전에는 원고의 결과를 바꾸지 않는다.

## 관련 기록

- [명령·결과 경로 빠른 찾기](REBUTTAL_COMMANDS_KO.md).
- [정리 전 계획·검증 이력](https://github.com/33modeling/offpolicy-misranking/blob/340bf7e7c7cad3f8a0a9bc229489ff9d4aa430a5/docs/LIMITATION_EXPERIMENTS_KO.md).
