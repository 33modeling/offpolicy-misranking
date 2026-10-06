# 추가 실험 목록

2026-10-06 기준. 코드 `master`. [실행·status·results](REBUTTAL_COMMANDS_KO.md) · [기존 결과 기록](EXPERIMENT_RESULTS_LEDGER_KO.md).

## 우선순위

기존 P1 결과를 먼저 수집하고 미완료분만 마무리한다. **신규 GPU는 원인 진단부터 MATH, 이후 MBPP 순서**로 배정한다.
작업 수는 두 데이터셋의 seeds 5-9를 합친 전체 설계 수다. 현재 남은 작업 수는 원격 status로 확인해야 한다.

| 순서 | 실험 | 목적·조건 | 작업 수 | 상태 |
| --- | --- | --- | ---: | --- |
| 먼저 수집 | P1 / E03·E04 | SR·Switch 독립 반복 k=1,2; fixed200 | 40 + 10 | 기존 결과 수집·미완료 확인 |
| 신규 1 | E14-E16 원인 진단 | t0/100/400 동일 상태에서 방향 정보·SR 정보·실제 학습 이득 비교 | 10 | 구현 완료, 신규 GPU 결과 미확보 |
| 2 | support | `direction_removed`, `sr_hold`, `sr_refresh_matched` | 30 | 구현 완료, 유효한 기존 결과 재사용 |
| 3 | E11 고정 전환 | fixed50/100/150/200/250 vs Switch | 50 | 구현 완료, 기존 fixed200 10개 포함 |
| 3의 후속 분석 | E13 | 균등 시점 기대값·LOSO 고정 일정 | 추가 GPU 0 | 전체 grid 결과 확보 후 분석 |
| 4 | E08 모델 일반화 | Qwen3.5-9B의 Random/SR/On-policy/Switch | 40 | 구현 완료, 실제 진행은 status 확인 |
| 신규 배정 보류 | E12 규칙 대조 | `switch_single`, `switch_consecutive` | 20 | 구현 유지, 기존 실행은 중단하지 않음 |

E14-E16은 **하나의 통합 실험**에서 얻는 세 분석이다. 표의 작업 수는 queue 사이에 중복이 있어 합산하지 않는다.
추가 RLOO는 신규 배정하지 않는다. 기존 queue·실행 중 작업·실험 조건은 유지한다.

## 1. 초반·후반 원인 진단

GRPO를 유지하고 seed별 별도 On-policy 경로를 **t0부터** 만든다. t0/100/400에서 모델과
AdamW 상태를 함께 복제해 여섯 branch를 비교한다. 시점마다 후보 40개를 중복 없이 뽑고
4개를 선택해 25 updates 동안 유지한다. 학습 응답은 매 update 문제당 8개씩 새로 생성한다.

| ID | 비교 | 확인할 내용 |
| --- | --- | --- |
| E14 | `on_policy` vs `direction_shuffle`, `random` | 초반 방향 정보의 실제 보상 이득이 후반에 줄어드는가 |
| E15 | `sr` vs `sr_shuffle`, `random`, `sr_fresh`, `on_policy` | 후반 SR 이득이 유효 GRPO 보상 group·현재 난이도와 연결되는가 |
| E16 | 독립 A/B 점수 측정과 branch 학습 전후 평가 | 점수 반복성·현재 gradient와 실제 학습 이득은 어떻게 연결되는가 |

A만 선별에 사용하고 B는 별도 진단으로 계측한다. 기존 SR-GC 40 대 40은 변경하지 않는다.
branch 종료 후 공통 경로의 모델·optimizer를 복구한다. 기존 prefix는 runtime/input 확인용이며,
t0의 초기 가중치로 복구하지 않는다.

seed당 물리적 update는 400 + 3시점 × 6branch × 25 = **850회**, 전체 평가 호출은 21회다.
기존 100-update 실험의 재현이 아니라 동일 상태의 국소 개입이다. SR도 배치를 25 updates
유지하므로 기존 매-update SR과 구분한다. On-policy 공통 경로에서의 효과를 분석한다.

[상세 설계](STAGE_MECHANISM_EXPERIMENTS_2026-10-06.md) ·
[실행](../scripts/run_srgc_mechanism.sh) ·
[학습·계측](../scripts/srgc_stage_mechanism.py) ·
[결과 분석](../scripts/srgc_stage_report.py)

## 2. 긴 학습 경로의 support 대조

기존 step-25 prefix에서 total step 275까지 이어간다.
후보 40개, 선택 4개, 문제당 8응답, 선택 배치 유지 간격 25 updates를 맞춘다.

| 조건 | 비교 대상 | 분리할 효과 |
| --- | --- | --- |
| `direction_removed` | `on_policy` | 같은 gradient 계산을 수행하되 무작위 순위로 선택 |
| `sr_hold` | 기존 `sr`, `on_policy` | SR 배치 유지 간격; 같은 유지 간격에서 SR과 gradient 점수 |
| `sr_refresh_matched` | `sr_hold` | 후보·유지 간격·동점 순서를 맞춘 현재 보상 갱신 |

`direction_removed`는 gradient 비용도 지출하므로 저비용 Random이 아니다.
과거 `sr_refresh`는 동점 순서가 달랐으며, 새 `sr_refresh_matched`와 결과를 섞지 않는다.

[실행](../scripts/run_srgc_support.sh) ·
[방향 대조](../scripts/srgc_direction_ablation.py) ·
[SR 대조](../scripts/srgc_sr_matched.py) ·
[결과 분석](../scripts/srgc_support_report.py)

## 3. 전환 시점과 일반화

**E11:** fixed50/100/150/200/250와 기존 Switch를 비교한다. fixedN은 update N까지 On-policy,
N+1부터 SR이다. 기존 fixed200 결과는 검증 후 재사용한다. 전체 validation queue는
고정 시점 50개와 보류 중인 규칙 20개이며, 이전 계획에 없는 고유 조건은 60개다.

**E13:** 다섯 고정 시점의 균등 선택 기대값과, 나머지 네 seed로 시점을 고르는 LOSO 분석이다.
전체 grid가 검증된 경우에만 계산한다. 추가 학습이나 독립 재현 실험으로 세지 않는다.

**E08:** Qwen3.5-9B의 네 arm을 비교한다. 자체 cache/prefix가 필요하며 OLMo 상태를 가져오지 않는다.
post-trained 모델이며 Base 모델로 표기하지 않는다. 기존 설치 환경은 임의로 바꾸지 않는다.

[전환 상세](SWITCH_ADDITIONAL_EXPERIMENTS_2026-10-06.md) ·
[고정 전환 실행](../scripts/run_srgc_switch_validation.sh) ·
[Qwen 실행 안내](QWEN35_SRGC_KO.md)

## 완료 및 후순위 기록

| ID | 실험 | 작업 수 | 처리 |
| --- | --- | ---: | --- |
| E01·E02 / P0 | MATH·MBPP의 기본 네 arm | 40 | **완료**, 2026-10-02 저자 확인; 재실행 제외 |
| E05 | 과거 후보 SR 갱신 `candidates` | 10 | 기존 구현·결과 보존 |
| E06 | 재전환 `switch_repeat` | 10 | 기존 구현·결과 보존 |
| E07 | 전체 pool SR 갱신 | 10 | 기존 구현·결과 보존 |
| E09 | 방향 제거·크기·교체 대조 | 30 | 제거 대조는 support에서 재사용, 나머지 후순위 |
| E10 | `sr_hold` | 10 | support에서 재사용 |
| E12 | 단일 음수·연속 음수 규칙 | 20 | 신규 배정 보류 |

기존 OLMo 120개 queue는 그대로다. 구현 완료와 GPU 실험 완료는 구분한다.
P0 완료 근거와 기존 수치는 [결과 기록](EXPERIMENT_RESULTS_LEDGER_KO.md)에 있다.

## 결과와 비용 기준

- 같은 seed·runtime·attention끼리 비교하고 원값, paired 차이, 평균·표본 SD, 유효 쌍 수를 보고한다.
- 실제 독립 평가셋 보상을 주 결과로 사용한다. 점수 상관이나 학습 로그를 평가 보상으로 대체하지 않는다.
- SR cache 생성 1회, cache 읽기·정렬, 선별, 학습, 평가, startup/checkpoint 비용을 구분한다.
- A/B 진단의 B 비용을 운영 선별 비용에 포함하지 않는다. inclusive 시간과 그 내부 phase를 중복 합산하지 않는다.
- 누락·미계측은 `null`로 남긴다. 미종료 timer가 있으면 완전한 총비용으로 표시하지 않는다.

## 과거 기록

[정리 전 실험 기록](https://github.com/33modeling/offpolicy-misranking/blob/340bf7e7c7cad3f8a0a9bc229489ff9d4aa430a5/docs/LIMITATION_EXPERIMENTS_KO.md)에
날짜별 계획·검증 기록을 보존한다. 현재 명령과 노드 배정은 [실행 안내](REBUTTAL_COMMANDS_KO.md)를 따른다.
