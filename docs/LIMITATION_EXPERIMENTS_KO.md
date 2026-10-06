# V6 Limitations 대응 실험: 우선순위·목적·실행

기준은 제출본 [V6 Discussion and Limitations 원문](https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/3d144ee666e480766e253d06221f6e29feaef25e/v6/overleaf/sections/discussion.tex)이다.
**V6에 적힌 한계별 대응 실험과, 별도로 요청한 메커니즘 실험을 구분한다. 동일-prefix replicate는 보조다.**
코드는 `master`에 있으며 아래 명령은 코드 저장소 루트에서 실행한다.

## 우선순위

| 순서 | V6에 명시된 한계 | 대응 실험 | 구현·진행 구분 |
| --- | --- | --- | --- |
| 완료 확인 | MATH·학습 seed 범위 제한; question bootstrap은 training-seed 변동을 측정하지 않음 | E01·E02: MATH·MBPP seeds 5-9 네 arm | P0 40개 완료, 결과·비용 검증; 재실행하지 않음 |
| 신규 1 | 온라인 전환을 한 backbone에서만 평가 | E08: Qwen3.5-9B의 네 arm | 구현됨, 실제 진행은 status 확인 |
| 2 | 공통 prefix의 중간 전환 시점·temporal confirmation 비교 부족 | E04·E11: 고정 전환 5시점; E12: 단일/연속 음수 규칙 | 구현됨; fixed200 재사용, 규칙 대조도 직접 대응 실험 |
| 3 | SR-cache refresh 미검증 | E05·E07·E10: 기존 cache vs matched 후보 갱신·pool 갱신 | 구현됨; 동점·유지 간격을 맞춘 support 대조 우선 |
| 4 | 학습 중 재전환이 필요할 수 있으나 선별 비용 증가 | E06: 단방향 Switch vs `switch_repeat` | 구현됨; 재전환 횟수·성능·추가 비용 비교 |
| 5 | local signal의 reference 민감성·불확실성; 장기 보상 예측 한계 | E09 방향 정보 대조, 기존 A/B 진단 검토 | 부분 대응만 구현됨; 직접 reference 변경·불확실성 규칙은 아래 미구현 항목 |
| 보조 | 같은 출발점 이후 학습 변동을 더 자세히 확인 | E03: 동일-prefix SR/Switch replicate | 기존 결과 수집, 신규 배정은 직접 대응 실험 이후 |
| 별도 요청·병행 | 초반 On-policy·후반 SR의 원인을 추가 설명 | E14-E16: 메커니즘 개입 | 저자 요청 실험; MATH 5노드, MBPP 5노드까지 독립 배정 |

기존 결과 수집은 GPU 재실행 없이 먼저 할 수 있다. **결과 수집과 replicate 신규 실행은 별개**다.
과거 문서의 P0/P1/P2는 운영용 분류였으며 V6의 실험명이나 중요도 등급이 아니다. 여기서는 V6 항목과 실험 ID로 구분한다.
V6 대응 실험은 위 순서로 정리한다. 별도로 요청한 메커니즘 실험은 병행 가능하며 임의로 중단하거나 후순위로 돌리지 않는다.
추가 RLOO는 배정하지 않는다.
기존 queue의 순서·진행 중인 작업·실험 조건은 바꾸지 않으며, 아래 명령으로 필요한 scope를 선택한다.
작업 수는 두 데이터셋의 seeds 5-9를 합친 설계 수다. 중복 조건을 합산하거나 현재 미완료 수로 읽지 않는다.

## 공통 실행 방법

통합 실행부는 `sh scripts/run_srgc_experiments.sh 데이터셋 실험 [run|status|results]`다.
예: `sh scripts/run_srgc_experiments.sh all mechanism`. 가능한 이름은
`sh scripts/run_srgc_experiments.sh list`로 확인한다. 아래 기존 개별 명령도 계속 사용할 수 있다.

- 빈 **4-H100 노드마다 선택한 실행 명령 하나**를 실행한다. seed·조건은 자동 배정된다.
- 아래 실행 예시는 MATH다. `math`를 `mbpp`로 바꾸면 MBPP, `all`로 바꾸면 양쪽을 배정한다.
- 단, 기존 P1/추가 arm의 `results`는 `all`을 받지 않는다. MATH·MBPP를 각각 조회한다.
- 같은 명령을 다시 실행하면 중단 지점에서 자동 재개한다. 실행 중인 checkout에 pull하지 않는다.
- 환경·캐시·저장은 기존 그룹 볼륨을 유지한다. 잠금·해시 검증을 임의로 해제하지 않는다.

## 1. 다른 모델에서 온라인 전환 재현

**V6 근거:** 온라인 전환을 한 backbone에서만 평가했다.

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

## 2. 공통 prefix의 전환 시점과 확인 규칙

**V6 근거:** 중간 전환 시점과 temporal confirmation의 기여를 분리하지 못했다.

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

`timing`은 고정 시점만 배정한다. 마지막 인자를 생략하면 규칙 대조를 포함한 70개 queue를 실행한다. 위 status는 고정 시점만,
results는 규칙 대조를 포함한 전체 validation 결과를 보고한다.

**볼 결과:** 각 고정 시점 대비 Switch의 seed별 보상·비용·실제 선별 횟수.
E13은 전체 grid로 균등 시점 기대값과 LOSO를 함께 계산한다. LOSO는 나머지 네 seed에서
시점을 정해 제외한 seed에 적용하므로 평가할 seed에 맞춘 사후 최적 시점 선택을 피한다.
추가 GPU 작업은 없으며 전체 grid가 검증되어야 집계한다.
[상세 설계](SWITCH_ADDITIONAL_EXPERIMENTS_2026-10-06.md) ·
[고정 전환 코드](../scripts/srgc_switch_fixed.py) · [분석 코드](../scripts/srgc_switch_validation_report.py).

### 확인 규칙 대조

**왜 필요한가:** SR-GC가 한 번 음수가 되는 것만으로 충분한지, 반복 확인과 현재 규칙의
예외 분기가 실제로 기여하는지 분리한다. V6가 명시한 confirmation 한계에 직접 대응한다.

**비교:** E12는 `switch_single`, `switch_consecutive`와 기존 Switch다.
각 방법의 자기 학습 경로에서 단일 reference와 기존 40 대 40으로 점검한다.
총 20개다. 고정 시점과 합하면 70개이며 기존 fixed200 10개를 포함한다. 점검 간격은 25로 유지하므로 간격 최적화 실험은 아니다.

```sh
sh scripts/run_srgc_switch_validation.sh math rules
sh scripts/run_srgc_sr_refresh.sh all status rules
sh scripts/run_srgc_switch_validation.sh all results
```

**볼 결과:** 전환 시점, 최종 보상, 전환 전 점검 비용의 차이.
[규칙 정의](SWITCH_ADDITIONAL_EXPERIMENTS_2026-10-06.md#e12-sr-gc-규칙-대조) ·
[규칙 코드](../scripts/srgc_switch_rules.py).

## 3. SR 캐시 갱신과 비용

**V6 근거:** SR-cache refresh를 검증하지 않았다.

**왜 필요한가:** 초기 캐시를 재사용한 SR의 결과와, 현재 정책으로 보상을 갱신한 SR의 결과가
달라질 수 있다. 캐시가 오래된 영향과 갱신으로 늘어나는 비용을 함께 측정한다.
주 비교는 `sr_refresh_matched - sr_hold`다. support의 방향 제거·유지 간격 대조는 조건 분리에 함께 사용한다.

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

**갱신 범위 추가 비교:** 후보 일부가 아니라 전체 pool을 갱신할 때의 보상과 비용을 비교한다.

```sh
sh scripts/run_srgc_sr_refresh.sh math pool
sh scripts/run_srgc_sr_refresh.sh all status pool
sh scripts/run_srgc_sr_refresh.sh math results
sh scripts/run_srgc_sr_refresh.sh mbpp results
```

E07 pool 갱신은 양쪽 10개다. 기존 E05 `candidates`도 구현되어 있으나 동점 처리가 달라
matched 비교를 대신하지 않는다. 기존 결과를 수집하고 새 주 비교는 위 support로 둔다.
[갱신 코드](../scripts/srgc_sr_refresh.py).

## 4. 재전환의 성능과 추가 비용

**V6 근거:** 학습이 진행되면 반복 전환이 필요할 수 있지만 선별 비용이 증가한다.

**왜 필요한가:** 한 번 SR로 전환한 뒤 끝내는 규칙이 충분한지, 이후 다시 On-policy로 돌아오는
편이 나은지 확인한다. 재전환은 추가 점검과 gradient 선별을 발생시키므로 보상뿐 아니라 총비용도 비교한다.

**비교:** 기존 Switch와 E06 `switch_repeat`. 같은 step-25 prefix에서 total step 275까지 학습한다.
repeat는 SR 전환 후에도 점검을 계속하며 반대 방향 신호에 따라 On-policy로 돌아올 수 있다.
양쪽 10개, 최대 5노드/데이터셋이다.

```sh
sh scripts/run_srgc_sr_refresh.sh math switch_repeat
sh scripts/run_srgc_sr_refresh.sh all status switch_repeat
sh scripts/run_srgc_sr_refresh.sh math results
sh scripts/run_srgc_sr_refresh.sh mbpp results
```

**볼 결과:** 최종 보상, 각 전환 step·횟수, 추가 점검·gradient 선별·학습·총비용.
재전환으로 실제 이득이 없으면 단방향 전환을 뒷받침하는 결과로 기록한다.
[재전환 코드](../scripts/srgc_switch_repeat.py).

## 5. Reference 민감성과 불확실성

**V6 근거:** projected local signal은 reference에 민감하고 장기 보상이나 optimizer 이동을
예측하지 않는다. A/B 부호 불일치와 jackknife 결과, point-estimate 규칙의 한계를 명시했다.

**현재 가능한 비교:** E09는 실제 validation 방향, 무작위 방향, gradient 크기,
방향 정보 제거를 비교한다. reference 방향 정보가 학습에 기여하는지 확인하는 **부분 대조**다.

```sh
sh scripts/run_srgc_sr_refresh.sh math direction
sh scripts/run_srgc_sr_refresh.sh all status direction
sh scripts/run_srgc_sr_refresh.sh math results
sh scripts/run_srgc_sr_refresh.sh mbpp results
```

세 조건 합계 양쪽 30개이며 `direction_removed` 10개는 support와 중복되므로 재사용한다.
[방향 대조 코드](../scripts/srgc_direction_ablation.py).

**아직 직접 대응하지 못하는 부분**

| V6의 남은 항목 | 현재 코드로 가능한 것 | 아직 없는 전용 실험 |
| --- | --- | --- |
| reference 민감성 | 실제 방향을 무작위 방향으로 교체하는 대조 | 여러 실제 reference 집합으로 온라인 Switch 전체 경로 비교 |
| point estimate·통계적 불확실성 | 기존 A/B 진단, 단일/연속 음수 확인 규칙 비교 | 신뢰구간을 사용하는 전환 규칙과 기존 규칙 비교 |
| check interval 미최적화 | 현재 점검 간격 25 유지 | 간격만 바꾸는 matched sweep |

무작위 방향 대조나 같은 reference의 응답 재추출을 실제 reference 집합 변경 실험이라고 쓰지 않는다.
확인 규칙 비교도 점검 간격 최적화나 불확실성 보정으로 부르지 않는다.
위 미구현 항목에 가짜 실행 명령을 붙이지 않는다. 별도 설계가 필요한 공백으로 남긴다.

## 6. 보조: 기존 결과 수집과 동일-prefix replicate

**위치:** V6의 training-seed 변동 문제에 대한 주 근거는 추가 seed의 P0다.
replicate는 같은 prefix 이후 무작위성만 바꾸므로 전체 학습 seed 반복을 대신하지 않는다.
기존 결과는 먼저 수집하되 replicate 신규 GPU 배정은 위 직접 대응 실험 이후로 둔다.

**비교:** E03은 동일 step-25 prefix에서 SR/Switch를 독립 sampling stream k=1,2로 반복한다.
E04는 fixed200과 기존 Switch를 비교한다. 두 데이터셋 합계 40 + 10개다.
fixed200은 2절의 전환 시점 실험에 포함해 재사용한다.

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

**선택 실행:** fixed200은 2절의 직접 대응 실험이다. replicate는 보조 실험을 배정할 때만 실행한다.

```sh
sh scripts/run_srgc_sr_refresh.sh math replicate
sh scripts/run_srgc_sr_refresh.sh math switch_fixed200
```

**볼 결과:** seed별 SR/Switch 보상 차이, 반복별 차이의 일관성, 실제 전환 시점과 비용.
노드 상한은 반복 20개/데이터셋, fixed200 5개/데이터셋이다.
[실행 코드](../scripts/run_srgc_sr_refresh.sh) · [자동 배정](../scripts/srgc_replicate_worker.py).

## 7. 별도 요청: 초반·후반 메커니즘

**위치:** 저자가 별도로 요청한 실험이다. V6 limitation 대응과 구분해 병행하며,
다른 실험이 끝날 때까지 기다릴 필요가 없다. 이미 진행 중인 작업도 그대로 유지한다.

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
JSON 출력은 `sh scripts/run_srgc_mechanism.sh all json`이다. results/json 모두
`<run root>/results/mechanism/results.json`에 원본을 포함한 수집본을 자동 저장한다.
support와 전환 비교도 각각 `results/support/`, `results/switch_validation/`에 자동 저장한다.
실제 경로는 `COLLECTED JSON / COLLECTED FILES`에서 확인한다.
원본은 `<run root>/seed-N/stage_mechanism-endpoint.json`, results의 `output=`으로 위치를 확인한다.

**규모:** 5개 seed 작업/데이터셋, 양쪽 최대 10노드. seed당 물리적 update는
공통 경로 400 + 3시점 × 6branch × 25 = 850회, 평가 호출 21회다.
SR도 배치를 25 updates 유지하는 국소 진단이며 기존 매-update SR이나 100-update 실험의 재현과 구분한다.
[상세 설계](STAGE_MECHANISM_EXPERIMENTS_2026-10-06.md) ·
[학습·계측 코드](../scripts/srgc_stage_mechanism.py) · [분석 코드](../scripts/srgc_stage_report.py).

## 8. 완료한 P0: 추가 학습 seed와 MBPP

**V6 근거:** 온라인 전환의 데이터·학습 seed 범위가 제한되어 있고 question bootstrap은
training-seed 변동을 측정하지 않는다. 추가 학습 seed의 실제 학습 결과로 보완한다.

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
- [정리 전 계획·검증 이력](https://github.com/33modeling/offpolicy-misranking/blob/6e4e7039c84a9a16db08f821d8326e09c81cab21/docs/LIMITATION_EXPERIMENTS_KO.md).
