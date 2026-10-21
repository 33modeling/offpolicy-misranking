# Switch 신규 실험: 목적·우선순위·실행 계획

## 2026-10-21: 수신 결과의 재분석 우선

저자는 mechanism 실험을 이미 수행했고 코드도 있다고 확인했다. 이를 신규 실험이나
재실행 1순위로 제안하지 않는다. N08, N12·N13과 rollout 분석 코드도 이미 있다.
현재 우선순위는 **모델·과제·학습 상태에 따라 선택법의 상대적 이점이 달라질 때
Switch가 성능과 비용을 얼마나 안정적으로 유지하는지 전체 결과로 분석하는 것**이다.
이는 저자가 정한 결론의 관점이며, V7 결론에도 상태·과제에 따라 유리한 선택법을
미리 알기 어렵다는 설명이 있다. 개별 대조의 유의성이나 seed 제외 분석을 주제로
대체하지 않는다. 반복성과 신호 진단은 이 전체 분석을 보완한다.
아래 10월 6일·18일의 신규 설계 순서는 당시 기록이며 현재 배정 순서로 사용하지 않는다.
새 학습·진단 GPU 작업, queue 변경, 원고·교정 사이트 변경은 하지 않았다.

### 주 분석: 상태 변화에 대한 성능·비용의 안정성

각 조건의 다섯 seed 평균을 사용하며, 모델 사이의 원래 정확도를 직접 합산하지 않는다.
조건별 최고 평균은 비교를 위한 사후 기준이며, 미래 조건에 대한 성능 보장이나
독립적인 실험 반복을 뜻하지 않는다. 아래는 네 방법을 빠짐없이 포함한 결과다.

| 모델·과제 | Random | On-policy | SR | Switch | Switch 순위 | 최고 평균 대비 Switch 차이 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OLMo MATH | 36.74 | 39.46 | 39.38 | 40.94 | 1 | 0.00 %p |
| OLMo MBPP | 34.88 | 36.84 | 41.06 | 41.11 | 1 | 0.00 %p |
| Qwen MATH | 78.96 | 80.74 | 81.98 | 82.10 | 1 | 0.00 %p |
| Qwen MBPP | 81.82 | 84.22 | 82.02 | 83.34 | 2 | 0.88 %p |
| Llama MATH | 60.82 | 64.28 | 65.32 | 64.48 | 2 | 0.84 %p |
| Llama MBPP | 62.04 | 61.34 | 62.90 | 63.18 | 1 | 0.00 %p |

최고 평균과의 차이는 **방법·조건별 평균**의 차이다. 모든 개별 seed나 새 모델에서
이 범위가 보장된다는 뜻은 아니다. 근소한 평균 차이도 통계적 우위로 바꾸어 쓰지 않는다.

- 고정 On-policy와 SR끼리 비교하면 유리한 방법은 조건에 따라 바뀐다. On-policy가
  높은 조건은 OLMo MATH와 Qwen MBPP이고, 나머지 네 조건은 SR이 높다.
- Switch는 여섯 조건 모두 평균 성능 1~2위이며 네 조건에서 최고 평균이다.
  각 조건 최고 평균으로부터 가장 크게 떨어진 정도는 Switch 0.88 %p,
  SR 2.20 %p, On-policy 4.27 %p, Random 6.23 %p다.
  분석의 중심은 이 **조건 변화에 따른 성능 저하 폭**과 비용이다.
- 같은 데이터에서 계산한 평균 순위는 Switch 1.33, SR 2.17, On-policy 2.67,
  Random 3.83이다. 순위는 효과 크기와 함께 읽으며 여섯 조건을 표본 수 30의
  독립 검정으로 바꾸지 않는다. Switch는 수신된 30개 seed 쌍 모두 Random보다 높다.

현재 원고에서 채택한 같은 성분·단위의 Qwen 시간 기록은 다음과 같다.
공통 prefix 이후 선별·학습·준비의 시간으로, 최초 cache·공통 prefix·평가 등은 제외한다.

| Qwen 과제 | On-policy 평균 시간 | Switch 평균 시간 | 평균 시간 감소 |
| --- | ---: | ---: | ---: |
| MATH | 33.59 h | 27.37 h | 18.5% |
| MBPP | 35.72 h | 31.80 h | 11.0% |

시간 비율은 반올림 전 값으로 계산했다. 원래 PARTIAL/audit 표기를 evidence에 유지한다.
MATH에서 Switch/On-policy의 seed별 시간 비율은 0.722~0.979,
MBPP에서는 0.772~1.022다. MBPP의 늦은 전환 두 seed에서 시간이 각각 약 1.7%,
2.2% 더 들었다. 따라서 비용 해석에는 평균과 seed별 범위를 함께 제시한다.
Llama는 중단·재시작 포함 시간과 부분 하한이 섞여 있어 동일한 다섯-seed 비용 평균으로
확장하지 않는다. 같은 모델의 학습을 새로 실행하는 대신 기존 비용 원장의 범위를 맞춘다.

기존 단계별 MATH/GRPO 결과에서는 Random 대비 On-policy/SR의 이득이
d0에서 +0.58/−0.60 %p, d400에서 +0.12/+1.74 %p로 바뀐다. 이는 단계에 따른
선별법 유불리 변화의 근거다. 다른 모델의 마지막 평균만으로 시간 중 상태의
급변 속도를 추정하지 않는다. 기존 RLOO 결과도 objective에 따른 유불리 차이의 근거로
연결한다. 별도의 실험 프로토콜을 새로운 독립 seed로 합치지 않는다.

이 관점에서 기존 결과로 먼저 정리할 것은 (1) 여섯 조건의 성능 순위·최고 평균과의 차이,
(2) 같은 범위의 비용과 성능을 함께 본 비교, (3) 단계별 선별법 효과와 실제 전환 이력의
관계다. 아래 seed 민감도와 신호 진단은 보조 분석이다. 원고·교정 사이트는 변경하지 않았다.

### 보조 1. 학습 seed 하나에 의존하는 결과인가

V7에 보존된 저자 최종표에서 동일 seed의 Switch−On-policy와 Switch−SR을 계산했다.
OLMo는 원래 정답 수/응답 수, Qwen·Llama는 전달받은 소수 한 자리 백분율을 사용한다.
각 행은 별도 모델·과제의 seed 5개이며 서로 다른 실험을 독립 seed처럼 합치지 않는다.
승수는 차이가 양수인 seed 수다. 단위는 %p다.

| 결과 묶음 | Switch−On-policy | 승수 | Switch−SR | 승수 |
| --- | ---: | ---: | ---: | ---: |
| OLMo MATH | +1.483 | 4/5 | +1.567 | 5/5 |
| OLMo MBPP | +4.267 | 5/5 | +0.050 | 3/5 |
| Qwen MATH | +1.360 | 4/5 | +0.120 | 3/5 |
| Qwen MBPP | −0.880 | 2/5 | +1.320 | 5/5 |
| Llama MATH | +0.200 | 3/5 | −0.840 | 1/5 |
| Llama MBPP | +1.840 | 3/5 | +0.280 | 2/5 |

seed를 하나씩 제외한 평균 차이는 OLMo MATH의 On-policy 대비 +1.135~+2.000,
Qwen MATH의 On-policy 대비 +1.075~+1.850, Qwen MBPP의 SR 대비 +0.925~+1.625다.
관측된 이득이 단 한 seed의 큰 값에만 의존하지 않는다는 설명에 사용할 수 있다.
반대로 Qwen MBPP의 On-policy 대비 −1.250~−0.275, Llama MATH의 SR 대비
−1.125~−0.625도 부호를 유지한다. 불리한 결과 역시 단 한 seed의 영향으로 지우지 않는다.

이것만으로 우연 가능성을 배제했다고 쓰지 않는다. 부호 대칭·교환가능성을 가정한
탐색적 양측 sign-flip은 5쌍에서 32개 부호 조합만 있으므로, 모두 같은 방향이어도
최소 p는 0.0625다. 계산한 paired t 95% 구간도 위 두 대조의 여섯 행 모두 0을 포함한다.
이는 동등성이나 무효의 증명도 아니다. 수천 개 응답을 독립 학습 반복으로 세거나,
사후에 단측 검정으로 바꾸거나, 유리한 모델·과제만 골라 확증 결과로 쓰지 않는다.
차이·구간·전체 seed를 함께 제시하고 이 분석이 탐색적이라는 점을 유지한다.

### 보조 2. 학습 단계만 아는 것보다 SR-GC가 더 알려 주는가

기존 9상태 진단(3 seeds × t0/t100/t400)을 사용했다. 한 seed를 평가할 때 같은 단계의
다른 두 seed에서 On-policy−SR 평균이 양수이면 On-policy, 아니면 SR을 선택하는
단계별 기준을 만들었다. 평가 대상 seed의 결과는 그 선택에 사용하지 않는다.
실제 online switching의 새 실행이나 새로운 전환 시점의 반사실적 성능 추정이 아니다.

| 기존 진단 신호 | 교차 검증 단계 기준 대비 선택 보상 차이 | 상태별 승/패/동률 |
| --- | ---: | ---: |
| A | −0.009 %p | 2/2/5 |
| B | +0.037 %p | 3/1/5 |
| A/B 평균 | +0.037 %p | 3/1/5 |

세 신호는 같은 자료의 반복 비교이며 독립 실험 세 개가 아니다. 9상태도 독립 seed
9개가 아니다. B/평균의 seed별 차이는 +0.111, +0.083, −0.083 %p다.
기존 정답 방향 7/9만으로 신호의 고유한 시점 선택 효과를 강하게 주장하지 않는다.
이 결과에서는 단계별 기준을 넘는 보상 차이가 작다. 미리 정한 선택 규칙의
실제 실행 결과와 해당 결정을 내리기 전의 기록을 연결하는 분석이 더 직접적이다.

### 보조 3. 기존 신호·선택·학습·평가 기록을 연결

- 기존 E11/E12 결과에서 같은 prefix·학습 길이·평가 조건의 고정 전환 및 단일/반복
  확인 대조를 분석한다. 전환 자체의 효과와 SR-GC로 시점을 고르는 효과를 구분한다.
  채택 제외된 fixed200은 다시 넣지 않고, 예전 Pair의 길이가 다른 branch를 정확히
  같은 budget의 대조로 쓰지 않는다. 코드의 존재를 실측 결과의 존재로 간주하지 않는다.
- 이미 구현된 N13으로 D40과 실제 선택 4문제의 D4가 독립 B에서도 같은 방향인지
  분석한다. 스칼라 D나 cosine에서 없는 vector를 복원하지 않는다. 로컬 수신본의
  29개 Switch check에 D4가 없다는 점과 서버의 원자료 존재 여부는 구분한다.
- 기존 information 결과에는 t0의 A/B 순위 상관과 실제 업데이트·probe 변화가 있다.
  MATH의 평균 순위 상관 0.0214, MBPP 0.0644와 실제 loss 변화까지 함께 해석한다.
  이미 분석한 t0 cosine을 다시 새 실험으로 제안하거나 단계 전체의 원인으로 확대하지 않는다.
- N08과 rollout 분석 코드를 재사용하되 출처가 일치하는 파일만 연결한다. 예전 수신
  OLMo JSON의 MATH seed 5 Switch는 37.5%, 최신 최종 정답 표는 39.5%다.
  동일 seed 번호만으로 그 JSON의 문제별 결과·D·전환 시점을 최신 표에 붙이지 않는다.
  최신 Qwen·Llama 요약에 없는 문제별 응답·중간 held-out 평가는 미측정으로 둔다.

[계산값·원본 SHA256·통계 범위](audits/received-results-robustness-2026-10-21.json)에
최종표 분석과 단계 기준 비교를 보존했다. 기존 실험의 성능 효과와 SR-GC 자체의
추가 정보를 구분하여 리뷰어의 우연 가능성 지적에 답하는 분석 순서다.

## 2026-10-18 추가 목록: 기존 실험과 겹치지 않는 질문

GitHub `master`의 기존 실험 목록·구현·수신 결과와 V7을 대조하여 **N12–N18 일곱 항목**을 추가했다.
2026-10-18 후속 요청으로 **N12·N13의 전용 CPU 분석·checkpoint 진단·추론/scoring 코드를 구현했다.**
[실행·checkpoint 재사용 안내](N12_N13_CHECKPOINT_DIAGNOSTICS.md)를 따른다. 로컬 CPU 검증과
실제 H100 실험은 구분하며 신규 GPU 측정을 실행한 상태는 아니다. **N14–N18은 설계 제안이다.**
아래 N01–N08의 구현 검증과 N09–N11의 설계 상태는 각각 2026-10-06/07 당시 기록이다.
기존 queue를 추가하거나 진행 중인 작업을 재시작한 변경은 아니다.

| 검토 순서 | ID·추가 질문 | 기존과 다른 점 | 먼저 필요한 작업 |
| --- | --- | --- | --- |
| 1 | [N12 데이터 구성 변화와 같은 문제의 학습 변화 분리](#n12) | 25-update 분포 집계에 고정 구성으로 표준화한 비교를 추가 | 저장 rollout·cache·ID 확인; CPU 재분석부터 |
| 2 | [N13 SR-GC의 40개 평균과 실제 선택 4개의 차이](#n13) | 전환 신호가 비교하는 대상과 실제 학습 배치가 다른 문제를 직접 점검 | 문제별 projected gradient 보존 여부 확인 |
| 3 | [N14 rollout 잡음과 불확실성을 고려한 전환](#n14) | 기존 단일/연속 음수 대조를 넘어 부호의 측정 변동을 분리 | 같은 상태의 독립 응답·gradient 진단 |
| 4 | [N15 On-policy 선택 4개를 25 updates 유지하는 효과](#n15) | SR cache의 노후화와 별개로 현재 gradient 순위의 유효기간 검사 | optimizer를 포함한 분기 상태 확인 |
| 5 | [N16 On-policy와 SR 사용량을 맞춘 순서 교환](#n16) | 두 방법의 사용 update 수를 같게 두고 적용 순서 변경 | 공통 t25에서 두 경로를 별도 실행 |
| 6 | [N17 다른 모델·과제에서 가져온 전환 시점](#n17) | 같은 과제의 seed 교차 검증 대신 cohort 사이의 전이 검사 | 채택한 고정 시점의 source/target 실측 결과 |
| 7 | [N18 학습 도중 문제 분포를 바꾸는 개입](#n18) | 동일 pool의 반복 대신 환경 변화를 의도적으로 만들어 적응성 검사 | 별도 입력 manifest·새 실험 경로 설계 |

**10월 18일 당시 제안은 N12와 N13 우선이었다.** N12는 MATH와 MBPP에서 선택하는 문제와
학습 신호가 실제로 어떻게 다른지, N13은 그 차이를 SR-GC가 제대로 대변하는지를 묻는다.
N14–N16은 관측한 차이를 설명하거나 개입으로 확인할 때 확장한다. N17은 전이 주장,
N18은 변화하는 데이터 환경에 대한 별도 검증으로 두며 새 전체 학습을 한꺼번에 배정하지 않는다.

E11–E13의 고정 전환·규칙 대조, E14–E16의 shuffle/fresh-SR 메커니즘,
N01–N08의 gradient·reference·batch 진단, N09–N11의 난도·다양성·응답 배분은
기존 계획으로 유지한다. N14는 기존 문서에 남아 있던 **미구현 불확실성 규칙을 구체화**한 것이다.
기존 실험을 새 번호로 다시 세거나 seed만 늘리는 목록이 아니다.
공통 출처·추출·기록 조건은 [추가 목록의 실행 전 확인](#followup-contract)을 따른다.

---

2026-10-07의 구체적 분석 목표는 **초반 On-policy의 학습 성능 이득이 생기는 이유**다.
[기존 근거와 원인 분석 순서](STAGE_MECHANISM_EXPERIMENTS_2026-10-06.md#초반에-왜-이득이-나는지-확인할-분석)에 따라
선택 문제·실제 학습 보상을 먼저 분석하고 N07/N02를 필요한 부분에 적용한다.
아래 전체 Switch 방법 비교의 우선순위를 초반 원인 분석의 순서로 대신 쓰지 않는다.

2026-10-07 추가 요청: [데이터 구성과 rollout 개입 N09–N11](DATA_ROLLOUT_INTERVENTIONS_2026-10-07.md).
현재 실행 중이라는 저자 보고는 메커니즘과 Qwen이다. 새 후속 설계는 난도를 맞춘 문제 분포,
reward·길이를 맞춘 응답 다양성, 고정 32응답의 배분을 조작한다.
N09–N11의 GPU 학습 runner는 아직 없으며 아래 N01–N08 구현·명령과 구분한다.

2026-10-06. [최근 논문 12편 조사](RECENT_PAPER_EXPERIMENT_SURVEY_2026-10-06.md)를 바탕으로 한 실험 구현·실행 안내.
**현재 상태: N01-N08 구현 완료. CPU 통합·분산 검증과 H100/NCCL 실측은 구분한다.**
N01-N08은 신규 실험 식별자이며
기존 E01-E16과 진행 중인 `mechanism`을 대체하지 않는다.

핵심 질문은 세 가지다. 더 저렴한 gradient 선별과 비교해도 전환이 유용한가,
선별 점수가 실제 학습 효과로 이어지는 관계가 단계별로 어떻게 달라지는가,
결과가 특정 reference·후보·batch 규모에 의존하는가를 확인한다.
성능과 비용을 함께 평가하며 SR-GC는 전환을 돕는 보조 지표로 유지한다.

## 지금 할 것과 기다릴 것

| 구분 | 실험 | 먼저 하는 이유·시작 조건 |
| --- | --- | --- |
| 신규 학습 우선 | **N01: 저비용 gradient 선별 대조** | 비싼 기존 scorer 때문에만 Switch가 유리한지 직접 확인 |
| 기존 결과로 병행 | **N08: 문제별 개선·퇴행 분석** | 받은 평가 JSON으로 평균 뒤의 차이를 확인; 새 학습 불필요 |
| 직접 한계 대응 | **N04: 실제 reference 교체**, **N05: 후보·batch 규모** | reference 구성과 작은 학습 규모에 대한 의존성을 각각 검증 |
| 메커니즘 결과 수신 후 | **N02: 실제 gradient·optimizer 진단**, **N03: 점수 구간별 학습** | 진행 중인 E14-E16을 먼저 해석한 뒤, 기존 결과로 답하지 못하는 부분만 추가 |
| 관측 차이 확인 후 | **N07: 구성 matched 대조** | 성공률·재노출 등의 차이가 실제로 있는 축을 통제 |
| 다음 외부 비교군 | **N06: ARCUS 계열 대조** | gradient 없는 동적 curriculum과 비교; 독립 adapted 구현이며 공식 코드 재현으로 부르지 않음 |

위 순서는 신규 작업 배정 기준이다. 기존 실험을 중단·재배정하지 않는다.
`fixed200`은 채택 제외를 유지하고, `replicate`는 보조 결과로만 둔다.

## 실행 조건 한눈에 보기

공통 기준은 MATH·MBPP, 기존 OLMo/GRPO 환경, seeds 5-9, 응답 8개다.
전체 경로 비교는 **t0에서 시작**하고, On-policy 재선별 간격은 25로 둔다.
기본 후보 40개는 매 선별 시점 400개 pool에서 중복 없이 뽑는다. 선택한 학습 문제는 4개다.
SR-GC는 **40 대 40, 실행당 reference 하나**를 유지한다.

| ID | 비교·개입 | 고정할 조건 | 핵심 출력 |
| --- | --- | --- | --- |
| N01 | 기존 On-policy / 출력층 feature 선별 / Switch | scorer 이외 모델·학습·응답·평가 조건 | reward 곡선, 선별·학습 시간, 목표·GPU 예산별 성능 |
| N02 | On-policy / SR / Random의 실제 학습 gradient·Adam update | 동일 checkpoint·optimizer, 학습 batch 크기 | raw·centered gradient, 실제·zero-gradient 차감 update, 평가 변화 |
| N03 | 점수 4구간별 4문제 / top-4 / Random-4 | 동일 상태에서 분기, 분기 내 선택 문제 유지 | 학습 전 및 1·5·25 updates 후 reward |
| N04 | reference 3집합, 각 집합으로 자체 Switch 경로 | reference 크기, 후보, 학습·평가 조건 | reference별 SR-GC·순위 변동, 전환 시점·reward·비용 |
| N05 | 후보 40/80/160에서 4개 선택; 별도로 후보 40에서 4/8/16개 학습 | 한 번에 한 축만 변경 | reward, prompt·응답·token 노출, GPU 비용 |
| N06 | ARCUS-adapted / Switch | 모델·pool·GRPO·평가·응답 cap | 생성량, 유효 group 비율, 성능·비용 |
| N07 | gradient top-4 / Random / 성공률 또는 재노출 분포 matched Random | 동일 상태·batch 크기, 한 번에 한 구성 축 | matched 전후 학습 이득·구성·비용 |
| N08 | 동일 seed·평가 문제의 방법별 결과 | 평가 ID·설정·실험 출처 | 문제별 증가·감소, 원시 응답이 있을 때만 pass@k |

N02·N03·N04 고정 상태 진단·N07의 비교 시점은 0/100/400이다.
**이 진단의 학습 경로와 본 실험의 275-update 경로는 다른 작업**이며, 비용도 합치지 않는다.
기존 메커니즘 checkpoint를 검증 없이 가져오지 않는다. 별도 `anchors` 작업은
t0부터 400 updates까지 진행해 비교 상태를 만든다. 따라서 기존 결과 수신 전에 이 작업을
새로 배정하지 않으며, 재사용 가능한 checkpoint가 있는지 먼저 확인한다.

N04는 100문제 validation pool에서 기존 reference와 같은 크기의 서로 다른 세 집합을 만든다.
현재 50문제 reference라면 세 집합을 모두 서로 겹치지 않게 만들 수 없으므로 실제 겹침을 기록한다.
후보·최종 evaluation 문제를 reference에 섞지 않으며, 없는 topic 라벨을 만들어 맞추지 않는다.

## 코드와 검증 상태

| 항목 | 위치 또는 상태 |
| --- | --- |
| 저장소·브랜치 | 코드 저장소 `offpolicy-misranking`, `master`; 논문 저장소에는 추가하지 않음 |
| 신규 코드 경로 | [`srgc_research/`](../srgc_research/) 및 [`scripts/run_srgc_research.sh`](../scripts/run_srgc_research.sh) |
| 기존 코드 보호 | 기존 학습·queue·E14-E16 실행 코드 수정 없음; 기존 Qwen 테스트의 MBPP fixture에 누락된 dataset만 보충 |
| 실행·상태·결과 수집 | 신규 전용 wrapper의 `run/status/results/json`; 기존 dispatcher와 별개 |
| 자동 재개 검증 | 여섯 trajectory의 모델·optimizer·원시 평가 기록 일치, 진단 단계별 복원, endpoint 재발행 확인 |
| 비용·결과 검증 | 비용 합계·미완료 timer, 원본 JSON 수집, 평가 ID·cache hash 오류 거부, 미완료 seed 표시 |
| 분산 검증 | 4-rank CPU/Gloo에서 scorer·rollout 재사용·GRPO/Adam 진단·checkpoint 복구 확인 |
| 최종 회귀 테스트 | 신규 45개 포함 754개 통과, 325 subtests 통과, 실패 0개; 별도 Qwen 환경 필요 9개 제외 |
| 정적·명령 검증 | Ruff, Python compile, `sh -n` 통과; MATH/MBPP `status/results` 경로 출력·빈 결과 처리 확인 |
| GPU 검증·실험 | H100/NCCL 실행 검증 및 신규 실험 실행하지 않음 |
| 기존 결과 N08 | MATH·MBPP 원본 JSON 각각 방법 쌍 30개 분석 완료; 새 학습 없음 |

검증에는 실제 작은 OLMo/LoRA 모델과 기존 GRPO·AdamW 경로를 사용한다.
단위 테스트의 생성 응답은 결정적으로 주입하며, H100의 메모리·속도·NCCL까지 검증했다는 뜻은 아니다.
재개 시 평가 응답 배열이 이후 학습 기록에 의해 변하던 참조 공유 오류와
binary float 응답의 pass@k 조합 계산 오류를 수정했다.
기존 결과·진행 중인 runtime을 다시 작성하거나 해시를 해제하지 않는다.

구현 위치: 조건·의존성 `design.py`, scorer/optimizer 계측 `backend.py`,
학습·진단 `study.py`, ARCUS 대조 `arcus.py`, 저장·복구 `storage.py`/`worker.py`,
노드 배정 `cli.py`/`launch.py`/`rank.py`, 결과·N08 `report.py`.

## 명령 형식

저장소 루트에서 `sh`로 실행하며, 별도 Python 옵션이나 `resume` 명령을 요구하지 않는다.
기존 Python 환경을 그대로 사용하고 새 패키지를 설치하지 않는다.

| 실험 | 실행 명령 형식 |
| --- | --- |
| N01 | `sh scripts/run_srgc_research.sh math n01 run` |
| N02 | `sh scripts/run_srgc_research.sh math n02 run` |
| N03 | `sh scripts/run_srgc_research.sh math n03 run` |
| N04 | `sh scripts/run_srgc_research.sh math n04 run` |
| N05 | `sh scripts/run_srgc_research.sh math n05 run` |
| N06 | `sh scripts/run_srgc_research.sh math n06 run` |
| N07 | `sh scripts/run_srgc_research.sh math n07 run` |
| N08 | `sh scripts/run_srgc_research.sh math n08 results /path/results.json` |

MATH는 `math`, MBPP는 `mbpp`를 쓴다. N01-N07의 `all`은 두 데이터셋을 배정한다.
작업별 기존 `PAIR_PYTHON`/`SWITCH_PYTHON`을 선택하므로 서로 다른 환경도 유지한다.
N08은 데이터셋별 JSON을 각각 분석한다.

상태·결과 명령은 해당 ID를 유지하고 마지막 인자만 바꾼다.

```sh
sh scripts/run_srgc_research.sh math n01 status
sh scripts/run_srgc_research.sh math n01 results
sh scripts/run_srgc_research.sh mbpp n01 results
```

빈 4-H100 노드에서 선택한 scope 하나만 실행하며, 여러 노드에서는 같은 명령으로
seed·조건을 중복 없이 배정한다. 이미 점유된 노드는 기존 작업을 중단하지 않고 `BUSY`로 종료한다.
실험 전체를 무조건 동시에 실행하는 기본값은 두지 않는다.
N05는 조건 15개/seed라 비용이 크다. 다른 scope와 같은 크기의 작업으로 안내하지 않는다.
노드 권장 수·예상 시간은 파일럿 계측 전 확정하지 않는다.

노드가 중단되면 **같은 명령을 다시 실행**한다. 매 학습 update와 진단 단계 종료 시
모델·optimizer·진행 상태를 저장하며, 그 사이 완료된 rollout 응답도 그룹 볼륨에서 복구한다.
자동 재개는 미완료 계산을 이어가는 것이지 실행 중인 노드를 다른 노드로 강제 이동하는 기능이 아니다.
완료 기준을 통과한 작업은 다시 돌리지 않는다. 반복 오류는 3회 뒤 `failed`로 표시한다.

N01의 On-policy는 N05 기본 조건과, N01의 Switch는 N06 대조와 동일 작업을 재사용한다.
진단용 `anchors`와 최초 SR cache도 scope 간 공유한다. 기존 v2/prefix 실험 결과와는 합치지 않는다.

## 결과와 비용 기록

신규 작업의 기본 저장 루트는 `$SRGC_STORAGE_ROOT/literature-v1`이다.
`SRGC_STORAGE_ROOT` 미지정 시 기존 그룹 볼륨의 `$OM_WORK/srgc-rebuttal` 아래를 사용한다.
user volume으로 대체하지 않으며 기존 결과 디렉터리에 덮어쓰지 않는다.

| 산출물 | 신규 루트 아래 위치 |
| --- | --- |
| 실험 출처·고정 입력 | `<dataset>/seed-<N>/manifest.json`, `inputs.json` |
| 최초 SR 캐시 | `<dataset>/seed-<N>/sr-cache.json` |
| 작업 결과·전체 원시 기록 | `<dataset>/seed-<N>/<condition>/endpoint.json` |
| 중단 복구 지점 | `<dataset>/seed-<N>/<condition>/state-latest.pt` |
| 세부 비용·실행 로그 | `<dataset>/seed-<N>/<condition>/cost-receipts/`, `task.log` |
| 전체 프로세스 점유 시간 | `<dataset>/seed-<N>/<condition>/invocations/`; 내부 phase와 중복 합산하지 않음 |
| 수집 JSON | `exports/<dataset>-<scope>-results.json` |
| N08 분석 | 입력 JSON 옆 `<입력파일명>-n08.json` |

`results`는 저장 경로를 출력하며 수집 JSON 안에 원본 결과를 포함한다.
미완료·오류·비용 누락을 완료 결과와 구분하고, 아직 시작하지 않은 seed도 누락하지 않는다.
평균에는 완료된 seed 목록과 전체 완료 여부를 함께 기록한다.
N02의 원시 gradient·update tensor는 checkpoint의 `state.vectors`에 저장하며,
JSON에는 norm·alignment·평가값과 해당 위치를 기록한다.
N08 기존 결과 분석 파일은 `/home/kms/results-n08.json`, `/home/kms/results_mbpp-n08.json`이다.

비용은 **최초 SR 캐시 / 선별용 후보·reference 생성 / gradient·feature 계산 /
정렬 / 학습 / 평가 / 저장 I/O**로 나눈다. SR 캐시 생성 비용과 점수 정렬 비용을 섞지 않는다.
최초 캐시는 seed·데이터셋별 별도 신규 측정 후 재사용하며, 기존 실행의 cache를 삭제·재생성하지 않는다.
Cold-start 비교에는 최초 캐시를 한 번 넣고, 재사용 비교에서는 이미 준비된 비용을 분리해 보여 준다.
연구용 반복 reference·centering·optimizer 진단 비용을 실제 Switch의 온라인 비용에 넣지 않는다.

목표·GPU 예산 비교가 필요하면 **최초 실행 전** `SRGC_RESEARCH_MATH_TARGET` 또는
`SRGC_RESEARCH_MBPP_TARGET`에 reward 목표(0-1), `SRGC_RESEARCH_GPU_BUDGETS`에
쉼표로 구분한 GPU-second 예산을 정한다. 이 값은 manifest에 고정한다.
미지정 시 학습·비용 곡선만 수집하며, 결과를 본 뒤 목표를 만들어 도달 시간을 채우지 않는다.
도달 시간은 실제 평가 시점 중 처음 관측된 값이며 그 사이를 보간하지 않는다.

## N01. Gradient 선별을 싸게 해도 전환할 이유가 남는가?

**참고:** [LESSER Sec. 4-5, App. A.4/F](https://arxiv.org/html/2610.03702v1).
논문의 출력층 feature 대체를 참고하되, 논문에 보고된 시간 배수를 우리 로그에 곱하지 않는다.

**비교군:** 기존 full-backprop 기반 On-policy continue, 출력층 gradient 기반 continue,
기존 Switch. 첫 단계에서는 Switch의 규칙이나 SR-GC 정의까지 동시에 바꾸지 않는다.
현재 baseline의 실제 학습 가능 parameter와 projection 범위를 명시하고,
논문의 dense full-gradient 구현과 동일하다고 부르지 않는다.

**고정:** 같은 시작 모델·optimizer, 후보/query 구성, scoring 응답 수, 선택 문제 수,
재선별 주기, 학습 objective·평가. Scoring representation만 교체하며 GRPO 학습 parameter는 유지한다.
같은 checkpoint의 feature 대조에서는 같은 scoring rollout을 쓴다.
학습 경로가 갈라진 뒤에는 각자의 현재 policy로 fresh rollout을 생성한다.
구현은 t0 전체 경로 세 개와 같은 응답을 재사용하는 `n01-features` 진단 한 개다.
출력층 feature는 현재 projection 차원에 맞춘 64×64 Rademacher 이중 투영(4,096차원)이며,
원 논문의 8,192차원 설정과 구분한다. Feature 진단 비용은 trajectory의 배포 비용에 더하지 않는다.

**측정:** 학습곡선, 최종 held-out reward, 사전에 정한 목표까지의 GPU 시간,
같은 누적 GPU 예산에서의 성능. 후보/query 생성, scorer forward/backward,
projection·ranking, 학습, SR 일회성 cache를 각각 기록한다.

**판정:** 싼 scorer가 실제로 빨라졌는데도 Switch가 성능·비용의 유리한 조합을 제공하면
전환의 가치가 비싼 backward 구현 하나에만 의존하지 않는다는 근거다.
계속 선별하는 쪽이 더 좋다면 그 조건을 그대로 보고한다. 비용만 이기고 성능이 낮으면 trade-off다.

**범위·규모:** 이번 구현은 t0에서 시작하는 전체 경로 비교다.
데이터셋·seed별 trajectory 3개와 같은-rollout 진단 1개다.
동일한 신규 baseline만 공유하며, 기존 prefix 결과를 전체 경로 결과로 합치지 않는다.

## N02. 점수의 방향과 실제 업데이트 방향은 같은가?

**참고:** [LESSER App. F.2](https://arxiv.org/html/2610.03702v1),
[MOPD 분석 Sec. 3-4, App. F](https://arxiv.org/html/2610.02179v1).

**질문:** 선택된 문제들의 cosine 평균이 높아도 실제 batch gradient는 다를 수 있다.
또한 서로 다른 gradient가 공통 Adam 이력 때문에 비슷한 update로 보일 수 있다.

**설계:** 보존된 초반·중반·후반 모델/optimizer 상태에서 On-policy, SR, Random이 고른
동일 크기 batch를 비교한다. 각 batch에 대해 다음 네 측정을 분리한다.

1. 실제 학습 loss로 계산한 batch gradient와 그 norm.
2. 독립 random batch의 평균 성분을 제거한 centered batch gradient.
3. 저장된 optimizer 상태를 적용한 실제 parameter update.
4. 같은 상태에서 현재 gradient를 0으로 둔 update를 뺀 증분 update.

증분은 `update(gradient, optimizer_state) - update(zero_gradient, optimizer_state)`다.
원고의 H 진단과 혼동하지 않도록 optimizer 상태를 H로 표기하지 않는다.
AdamW moments, step counter, weight decay, clipping, loss reduction과 정밀도를 맞춘다.
`grad=None`으로 update를 생략하는 것은 zero-gradient 대조가 아니다.

**측정:** 방법 간 raw/centered alignment, update norm, 증분 update 방향,
한 번 업데이트한 뒤 독립 평가의 loss·reward 변화. Loss 진단은 frozen probe 응답에서,
reward 평가는 업데이트된 policy의 새 응답에서 측정하고 둘을 구분한다.
진단 branch는 실행 후 원래 상태로 복구한다. 본 학습 optimizer는 바꾸지 않는다.

**판정:** 어느 단계에서 선별 점수, 실제 batch 방향, optimizer 처리, 실제 성능의 연결이
약해지는지 확인한다. 방향 유사성만으로 인과나 장기 성능을 결론내리지 않는다.

**기존과 차이:** 현재 mechanism은 선택 문제별 B cosine/dot/norm 평균을 저장한다.
합산 gradient vector나 optimizer update 진단은 아니다.
Checkpoint와 원시 vector가 없으면 기존 JSON만으로 복원할 수 없고 별도 계측이 필요하다.
Centering 평균은 비교 batch와 독립적으로 추정하며, 불안정하거나 norm이 0인 측정은 따로 표시한다.

## N03. 높은 선별 점수가 실제 학습 이득을 예측하는가?

**참고:** [LESSER의 similarity-bin 학습, Sec. 4/App. C](https://arxiv.org/html/2610.03702v1),
[Quality-Utility의 동일 문제 통제, Sec. 4-6](https://arxiv.org/html/2606.16152v1).

**설계:** 동일 checkpoint의 후보 40개에 현재 policy의 gradient 점수를 매긴다.
정렬한 후보를 10개씩 네 구간으로 나누고 각 구간에서 4개를 무작위 선택한다.
별도로 기존 top-4, Random-4를 비교한다. 각 분기는 모델·optimizer를 같은 상태로 복원한다.
이 수치는 신규 설계의 시작안이며 기존 runner 설정을 바꾸지 않는다.

점수를 구간으로 나눌 때 동점은 사전 고정 순서로 처리하되, 실제 score 범위와 동점 비율을 보고한다.
전부 같은 점수인데 서로 다른 정보 구간이 생긴 것처럼 해석하지 않는다.
선정된 4개를 유지하며 매 update 문제당 8개 응답을 새로 생성한다.
학습 전과 1·5·25 updates 후를 비교해 즉시 효과와 여러 번 학습한 효과를 구분한다.

**측정:** 구간별 독립 평가 reward 증가, selection score와 개선량의 순서 관계,
유효 reward group 비율, gradient/update norm, 학습·평가 시간.
같은 상태의 여러 후보 추출을 사용해 한 batch의 우연을 구분하되 독립 training seed로 세지 않는다.

**판정:** 높은 구간의 이득이 초기에는 크고 후기에는 줄어드는지, 국소 loss 개선은 유지되지만
held-out reward로 이어지지 않는지 구분한다. 단조 관계가 유지되면 그 결과도 그대로 기록한다.

**기존과 차이:** E16은 selector가 고른 집합끼리 비교한다.
이 제안은 **점수 순위의 여러 구간을 실제로 학습**해 점수의 예측력을 검사한다.
기존 top-4/Random 결과는 시작 상태·후보·난수·평가가 같을 때만 재사용한다.
기본 분기 6개를 세 시점에서 25 updates씩 실행하면 후보 추출 1회당 450 branch updates다.
구현 기본값은 시점마다 서로 다른 후보 추출 3회로, seed·데이터셋당 **1,350 branch updates**다.
`draw-0/1/2`는 같은 학습 상태에서의 후보 반복이지 새 training seed가 아니다.
Carrier·scoring·평가 비용은 별도이므로 가벼운 무학습 분석으로 안내하지 않는다.

## N04. Reference 문제를 바꾸어도 판단과 성능이 유지되는가?

**참고:** [FAC의 지표 오차·전이 검증, App. L](https://arxiv.org/html/2602.10388v4),
[TESSY의 교차 학습, Sec. 4.2](https://arxiv.org/html/2604.14164v2).
Reference 교체 자체는 이 논문들의 실험이 아니라 Switch의 한계에 맞춘 신규 제안이다.

**설계 1, 고정 상태 진단:** 같은 크기의 reference 집합 3개를 미리 정한다.
후보 pool과 최종 evaluation에서 분리하고, 가능하면 주제·난도 구성을 맞춘다.
Reference끼리 불가피하게 겹치면 겹친 비율을 보고한다. 후보와 candidate rollout은 고정하고
reference 구성만 바꿔 ranking, SR-GC 값·부호, top-k를 비교한다.
별도로 같은 reference에서 rollout만 재추출한 변동을 측정해 두 불확실성을 구분한다.

**설계 2, 실행 검증:** 각 reference를 사용하는 Switch를 각자의 학습 경로에서 실행한다.
**한 실행에는 reference 하나만 사용한다.** 세 reference의 투표나 A/B 동시 선택 규칙을 만들지 않는다.
같은 경로의 신호만 재계산해서 다른 경로의 최종 성능을 추정하지 않는다.

**측정:** 부호 일치율, 순위 상관, top-k 겹침, 실제 전환 시점, 최종 reward·비용의 변화.
Reference 변경에 의한 변동과 rollout 재추출에 의한 변동을 나란히 보고한다.

**판정:** Reference가 달라도 성능·비용 결론이 유지되는지 확인한다.
신호 부호가 자주 바뀌어도 최종 성능은 안정적일 수 있으므로 둘을 별도 판정한다.
이 실험은 기존 A/B 진단이나 무작위 reference 방향 대조를 반복하는 것이 아니다.

## N05. 후보 수와 학습 batch 규모를 분리해서 늘리기

**참고:** [ARCUS App. F](https://arxiv.org/html/2609.38018v1),
[Lev Sec. 4.3](https://arxiv.org/html/2609.32484v1),
[LESSER의 선택 budget 비교](https://arxiv.org/html/2610.03702v1).

**질문:** 현재처럼 40개에서 4개를 골라 학습하는 조건에서만 전략 간 차이가 나타나는가?

| 별도 대조 | 시작 설계안 | 고정할 항목 | 해석 |
| --- | --- | --- | --- |
| 후보 범위 | 후보 40/80/160, 선택 4 | 전체 pool 400, 학습 문제 수·응답 수·주기 | 더 넓게 선별하는 비용이 학습 개선으로 돌아오는가? |
| 학습 batch | 후보 40, 선택 4/8/16 | 후보 범위, 문제당 응답 8 | 작은 학습 batch에서만 관측된 현상인가? |

두 축의 전체 조합을 처음부터 돌리지 않는다. 최소 규모 점검 후 필요한 축만 확장한다.
후보 순서는 미리 고정한 공통 무작위 순열을 이용해 작은 후보군이 큰 후보군의 부분집합이 되게 하고,
같은 후보군 안에서는 중복 문제를 뽑지 않는다. 학습량이 달라지는 대조는 누적 prompt 노출·응답·
token 기준과 실제 GPU 예산 기준을 모두 보고한다. 같은 update 수를 같은 계산량이라고 부르지 않는다.

**규칙 보존:** SR-GC의 40 대 40 점검은 그대로 둔다.
후보 규모 확장의 1차 비교는 selector 자체에 집중한다. Switch까지 확장할 때는
선별 후보군과 별도의 40 대 40 진단 집합을 명시하고 비용도 별도로 계측한다.

**측정·판정:** 규모에 따른 방법 간 reward 차이·변동성과 비용 곡선을 본다.
큰 batch에서도 차이가 유지되는지, 또는 평균화로 사라지는지 확인한다.
규모 하나에서 유리한 결과만 골라 기존 결론을 강화하지 않는다.

## N06. 동적으로 성공률을 추정하는 강한 비교군

**참고:** [ARCUS의 belief·pacing·sampling 대조](https://arxiv.org/html/2609.38018v1),
[DataMaster의 부분 모듈 결합](https://arxiv.org/html/2608.10579v1).

**질문:** Gradient 계산을 중단하는 Switch가, 학습 이력만으로 다음 문제를 고르는
동적 curriculum과 비교해 어떤 성능·비용 차이를 만드는가?

**설계:** ARCUS의 성공률 상태 추정, drift, target pacing, post-rollout filtering을
논문 식 (2)-(5)·Appendix D를 참고해 독립 구현한 **ARCUS-adapted**와 기존 Switch를 비교한다.
공식 코드 재현 실험으로 부르지 않는다. 이미 있는 matched fresh-SR 결과는
단순 갱신과 동적 추정의 차이를 읽는 보조 대조로 쓴다. 초기 policy부터 비교하는 것을 주 설계로 둔다.
기존 prefix에서 시작하는 결과를 사용하면 조건부 비교라고 따로 표시한다.

**통제:** 같은 모델·학습 pool·held-out 평가·보상·응답 cap을 사용한다.
ARCUS의 후보 공급이나 그룹 filtering을 기존 40-to-4 규칙으로 바꾸면 원래 ARCUS가 아닌
adapted 조건임을 명시한다. Fresh-SR을 ARCUS라고 이름만 바꾸지 않는다.
동일 update 수 외에 실제 생성 token·GPU 예산을 맞춘 결과도 제시한다.
이번 adapted 설정은 batch 4, 응답 8, 후보 여유율 25%로 매 round 5문제의 응답을 생성하고,
mixed-reward group 중 최대 4개를 학습한다. **생성한 응답을 그대로 학습에 재사용**한다.
유효 group이 없으면 optimizer update를 생략하고 sampling rounds와 실제 update 수를 따로 기록한다.
40개에서 4개를 고르는 gradient selector와 생성량이 같다고 설명하지 않는다.

**측정·판정:** 목표별 도달 시간, 학습곡선, 유효 group 비율, 생성·추정·학습 비용.
더 나은 방법이 조건별로 달라지면 그 경계를 결과로 삼는다.
N01보다 구현 변경이 크므로 독립 구현의 결과는 별도 대조로 보고하고, H100 파일럿 비용을 먼저 확인한다.

## N07. 난도·다양성·재노출의 영향을 분리하기

**참고:** [Lev의 혼합 구성 통제](https://arxiv.org/html/2609.32484v1),
[FAC의 무작위 feature 대조](https://arxiv.org/html/2602.10388v4),
[Saturated Data의 신호 크기와 내용 대조](https://arxiv.org/html/2609.33126v1).

**1단계, 관측:** 기존 선택 이력에서 고유 문제 수, 재선택 빈도, cached/current 성공률,
실제 학습의 mixed-reward group 비율, 문제 유형과 길이를 비교한다.
MATH의 알려진 topic·difficulty를 활용하고 MBPP에 없는 라벨을 임의로 있다고 가정하지 않는다.
캐시 성공률은 현재 난이도의 정답이 아니다.

**2단계, 개입:** 관측 차이가 있는 축 하나만 고정한다. 예를 들어 현재 성공률 구간별 선택 수를
맞춘 뒤 gradient 상위 선택과 구간 내부 무작위 선택을 비교한다. 다른 대조에서는 재노출 횟수나
주제 구성을 맞춘다. 모든 축을 한꺼번에 맞춰 비교 가능한 후보가 사라지는 설계는 피한다.
같은 checkpoint·optimizer에서 같은 크기의 batch로 짧게 학습한다.

**측정·판정:** 조건을 맞추기 전후의 reward 증가 차이, 실제 group 신호, 선택 분포와 비용.
난도를 맞추어도 gradient 이득이 남으면 난도 외 정보의 기여를 지지한다.
차이가 줄어들면 그 축이 유력한 설명이지만 전체 효과의 유일한 원인으로 단정하지 않는다.
Matching에 성공하지 못한 비율과 제외된 후보도 보고한다.

**기존과 차이:** 현재 shuffle은 점수-문제 연결 전체를 끊는다.
이 실험은 난도나 노출량 같은 한 축을 맞춘 상태에서도 추가 방향 정보가 유효한지 본다.
Current 성공률을 맞추기 위한 probe 비용은 진단 비용이며 배포용 selector 비용과 분리한다.

## N08. 평균 뒤의 문제별 개선과 해결 범위

**참고:** [Sampling SFT의 pass@k·능력 보존 분석](https://arxiv.org/html/2610.02140v1),
[Quality-Utility의 지표와 실제 효용 구분](https://arxiv.org/html/2606.16152v1).

**질문:** 비슷한 평균 reward가 같은 문제를 잘 푼다는 뜻인가?
Switch가 이미 풀던 문제의 성공 확률만 올리는지, 풀 수 있는 문제 범위도 바꾸는지 확인한다.

**설계:** 동일 평가 문제에서 방법별 정답률 변화를 짝지어 비교하고,
개선된 문제·퇴행한 문제·변화가 작은 문제의 분포를 보여 준다.
문제당 원시 binary 응답 결과와 sampling 조건이 보존돼 있으면 같은 응답 예산의
평균 정답률과 pass@k를 별도로 계산한다. 집계값만으로 지원하지 않는 k의 결과를 만들지 않는다.
필요한 원시 기록이 없으면 새 학습 대신 보존 checkpoint의 평가만 추가한다.

**측정·판정:** 문제별 성공률 변화, 주제별 손익, 관측 가능한 k 범위의 해결 비율.
기존 평가와 분리된 OOD 평가를 확장할 경우 모든 방법을 같은 조건으로 평가한다.
유한 횟수의 실패는 능력이 없다는 증명이 아니며, 작은 차이의 문제를 확정적인 획득·망각으로 분류하지 않는다.
반복 응답의 표본 오차와 training-seed 변동을 구분한다.

**역할:** 새 학습을 많이 늘리지 않고 성능 차이를 설명하는 보조 분석이다.
Switch의 학습 단계별 원인을 이 분석만으로 입증하지는 않는다.

## 공통 비용·평가 규칙

- 비교의 기본값은 기존 모델·GRPO·seed 5-9·MATH/MBPP다. 새 seed 반복을 먼저 늘리지 않는다.
  Pilot seed·상태는 실행 전에 정하고 유리한 결과만 골라 본 실험에 포함하지 않는다.
- 학습 상태를 분기할 때 모델뿐 아니라 optimizer 상태·결정적 rollout seed·후보/동점 순서를 보존한다.
  기존 constant-learning-rate GRPO를 유지하며 별도 scheduler를 추가하지 않는다.
  학습용 rollout은 각 분기의 현재 policy에서 새로 생성한다. 진단용 고정 응답과 구분한다.
- 목표 성능, 평가 간격, 비교 예산은 결과를 보기 전에 정한다. 최종 test로 threshold를 조정하지 않는다.
  목표 미도달과 예산 미완료는 그대로 표시한다. 기록 없는 시점의 성능을 임의 보간하지 않는다.
- SR cache 최초 생성·채점·저장, 후보/query scoring rollout, gradient/feature 계산,
  ranking, 실제 학습, 평가, checkpoint I/O의 계측 범위를 명시한다.
- Cold-start 총비용과 이미 준비된 cache 재사용 비용을 나란히 보고한다.
  공유 준비 비용을 실행 자원 집계에서는 한 번만 세고, 단독 배포 비용에는 필요한 준비를 포함한다.
  포함 관계인 phase와 세부 stage를 중복 합산하지 않는다.
- 배포용 실행은 단일 reference를 쓴다. 별도 B·centering·optimizer probe·reference 반복은
  연구 진단 비용으로 구분하며, 진단 시간이 무료라는 뜻은 아니다.
- GPU 수와 점유 시간, wall-clock과 GPU-hours를 구분한다. H100 실측 전 예상 완료 시간을 단정하지 않는다.
  누락된 timer는 0이 아니며 평가 완료와 비용 계측 완료도 별개다.
- 데이터셋 안에서 같은 seed의 방법 간 차이를 먼저 계산한다. 같은 seed 번호라고 MATH와 MBPP를
  대응 표본으로 취급하지 않는다. 문제/응답 반복을 독립 training seed로 세지 않는다.

## 기존 실험과 변경 범위

- [진행 중인 mechanism E14-E16](STAGE_MECHANISM_EXPERIMENTS_2026-10-06.md)은 유지한다.
  Shuffle, fresh-SR, 단계별 실제 학습 효과를 새 아이디어로 다시 실행하지 않는다.
- Qwen 모델 확장, matched cache refresh, 재전환 등은 [기존 실행 안내](LIMITATION_EXPERIMENTS_KO.md)에 있다.
  N01-N08과 같은 이름이나 완료 상태를 붙이지 않는다.
- Fixed200 채택 제외와 동일-prefix replicate의 보조 위치는 유지한다. 원본 결과는 삭제하지 않는다.
- 신규 코드는 별도 실험 식별자·결과 경로에 구현했다. 기존 환경·queue·cache·checkpoint와
  V6/V7 원고·PDF·웹 게시본은 변경하지 않는다.
- CPU 검증과 실제 GPU 실험 완료를 구분한다. 원고에 신규 성능·시간 값을 넣은 상태가 아니다.

---

<a id="followup-contract"></a>

## N12–N18의 근거와 실행 전 확인

**점검한 출처.** 코드 저장소의 확인 기준은
[`2cc2f9e`](https://github.com/33modeling/offpolicy-misranking/tree/2cc2f9ee3bb14e708622d4bf9c0fef5522a2f4ca)다.
[기존 한계 대응 E01–E16](LIMITATION_EXPERIMENTS_KO.md),
[전체 실험 명세와 Pair](EXPERIMENTS_COMPLETE_GUIDE_KO.md),
[N09–N11](DATA_ROLLOUT_INTERVENTIONS_2026-10-07.md),
[Information t0 결과](INFORMATION_RESULTS_2026-10-10.md),
[저장 rollout 분석·기록 명세](STAGE_ROLLOUT_ANALYSIS_KO.md),
[현재 선별·학습 구현](../srgc_rebuttal/srgc.py)을 읽었다.
원고는 GitHub와 일치하는 V7
[`01c105e`](https://github.com/33modeling/offpolicy-misranking-paper-v2/tree/01c105e6f9452a91d44a794f39b5858f9777912c/v7)의
[SR-GC 정의](https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/01c105e6f9452a91d44a794f39b5858f9777912c/v7/overleaf/sections/sr_gc_method.tex),
[한계](https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/01c105e6f9452a91d44a794f39b5858f9777912c/v7/overleaf/sections/discussion.tex),
[결론](https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/01c105e6f9452a91d44a794f39b5858f9777912c/v7/overleaf/sections/conclusion.tex)을 확인했다.

V7에는 Qwen MATH seed 5–9 endpoint가 반영돼 있지만 Qwen MBPP는 결과 미수신으로
기록돼 있다. 저자가 말한 MATH의 이른 전환·MBPP의 늦은 전환은 **검증할 관측·가설**이다.
Qwen MBPP 원본과 중간 곡선을 받기 전에는 두 과제의 전환 차이를 확인된 결과로 쓰지 않는다.
Information의 현재 수신 결과는 OLMo t0 진단이다. 그 결과에서 cosine과 실제 probe loss
변화가 일치하지 않았으므로 cosine·mixed 비율 하나로 초반 우위의 원인을 확정하지 않는다.

| 항목 | 이번 추가 설계의 기본값·조건 |
| --- | --- |
| 범위 | 먼저 OLMo MATH seed 5의 자료·실행 가능성을 점검한다. 이 pilot은 탐색으로 표시하고, 본 비교로 확장할 때 동일 설계의 seeds 5–9 전부를 보고한다. 모델·과제 확장은 별도 cohort로 둔다. |
| 주 경로 | 검증된 공통 t25 이후 total t275까지의 protocol을 우선 재사용한다. N01–N08의 별도 t0/100/400 경로와 합치지 않는다. |
| 상태 진단 시점 | 기본 t25/125/250. 결과를 보고 유리한 시점으로 교체하지 않으며, 별도 전환 전후 탐색은 사후 분석으로 표시한다. |
| 분기에 필요한 상태 | 실제 가중치와 optimizer moments·step, selector 상태, sampling/RNG·동점 순서, plan/input/runtime hash를 함께 확인한다. 평가용 weights-only `.pt`는 학습 재개의 대체물이 아니다. optimizer가 없으면 `MISSING_OPTIMIZER`로 두고 해당 분기를 보류한다. |
| pool | 해당 실행의 input manifest에 저장된 candidate ID 전체를 사용한다. 과거 OLMo MBPP의 512개와 현재 다른 runner의 pool 크기를 일괄 400개로 가정하지 않는다. candidate/reference/held-out ID는 분리한다. |
| 후보와 학습 | 별도 개입이 없는 한 후보는 40개, 실제 학습은 4문제×8개 fresh 응답이다. 후보 추출·문제 유지·재추출 시점은 각 항목에 적는다. 과거 응답 재사용은 진단으로 구분한다. |
| 중간 기록 | 모든 신규 학습 arm/model에서 매 update 학습 배치 보상과 최소 25 updates마다 고정 held-out 보상·재평가 가능한 checkpoint를 보존한다. 분기 시작/종료도 평가한다. 기존 기록 주기가 다르면 섞지 않고 그대로 표시한다. |
| 원시 기록 | 후보·선택·reference·평가 ID, 응답별 reward/token·sampling seed, 사용한 gradient/vector, 종료 이유와 비용 receipt의 출처를 남긴다. 재시도·공통 prefix를 중복 집계하지 않으며 학습 보상과 held-out 보상을 별도 내보낸다. |
| 누락 처리 | compact export만 보고 미저장으로 판정하지 않는다. 원래 progress/history·rollout JSON.gz·checkpoint를 확인한다. 없는 평가·vector·log-probability는 `null`/미측정이며 보간하지 않는다. |
| 자원·비용 | 아래 update/응답 수는 설계 수량이며 GPU-hours 예측이 아니다. 실제 생성·scoring gradient·학습·평가·cache·저장 비용을 분리하고, 계측이 겹치는 phase/stage를 더하지 않는다. |
| 실행 상태 | N12·N13은 `srgc_research/followup/`에 CPU 분석·checkpoint 감사·추론/scoring을 구현했다. 신규 GPU 실측은 미실행이다. N14–N18은 미구현 설계다. 진행 중인 MATH→MBPP 작업·frozen runtime은 변경하지 않는다. |

<a id="n12"></a>

## N12. 데이터 구성 변화와 같은 문제의 학습 변화를 분리하기

**구현:** [N12·N13 실행 안내](N12_N13_CHECKPOINT_DIAGNOSTICS.md). `n12-report`, `n12-panel`,
`n12-panels`, `collect --experiment n12`를 사용한다. optimizer 없는 평가 archive도 진단 가능하다.

**질문:** 후반 mixed-reward group이 줄거나 늘어나는 이유가 다른 문제를 골랐기 때문인가,
같은 문제에 대한 정책의 응답 분포가 변했기 때문인가? MATH와 MBPP의 전환 차이가
어떤 데이터 구성 변화와 함께 나타나는가?

**먼저 저장 자료로 하는 분석.** 각 arm의 실제 학습 group을 25-update 창으로 나눈다.
초기 cache의 정답 수 0/8, 1–3/8, 4/8, 5–7/8, 8/8을 고정 strata로 사용한다.
이는 초기 정책의 유한 표본 성공률이며 실제 난도 label이 아니다. 한 문제를 반복 학습한
group도 실제 노출 횟수로 세되 고유 문제 수는 따로 보고한다.

각 stratum의 mixed 비율을 구한 뒤, 해당 seed의 **전체 candidate pool에서 사전에 정한
stratum 비중**으로 가중한 표준화 mixed 비율을 계산한다. 실제 선택 비중으로 계산한
원래 mixed 비율과 나란히 비교한다. pool 비중이 양수인 stratum의 학습 기록이 없으면 전체 표준화 값은
미측정으로 두며, 남은 strata만 재정규화해 완전한 비교처럼 제시하지 않는다.
성공률·응답 길이에도 같은 비교를 적용하고 원본 topic/level이 있을 때만 해당 축을 추가한다.

**같은 문제를 보는 보완 진단.** 기록만으로 문제 구성과 상태 변화를 구분하기 어려울 때는
candidate pool에서 진단용 20문제를 균등·중복 없이 **한 번** 뽑아 ID를 고정한다.
각 arm의 t25/125/250 정책에서 같은 20문제에 각각 8개의 새 응답을 생성한다.
20문제는 학습 pool 안의 진단 패널이며 고정 held-out 평가셋이 아니다. 이 응답·결과를
선별이나 학습에 사용하지 않는다. 시점마다 문제를 다시 뽑지 않고 응답만 새로 뽑는다.
공통 t25가 같은 정책이면 측정은 한 번만 수행·계상한다. 이 단계는 학습 없이 추가 추론이 필요하다.

**출력·판정.** 원래/표준화 mixed 비율과 정답 수 histogram, 고유 문제·재노출 수,
같은 문제의 성공률·길이 변화, 실제 held-out 곡선을 별도 패널로 만든다.
표준화 후 차이가 줄면 초기 성공률 구성의 기여를 시사하고, 고정 패널에서도 변하면
같은 문제의 정책 변화도 관측된 것이다. 두 결과 모두 관찰 근거이며 초기 성공률 bins
안의 차이까지 통제하거나 인과를 증명하지 않는다.

**기존과 차이·비용.** N07과 기존 `windows25.csv`의 구성 집계를 재사용하되
고정 비중 표준화와 고정 문제 패널을 추가한다. CPU 1단계는 새 생성·학습이 없다.
보완 패널의 추가 응답 수는 `20×8×실제로 다른 정책 checkpoint 수`이며 별도 계측한다.

<a id="n13"></a>

## N13. SR-GC의 40개 평균과 실제 선택 4개는 같은 판단을 주는가?

**구현:** [N12·N13 실행 안내](N12_N13_CHECKPOINT_DIAGNOSTICS.md). `n13-report`,
`collect --experiment n13`를 사용한다. 저장 projected vector는 CPU 재계산하고,
vector가 없으면 검증한 기존 weights에서 새 scoring을 별도 비용으로 측정한다.

**질문:** 후보 40개 평균이 SR 비교 40개보다 불리해졌다는 신호가,
그 후보에서 gradient로 고른 상위 4개도 불리하다는 뜻인가?
V7은 두 대상이 다르다고 명시하지만 그 차이의 크기는 아직 직접 비교하지 않았다.

**동일 상태·동일 후보 진단.** 실제 check의 후보 `C40`, SR 비교 `S40`, reference ID와
문제별 projected gradient `g`, 평균 reference gradient `v`를 사용한다.
같은 `C40`에서 기존 cosine 상위 4개 `ON4`와 기존 cached-SR 순서의 상위 4개 `SR4`를 고른다.
동점 순서는 해당 실행의 기록을 따른다. 후보를 새로 뽑거나 SR4를 전체 pool 상위 4개로 바꾸지 않는다.

```text
D40 = dot(v, mean(g[C40]) - mean(g[S40]))   # 원래 SR-GC
D4  = dot(v, mean(g[ON4]) - mean(g[SR4]))   # 같은 후보의 선택 배치 진단
```

`S40`은 실제 진단 집합 그대로이며 `SR4`와 동일한 대상이 아니다.
현재 구현의 SR 비교 집합은 이미 학습 배치에 사용된 ID 이력을 고려하지만,
실제 SR 학습 배치는 그 update의 후보 40개 안에서 cache 순서로 4개를 고른다.
따라서 진단 집합·학습 후보·학습 사용 이력을 각각 보존한다. 진단 preview로 학습 이력을 소비하지 않는다.

**언제·무엇을 재사용하는가.** 우선 기존 check 전부에서 자료 coverage를 점검한다.
원시 projected vector가 보존돼 있으면 CPU에서 재계산할 수 있다. 저장된 `d`, cosine,
reward, token ID만으로는 D4를 복원할 수 없다. vector가 없으면 t25/125/250의 검증된
정책으로 별도 scoring 진단을 실행하고 새 측정으로 표시한다. reference는 실행당 원래 50개
하나이며 새 reference 집합을 만들지 않는다. 원래 응답을 정확히 재사용한 경우와
새 응답으로 재측정한 경우는 별도 결과다.

독립 scoring B가 있다면 A로 고른 **같은 ON4/SR4**를 B의 gradient로 다시 비교한다.
B에서 다시 top-4를 골라 A의 선택을 검증한 것처럼 쓰지 않는다. 동일 상태의 실제
1-update 분기·독립 평가가 이미 있다면 연결하고, 없으면 효용의 직접 측정은 미확인으로 둔다.
25-update 분기 결과와의 관계는 더 긴 경로에 대한 보조 분석으로 분리한다.

**출력·판정.** D40/D4 부호 일치·불일치 수, ON4/SR4 ID overlap, A/B 차이,
전환 시점 주변의 실제 평가 변화, 유효 비교 수를 출력한다.
D40<0인데 D4>0인 사례가 많으면 후보 평균과 선택 배치 사이의 간극을 보여 준다.
D4가 더 잘 맞아도 AdamW의 실제 update나 장기 reward 예측력이 입증된 것은 아니다.
**기존 SR-GC 정의나 진행 중인 Switch의 규칙을 D4로 교체하지 않는다.**

**기존과 차이·비용.** N02는 실제 optimizer 이동, E16은 선택 점수의 재현성을 본다.
여기서는 SR-GC가 비교하는 집합과 학습에 쓰는 집합의 차이를 직접 검증한다.
새 scoring이 필요하면 후보/SR 합집합과 reference의 생성·backward 비용을 계상한다.

<a id="n14"></a>

## N14. 부호 변화가 rollout 잡음인가, 반복되는 상대 우위 변화인가?

**질문:** SR-GC가 0 근처에서 바뀔 때 실제 상대 우위가 변한 것인가,
후보 응답과 reference 응답의 재추출만으로도 그 정도 변동이 생기는가?
기존 t0 Information 결과는 A/B에서 둘을 함께 재추출했으므로 두 잡음원을 분리하지 못했다.

**학습 없는 1단계.** t25/125/250의 같은 정책에서 후보 C40, 실제 SR 비교 S40,
reference 50개 ID를 고정한다. 후보/SR 합집합의 독립 응답·gradient를 A1/A2로 두 번,
같은 reference의 독립 응답·gradient를 B1/B2로 두 번 얻는다. 문제당 응답은 매번 8개다.
같은 acquisition 안에서 겹치는 후보/SR 문제는 한 번 계산하여 재사용한다.
`D(A1,B1)`, `D(A1,B2)`, `D(A2,B1)`, `D(A2,B2)`를 계산하여 각각 어느 재추출에서 부호가 바뀌는지 본다.
네 조합은 gradient를 공유하므로 독립 반복 네 개나 training seed 네 개로 세지 않는다.
문제 ID와 reference 구성은 바꾸지 않는다. N04의 reference 집합 교체와 다른 진단이다.

**불확실성 규칙으로 확장할 조건.** 부호 변동이 확인되고 원시 vector가 있을 때만
문제 단위 재표집을 이용한 조건부 구간을 설계한다. 후보/SR overlap의 기여 상쇄와
reference가 공통 방향이라는 의존 구조를 보존한다. 재표집 1,000회·구간 수준·처리할
퇴화 사례를 개발 자료에서 고정한다. 이는 측정된 문제/응답에 조건부인 구간이며
두 번의 acquisition만으로 명목 coverage가 검증됐다고 주장하지 않는다.

실제 새 학습 비교를 한다면 **두 연속 check에서 D<0인 `switch_consecutive`**와
**두 연속 check에서 구간 상한<0인 규칙**을 주 비교로 둔다. 둘 다 check 간격 25와
40대40·단일 reference를 유지해 temporal rule까지 동시에 바뀌지 않게 한다.
기존 Switch는 보조 대조로 둔다. threshold와 구간 수준을 최종 held-out reward에 맞춰 고르지 않는다.
한 경로의 D를 replay하여 다른 규칙의 최종 reward를 추정하지 않고, 각 규칙을 자기 경로에서 실행한다.

**출력·판정·비용.** 후보/reference 재추출에 따른 부호 변동, 실제 전환 지연,
미전환 수, 최종 reward와 추가 scoring 비용을 함께 보고한다.
불확실성 규칙이 늦게 전환해 성능·비용이 나빠지는 경우도 결과다.
진단용 A/B 비용과 배포 규칙에 필요한 획득·재표집 비용을 구분한다.
기존 E12의 단일/연속 음수 실험을 이미 신뢰구간 규칙의 검증이라고 부르지 않는다.

<a id="n15"></a>

## N15. On-policy로 고른 4개를 25 updates 유지하는 동안 순위가 낡는가?

**질문:** 후반 On-policy의 이득이 줄어드는 이유 중 하나가 25-update 동안 같은
선택 문제를 유지하는 운영 방식인가? 초기 SR cache의 노후화와 별개의 질문이다.

**추출·분기.** 우선 공통 t25에서 후보 40개를 한 번 뽑는다. 같은 모델·optimizer와
같은 후보 ID에서 두 arm을 각각 25 updates 실행한다. t125/t250 확장은 완전한 optimizer
상태가 있을 때만 같은 설계로 한다. 각 branch 동안 후보 40개는 그대로 유지한다.

| Arm | 문제 선별 시점 | 실제 학습 |
| --- | --- | --- |
| Hold-25 | 분기 시작 때 cosine top-4를 한 번 선택 | 같은 4문제로 25 updates, 매 update 새 8응답 |
| Refresh-5 | 분기 내부 update 0/5/10/15/20 직전 현재 정책으로 같은 40개를 재평가 | 각 5-update 구간의 top-4로 학습, 매 update 새 8응답 |

둘 다 reference 50개 ID·학습률·objective·학습량을 유지한다. Refresh-5는 매 refresh
scoring 응답을 새로 생성한다. 후보까지 새로 뽑지 않아 후보 접근 범위 확대를 섞지 않는다.
이는 **후보를 고정한 국소 cadence 진단**이며 기존 전체 On-policy 경로를 그대로 재현한 것은 아니다.
Hold-25의 중간 순위도 측정하려면 별도 진단으로 기록하며 그 비용을 운영비에 숨기지 않는다.

**출력·판정.** 선택 ID turnover, 시작 top-4의 이후 점수, 실제 학습 mixed 비율,
분기 전후 고정 held-out reward, 생성 token과 선별/학습 비용을 비교한다.
Refresh-5가 좋아지면 선택 유지 간격이 해당 상태의 성능에 기여한다는 근거다.
순위만 바뀌고 reward가 같으면 자주 재선별해야 한다는 근거가 아니다.
SR의 cache 갱신 효과나 더 싼 GPU 실행이라고 해석하지 않는다.

**규모.** seed·anchor당 두 arm 합계 50 branch updates다. 첫 pilot은 t25 하나이며,
세 anchor로 확장하면 150 updates다. 추가 scorer·평가 비용은 이 수량에 포함되지 않는다.
N05의 후보/batch 규모 변화 및 E05/E07/E10의 SR 갱신과 구분한다.

<a id="n16"></a>

## N16. On-policy와 SR 사용량이 같아도 적용 순서가 중요한가?

**질문:** Switch의 성능이 두 baseline 보상의 평균처럼 결정되는가,
앞선 학습 경로가 뒤의 selector 효용을 바꾸는가?
먼저 기존 곡선의 같은 측정 step에서 `Switch−On-policy`, `Switch−SR`를 표시한다.
두 baseline 사이에 있다는 수학적 제약은 없으며, 관측된 범위를 넘었는지는 실제 값으로만 판정한다.

**추가 개입.** 같은 seed의 검증된 t25 모델·optimizer에서 아래 두 경로를 실행한다.

| 경로 | updates 26–150: 125회 | updates 151–275: 125회 |
| --- | --- | --- |
| ON→SR | On-policy | SR |
| SR→ON | SR | On-policy |

두 경로 모두 On-policy 125회, SR 125회다. On-policy는 25 updates마다 40개를
새로 뽑아 top-4를 유지하고, SR은 매 update 후보 40개를 새로 뽑아 cached-SR top-4를 고른다.
어느 구간이든 실제 학습은 매 update 4×8 fresh 응답이다. 전환 시 모델·optimizer를
유지하고 재초기화하지 않는다. SR→ON은 새 대조 arm이며 기존 단방향 Switch를 변경한 것이 아니다.
SR-GC로 시점을 정하지 않으며 기존 cache와 원래 reference를 사용한다.

**비교·판정.** 같은 protocol의 계속 On-policy·계속 SR·adaptive Switch 결과가 있다면
같은 종료점/중간 평가에서 나란히 비교한다. 순서 간 차이는 사용량이 같아도 경로가
중요하다는 근거다. 같은 문제·응답 노출이나 같은 GPU 비용까지 맞춘 것은 아니다.
OP/SR 순서가 선택 데이터와 정책을 함께 바꾸므로 순수 optimizer 효과나 양의 시너지로 단정하지 않는다.
Switch가 두 baseline보다 높다는 endpoint만으로 적절한 전환 신호의 기여까지 입증하지 않는다.

**규모·기존과 차이.** 새 두 경로는 seed당 500 continuation updates다.
E11은 ON→SR의 시점을 바꾸며 두 방법의 사용량도 바뀐다. 이 설계는 사용량을 고정한
순서 대조다. 기존 fixed150과 ON→SR이 시작 상태·runtime·학습 절차까지 같으면 그 경로는
검증 후 재사용한다. `fixed200`은 재도입하지 않는다.

<a id="n17"></a>

## N17. 다른 모델·과제에서 정한 전환 시점도 그대로 통하는가?

**질문:** 한 과제에서 좋은 전환 시점을 알아도 다른 과제나 backbone에 그대로 적용할 수 있는가?
상대 우위의 시점을 미리 알기 어렵다는 V7 결론을 **cohort 사이의 전이**로 검사한다.

**source에서 한 번 선택.** 기존 채택 대상 fixed50/100/150/250 중 source cohort의
평균 최종 reward가 가장 높은 시점 하나를 고른다. 동률은 이른 시점으로 정한다.
source에 채택한 grid가 빠져 있으면 `INCOMPLETE_SOURCE_GRID`로 두며 최선값을 지어내지 않는다.
선택 기준·source 파일 hash·선택한 시점·동결 시각을 target 분석 전에 저장한다.

**target에서 변경 없이 적용.** 1차 비교는 같은 모델의 MATH→MBPP이고,
2차는 같은 과제의 OLMo→Qwen/Llama/Gemma다. 한 번에 모델과 과제를 모두 바꾸지 않는다.
target 자체의 t25 prefix·cache·reference·입력 pool·token 설정을 사용하되 source의
전환 update만 가져온다. target reward를 보고 시점을 다시 고르지 않는다.
target에서 그 시점으로 실제 학습한 결과와 기존 adaptive Switch를 비교한다.
해당 fixed 경로가 없으면 별도 실행이 필요한 항목으로 남기며 두 baseline 곡선을 이어 붙이지 않는다.

**출력·판정.** source에서 고른 시점, target의 실제 Switch 경계, 방법별 최종 reward·
실측 비용, 같은 target seed의 paired 차이를 표로 만든다.
전이한 일정이 나빠지고 adaptive가 유지되면 그 source/target 범위에서 적응 판단의 가치를 지지한다.
고정 시점도 잘 통하면 그 사실을 보고한다. 이미 target 결과를 본 경우는 사후 전이 분석으로
표시하고 새로운 독립 검증이라고 부르지 않는다. Qwen MBPP 등 미수신 자료는 대기 상태다.

**기존과 차이·비용.** E13은 같은 cohort 안의 leave-one-seed-out이다.
여기서는 과제 또는 모델 전체를 target으로 남긴다. 실제로 맞는 fixed 결과가 모두 있으면
추가 학습 없이 분석할 수 있지만, 누락 경로가 있으면 그 수와 비용을 별도로 산정한다.
다른 모델의 절대 GPU 시간을 같게 만들거나 실행 속도 차이를 가상으로 빼지 않는다.

<a id="n18"></a>

## N18. 학습 중 문제 분포가 바뀌어도 전환 전략이 유효한가?

**질문:** 기존처럼 같은 후보 모집단을 유지할 때뿐 아니라, 학습 도중 사용할 수 있는
문제가 바뀔 때도 단방향 Switch가 유효한가? 다시 On-policy가 필요한 상황도 있는가?

**문제 집합을 미리 고정.** 같은 과제의 원래 candidate pool을 초기 cache 성공률이
낮은 절반 L과 높은 절반 H로 나눈다. 동점은 사전 고정 순서로 처리하고 실제 histogram을 저장한다.
두 집합은 겹치지 않으며 각각 최소 40개여야 한다. 초기 성공률의 차이를 조작하는 것이지
현재 모델의 정확한 난도를 안다고 가정하지 않는다. prompt 길이·실제 topic 구성이 다르면
함께 보고하고, matching에 실패한 조건을 난도만 통제한 비교라고 부르지 않는다.

**분포를 바꾸는 시점.** 공통 t25에서 시작하여 updates 26–150에는 L, 151–275에는 H만
후보로 허용하는 L→H와 반대 H→L을 각각 만든다. 두 phase는 125 updates씩이다.
각 phase 안에서 후보 40개는 기존 간격/규칙으로 새로 추출하며, 경계 직후 후보·active selection을
새 모집단에 맞춰 갱신한다. 초기 cache는 새로 만들지 않고 해당 집합의 기록을 읽는다.
reference 50개와 고정 held-out ID는 전 구간 유지한다. SR 비교 40개도 현재 허용 집합에서 구성한다.

**대조군.** 각 분포 순서에서 계속 On-policy, 계속 SR, 기존 단방향 Switch,
E06 계열의 `switch_repeat`를 별도 입력·실험 ID로 비교한다.
On-policy와 전환 전 Switch의 기본 재선별/check 간격은 25, 학습은 4×8 fresh 응답이다.
재전환 arm은 자기 경로에서 SR 이후에도 정해진 check를 실제 수행하고 비용을 기록한다.
현재 E06의 전환·확인 규칙을 명시하여 재사용하되 새 분포에서 실제 실행해야 한다.
과거 E06 결과를 새 입력의 결과로 복사하거나 그 queue를 덮어쓰지 않는다.

**출력·판정.** 분포 경계 전후의 선택 ID·현재 성공률·mixed group·SR-GC,
고정 held-out reward와 재적응 지연, 전환 횟수·전체 비용을 비교한다.
단방향 Switch가 새 분포에서 회복하지 못하고 재전환이 도움이 되면 단방향 규칙의 한계다.
재전환 비용만 늘면 단방향 유지의 실용적 근거가 된다. Switch가 항상 좋아야 한다는
판정 기준이나 실패 사례를 제외하는 규칙을 만들지 않는다.

**규모·위치.** 두 분포 순서×네 arm×250 continuation updates로 seed당 2,000 updates다.
학습량이 큰 후속 개입이므로 N12–N17의 결과와 자원 여유를 확인한 뒤 검토한다.
N09의 같은 checkpoint·난도 matched 비교, 기존의 정적인 MATH/MBPP mixed-pool과 다른 실험이다.
새 manifest와 별도 결과 경로에서만 설계하며 현재 학습 입력은 수정하지 않는다.
