# Switch 추가 실험: 전환 시점과 반복 확인의 기여

작성: 2026-10-06. 코드 `master`. **계획·구현 기록이며 신규 GPU 결과가 아니다.**
기존 MATH/MBPP P0 seeds 5-9 완료 기록은 유지한다. 원격 노드의 현재 진행 상태는
이번 작업에서 조회하지 않았으므로, 아래 개수를 현재 미완료 개수로 해석하지 않는다.

전체 순서는 [실험 목록](LIMITATION_EXPERIMENTS_KO.md#우선순위)을 따른다.
E11 고정 전환과 E12 규칙 대조는 V6의 timing·temporal confirmation 한계에 직접 대응한다.
기존 실행은 유지하며, 별도 요청된 메커니즘 실험과 구분한다.

## 1. 무엇을 검증하는가

Switch의 질문은 좋은 선별 점수가 실제 학습 이득을 계속 보장하는지, 언제 전략을
바꾸는 편이 성능과 계산 비용에 유리한지다. Quality-Utility Paradox와는 점수와 실제
효용의 구분, TESSY/Echo-GRPO와는 학습자에 따른 효용이라는 동기로 연결된다.
그 논문들의 response 재작성 방법을 이번 prompt-selection 실험에 섞지 않는다.
Echo의 clipping 원인을 Switch의 원인으로 가정하지 않는다.

| ID | 질문 | 조건 | MATH + MBPP 작업 수 |
| --- | --- | --- | ---: |
| E11 | SR-GC 전환이 고정 일정보다 유리한가 | `switch_fixed50/100/150/200/250` vs 기존 `switch` | 50 |
| E12 | 반복 확인과 현재 규칙의 예외 분기가 필요한가 | `switch_single`, `switch_consecutive` vs 기존 `switch` | 20 |
| E13 | 임의 시점이나 다른 시드로 정한 시점과 비교하면 어떤가 | 균등 시점 기대값, LOSO 고정 일정 | 추가 GPU 0 |

새 validation queue는 E11+E12 **70개**다. 기존 fixed200 10개가 포함되며 검증된
완료 결과는 다시 학습하지 않는다. 따라서 기존 전체 계획에 없는 고유 조건은
fixed50/100/150/250와 두 rule, **60개**다. 기존 support 30개도 중복 생성하지 않는다.
기존 120개 전체 queue와 support 30개 queue의 구성·순서는 변경하지 않았다.
독립 반복·SR refresh·다른 모델의 기존 계획을 취소하거나 완료로 바꾸지 않는다.

## 2. 고정 조건과 정확한 절차

- 데이터셋: MATH와 MBPP를 별도 집계한다. 학습 seed 5, 6, 7, 8, 9를 모두 사용한다.
- 모델·revision·LoRA·GRPO·학습률·token budget·채점기·평가셋은 각 frozen plan/input을 따른다.
- 각 seed의 검증된 **total step 25 prefix**에서 optimizer와 sampling 상태까지 복구하여
  total step 275까지 이어간다. P0/prefix/cache를 새로 만들지 않는다.
- On-policy 구간은 25-update마다 기존 pool에서 후보 40개를 뽑아 현재 정책 응답과
  gradient로 상위 4개를 선별하고 다음 refresh까지 유지한다. 학습 rollout은 매 update 새로 생성한다.
- 각 문제의 응답 수는 8개다. 후보 집합과 사용 문제 제외·pool 순환 방식은 기존 엔진을 유지한다.
- SR-GC는 후보 40개와 SR 비교 40개를 비교한다. 학습 batch 4개와 혼동하지 않는다.
  ranking reference는 기존 단일 집합 하나이며 A/B 두 reference를 추가하지 않는다.
- SR로 전환한 뒤의 후보 추출·상위 4개 선택은 기존 Switch와 같다. 전환 후에는
  fresh 선별 gradient와 SR-GC 점검을 중단한다. 학습용 rollout/backward는 계속한다.
- 동일 seed는 같은 prefix·추출 규칙·난수 stream의 쌍이다. 정책이 갈라진 이후에도
  생성한 답변이나 학습 문제 이력이 동일하다고 가정하지 않는다.

이 설계는 **step 25 이후의 전환·유지 효과**를 검증한다. 초기 0-25에서 gradient
선별이 필수였다는 인과 주장을 이 결과로 만들지 않는다. 전체 학습의 출발점은 t0이며,
step 25는 이미 수행한 공통 경로를 재사용하는 분기점이다.
기존 학습 seed가 공유하는 데이터 split을 새로운 독립 데이터셋 반복으로 세지 않는다.

### E11: 고정 전환

전환 시점은 결과를 보기 전에 이 계획의 50, 100, 150, 200, 250으로 고정한다.
`switch_fixed100`은 update 100까지 On-policy, **update 101부터 SR**다.
다른 고정 시점도 같은 경계 의미다. D를 계산하거나 D를 보고 전환하지 않는다.
과거 `f581...` 엔진은 On-policy refresh에서도 SR 비교 집합의 gradient를 구한다.
그 prefix를 재사용하는 고정 대조도 해당 절차와 비용을 유지하며 최신 엔진과 합산하지 않는다.
기존 fixed engine의 경계 refresh를 그대로 실행·계측하므로, 미리 정한 시점의
마지막 선별을 생략해 최적화한 최소 비용으로 해석하지 않는다.
fixed200 원본 결과에 protocol 필드가 없으면 최종 progress의 실제 학습 selector
경계를 검사한 뒤 비교에 사용한다. 기존 결과를 새 의미로 재명명하지 않는다.

### E12: SR-GC 규칙 대조

| Arm | 전환 조건 | 역할 |
| --- | --- | --- |
| 기존 `switch` | 기존 temporal rule 그대로 | 수정하지 않는 주 방법 |
| `switch_single` | 자기 경로에서 처음 관측한 `D < 0` | 시간적 확인이 없는 대조 |
| `switch_consecutive` | 연속한 두 예정 점검에서 `D < 0` | 기존 규칙의 음수-비음수-음수 예외 분기를 제거 |

0은 음수가 아니다. 점검 간격이 빠지면 연속성은 초기화된다. 전환을 끝내 유발하지
않으면 On-policy로 끝까지 학습하고 전환 시점은 null로 남긴다. 다른 arm에서 얻은
D나 전환 시점을 가져오지 않는다. 자기 check 이력, rule protocol, window,
전환 step을 checkpoint/endpoint에 저장하며 다른 rule로 resume하는 것을 거부한다.

### E13: 추가 학습 없는 두 비교

1. **균등 시점 기대값:** 같은 seed에서 다섯 고정 시점 결과를 각각 확률 1/5로
   선택하는 정책의 reward/비용 기대값이다. 다섯 결과가 모두 있을 때만 계산한다.
   새로 실행한 Random arm, 평균 시점에서 실행한 모델, 추가 training seed가 아니다.
2. **LOSO 고정 일정:** 네 seed에서 최종 reward 평균이 가장 높은 고정 시점을
   고른 뒤 제외한 한 seed에서 Switch와 비교한다. 동률이면 이른 시점을 선택한다.
   제외한 seed의 결과는 시점 선택에 쓰지 않는다. 다섯 seed와 다섯 시점 전체가
   검증되어야 집계한다. reward로 선택하며 비용까지 사후 가중한 목적함수를 만들지 않는다.

LOSO는 기존 시드들에 대한 교차 검증 분석이지 새로운 독립 confirmation cohort가 아니다.
결과를 보고 고른 seed별 최적 시점과 Switch를 비교해 우위를 주장하지 않는다.
개별 고정 시점 결과와 전체 grid를 우선 보고, 두 파생 대조를 보조 분석으로 표시한다.

## 3. 평가·비용·판정

주 결과는 같은 종료 step의 독립 평가셋 reward다. 모든 seed의 원값, 같은 seed의
`Switch - comparator`, 평균, 표본 SD, 승/동률과 유효 쌍 수를 기록한다.
평가 응답들을 독립 학습 반복으로 세지 않는다. 누락은 0으로 채우지 않는다.
model/runtime generation과 attention이 맞는 결과만 쌍으로 비교하고,
implementation/attention이 다른 seed들은 합쳐 평균 내지 않는다.
원래 P0 endpoint에 attention 항목이 없으면 final checkpoint를 CPU memory map으로
읽어 복구한다. tensor를 GPU에 올리거나 checkpoint를 다시 쓰지 않는다. 최종 metadata도
없으면 reward 원값은 표시하되 검증되지 않은 쌍의 평균은 만들지 않는다.

비용은 횟수로 추정하지 않고 기존 동기화 phase/invocation receipts를 합산한다.

| 보고 성분 | 내용 |
| --- | --- |
| SR cache build | 최초 cache 생성의 inclusive 비용. 기존 cache를 읽고 정렬하는 prep와 구분 |
| Selection | 현재 정책 candidate/reference rollout, reward, gradient, cosine ranking, SR-GC 산술 등 계측된 선별 작업 |
| Training | 매 update의 학습 rollout/reward/gradient/optimizer |
| Preparation | selector 초기 설정과 cache 정렬 등 기존 계측 범위 |
| Evaluation | 보고용 독립 평가 비용, core 학습 비용과 별도 |
| Checkpoint/startup | checkpoint 읽기·쓰기, 모델 로드 등 |
| Core | selection + training + preparation |
| Continuation inclusive | continuation process의 계측 총합. 내부 phase 합을 다시 더하지 않음 |
| Protocol cold inclusive | continuation + 공통 prefix + SR을 쓰는 arm의 cache 생성 1회 |

Random/On-policy에 SR cache 생성비를 배정하지 않는다. 이 cold total은 **공통
On-policy prefix를 포함한 현재 실험 프로토콜**의 합계이며, t0부터 SR만 수행한
독립 전략 비용으로 부르지 않는다. inclusive에는 계측된 평가/시작/저장이 포함되지만
프로세스 밖 node admission이나 큐 대기는 포함되지 않는다.
cache가 외부에서 복사되어 최초 생성 receipt가 없으면 cold total은 null이다.
그때도 continuation 계측과 보상은 보고하며 캐시 미계측을 0초라고 쓰지 않는다.
실패·재시작 비용은 기존 ledger가 포함하고 미종료 timer가 있으면 완전한 총합으로 표시하지 않는다.
JSON에는 exclusive stage 시간·호출 횟수와 실제 selection/check 횟수도 남긴다.

해석은 결과에 따른다. Switch가 고정 일정들과 비슷하면 SR-GC가 특별히 우수한
전환 시점을 찾았다고 주장하지 않는다. single과 current가 비슷하면 반복 확인의
추가 기여가 입증되지 않은 것이다. 반대로 성능과 비용을 함께 개선하는 결과가
나오면 해당 protocol/도메인에서의 근거로 보고한다. 동일 update 비교를 동일 GPU
budget 실험이나 목표 성능 도달 시간 실험으로 바꾸어 부르지 않는다.

## 4. 실행·status·results

코드 저장소 루트에서, **빈 4-H100 노드마다 같은 명령을 한 번** 실행한다.
환경·모델·캐시 경로를 바꾸지 않는다. 저장은 기존 `/group-volume` 경로를 사용한다.

```sh
sh scripts/run_srgc_switch_validation.sh all timing
sh scripts/run_srgc_sr_refresh.sh all status timing
sh scripts/run_srgc_switch_validation.sh all results
```

`all` 대신 `math`/`mbpp`를 쓰면 해당 도메인만 처리한다. 시드 지정은 필요 없다.
`timing`은 고정 시점만, `rules`는 확인 규칙만 실행한다.
마지막 인자를 생략하면 두 대조를 합친 70개 queue를 실행한다. 필요한 scope를 명시한다.
`json`은 결과·원값·단계별 비용·검증 오류를 JSON으로 출력한다.
위 status는 고정 시점만 조회한다. results는 규칙 대조를 포함한 전체 validation을 보고한다.
`status/results/json`은 CPU 읽기 전용이며 학습, cache 생성, 잠금 변경을 하지 않는다.
진행 표의 `reported_running`은 heartbeat 기록이며 원격 프로세스 생존 확인은 아니다.

검증된 완료 결과는 skip한다. 중단된 작업은 같은 명령으로 model/optimizer/선별 상태를
자동 복구한다. 기존 1-update checkpoint와 scoring rollout cache를 그대로 사용한다.
새 retry/잠금/환경 구현을 만들지 않고 기존 task/GPU lease와 supervisor를 재사용한다.
정상 실행 중인 노드에 pull하거나 worker를 하나 더 띄우지 않는다.

`timing`의 노드 상한은 모든 prefix가 준비되고 미완료인 경우 **25노드/도메인, 양쪽 50노드**다.
규칙 대조까지 포함한 전체 queue의 상한은 35노드/도메인, 양쪽 70노드다.
4노드라면 각 노드에 같은 명령을 실행하여 4개씩 처리한다. 이는 작업 독립성의
상한이지 실제 필요 노드 수 또는 이 규모의 공유 볼륨/NCCL 부하 검증 결과가 아니다.
현재 남은 수와 소요 시간은 실제 status와 새 phase 로그를 확보한 뒤 산정한다.

MBPP의 과거 prefix는 그 당시 채점기 v2를 유지한다. 신규 v3 채점 결과와 섞지 않는다.
이 추가 대조가 과거 reward 채점의 타당성을 새로 인증하지는 않는다.

## 5. 코드 위치

- [Shell 진입점](../scripts/run_srgc_switch_validation.sh)
- [기존 자동 배정에 추가한 scope](../scripts/srgc_replicate_worker.py): `switch_validation`, `timing`, `rules`
- [고정 시점 engine](../scripts/srgc_switch_fixed.py): 기존 구현 재사용
- [새 temporal-rule engine](../scripts/srgc_switch_rules.py)
- [공통 학습·resume·계측 runner](../scripts/srgc_sr_refresh.py)
- [검증·시드 쌍·LOSO·비용 보고](../scripts/srgc_switch_validation_report.py)
- [새 회귀 테스트](../srgc_rebuttal/tests/test_switch_validation.py)
- [기존 support 계획](LIMITATION_EXPERIMENTS_KO.md)

V6/V7 TeX, PDF, 이미 받은 결과와 웹 게시본은 변경하지 않는다.
실제 GPU 완료 전에는 이 문서의 조건을 원고의 새 결과로 기입하지 않는다.

## 6. 검증 기록

2026-10-06 최종 코드 기준:

- 새 테스트 **22개 통과**. 규칙별 trigger, 음수-비음수-음수 구분, 중간 재개,
  전환 후 scoring 중단, 과거 두 runtime의 prefix 복구, 70개 중복 없는 배정,
  기존 120/30개 queue 보존, shell 인자, 잘못된 endpoint 제외, LOSO 누출 방지,
  최초 cache 단일 합산, 누락 비용, CPU checkpoint metadata를 검사했다.
- 전체 `srgc_rebuttal/tests`: **474 passed, 9 skipped**, 추가 subtests 303 passed.
  subtests를 독립 테스트 수에 더하지 않는다. skip 9개는 별도 Transformers 5가
  필요한 기존 Qwen 테스트다. 이번 OLMo 추가 실험의 새 테스트는 skip하지 않았다.
- 테스트 환경: CPU PyTorch 2.14.0, Transformers 4.57.6, PEFT 0.21.0. 기존 임시
  환경을 사용했으며 H100 노드 환경이나 requirements를 변경하지 않았다.
- shell 문법, 변경 Python compile, Ruff `F821/F823/F811`, 신규 파일 `F401`,
  scoped diff 검사 통과. frozen core/input/plan/runtime에는 변경이 없다.
- V6 보존 검사: 271개 제출 파일 불변. 원고 변경은 실험/리뷰 계획 MD 두 개뿐이다.

로컬에는 `/group-volume`과 H100 할당이 없어 실제 학습·원격 NCCL·그룹 볼륨 부하를
검증하지 않았다. 실제 shell `status/results`도 group volume 부재를 명시하고 종료하며
user volume으로 대체하거나 새 실험을 시작하지 않는 것을 확인했다.
최종 로컬 검사 기록: `/tmp/switch-validation-20261006-final.xml`.
