# 학습 단계와 데이터셋에 따른 선별 효과의 차이를 분리하는 실험

2026-10-06. 코드 `master`. **계획 및 구현이며 GPU 측정 결과가 아니다.**
RLOO를 추가하는 실험이 아니다. GRPO를 유지하고 선별 정보만 조작한다.

**2026-10-06 상태:** 저자가 메커니즘 결과가 나오면 전달하기로 했다. 현재는 결과 수신 대기이며,
전달받은 뒤 단계별 보상 증가·대조 효과·비용을 분석한다. 기존 실행은 변경하거나 재실행하지 않는다.
fixed200은 비교 분석·논문 반영 대상에서 제외하지만, 이 메커니즘 실험과 온라인 Switch는 유지한다.

## 무엇을 알아보는 실험인가

핵심 질문은 **“학습 단계와 데이터셋에 따라 유리한 선별 방법이 달라지는 이유가 무엇인가?”**다.
특히 MATH와 MBPP에서 관찰한 서로 다른 양상을 설명해야 한다. 두 데이터셋 모두
초반 On-policy·후반 SR이라는 같은 패턴을 보인다고 전제하지 않는다.
같은 모델 상태에서 선별 점수의 문제별 정보를 그대로 쓰거나 일부러 섞은 뒤,
실제로 학습시켜 평가 보상이 얼마나 달라지는지 본다.

세 가지를 확인한다.

1. **초반의 이득이 문제에 맞는 gradient 방향 점수에서 오는가?**
   On-policy가 계산한 점수 값들은 그대로 두고, 그 점수를 다른 문제에 무작위로
   붙인 대조군과 비교한다. 예를 들어 원래 문제 A에 붙은 높은 점수를 문제 B에
   옮기는 식이다. 점수의 분포는 같지만 문제와 점수의 연결이 깨진다.
   원래 연결을 쓴 쪽이 더 잘 학습되고 그 차이가 초반에 크다면,
   현재 gradient 점수의 문제별 정보가 초기 이득에 기여한다는 설명을 지지한다.
2. **후반의 SR 이득이 학습 신호가 있는 문제를 고르는 데서 오는가?**
   SR은 캐시에 기록된 성공률이 0.5에 가까운 문제를 고른다. 실제 학습에서
   그 문제들의 8개 응답에 성공과 실패가 섞이는지, 보상도 더 오르는지 함께 본다.
   SR 점수를 문제 사이에서 섞은 대조군은 문제별 SR 정보의 기여를 확인한다.
   현재 정책으로 성공률을 다시 측정한 `sr_fresh`와의 비교는 오래된 캐시의
   영향을 확인한다. 혼합 응답 비율이 높다는 사실만으로 성능 향상을 확정하지 않는다.
3. **선별 점수가 잘 반복되는 것이 실제 학습에도 도움이 되는가?**
   독립 응답으로 점수를 두 번 측정해 순위가 얼마나 반복되는지 확인하고,
   같은 출발점에서 실제 학습한 뒤의 보상 증가와 함께 해석한다.
   두 번째 측정은 진단에만 쓰며 선택이나 전환 결정에 넣지 않는다.

## 실제로 무엇을 실행하는가

각 데이터셋의 seed 5–9마다 별도 On-policy 학습 경로를 만들고,
**0·100·400 step의 상태에서 아래 과정을 각각 수행**한다.

1. 모델과 optimizer 상태를 저장하고 공통 후보 문제 40개를 뽑는다.
2. 그 상태를 여섯 방법에 똑같이 복원한다. 각 방법이 후보 40개에서 문제 4개를
   고르고, 그 4개를 유지하며 25 updates 학습한다. 매 update마다 문제당
   8개 응답을 새로 생성하므로 학습에는 매번 32개 응답을 사용한다.
3. 별도 평가셋에서 **학습 후 보상 − 같은 상태의 학습 전 보상**을 계산한다.
   여섯 분기의 결과를 비교한 뒤 원래 공통 경로로 돌아가 다음 시점까지 학습한다.
   앞서 진단한 분기의 학습 결과가 다음 시점의 출발 상태를 바꾸지 않는다.

핵심 결과는 **데이터셋별·단계별 보상 증가와 점수 정보를 섞었을 때 사라지는 이득**이다.
On-policy와 SR의 정보 이득이 각 데이터셋에서 어떻게 변하는지 따로 확인한다.
예상한 패턴이 나오지 않으면 해당 원인 설명을 지지하지 않는 것으로 해석한다.
공통 On-policy 경로의 세 상태에서 수행하는 25-update 국소 실험이므로,
기존 Switch의 전체 학습 효과는 별도의 전환 실험 결과와 함께 판단한다.

## MATH와 MBPP의 차이를 해석하는 방법

이번에 확인한 추가 시드 5–9의 종점 평균은 **MATH에서 On-policy가 가장 높고,
MBPP에서 SR이 가장 높다.** Switch도 MATH에서는 On-policy보다 낮지만
MBPP에서는 On-policy보다 높고 SR보다 낮다. 이는 두 수집본의 관찰 결과다.
종점 순위만으로 초반·후반의 순서나 원인을 확정하지 않는다.
확인한 파일은 2026-10-06의 `results.json`(SHA-256 `7718fdcf0f22475d…`)과
`results_mbpp.json`(`2c4bc214b5af506e…`)이다. 아래는 이 결과를 보고 정리한 설명 후보이며
확정된 원인이나 관찰 전에 등록한 예측으로 표시하지 않는다.

메커니즘 실험은 아래 세 설명을 기존 여섯 대조군과 측정값으로 구분한다.
추가 시드나 고정 200-step 대조를 더 만드는 계획이 아니다.

| 확인할 설명 | 기존 실험에서 비교할 것 | 설명을 지지하려면 볼 결과 |
| --- | --- | --- |
| 현재 gradient 점수가 유용한 정도가 데이터셋마다 다른가 | 시점별 `on_policy - direction_shuffle`과 `on_policy - random`; 독립 B 점수·상위 4개 반복성 | MATH에서 문제별 gradient 점수의 학습 이득이 더 크고, MBPP에서는 작거나 시점에 따라 줄어드는지 확인. 상관이나 overlap 차이만으로 결론 내리지 않음 |
| SR이 실제로 학습 신호가 있는 문제를 고르는 정도가 다른가 | 시점별 `sr - sr_shuffle`, `sr - random`, `sr - on_policy`; 실제 25-update 학습의 혼합 응답 group 비율 | MBPP의 SR 보상 이득과 혼합 group 비율이 함께 높고, 점수를 섞으면 이득이 줄어드는지 확인. MATH에서도 같은 연결이 나타나는지 별도 비교 |
| 고정 SR 캐시가 현재 정책의 난이도를 반영하는 정도가 다른가 | `sr_fresh - sr`과 독립 B 측정의 cache-current 성공률 차이 | MATH에서 캐시 오차가 더 크고 fresh-SR이 학습 이득을 회복하는지, MBPP에서는 그 차이가 작은지 확인. 반대 결과도 그대로 보고하며 캐시 노후화를 미리 원인으로 정하지 않음 |

분석은 **각 데이터셋 안에서 같은 seed·같은 출발 상태의 방법 간 보상 증가 차이**를
먼저 구한다. 이후 MATH와 MBPP의 단계별 평균 효과·seed별 값·표본 SD를 나란히 본다.
예를 들어 `on_policy - direction_shuffle`의 초반 대비 후반 변화와
`sr - on_policy`의 단계별 변화를 각각 비교한다. 같은 번호의 seed라는 이유로
서로 다른 데이터셋의 문제나 실험 경로를 한 쌍으로 취급하지 않는다.

현재 코드는 데이터셋별 대조 효과를 요약하고, 위 해석에 필요한 응답·점수·캐시 차이를
원본 결과에 저장한다. 데이터셋 간 차이를 자동으로 원인 판정하는 출력은 없다.
두 데이터셋의 결과를 모두 확보한 뒤 위 비교에 따라 해석한다.

이 설계가 뒷받침할 수 있는 것은 **선별 정보의 학습 기여가 데이터셋·단계에 따라
달라지는지**다. 혼합 group 비율은 보조 측정이므로 그것이 성능 차이를 일으킨
매개 원인이라고 단정하지 않는다. MATH와 MBPP는 문제 분포·채점기 등도 다르므로
차이를 단순히 “수학 대 코딩”의 본질적 차이라고 일반화하지 않는다.
모델·objective·응답 수·학습 budget·실제 채점기 버전을 함께 확인한다.

또한 공통 경로는 On-policy이고 진단에서는 여섯 방법 모두 선택한 4개를 25 updates
유지한다. 이 결과만으로 기존 매-update SR이나 Switch 전체 경로의 종점 차이가
완전히 설명됐다고 쓰지 않는다. 먼저 국소 효과가 관찰된 종점 차이와 같은 방향인지
확인하고, 연결되지 않으면 원인 설명이 미완료라고 남긴다.

## V7을 다시 읽고 확인한 공백

- `learning_results.tex`와 `training_table.tex`: MATH-300에서 Random 대비 이득은
  t0의 On-policy +0.58 pp / SR -0.60 pp, t400의 On-policy +0.12 pp / SR +1.74 pp다.
  이것은 단계에 따른 평균 순서 변화이며 그 원인의 인과 검증은 아니다.
- `learning_results.tex`와 `discussion.tex`: 초반 방향 정보의 이점을 가능한 해석으로
  제시하지만, 동일한 학습 상태에서 그 정보를 제거한 개입 결과는 없다.
- `appendix_theory.tex`: 점수 반복성에 대한 감쇠 법칙은 선택 점수의 이득을 다룬다.
  이 법칙만으로 GRPO 보상 개선이나 후반 SR 우위를 증명할 수 없다.
- `appendix_protocols.tex`: 선별은 LOO 및 response-token 합, 학습은 GRPO 보상 표준화와
  response-token 평균이다. 선별용 dense gradient와 LoRA/AdamW 업데이트도 같지 않다.
  따라서 cosine이나 D의 크기를 실제 학습 효과로 치환하지 않는다.
- 기존 `direction_removed`, `sr_hold`, `sr_refresh_matched`는 유용하지만 t25부터 긴
  경로 전체를 비교한다. 초반과 후반에서 달라지는 효과를 동일 상태에서 분리하지 못한다.

## 실행 순서

이 실험은 저자가 별도로 요청한 메커니즘 검증이다. V6 limitation 대응과 병행 가능하며
임의로 후순위로 내리거나 기존 작업을 중단하지 않는다. MATH 최대 5노드, MBPP 최대 5노드다.
V6 대응과의 구분은 [실험 목록](LIMITATION_EXPERIMENTS_KO.md#우선순위),
명령과 노드 배정은 [실행 안내](REBUTTAL_COMMANDS_KO.md)를 따른다.
아래 명령은 mechanism queue만 배정한다. 기존 P1·support와 진행 중인 작업은 유지한다.

## 동일 상태에서의 개입

MATH와 MBPP 각각 seed 5-9. 각 seed에서 별도 On-policy 공통 경로를 **t0부터** 만든다.
모델·LoRA·GRPO·optimizer·응답 수·projection·채점기는 해당 seed의 검증된 runtime을 따른다.
기존 prefix는 runtime/attention/input 식별을 위해 확인할 뿐, 초기 모델 가중치로 복구하지 않는다.
P0나 그 prefix를 덮어쓰거나 P0 결과를 새로 만드는 작업이 아니다.

t0, t100, t400에서 모델과 optimizer를 함께 복제한다. 시점은 V7의 단계 구분에 맞췄다.
각 시점에서 공통 pool의 **40개를 중복 없이 균등 추출**하고, 아래 여섯 방법으로
4개를 선택한다. 같은 4개를 25 updates 동안 유지하고, 매 update 문제마다 **8개 새 응답**을
생성한다. 방법 간 후보·유지 간격·학습 budget·평가셋·난수 규칙을 맞춘다.
후보 추출은 시점마다 새로 하며 처음 뽑은 40개만 전체 경로에 고정하지 않는다.

| 진단 arm | 바꾸는 것 | 비교 질문 |
| --- | --- | --- |
| `on_policy` | A의 현재 gradient cosine 상위 4개 | 현재 방향 점수가 실제 학습 이득을 주는가 |
| `direction_shuffle` | 같은 cosine 값을 후보들 사이에서 무작위로 섞음 | 점수 분포는 유지하고 문제-방향 정보의 연결을 끊으면 이득이 사라지는가 |
| `sr` | 고정 cache의 성공률이 0.5에 가까운 4개 | 후반에서도 유효한 학습 문제를 고르는가 |
| `sr_shuffle` | 같은 cached-SR 점수를 후보 사이에서 무작위로 섞음 | SR의 문제별 정보가 필요했는가 |
| `sr_fresh` | 같은 초기 상태에서 A의 새 보상으로 성공률만 갱신 | 고정 cache와 현재 난이도의 차이가 영향을 주는가 |
| `random` | 같은 40개에서 무작위 4개 | 선별 없는 기준 |

이는 **배치 유지 간격을 맞춘 SR 진단**이며 기존 매-update SR arm과 동일한 운영 방식으로
부르지 않는다. 각 시점의 branch들은 복제한 동일 상태에서 출발한다. 진단 학습 종료 후
공통 경로의 모델/optimizer를 그대로 복구하므로 진단 branch가 다음 시점에 영향을 주지 않는다.
공통 경로가 On-policy라는 조건은 남는다. 모든 학습 경로나 모델에 대한 보편적 원인으로
일반화하지 않는다. 기존 원고의 100-update continuation을 그대로 재현하는 실험도 아니다.

## E14: 초반 방향 정보의 기여

주 비교는 `on_policy - direction_shuffle`의 독립 평가셋 보상 차이다.
같은 seed에서 후반 차이에서 초반 차이를 뺀 **단계별 효과 차이**도 계산한다.
초반 양의 차이가 후반에 줄어드는지 직접 확인한다. `on_policy - random`도 함께 보고한다.
초반 차이가 없으면 방향 정보가 초기 우위를 설명한다는 해석을 지지하지 않는 결과다.

## E15: 후반 SR과 유효한 GRPO 학습 신호

`sr - sr_shuffle`, `sr - random`, `sr - on_policy`, `sr_fresh - sr`를 시점별로 비교한다.
각 branch의 **실제 학습 rollout**에서 성공과 실패가 섞인 8-response group 비율을 기록한다.
모든 보상이 같은 group은 현재 GRPO의 reward advantage가 0이다. 다만 AdamW momentum이
있으므로 이 사실을 optimizer 이동이 정확히 0이라는 뜻으로 쓰지 않는다.

현재 성공 확률이 p이고 K개 응답이 독립인 Bernoulli 표본이면 혼합 group 확률은
`1 - p^K - (1-p)^K`다. 이는 현재 p가 0.5 근처일 때 가장 크다. 그러나 cached-SR은
초기 정책의 추정 성공률을 쓰므로 현재 p가 0.5에 가깝다는 보장은 없다. 바로 이 연결을
실제 현재 보상과 cache 갱신 대조로 검증한다. 혼합 group 확률 식 자체가 SR 우위의 증명은 아니다.

SR이 후반에 보상을 더 올리면서 유효 group 비율도 높이고, 점수 shuffle에서 이득이
줄어드는지 확인한다. group 비율 하나만으로 개선을 확정하지 않는다. 캐시-현재 성공률 차이와
fresh-SR 대조로 캐시의 오래됨도 분리한다. 이 실험은 왜 SR을 **후반에 선택할 가치가 있는지**를
분석하며 t0부터 SR만 쓴 전체 학습 전략의 우위를 새로 주장하지 않는다.

## E16: 선별 점수와 학습 이득의 연결

같은 체크포인트에서 독립 응답으로 A와 B를 각각 측정한다. 각 block은 후보 40개와
입력에 지정된 **하나의 ranking-validation 집합**을 사용한다. 문제당 8응답이다.
후보 scoring은 기존대로 4응답 LOO subgroup 두 개, validation은 8응답 group을 사용한다.
A만 선별에 사용한다. B는 선택에 쓰지 않는 별도 진단이며 배포용 두-reference 규칙이 아니다.
기존 SR-GC의 40 대 40 계산·규칙·비용은 바꾸지 않는다.

점수 A/B 상관, 상위 4개 overlap, B에서의 선택 집합 cosine·dot·gradient norm,
현재 성공률·혼합 보상 group 비율·cached/current 차이를 저장한다.
같은 상태의 학습 전후 독립 평가셋 reward 차이를 반드시 같이 본다.
상수 점수의 상관은 `null`이며 0으로 대체하지 않는다. top-4 repeatability와
실제 학습 효과가 반대일 수도 있으므로 안정성만으로 성능을 판정하지 않는다.

## 평가와 비용

평가는 각 시점 학습 전 1회, 여섯 branch의 25-update 학습 후 각각 1회다.
기존 독립 평가셋 전체와 문제당 8응답을 사용한다. 평가 전후와 방법 간 난수 규칙은
공유하지만, 달라진 정책에서 응답이 동일하다는 뜻은 아니다. 평가값은 선별에 쓰지 않는다.
seed별 원값과 paired 차이, 평균·표본 SD, 유효 seed 수를 출력하며 runtime/attention이
다른 seed를 섞지 않는다. prompt·response 수를 독립 학습 seed 수로 세지 않는다.

모든 계측은 기존 동기화 phase/exclusive-stage/invocation ledger를 사용한다.
공통 경로, A 획득, B 진단, 순위 계산, 각 branch 학습, 평가, checkpoint, startup을 구분한다.
GPU-s는 할당 GPU 수를 곱한 경과 시간이며 kernel-active time이 아니다.
기존 SR cache 생성 receipt도 **1회 별도 표시**한다. 이미 만든 cache는 다시 생성하지 않는다.
생성 기록이 없으면 `null`이며 0초로 쓰지 않는다. cache read/rank와 생성 비용을 구분한다.

이 실험의 A/B 공통 획득 비용을 각 방법의 운영비로 복제해 붙이지 않는다.
branch별 출력의 training 비용은 순수 branch 학습이며 전체 운영 비용이 아니다.
`successful` 열은 채택된 update/진단의 비용이고, 폐기된 시도·재시작의 실측 비용은 전체 ledger에 포함한다.
`json`에는 실험 전체 phase, exclusive stage, 재시작 비용, inclusive session과 cache 원값이 있다.
inclusive 안에 phase가 포함되므로 두 합계를 다시 더하지 않는다.
중단된 timer가 있으면 완전한 총비용으로 표시하지 않는다. 기존 Figure 3은 수정하지 않는다.

## 실행, status, results

코드 저장소 루트에서 빈 **4-H100 노드마다 같은 명령을 한 번** 실행한다.
먼저 MATH를 권장한다. MBPP는 `math`를 `mbpp`로, 양쪽 동시 배정은 `all`로 바꾼다.

```sh
sh scripts/run_srgc_mechanism.sh math
sh scripts/run_srgc_mechanism.sh math status
sh scripts/run_srgc_mechanism.sh math results
sh scripts/run_srgc_mechanism.sh math json
```

원본은 기존 active group-volume run의 `seed-N/stage_mechanism-endpoint.json`이다.
중간 상태는 `stage_mechanism-progress.json`, 재개 파일은 `stage_mechanism-latest.pt`,
phase 비용은 `cost-receipts/stage_mechanism/`, inclusive 비용은 `invocations/stage_mechanism/`이다.
`results` 첫 부분에 seed별 실제 `output=` 경로를 표시한다. `results/json`은
`<run root>/results/mechanism/results.json`과 `exports/<수집 시각-ID>/raw/`에
통합 JSON·검증된 원본 사본을 자동 저장한다. `COLLECTED JSON / COLLECTED FILES`로 위치를 확인한다.
`json`의 stdout에는 JSON만, 저장 경로는 stderr에 출력한다. 별도 리다이렉션은 필요 없다.

한 작업은 한 seed의 모든 시점·대조를 담당한다. **5작업/도메인, 총 10작업**이며
최대 동시 독립 배정은 5노드/도메인, 양쪽 10노드다. 한 seed 안에서는 모델을 공유하여
순차 진단한다. 기존 120/30/70개 queue는 재구성하지 않는다.

한 seed의 물리적 update는 공통 경로 400 + 3시점 × 6대조 × 25 = **850회**다.
주 정책이 850스텝 학습됐다는 뜻이 아니다. 평가 21회와 A/B 점수 획득 비용도 든다.
GPU 시간은 실제 로그 전에는 단정하지 않는다. `status`는 물리적 진행량과 policy 시점,
현재 대조·branch step을 함께 표시한다.

매 학습 update와 진단 단계 완료 시 group-volume에 checkpoint한다. 같은 명령을 다시
실행하면 모델·optimizer·측정값·진단 진행 위치를 복구한다. scoring/evaluation 응답은 기존
prompt별 rollout cache를 재사용한다. 사용자 볼륨 저장이나 migration은 추가하지 않는다.
`status`는 읽기 전용이고 `results/json`은 수집본만 저장한다. 학습·원본·잠금을 바꾸지 않는다.
정상 실행 중인 checkout에 pull하거나 worker를 중복 실행하지 않는다.
과거 MBPP prefix에 연결된 실행은 그 당시 채점기 버전을 유지한다. v3 결과와 혼합하지 않는다.

## 코드 위치와 검증

- [실행](../scripts/run_srgc_mechanism.sh)
- [개입·복구·계측](../scripts/srgc_stage_mechanism.py)
- [검증·단계별 paired 분석](../scripts/srgc_stage_report.py)
- [기존 노드 배정](../scripts/srgc_replicate_worker.py): `mechanism` scope
- [회귀 테스트](../srgc_rebuttal/tests/test_stage_mechanism.py)

로컬 CPU 통합 테스트는 실제 sampler/AdamW toy backend로 복구·모델/optimizer 분리·평가 누출
방지·원값 검증·shell dispatch를 확인한다. toy 결과는 논문 실험이 아니다.
로컬에는 H100/group-volume 할당이 없어 실제 GPU 학습 완료와 다중 노드 부하 실측은 하지 않았다.
논문 레포에는 분석/계획 MD만 추가한다. TeX·PDF·웹·V6 결과는 변경하지 않는다.

검증 기록: 신규 45 tests 통과, 전체 SRGC suite 519 passed / 9 skipped / 303 subtests passed.
skip은 기존 별도 Transformers 5 Qwen 환경 조건이며 새 테스트는 skip하지 않았다.
실제 소형 OLMo·LoRA CPU backend, 두 archived runtime, 모든 진단 phase의 재개,
원래 P1 results와의 분리, 잘못된 endpoint 제외를 검사했다. shell 문법·Python compile·
Ruff·diff 검사 통과. V6 제출 파일 271개 불변. 전체 기록은
`/tmp/stage-mechanism-20261006-final.xml`이다.
