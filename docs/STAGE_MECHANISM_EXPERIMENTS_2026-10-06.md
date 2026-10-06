# 초반 On-policy와 후반 SR의 학습 효과를 분리하는 실험

2026-10-06. 코드 `master`. **계획 및 구현이며 GPU 측정 결과가 아니다.**
RLOO를 추가하는 실험이 아니다. GRPO를 유지하고 선별 정보만 조작한다.

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
