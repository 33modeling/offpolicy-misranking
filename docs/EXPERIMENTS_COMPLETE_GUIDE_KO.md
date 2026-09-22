# 전체 실험 명세와 완료 기준

작성 기준: 2026-09-22. 대상: 이 저장소의 주 실험, 후속 비교 실험,
측정 보강, 분석 전용 실험, 보존된 이전 설계, 새 MBPP off-policy 코드.

이 문서는 **실험 자체**를 설명한다. 장애 이력, 코드 수정 내역, 배포 일지는
다루지 않는다. 명령은 실험을 구분하기 위한 진입점이며, 이 문서 작성으로 GPU
실험을 새로 실행한 것은 아니다. 실행 중인 서버의 현재 상태를 조회한 기록도 아니다.

이 파일은 Git에 남기는 상세 연구 기록이며 논문 본문이 아니다. 실험 재현에 필요한
입출력·완료 조건·출처는 여기 남기고, 논문에는 연구 질문과 비교 조건, 측정 결과,
해석에 필요한 내용만 간결하게 반영한다. 운영·검증 설명을 그대로 원고에 옮기지 않는다.

숫자는 세 종류로 구분한다.

- **설계 수**: 코드나 동결 설정에 지정된 seed, 분기, 평가점의 수.
- **수신한 측정값**: 출처가 확인된 결과 파일에 실제로 있는 값.
- **미확인 값**: 아직 이 문서에서 결과 파일을 대조하지 않은 값. 빈칸은 0이 아니다.

기본값보다 이미 실행된 실험의 `run_config.json`, `switch.json`, `pair.json`,
`experiment.json` 등 **동결된 설정이 우선**한다. 아래 계획 수를 현재 완료 수로
읽으면 안 된다. 과거 문서의 설계와 현재 실행 설정이 다르면 현재 설정을 따른다.

## 1. 이 연구가 나누어 묻는 질문

1. 기존 정책의 응답으로 계산한 점수가 현재 정책의 데이터 순위를 얼마나 보존하는가?
2. 그 순위를 평가하는 기준 자체는 유한한 응답 수에서 얼마나 재현 가능한가?
3. 점수가 높은 데이터를 고르면 실제 후속 학습 성능도 좋아지는가?
4. 학습 성능이 좋아지더라도 선택 비용까지 내면 계산 효율도 좋아지는가?
5. 같은 학습 상태에서 비싼 on-policy 선택과 저렴한 cached 선택 중 무엇이 유리한가?
6. 선택한 데이터의 효과가 다른 학습 목적함수나 모델·도메인에서도 유지되는가?

중심 구분은 **데이터 선택의 유용성**, **선택기의 지속적인 우위**, **계산 효율**이다.
한 가지를 관측했다고 나머지까지 입증된 것은 아니다. H는 비용을 비교하기 위한
판단 기준이며, 자동으로 최적 전환 시점이나 보편적 우월성을 보증하지 않는다.

## 2. 용어와 실험 단위

| 용어 | 정확한 의미 | 혼동하면 안 되는 것 |
| --- | --- | --- |
| behavior policy, beta | 기존 응답을 생성한 정책 | 모든 drift에서 새로 만든 현재 정책 |
| current policy, pi | 해당 학습 체크포인트의 정책 | behavior 정책과 항상 같다는 가정 |
| drift, d | 주 매트릭스에서 누적 정책 업데이트 수 | 시간, 데이터셋 변화량의 직접 측정 |
| selected prefix | 선택된 데이터로 선행 학습한 공통 이력 | 서로 독립적으로 학습한 비슷한 모델 |
| state | seed와 공통 prefix checkpoint로 정한 분기 상태 | 독립 seed 하나 |
| arm, 분기 | 같은 상태에서 실행하는 한 비교 조건 | GPU 하나 또는 평가 shard 하나 |
| shard | 한 계산을 GPU별로 나눈 부분 작업 | 독립 실험 반복 |
| ranking validation | 선택 점수의 기준 gradient를 만드는 자료 | 마지막 성능 평가용 test 자료 |
| eval | 학습 후 정책의 응답 생성·채점 | 추가 학습 |
| curve | 여러 저장 체크포인트의 성능 평가 | 다시 처음부터 학습하는 작업 |
| results | 현재 저장된 측정·출처·누락 정보를 내보내기 | 누락된 GPU 계산을 대신 수행 |
| DONE | 해당 실험의 필수 산출물 검증까지 완료 | 시간 한도 도달, 로그의 100%, 프로세스 종료 |

논문 표현과 내부 이름은 다음처럼 연결한다. 파일명을 바꾸어 과거 결과의 정체성을
바꾸지 않는다.

| 논문에서 쓸 표현 | 내부 키 | 의미 |
| --- | --- | --- |
| On-policy gradient alignment | `fresh_r`, 일부 분석의 `fresh` | 현재 정책 응답과 gradient 기반 점수 |
| Cached success-rate | `difficulty`, `passrate_beta` | 저장된 보상에서 구한 성공률 기반 점수 |
| Uniform random | `random`, `random_full`, `random_reduced` | 같은 크기 무작위 부분집합 |
| Adaptive selector | Pair의 `adaptive` | 고정된 판단으로 on-policy 또는 cached 실행 |
| Gate policy | 기존 Switch의 `gated` | 해당 프로토콜의 선택 대 random 판단을 실제 실행 |
| Off-policy estimators | `g00`, `g10`, `g01`, `g11` | 기존 응답에 서로 다른 중요도 교정을 적용 |

Cached success-rate의 기본 점수는 `-abs(p_hat - 0.5)`이다. 이는 저장된 응답의
성공률이며, 새로운 현재 정책 평가가 아니다. `passrate_beta`의 beta는 behavior
정책을 뜻한다. MoPPS의 Beta posterior와 같은 방법이 아니다.

## 3. 전체 목록

아래 수는 기본 설계 기준이다. 서로 다른 행을 합산해 하나의 독립 실험 수로 쓰지 않는다.

| 계열 | 핵심 산출물 | 새 학습 | 기본 설계 단위 |
| --- | --- | --- | --- |
| OLMo 주 매트릭스 | 순위·일치도·신뢰도·교정 추정량 | 양의 drift에서 수행 | 2 datasets x 5 seeds x 4 checkpoints = 40 points |
| Qwen 9B 복제 | 다른 모델 조건의 같은 측정 | 수행 | 40 points |
| Qwen 2B/4B 및 27B 설정 | 규모·모델 확장 | 실행 시 수행 | 각 설정별 별도 매트릭스 |
| 도메인 확장 | 논리·과학·지식의 같은 측정 | 수행 | 도메인과 모델별 별도 추정 |
| E1 | 세밀한 drift 곡선 | 수행 | MATH 2 seeds, 8 checkpoints |
| E2/E3 | 순위 반전, margin 조건 | 없음 | 완료된 매트릭스 점 재분석 |
| E4 및 gain-law 합성 | 제어 가능한 합성 환경의 진단 | LLM 학습 없음 | 합성 조건별 반복 |
| E5 | 고정 부분집합의 후속 학습 보상 | 수행 | 버전별 arms·업데이트 수 구분 |
| 외부 benchmark | 저장된 정책의 추가 성능 | 없음 | 정책 x benchmark x 평가 shard |
| Reference axes | 응답 예산별 새 reference의 신뢰도 | 없음 | 데이터셋별 5 예산 조건 x 5 반복 |
| Off-policy split-half | 재사용 추정량의 반쪽 점수 | 없음 | source point x 4 estimators |
| 새 MBPP off-policy | Figure 2용 MBPP calibration | **없음** | 6 source points, 24 estimator rows |
| Method choice | 방법 선택 기준별 후속 보상 | 수행 | 기본 40 trained arms |
| Mixed pool | 이질 후보 풀이 있는 양성 대조 | 수행 | 별도 혼합 풀 실험 |
| Additive/TayPO-2 | 두 추가 추정량의 점수·gain | 없음 | 기본 MATH d400, 3 seeds |
| Low-order reuse | 저차 추정량의 선택·후속 보상 | 수행 | 기본 5 seeds x 3 arms |
| 이전 gate 계열 | 진단 후 선택 또는 random | 설계별 수행 | gate 버전별 별도 실험 |
| Selected-prefix Switch | 공통 학습 이력에서 선택 대 random | 수행 | 조건당 18 DEV + 30 TEST = 48 branches |
| MBPP quality | 선택 비용을 별도 계측한 학습 성능 | 수행 | 기본 48 branches |
| MBPP repair follow-up | 기존 결과 재사용과 보완 실행 | 필요한 분기만 | 별도 48칸 cohort, 48회 신규 학습 아님 |
| Selector Pair | on-policy 대 cached의 실측 비용 비교 | 수행 | 18 DEV + 24 TEST = 42 continuations |
| MoPPS 비교 | online selector 대 실제 Gate | 수행 | 6 TEST states x 2 새 arms = 12 continuations |
| RLOO | 고정 선택의 다른 learner로의 이전 | 수행 | 18 continuations + 6 parent evaluations |
| CFCS 로컬 연구 | 교차적합 선택기의 합성 연구 | LLM 결과와 분리 | 미추적 연구 코드, 정식 실행과 분리 |

## 4. 주 매트릭스: OLMo GRPO와 off-policy 순위

근거: `configs/olmo3_rlzero.json`, `configs/olmo3_rlzero_h100.json`,
`scripts/run_point.sh`, `src/experiment.py`, `src/grads.py`, `src/train_policy_grpo.py`.

### 4.1 비교 조건

- 모델: 저장소 설정에 revision이 고정된 `allenai/Olmo-3-1025-7B` base.
- 데이터: MATH-500, MBPP. 후보 수는 각각 400, 512; ranking validation은 100.
- Seed: 0, 1, 2, 3, 4. Checkpoint: 0, 25, 100, 400.
- 가족 단위는 dataset x seed. 가족 내 checkpoint는 연속된 한 학습 이력이다.
- d0의 behavior 응답을 이후 drift에서도 재사용한다. 양의 drift는 현재 정책을
  학습시키는 것이며, 매번 beta cache를 새 정책 응답으로 교체하는 것이 아니다.
- 양의 checkpoint는 선행 adapter와 optimizer를 이어받는다.

### 4.2 한 점의 처리 순서

1. 후보·validation 분할과 모델·학습·생성 설정을 고정한다.
2. behavior 응답을 후보당 8개 확보한다. 검증된 공통 cache는 재사용한다.
3. 양의 drift이면 해당 누적 업데이트까지 GRPO 학습한다. d0은 생략한다.
4. 현재 정책으로 후보당 32개, validation당 8개 응답을 생성하고 채점한다.
5. validation gradient와 후보의 micro-group gradient를 계산한다.
6. 기존 behavior 응답으로 g00/g10/g01/g11 점수를 계산한다.
7. GPU shard를 병합하고 coverage·출처·점수 형식을 검증한다.
8. 순위·일치도·신뢰도와 통계 요약을 저장한 뒤 point를 완료 처리한다.

### 4.3 학습 규격

4 GPU ranks 각각 한 프롬프트에서 8응답을 생성한다. 한 업데이트의 표본 수는
4프롬프트, 32응답이다. 기본 OLMo 설정은 one epoch, learning rate `1e-5`,
clip epsilon `0.2`, reference KL coefficient `0`, LoRA rank 16/alpha 32,
`q_proj`·`v_proj`, gradient norm cap 1이다. 최대 새 토큰은 2048,
temperature와 top-p는 1이다.

GRPO advantage는 `(reward - group_mean) / (group_std + 1e-4)`다.
생성 시점의 old log-prob를 저장하고 PPO형 clipped surrogate를 계산한다.
단, 한 그룹에 업데이트가 한 번인 주 설정에서는 업데이트 전 ratio가 거의 1이다.
따라서 clipping이 실제 trust region으로 작동했다고 단정하지 않는다.

### 4.4 점수와 독립 평가

한 후보의 fresh 32응답을 4응답 micro-group 8개로 나눈다. R에 4그룹,
A/B에 각각 2그룹을 배정한다. Validation 100개도 R=50, A=25, B=25로 나눈다.
R로 순위를 만들고 A/B로 선택된 데이터의 점수와 신뢰도를 평가한다.
기준 점수도 유한 표본의 추정량이며 정확한 참값이라고 부르지 않는다.

gradient는 마지막 4개 층을 사용하고 기본 4096차원 projection을 적용한다.
상위 선택 비율은 10%다. 실제 정수 선택 수는 해당 함수·동결된 부분집합을 따른다.
기본 추정량은 clipping cap 10을 적용하므로 unclipped 이론과 관측량을 구별한다.

토큰 비를 `r_t`, 앞부분 곱을 `P_t`, 뒷부분 곱을 `S_t`라 하면 교정 가중은
g00=`r_t`, g10=`P_t r_t`, g01=`r_t S_t`, g11=`P_t r_t S_t`다.
실제 gradient에는 LOO advantage, clipping, projection 등 코드의 측정 규칙이
추가된다. full correction이라는 이름만으로 유한 표본 분산이나 순위 품질이
항상 좋아진다고 말할 수 없다.

### 4.5 완료와 해석

checkpoint 파일만 있으면 학습 산출물 일부가 있다는 뜻이다. 응답, oracle 및
off-policy 점수, 정확한 후보 coverage, 분석 산출물까지 검증돼야 point 완료다.
40개 점은 독립 40개 seed가 아니다. 같은 가족의 여러 checkpoint는 종속된다.
순위 회복만으로 후속 학습 성능 또는 총비용 우위를 결론내리지 않는다.

## 5. 신뢰도·reference 예산·gain calibration

### 5.1 기존 응답의 재분석

`kcurve_floor.py`, `kcurve_all.py`, `measurement_ceiling.py`,
`reliability_trajectory.py` 등은 저장된 산출물의 신뢰도와 측정 범위를 분석한다.
기존 표본의 부분집합 재분석과 새 표본을 생성한 독립 반복을 구분한다.
큰 bootstrap 횟수는 독립 학습 seed 수를 늘리지 않는다.

### 5.2 독립 reference 생성

`scripts/run_reliability_budget.sh`는 d0에서 학습 없이 새 응답을 생성한다.
기본 fresh K=64, validation K=32이며 원본과 구분되는 RNG·출력 경로를 사용한다.
생성 → validation/oracle gradients → 병합 → 신뢰도 분석 순서다.

`scripts/run_reference_axes.sh`는 한 축씩 바꾼다.

| 후보 응답 K | Validation 응답 K | 목적 |
| ---: | ---: | --- |
| 32 | 8 | 새 기준 반복 |
| 64 | 8 | 후보 reference 예산 증가 |
| 128 | 8 | 후보 reference 예산 추가 증가 |
| 32 | 16 | validation 예산 증가 |
| 32 | 32 | validation 예산 추가 증가 |

각 dataset에 반복 ID 0..4를 사용한다. 이 25조건은 25회 정책 학습이 아니다.
응답 생성과 gradient 계산은 GPU 작업이며, 학습이 없다고 비용이 없는 것은 아니다.

### 5.3 관측 gain과 계산 예측

`src/gain_vs_reliability.py`는 반쪽 점수 a/b를 각각 표준화한다.
a 상위 k의 b 평균과 b 상위 k의 a 평균을 구한 뒤 평균한다.
이 값이 관측된 cross-half gain이다. 실제 후속 학습 reward와 단위가 다르다.

```text
rho_half = corr(a, b)
gain_observed = (mean(z_b[topk(a)]) + mean(z_a[topk(b)])) / 2
gain_predicted = rho_half * c(k, n)
```

`c(k,n)`은 표준정규 n개 중 상위 k 평균의 기대값을 Monte Carlo로 계산한다.
따라서 `predicted` 열은 실측으로 바꾸어 부르지 않는다. Latent score에 관한
`sqrt(rho)` 관계와 관측 cross-half gain의 `rho` 관계도 서로 바꾸어 쓰지 않는다.
상수인 반쪽 점수처럼 상관을 정의할 수 없는 경우 결측으로 남긴다.

## 6. 아까 추가한 MBPP off-policy calibration

근거: [상세 명세](MBPP_OFFPOLICY_CALIBRATION.md),
`scripts/run_mbpp_offpolicy.sh`, `src/mbpp_offpolicy_followup.py`,
`src/stale_splithalf.py`.

### 6.1 필요한 이유

MBPP continuation의 최종 reward와 Figure 2의 off-policy 점수 calibration은
다른 데이터다. 전자의 실험이 끝났다고 후자의 점수가 생기지는 않는다.
기존 모델·응답을 재사용해 빠진 반쪽 점수를 계산하는 후속 실험이다.

### 6.2 범위와 입력

- MBPP seed 0, 1, 2; checkpoint 0, 400: 총 6 source points.
- 각 point의 512개 후보와 후보당 기존 behavior 응답 8개.
- g00, g10, g01, g11: point당 4종, 총 24 calibration rows.
- 기존 모델, adapter, ranking validation 방향, projection과 clipping 설정.
- 원본 `DONE`, dataset/seed/drift, 응답 인덱스 coverage, 파일 hash를 확인한다.

24행은 24개 독립 학습 결과가 아니다. 같은 point의 네 추정량은 입력을 공유한다.

### 6.3 실행 순서

1. 여섯 원본 점의 필수 산출물과 모델 파일을 확인한다.
2. 입력 및 점수 계산 코드의 hash를 새 측정 계약에 묶는다.
3. 각 후보의 8응답을 원래 응답 인덱스 기준 4+4로 나눈다.
4. 각 반쪽 내부에서 LOO advantage를 계산한다.
5. 고정된 validation 방향으로 네 추정량의 반쪽 점수를 계산한다.
6. 각 shard에서 일부 full-group 점수를 다시 계산해 기존 점수와 차이를 기록한다.
7. 모든 후보의 반쪽 점수·유한성·설정·검사 기록을 확인한다.
8. 반쪽 상관, 관측 gain, 계산 예측과 원자료·출처를 TXT 하나로 내보낸다.

**추가 학습도 새 응답 생성도 없다.** 기존 응답의 확률·gradient를 다시 계산하므로
GPU는 필요하다. 일부 full-group 비교는 shard당 4개, point당 16개이며,
차이를 기록했다는 사실만으로 전체 수치 동등성을 인증한 것은 아니다.

상위 선택 수는 512개 중 51개다. 완료하려면 네 추정량 모두 512개 후보를
빠짐없이 포함해야 한다. 중간 결과는 내보낼 수 있지만 완료로 계산하지 않는다.
한 point는 4-GPU allocation 하나에서 계산한다. 여섯 점의 독립 작업이 모두
남아 있다면 최대 여섯 allocation에 나누어 처리할 수 있다.

### 6.4 실행과 내보내기

```bash
bash scripts/run_mbpp_offpolicy.sh
bash scripts/run_mbpp_offpolicy.sh status
bash scripts/run_mbpp_offpolicy.sh results
```

단일 파일: `~/mbpp-offpolicy-results.txt`. 미완료면 results가 비정상 종료 코드를
반환할 수 있어도 현재의 부분 결과 TXT는 작성한다.

이 문서 기준 **코드 준비 완료, 새 GPU 측정값은 이 문서에서 수신 확인하지 않음**.
논문의 미측정 MBPP calibration 값은 비워 둔다. MATH 숫자나 MBPP 최종 reward로
채우지 않는다. 기존 MATH 재채점은 `scripts/run_stale_splithalf.sh` 계열이다.

## 7. E1-E6와 후속 학습 성능

근거: [확장 명세](EXTENSIONS_2026-09-07.md), [Reduced E5](E5_REDUCED_RUN.md).

### 7.1 E1: 세밀한 drift

기본 MATH seed 0, 1에 `0/5/10/25/50/100/200/400`의 별도 연속 이력을 만든다.
기존 checkpoint 사이의 변화를 더 자세히 보는 실험이다. 정책 학습뿐 아니라
각 점의 fresh 응답·gradient·점수 계산을 반복하므로 400회 학습만의 비용이 아니다.
원본 매트릭스 경로를 덮어쓰지 않는다. 진입점은 `scripts/run_drift_curve.sh`다.

### 7.2 E2/E3: 순위 반전과 margin

E2 `src/reversal_matrix.py`는 저장된 점수의 순위 반전 빈도와 경계 부근을 요약한다.
E3 `src/margin_condition.py`는 margin 충분조건의 성립 비율을 계산한다.
대수적으로 성립하는 함의의 확인과 현실 데이터에서 그 조건이 자주 만족되는지는
다른 질문이다. 조건 확인을 새 학습 효과로 보고하지 않는다.

### 7.3 E4: 합성 풀

`src/synthetic_pool_study.py`에서 제어된 분포·noise·예산 조건을 바꾼다.
알려진 모집단이나 생성 규칙과 관측 추정량의 차이를 볼 수 있지만,
합성 결과를 OLMo나 MBPP에서 관측한 수치로 제시하지 않는다.

### 7.4 E5: 고정 선택 후 학습

원래 확장의 7 selectors x 50 updates와 이후 reduced E5를 구분한다.
Reduced E5 기본 설명은 MATH seed 0/1/2, d0/d400에서
`random`, `passrate_beta`, `fresh_r`, `g11`의 100회 추가 GRPO다.
후속 d100·gate 조건이 있는 결과는 실제 frozen contract별로 별도 표시한다.

절차는 source 검증 → 점수에 따른 부분집합 확정 → 동일 parent에서 각 arm 학습
→ 별도 test 평가 → arm 간 paired difference 계산이다.
기본 test는 후보와 ranking validation에 겹치지 않는 MATH-train 300문제,
문제당 8응답이다. 문자열 기반 비중복을 의미하며 의미적 중복까지 완전 제거했다는
주장은 별도의 증거가 필요하다.

d0은 base에서 시작하고 d400은 원본 adapter와 optimizer를 계승한다.
동일 업데이트 수의 비교는 데이터 선택의 학습 효과를 보지만 동일 총GPU비용을
보장하지 않는다. 저장된 선택 점수의 역사적 계산 비용을 0으로 놓지 않는다.

### 7.5 E6: 민감도

`src/rescore_variants.py`는 기존 응답으로 behavior 표본 수나 clipping 설정을
달리해 다시 점수를 계산한다. `sensitivity_tables.py`와
`diagnostics_vs_retention.py`가 비교 표를 만든다. 기본 variant 예시는
`bk2`, `bk4`, `clip3`, `clip30`이다. 새 generation이나 학습 없이 GPU 재채점할
수 있지만, 원본 점수 파일을 다른 조건으로 덮어써서는 안 된다.

## 8. 외부 benchmark와 기존 eval의 차이

`scripts/run_e5_bench.sh`는 완료된 E5 정책을 추가 문제 집합에서 평가한다.
기존 기본 묶음은 AIME24, AIME25, AMC23, GSM8K, MATH-rest다.
GSM8K와 MATH-rest는 기본 200문제 고정 표본, 기본 응답 수는 문제당 8개다.
실제 실행의 표본·K·평가 seed는 저장된 설정으로 확인한다.

새 정책을 학습하지 않는다. Parent와 각 완료 arm을 로드하고 응답 생성·채점·병합을
수행한다. 각 benchmark별 reward, parent/random 대비 차이, 비용을 남긴다.
기존 300문제 eval이 끝나도 이 추가 benchmark는 자동으로 측정됐다고 볼 수 없다.
반대로 외부 benchmark가 없다고 이미 검증된 내부 held-out reward가 없는 것도 아니다.
다만 일반화 주장의 범위는 확보한 평가에 한정해야 한다.

이 MATH benchmark 묶음을 MBPP 실험에 그대로 적용한 것으로 쓰지 않는다.
MBPP는 별도의 코드 실행 기반 평가 문제와 프로토콜을 확인해야 한다.

## 9. Selected-Prefix Switch: 48개가 의미하는 것

근거: `src/selection_switch.py`, `src/selection_switch_gpu.py`,
[Switch 설계](SELECTION_SWITCH_EXPERIMENT.md).

### 9.1 공통 시작 상태

On-policy 선택으로 학습한 인증된 prefix 25/50/100에서 분기한다.
각 비교는 같은 모델뿐 아니라 optimizer와 과거 학습 이력을 공유해야 한다.
세 분기점은 한 실행에서 순차적으로 세 번 판단한 결과가 아니라,
공통 이력의 각 상태에서 시작하는 별도 continuation이다.

### 9.2 분기 수와 대조군

| 구간 | Seeds | 분기점 | 각 상태의 arms | 개수 |
| --- | --- | --- | --- | ---: |
| DEV | 0,1,2 | 25,50,100 | selection_reduced, random_reduced | 18 |
| TEST | 3,4 | 25,50,100 | selection_full, random_full, gated, selection_reduced, random_reduced | 30 |
| 합계 | 5 seeds | 15 states | 위 구성 | 48 |

Full controls는 진단 없이 해당 고정 방법을 쓰는 대조군이다. Reduced controls는
진단 비용을 지불했을 때의 비교를 분리한다. Gated는 고정된 판단을 실제 실행한
별도 분기다. 결과를 본 뒤 유리한 control을 gated 결과로 복사하지 않는다.

### 9.3 판단 입력과 개발·검증 분리

입력은 최근 reward, 최근 active group 비율, 저장된 성공률 표준편차,
prefix updates의 로그다. 최근 통계의 기본 창은 20 updates다.
기본 진단은 30초 wall cap과 남은 GPU예산의 1% 제한을 둔다.
현재 상태 이후의 reward를 판단 입력으로 사용하지 않는다.

기본 ridge는 alpha=1, threshold=0이며 DEV 9개 상태에서 학습한다.
TEST 고정 대조군은 결과가 gate fitting에 유입되지 않도록 유지하면 독립 수행할
수 있다. **학습된 Gate의 TEST 실행**은 모델과 결정이 먼저 동결돼야 한다.
따라서 모든 validation이 하나의 전역 순서로 직렬 실행되는 것은 아니지만,
adaptive 결정의 통계적 의존성까지 없앨 수는 없다.

### 9.4 비용과 목표의 서로 다른 버전

`accounting=budget`은 선택 계산을 deployment 예산에 포함한다.
`accounting=matched`는 선택 계산을 별도 ledger에 두고 진단·학습 allocation을
비교한다. 후자는 같은 총비용의 비교가 아니다. 실제 업데이트 수도 자동으로 같지 않다.

`gate=final`의 label은 고정 예산 후 reward 차이다.
기존 `gate=convergence`의 label은 두 endpoint의 작은 값을 공통 목표로 정하고,
저장된 곡선 사이를 보간한 update 수와 평균 random update 비용으로 계산한다.
이 값은 **사후 목표와 비용 환산을 사용한 분석량**이다. 사전 목표에서 실제 최초
도달 비용을 측정한 값으로 표현하지 않는다. Pair는 이 차이를 직접 보강한다.

### 9.5 분기 하나의 순서와 DONE

1. 입력·parent·학습 설정 확인.
2. 해당 arm에 필요한 진단 또는 고정 decision 확인.
3. On-policy이면 validation 응답·gradient와 후보 응답·gradient·점수 계산.
4. Cached이면 저장된 성공률 점수 계산, random이면 동일 크기 무작위 선택.
5. 학습 부분집합 고정.
6. 예산 종료 규칙에 따른 학습 및 checkpoint 저장.
7. 최종 정책 eval: 응답 생성·채점·네 shard 검증 및 result 저장.
8. Convergence 조건이면 필요한 저장 checkpoint들의 curve 평가.
9. 최종 result와 요구된 curve까지 검증되면 branch DONE.

Gradient 계산 중 `validation`은 7번의 성능 eval이 아니다.
모델 업데이트가 종료됐다는 사실만으로 7~9번을 완료 처리하지 않는다.

## 10. MBPP 본 조건과 보완 cohort

### 10.1 현재 기본 조건

기본 `scripts/run_mbpp_experiments.sh`는 `quality`를 대상으로 한다.
Selector/accounting/gate는 `fresh_r/matched/convergence`다.
선택 비용은 별도 계측하며, 진단·학습 예산과 최종 보상을 비교한다.
MBPP cap은 MBPP의 calibration과 frozen contract를 따른다.
다른 데이터셋의 GPU초를 임의로 대입하지 않는다.

기본 48 branches이고 seed/분기점 구성은 위 Switch와 같다.
최종 eval은 문제당 K=8. Curve는 기본 중간 checkpoint 3개, K=4다.
시작 정책 평가와 최종 endpoint를 포함하면 보통 최대 5개 곡선 위치지만,
실제로 존재하는 고유 checkpoint 수와 학습 종료 상태에 따라 줄어들 수 있다.

이 Gate는 **on-policy 대 random**이다. Cached와 직접 선택하는 Pair가 아니다.
기존 `fresh` 비용 포함 조건, `difficulty` 조건, `long` 조건은 별도 결과로 보존한다.
현재 48칸에 다른 조건의 결과를 섞어 빈칸을 채우지 않는다.

### 10.2 보완 결과의 출처

2026-09-22 수신 파일 `mbpp_results_1.txt`의 확인된 SHA-256:

```text
7641b2ef5bbfae9a1eef6d2377a1585aaef6975d6bb36611a612a5a339987caa
```

해당 파일의 별도 repair cohort는 48 endpoints와 48 curves를 포함한다.
구성은 **기존 37개 재사용 + 5개 rerun + 6개 dependent gated**다.
새 독립 반복 48개 또는 원래 동일예산 시도 48개 성공으로 바꾸어 쓰지 않는다.
원본 시도 비용과 추가 실행 비용은 분리하고, 알 수 없는 과거 비용을 0으로 놓지 않는다.

수신 분석의 TEST full-control 6개 상태 평균:

| 비교 대상 | 평균 reward (%) | Full random 대비 차이 (percentage points) |
| --- | ---: | ---: |
| Full on-policy | 33.576389 | +0.791667 |
| Full random | 32.784722 | 0 |
| Gate policy | 33.451389 | +0.666667 |

이는 해당 보완 cohort의 기술통계다. 같은 seed의 세 상태를 독립 seed처럼 계산하지
않는다. On-policy가 유리한 학습 성능은 비용 회수 또는 H 판단의 성공과 별개다.
표의 수치는 `RESULTS_EXPORT_AUDIT_2026-09-22.md`와 원고에 반영된 보완 분석을
가리키며, 새 실행 상태를 추정한 수치가 아니다.

## 11. Selector Pair: 실측 H 보강

근거: `src/selector_pair.py`, `src/selector_pair_gpu.py`,
`src/selector_pair_train.py`, [Pair 상세 명세](SELECTOR_PAIR_RUN.md).

### 11.1 무엇을 직접 비교하는가

같은 모델·optimizer·prefix·후보 풀·평가 규칙에서 on-policy와 cached를 비교한다.
Random은 별도 기준이며, 주 H의 상대는 cached다.

```text
C_on(tau)     = 관측 checkpoint에서 목표를 처음 확인할 때까지의 on-policy 비용
C_cached(tau) = 관측 checkpoint에서 목표를 처음 확인할 때까지의 cached 비용
H(tau)        = C_cached(tau) - C_on(tau)
H > 0         : 해당 비교에서는 on-policy 비용이 작음
H <= 0        : 해당 비교에서는 cached 비용이 작거나 같음
```

목표는 원칙적으로 continuation 전에 동결한다. 실제 preregistration 여부를 주장하려면
설정의 생성·동결 기록이 결과 관측보다 앞서는지도 확인해야 한다.
기본 목표 reward는 0.35, 학습 cap은 분기당 87,120 GPU초다.
24.2 GPU시간은 4 GPU에서 순수 나눗셈상 6.05시간이지만, scoring·eval·curve까지
포함한 전체 wall-time 예측이 아니다. 이미 동결된 실험의 값은 기본값으로 교체하지 않는다.

### 11.2 정확한 42개 구성

| 구간 | 상태 수 | Arms | Continuations |
| --- | ---: | --- | ---: |
| DEV: seed 0/1/2 x t25/50/100 | 9 | on-policy, cached | 18 |
| TEST: seed 3/4 x t25/50/100 | 6 | on-policy, cached, random, adaptive | 24 |
| 합계 | 15 | 위 구성 | 42 |

한 분기만 끝나면 결과 하나는 있을 수 있지만 paired H는 아직 없을 수 있다.
DEV의 독립 학습 seed는 3개, TEST는 2개다. 42는 노드 수·seed 수·평가점 수가 아니다.

### 11.3 전체 실행 순서

1. 원본의 인증된 prefix와 두 selector의 상태 일치를 검증한다.
2. DEV의 두 selector 분기를 학습·eval·curve까지 수행한다.
3. 독립 TEST 고정 대조군 on-policy/cached/random도 배정 가능한 상태에서 수행한다.
4. DEV 아홉 상태의 유효한 paired cost label을 검증한다.
5. 고정 ridge를 학습하고 TEST 여섯 상태의 결정을 동결한다.
6. Adaptive가 선택한 selector를 자기 디렉터리에서 실제로 계산·학습·평가한다.
7. TEST matched comparisons, 선택 횟수, 비용·reward·미도달 상태를 보고한다.

Adaptive는 선택된 고정 대조군의 결과를 복사하지 않는다. 계산·학습을 실제 실행한다.
DEV label이 모두 유효해야 하는 현재 fit 규칙 때문에, 미도달이 있으면 고정 대조군의
관측 결과가 있어도 adaptive fitting까지 완료되지는 않을 수 있다.
성공한 DEV 상태만 골라 모델을 학습해서 이 조건을 우회하지 않는다.

### 11.4 Curve가 오래 걸리는 이유와 범위

Pair 기본 중간 평가점은 9개, eval K는 8이다.
시작 정책, 고유 중간 checkpoint들, 최종 endpoint를 합치면 보통 최대 11위치다.
정확한 개수는 실제 저장된 checkpoint로 정한다. 최종 endpoint는 이미 계산한 eval을
재사용하므로 11번의 새 eval을 무조건 추가하는 것이 아니다.

Curve 한 위치에서도 모델 로드 → 평가 응답 생성 → 채점 → shard 검증이 필요하다.
한 위치의 응답 생성이 100%여도 다른 위치는 남을 수 있다.
이 과정에 optimizer update는 없지만 여러 정책의 추론 비용이 든다.
자기 중간 checkpoint를 먼저 처리하고 공유 시작 정책은 뒤에 확인하는 경로다.
다른 분기가 시작 정책 평가를 소유하면 그 작업을 중복 실행하지 않는다.

### 11.5 실측과 추정의 구분

Checkpoint의 저장 시각과 같은 allocation의 비용 기록을 연결한다.
서로 다른 서버의 wall clock을 직접 빼거나 평균 step 시간으로 checkpoint 비용을
대신 만들지 않는다. Scoring이 별도 ledger에 있더라도 H 비용에는 다시 포함한다.
실패·재시도와 모델 시작 등의 할당 비용도 관련 범위에 포함한다.
Offline 평가 비용은 별도로 보고하고, 학습·선택 비용과 섞지 않는다.

목표 도달은 **평가한 checkpoint 중 처음 확인한 위치**다. 평가 사이의 연속 시간에서
실제 최초로 목표를 넘은 순간이나 전역 최소 비용을 측정한 것이 아니다.
기본 Pair는 사이를 보간해 도달 비용을 만들어내지 않는다.
목표가 시작 모델 이하인 경우와 관측 종료까지 미도달인 경우를 구분한다.
미도달은 무한비용·0점·cached/on-policy의 자동 패배로 바꾸지 않는다.

### 11.6 병렬 작업의 의미

노드는 분기를 나눠 수행하고 분기 내부는 기본 4 GPU로 작업한다.
개발 18분기와 고정 TEST 대조군 18분기는 조건이 준비되면 독립적으로 수행 가능하다.
Adaptive 6분기는 fitting·decision 동결 뒤에 열린다.
따라서 42개 전체가 처음부터 모두 실행 가능하다는 뜻은 아니다.
가용 작업 수의 상한과 클러스터 GPU 할당량은 다르다.

## 12. RLOO: 학습 방식만 바꾼 후속 비교

근거: `src/rloo_experiment.py`, `src/train_policy_rloo.py`,
[RLOO 상세 명세](RLOO_EXPERIMENT.md).

### 12.1 설계

- MATH d0/d400, seed 0/1/2: 여섯 source points.
- Random, cached `passrate_beta`, on-policy `fresh_r`: 점당 세 arms.
- Arm당 정확히 100회 추가 RLOO updates: 18회 continuation.
- 별도 parent 평가 여섯 개. 이것을 학습 6개로 합산하지 않는다.
- 고정된 기존 선택 부분집합과 E5의 비중복 300문제, 문제당 8응답을 재사용한다.

RLOO는 `A_i = r_i - mean(r_j, j != i)`와 sequence-sum REINFORCE loss를
사용한다. 기본 one epoch다. 기존의 GRPO score로 선택한 데이터를 그대로 사용하며,
RLOO gradient selector를 새로 측정하는 실험이 아니다.

d0은 base에서 시작한다. d400은 GRPO로 학습된 실제 parent adapter와 optimizer를
계승해서 401~500 updates를 RLOO로 수행한다. 전체 500회가 RLOO였다고 말하지 않는다.
모델·부분집합·생성·평가·optimizer 설정을 맞추고 continuation objective를 바꾼다.

### 12.2 순서와 완료

1. 여섯 source points, 기존 선택, 평가 문제와 설정을 검증·고정한다.
2. 한 parent evaluation을 각 source point의 공통 baseline으로 확보한다.
3. 각 arm을 100회 학습하고 정책·optimizer·lineage 기록을 저장한다.
4. 학습한 정책으로 evaluation 네 shards를 계산·검증한다.
5. Random/cached/parent 대비 reward 차이를 계산한다.
6. 필요한 parent·trained-arm 평가가 모두 검증되면 point 요약을 완성한다.

Parent evaluation과 독립 arm의 실행 순서가 반드시 위 번호로 직렬화되는 것은 아니다.
최종 point 비교에 모두 필요하다는 논리적 순서다.
**현재 RLOO에는 Pair처럼 여러 중간 checkpoint를 평가하는 curve 단계가 없다.**
100 updates가 끝나도 eval이 남으면 branch DONE이 아니다.
Phase timeout은 장애 방지 한도이며 예산을 소진하면 정상 완료시키는 규칙이 아니다.

### 12.3 해석

고정된 데이터 선택이 다른 learner에서도 도움이 되는지를 본다.
RLOO-native selector의 우위, H 경계, MBPP 일반화까지 보인 것은 아니다.
새 실행 비용에 과거 선택 계산이 포함되지 않았으면 총비용 우위를 주장하지 않는다.
Question bootstrap은 해당 trained policy pair에 조건부이며 학습 seed 불확실성을
대체하지 않는다. Random보다 좋다는 결론도 checkpoint·seed별 실제 대비를 확인한다.

## 13. MoPPS와 추가 선택기 비교

### 13.1 MoPPS online selection

근거: [MoPPS 실험 명세](MOPPS_COMPARISON.md), `src/mopps_comparison_gpu.py`.
TEST seed 3/4 x prefix 25/50/100의 여섯 상태에서 `mopps`와 `random_online`을
추가한다. 총 12 continuations이며 기존 48개 Switch를 다른 이름으로 다시 센 것이 아니다.

주 비교는 동일 시작 상태·동일 총 deployment cap의 **실제 Gate minus MoPPS**다.
MoPPS는 Beta(1,1) 시작, 목표 성공률 0.5, decay 1, 후보 배수 16을 사용한다.
한 업데이트에 네 프롬프트를 선택하고 새 reward로 posterior를 갱신한다.
초기 posterior에 미래 결과나 다른 branch의 보상을 주지 않는다.

Online random은 동일 온라인 후보 접근의 효과를 분리한다. 원래 fixed-subset
random과 같지 않다. 매 step 후보를 바꾸는 방식과 고정 10% subset을 쓰는 방식의
차이를 숨기지 않는다. MoPPS의 원 논문 전체 학습 recipe를 재현했다고 하지 않는다.
완료된 MoPPS만 있고 실제 Gate 결과가 없으면 주 비교는 아직 미완료다.

### 13.2 Additive와 terminal TayPO-2

근거: [추정량 명세](ADDITIVE_CORRECTION.md), `src/additive_experiment.py`.
기본 MATH d400 seed 0/1/2의 기존 응답으로 GPU 재채점한다. 새 학습·generation은 없다.

```text
c(w) = min(C, max(1/C, w))
gadd weight = c(r_t P_t) + c(r_t S_t) - c(r_t)
tay2_terminal weight = c(r_t) * (1 + sum_{u != t}(c(r_u) - 1))
```

Clipping 후 합성하는 정의다. 음수 weight도 유지한다. 확률분포라고 해석하지 않는다.
기존 R 방향으로 선택하고 독립 A/B 점수로 평가한다. 비교 출력은 점수 gain과
agreement이며, 새 downstream reward가 아니다. Full-pool coverage와 frozen subsets를
확인하고 `complete.json` 등을 발행해야 재채점 완료다.

### 13.3 Low-order reuse

근거: [Low-order 명세](LOW_ORDER_REUSE_RUN.md), `src/low_order_experiment.py`.
기본 MATH d100 seed 0..4, arms `random`, `pair_u2`, `low_order`,
100회 추가 GRPO와 비중복 300문제 x K8 평가다.

Validation 방향 계산 → 후보 derivative 수치검사 → 기존 8응답으로 scoring →
부분집합 고정 → 각 arm 학습 → baseline/arm 평가 순서다.
선택 단계의 후보 응답은 재사용하지만 학습·평가에서는 새 응답을 생성한다.
기본 중앙차분은 autograd와 step/half-step 검사를 통과해야 한다.
근사 optimizer geometry는 실제 전체 AdamW update와 동일하다고 주장하지 않는다.
동일 업데이트 비교이지 matched total cost 실험은 아니다.

### 13.4 Method choice

근거: [Method choice 명세](METHOD_CHOICE.md), `src/method_choice.py`.
기본 MATH d100 다섯 seed에서 일치도 기준과 독립 alignment 기준의 방법 선택을
비교한다. 둘 다 g00/g10/g01/g11 중 선택한다.
기본 7 subset arms와 full-pool control, arm당 200 updates, 500 test questions x K32다.
총 40 trained arms, 8,000 updates, parent 포함 45 policy evaluations다.

선택에 A/B를 사용했다면 그 winning A/B 점수는 더 이상 독립 성능 검증이 아니다.
방법 선택을 먼저 고정하고 별도 test reward로 검증한다. Random 10% subset은
full-pool control의 대체물이 아니다. 누락 seed를 빼고 평균 분모를 바꾸지 않는다.

### 13.5 Mixed pool

근거: `src/mixed_pool.py`, `scripts/run_mixed_pool.sh`.
기본 MATH 후보와 다른 과제 후보(MBPP)를 섞고 validation/test는 MATH에 둔다.
무작위 선택이 과제와 무관한 후보에 예산을 쓰는 상황에서 선택의 효과를 보는
양성 대조다. Pool 구성 → 새 점 측정 → subset 선택 → continuation → MATH 평가를
수행한다. 동질적인 원본 MATH 풀의 효과와 같은 모집단으로 합치지 않는다.

## 14. 이전 gate 설계의 보존 범위

서로 이름이 비슷하지만 아래 실험들은 같은 방법의 같은 결과가 아니다.

| 설계 | 진단·판단 | 학습·평가 | 현재 해석 범위 |
| --- | --- | --- | --- |
| One-shot selection gate | 전체 behavior cache 분포, 얕은 tree | 선택 또는 random을 고정하고 continuation | 초기 일회성 gate 설계 |
| Light gate v2 | Cached 점수의 반쪽 상관·분포에 대한 고정 heuristic | 기본 d100 다섯 seed, gate/random 10 branches | Heuristic 검증, 보장된 선택 규칙 아님 |
| Fixed checkpoint gate | 40개 pilot에서 독립 behavior 응답을 생성해 g11 재현성 측정 | 동일 subset이면 검증된 E5 outcome 사용 가능 | 후속 설계와 분리된 이전 실험 |
| Net-gain gate v3 | Cache+최근 학습 통계로 비용 포함 최종 reward 차이 예측 | 진단·선택 비용을 cap에 포함 | 저차/difficulty 선택기의 고정 예산 gate |
| Offline gate decision | 저장된 반쪽 점수에 고정 규칙 적용 | 새 학습 없음 | 실제 실행한 adaptive policy와 다름 |

세부 명세: [One-shot](ONE_SHOT_SELECTION_GATE.md), [Light](LIGHT_GATE_V2.md),
[Fixed](FIXED_CHECKPOINT_GATE.md), [Net-gain](NET_GAIN_GATE_V3.md).
예전 문서의 주장을 새 Pair 실측의 결론처럼 인용하지 않는다.
상관이 높다는 사실만으로 selection reward 또는 비용 이득을 보증하지 않는다.

## 15. 모델·규모·도메인 확장

### 15.1 Qwen 계열

현재 저장소의 9B 설정은 `Qwen/Qwen3.5-9B` post-trained 모델을 text-only로
사용한다. MATH/MBPP, 다섯 seed, 0/25/100/400의 40 points다.
OLMo base와 비교하면 모델 크기뿐 아니라 사전 post-training, 구조, prompt template,
adapter 대상이 함께 다르다. 순수 크기 효과라고 해석하지 않는다.

2B/4B 설정은 각각 세 seed의 별도 매트릭스다. 보존된 27B 설정도 별도 경로이며,
9B 실행을 했다고 자동으로 27B까지 돌거나 그 결과가 생기는 것이 아니다.
설정이 존재한다는 사실과 실제 결과 수신·검증 여부를 구분한다.
근거: `configs/qwen35_{2b,4b,9b}_grpo.json`, `configs/qwen38_27b_grpo.json`.
모델 이름은 로컬 frozen config의 식별자를 기록한 것으로 현재 외부 배포 현황을
새로 조사했다는 뜻이 아니다.

### 15.2 도메인 확장

`generalization_logic/science/knowledge.json`은 OLMoE-1B-7B와 Qwen2.5-14B를
각각 KK, ARC-Challenge, MMLU-Pro nonmath에 적용하는 별도 설정이다.
각 model/domain cell은 세 seed, 네 checkpoints, 후보 512·validation 100을 쓴다.
기본 2 models x 3 domains x 3 seeds x 4 checkpoints = 72 points다.
한 모델에서 세 도메인을 섞어 학습하는 실험으로 해석하지 않는다.

`olmo3_domains_grpo.json`의 OLMo 도메인 확장은 별도 설정이다. 서로 다른 모델
계열의 도메인 결과를 동일 cohort로 합치지 않는다. 각 domain은 보상·문제 형식·응답
길이가 달라 과제별 측정으로 보고한다. 주 MATH/MBPP 수치를 대신 채우지 않는다.

### 15.3 합성·측정 감사와 CFCS

`measurement_audit.py`는 비선형 cosine과 finite-budget reference의 성질을
CPU 합성 또는 저장 산출물로 분석한다. Population cosine과 유한 표본 기대값을
구분한다. 소규모 calibration 실행은 명목 신뢰구간 coverage의 실증 보장이 아니다.

CFCS는 `correction_selector.py`, `correction_selector_study.py`와
`CFCS_PROPOSAL.md`, `CFCS_RESULTS_2026-09-11.md`에 있는 로컬 연구다.
이 문서 작성 시 해당 파일들은 미추적 상태이며, 이 문서 추가가 그 코드의 배포나
실제 LLM GPU 실험 실행을 의미하지 않는다. 기존 정식 실험과 결과를 합치지 않는다.

## 16. 진행률과 완료 판정을 읽는 법

### 16.1 서로 다른 분모

- Suite 완료율: 검증된 완료 branches / 계획 branches. MBPP quality는 48, Pair는 42.
- 학습량: 완료한 optimizer updates. RLOO처럼 총량이 고정되면 완료/100으로 표시 가능.
- 선택 계산: 완료 후보 수 / 필요한 후보 수. GPU shard 전체를 합쳐야 한다.
- 응답 생성: 처리한 prompt 수 / 해당 평가의 prompt 수. Prompt당 K응답과 구분한다.
- Curve coverage: 평가가 끝난 고유 checkpoint 수 / 필요한 checkpoint 수.
- 시간: 경과 wall-time, GPU allocation 비용. **작업 완료율의 분자가 아니다.**

Budget-stopped 학습은 최종 업데이트 수가 사전에 정해지지 않을 수 있다.
그때 경과시간을 100분율로 바꾸어 학습 완료율이라고 표시하지 않는다.
확인된 `업데이트 N회 완료`를 보여주고 최종 저장·eval·curve의 완료를 따로 판단한다.

현재 node 표의 각 샘플 진행률은 해당 로그에 있는 처리량이다. Curve에서 한 평가의
응답 생성이 100%여도 전체 checkpoint coverage가 100%라는 뜻이 아니다.
전체 curve checkpoint 완료 수를 표시하지 않는 화면에서 그 값을 추정해서 읽지 않는다.
로그가 없거나 오래됐으면 현재 처리량은 미확인이다. 시간이 지난 것만으로 정상
진행 또는 정지를 확정할 수 없다.

### 16.2 무엇이 있어야 정말 끝났는가

| 실험 | 학습 종료 후 남을 수 있는 일 | 완료 판정에 필요한 결과 |
| --- | --- | --- |
| 주 매트릭스 | Fresh generation, gradients, scoring, 분석 | 필수 계약·coverage·산출물과 DONE |
| E5 | 최종 eval 및 paired 분석 | Parent/arms 정책과 검증된 평가 |
| Switch final | 최종 eval, 결과 봉인 | 유효한 final result |
| MBPP quality | 최종 eval와 중간 curve | Final result와 요구된 curve |
| Pair 한 분기 | 최종 eval, 중간·시작 curve, 비용 연결 | 검증된 curve와 result 및 비용 기록 |
| Pair 비교 | 대응 분기 및 DEV fit/TEST decision 조건 | 유효한 matched contrast; 미도달 별도 |
| RLOO | 최종 eval 네 shards, parent 비교 | 100 updates와 검증된 평가 |
| MBPP off-policy | 점수 병합·coverage·calibration export | 여섯 점의 네 추정량과 검사 기록 |
| 외부 benchmark | 각 dataset/shard 응답·채점·병합 | Benchmark별 검증된 측정값 |

## 17. 비용·통계·논문 해석 원칙

1. **동일 업데이트**는 동일 총비용이 아니다. 선택·진단·시작·평가 비용을 구분한다.
2. **동일 학습 cap**도 동일 실제 업데이트 수 또는 동일 총비용을 뜻하지 않는다.
3. **동일 총 deployment cap의 endpoint 비교**와 **고정 목표까지의 비용 비교**는
   답하는 질문이 다르다. 둘 중 하나의 결과로 다른 결과를 대신하지 않는다.
4. Allocated GPU-seconds는 GPU 할당 개수 x 경과시간이다. 순수 kernel time이나
   FLOPs가 아니다. CPU 작업 중 GPU를 점유한 시간도 해당 계측에 들어갈 수 있다.
5. 실패와 재시도 비용을 지우지 않는다. 알 수 없는 과거 비용은 빈칸과 제한사항으로 남긴다.
6. Saved checkpoint reward는 관측 위치의 성능이다. 사이 구간의 실제 최초 도달을
   관측했다고 단정하지 않는다.
7. 사후 endpoint에서 만든 목표는 사전 목표라고 표현하지 않는다.
8. 미도달은 관측 범위 내 미도달이다. 무한비용 또는 0 reward가 아니다.
9. 같은 seed의 checkpoint들을 독립 반복으로 세지 않는다.
10. Question bootstrap과 seed 간 변동은 다른 불확실성이다.
11. Cached가 대부분 유리하면 그 사실을 기술한다. 정확한 switching time을 찾아냈다는
    주장이나 상한으로 바꾸지 않는다.
12. 선택 성능이 약하거나 null 결과이면 그대로 유지한다. 유리한 상태만 뽑아 요약하지 않는다.
13. 새 실측이 계산 근사를 대체하면 표에서 근사의 지위를 명확히 바꾸되 출처를 보존한다.
14. Figure 번호보다 측정량 이름을 기준으로 연결한다. 원고 편집으로 번호는 바뀔 수 있다.

## 18. 실행·결과 명령과 파일

아래는 현재 주로 쓰는 실험의 Bash 진입점이다. 실행 명령은 빈 4-GPU allocation에서
사용한다. Status/results는 학습을 새로 시작하기 위한 명령이 아니다.

### MBPP

```bash
bash scripts/run_mbpp_experiments.sh
bash scripts/run_mbpp_experiments.sh status
bash scripts/run_mbpp_experiments.sh results
```

결과: `~/mbpp-results.txt`. Original과 repair가 있으면 별도 cohort로 기록한다.

### Selector Pair

```bash
bash scripts/run_selector_pair.sh
bash scripts/run_selector_pair.sh status
bash scripts/run_selector_pair_results.sh
```

결과: `~/selector-pair-results.txt`. 부분 결과와 완성된 paired comparison을 구별한다.
기본 Pair dataset은 MATH다. 이 명령 하나로 MBPP Pair까지 자동 실행하는 것이 아니다.

### RLOO

```bash
bash scripts/run_rloo.sh
bash scripts/run_rloo.sh status
bash scripts/run_rloo.sh results
```

결과: `~/rloo-results.txt`. 18 trained arms와 6 parent evaluations를 따로 본다.

### 새 MBPP off-policy

```bash
bash scripts/run_mbpp_offpolicy.sh
bash scripts/run_mbpp_offpolicy.sh status
bash scripts/run_mbpp_offpolicy.sh results
```

결과: `~/mbpp-offpolicy-results.txt`. 현재 결과가 없더라도 이전 TXT를 최신 측정으로
오인하지 않도록 coverage와 source hash를 확인한다.

### 그 밖의 계열

| 실험 | Bash 진입점 | 상세 명세 |
| --- | --- | --- |
| OLMo 주 매트릭스 | `scripts/run_olmo3_rlzero.sh` | [OLMo runbook](OLMO3_RLZERO_RUNBOOK.md) |
| Qwen 9B | `scripts/run_qwen35_9b.sh` | [Qwen 9B](QWEN35_9B_RUNBOOK.md) |
| Qwen 27B 보존 설정 | `scripts/run_qwen38_27b.sh` | [Qwen 27B](QWEN38_27B_RUNBOOK.md) |
| 추가 모델·도메인 | `scripts/run_additional_experiments.sh`, `scripts/run_followup.sh` | [확장 설계](FOLLOWUP_GENERALIZATION_DESIGN.md) |
| E1-E6 | `scripts/go_extensions.sh`, `scripts/run_drift_curve.sh` | [E1-E6](EXTENSIONS_2026-09-07.md) |
| E5와 public benchmark | `scripts/run_e5.sh`, `scripts/run_e5_bench.sh` | [Reduced E5](E5_REDUCED_RUN.md) |
| Reference axes | `scripts/run_reference_axes.sh` | [Reference sampling](REFERENCE_AXES_RUN.md) |
| 신뢰도 예산 | `scripts/run_reliability_budget.sh` | 이 문서 5절 및 스크립트 계약 |
| Off-policy 반쪽 점수 | `scripts/run_stale_splithalf.sh` | 이 문서 5~6절 |
| Gain-law 및 합성 | `scripts/run_gain_law.sh` | 이 문서 5절 |
| CPU 측정 감사 | `scripts/run_measurement_audit.sh` | [Method choice](METHOD_CHOICE.md) |
| CPU 비용 정리 | `scripts/run_cost_accounting.sh` | 추정 비용과 직접 계측 구분 필수 |
| CPU 분석 묶음 | `scripts/run_analyses.sh` | 새 GPU 실험이 아니라 저장 자료 분석 |
| Method choice | `scripts/run_method_choice.sh` | [명세](METHOD_CHOICE.md) |
| Mixed pool | `scripts/run_mixed_pool.sh` | [E5의 mixed pool 절](E5_REDUCED_RUN.md) |
| Additive/TayPO-2 | `scripts/run_additive.sh` | [명세](ADDITIVE_CORRECTION.md) |
| Low-order | `scripts/run_low_order.sh` | [명세](LOW_ORDER_REUSE_RUN.md) |
| Selected-prefix Switch | `scripts/run_selection_switch.sh` | [명세](SELECTION_SWITCH_EXPERIMENT.md) |
| Switch 변형 | `scripts/run_switch_quality.sh`, `scripts/run_switch_difficulty.sh`, `scripts/run_switch_hard.sh`, `scripts/run_switch_long.sh`, `scripts/run_switch_mbpp.sh` | 실행된 selector/accounting/gate별 분리 |
| MoPPS | `scripts/run_mopps_comparison.sh` | [명세](MOPPS_COMPARISON.md) |
| 이전 gate들 | `scripts/run_selection_gate.sh`, `scripts/run_light_gate.sh`, `scripts/run_fixed_gate.sh`, `scripts/run_net_gain_gate.sh`, `scripts/run_gate_decision.sh` | 이 문서 14절 |

옵션이 필요한 과거 실험은 링크된 frozen 명세를 따른다. 이 목록은 모든 과거 실험을
한꺼번에 새로 실행하라는 요청이 아니다. 사용자 서버의 실제 잔여량·GPU 할당량이나
완료 시각을 이 문서의 설계 수로 추정하지 않는다.

## 19. 결과를 추가할 때 반드시 붙일 정보

- 실험 이름·root·protocol hash·코드 revision 및 결과 파일 hash.
- 모델·dataset·seed·시작 checkpoint·arm.
- 계획 수, 관측 수, 검증된 완료 수와 미측정 항목.
- 신규 학습인지, 기존 checkpoint 평가인지, 기존 응답 재채점인지.
- 측정 reward/score/cost의 정의와 단위.
- 비교 상대, 동일 조건의 확인 항목과 달라진 조건.
- 비용에 포함한 항목, 제외한 역사적 비용, 미확인 비용.
- 재사용·rerun·새 branch 여부와 원본 실험과의 관계.
- 표본 단위, paired 계산 방식, 불확실성의 조건부 범위.
- 주장 가능한 결론과 아직 주장할 수 없는 결론.

이 정보가 없는 단독 평균이나 진행률 숫자로 논문의 실험 결론을 대체하지 않는다.
