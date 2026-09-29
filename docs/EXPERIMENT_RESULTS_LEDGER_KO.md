# 수행한 실험과 결과 기록

확인일 **2026-09-28**. V7 작성에 사용할 기존 결과와 추가 실험의 현재 확인 상태를
모은다. [새 실험 명령·CPU/GPU·노드 수](REBUTTAL_COMMANDS_KO.md),
[전체 실험 후 V7 완성 일정](REVIEW_SCHEDULE_2027_KO.md)과 함께 본다.

원고 저장소 `344856e3f7d4cdc90cba662bfff6d48b19ad8438`의 V7 TeX와 보존 evidence를
직접 읽었다. 아래 원고 링크는 이 commit에 고정했다. 보존된 JSON의 step-275 보상·비용과
SR-GC 진단은 일부 재계산했으며, 원격 GPU의 현재 파일을 다시 수집한 것은 아니다.
과거 답변의 “완료” 표현만으로 새 실험 결과를 채우지 않았다.

**단위:** reward는 %, reward 차이는 percentage points(pp), 비용은 명시한 GPU-h/GPU-s.
서로 겹치는 정책·평가·진단을 더해 전체 독립 실험 수로 세지 않는다.

## 1. 기존 실험 요약

| 기록 ID | 실험 / 질문 | 확인된 결과 | 확인 수준 |
| --- | --- | --- | --- |
| H01 | MATH에서 selection이 random보다 좋은가, 단계에 따라 달라지는가? | 초기 On-policy +0.58 pp, 후기 SR +1.74 pp; 중간 Off-policy +1.51 pp | V7의 seed별 학습 결과와 평균 표 확인 |
| H02 | 고정 step에서 SR로 전환하면 어떤가? | 개발 18개 실행; 유사 길이 8쌍 중 SR 6승, 평균 +1.11 pp | 42-endpoint export의 개발 부분과 원고 확인 |
| H03 | 기록된 seed 3/4 전환 결과 | step 275에서 Switch 평균 35.02%; On-policy 대비 +2.48, SR 대비 +0.96 pp | step-275 JSON 재계산; 실행 이력 구분 필요 |
| H04 | 동일 목표 보상 도달 비용 | 35% 목표의 두 paired 비교에서 SR 비용 차이 −9.23, −8.70 GPU-h | checkpoint 연결 비용; 한 SR 비용은 재구성 |
| H05 | MATH 고정 GPU 예산 | 보존 reward 차이 On-policy −1.74, SR +0.83 pp | **관련 선택 비용은 교정 원자료 대기** |
| H06 | MBPP selection/random | Full held-out 6상태 평균 +0.79 pp; 비용이 있는 4상태 +0.76 pp | 서로 다른 coverage를 분리해 확인 |
| H07 | RLOO objective 대조 | 27개 학습 결과; 세 시작 checkpoint 모두 평균 SR > On-policy | 원고 표 및 추가 d100 export 확인 |
| H08 | SR-GC 사후 진단 | 9상태 중 부호 일치 7; Pearson 0.058, Spearman 0.033 | 9상태 JSON 확인; Pearson 재계산 |
| H09 | Single-check로 전략 선택 | SR 대비 0/6승, On-policy 대비 3/6승; 평균 −2.17/−0.99 pp | 별도 6상태 endpoint 비교 |
| H10 | 반복 checkpoint 진단 | seed별 11 checks; 평균 reference 시계열에서 신호 step 125/100 | 사후 reference 민감도 분석과 전환 결과 분리 |
| H11 | 점수 신뢰도·selection gain | Off-policy gain이 Gaussian 예측보다 큰 행 MATH 22/24, MBPP 13/24 | score 진단; 후속 학습 reward가 아님 |
| H12 | Correlation gate pilot | 초기 MATH 평균 이득 음수, d100에서 고정 SR보다 세 seed 모두 낮음 | 기존 학습 pilot 결과; SR-GC와 다른 규칙 |
| H13 | SR-GC 집계 CPU 시간 | 1회 18.009815 ms wall / 0.935607 ms process CPU | 모의 벡터의 산술만 측정 |

이 표의 “확인”은 보존 자료와 수치를 확인했다는 뜻이다. 전체 원격 checkpoint 계보,
모든 비용 receipt 또는 해당 논문의 해석까지 새로 인증했다는 뜻은 아니다.

## 2. H01 — MATH selection / Random, public benchmark

OLMo-3 7B, GRPO, 시작 checkpoint d=0/100/400, 각 seeds 0–2에서 100 additional
updates 후 비교한 기록이다. Random/SR/On-policy/Off-policy의 36개 학습 결과가
seed별 표에 있으며 correlation-gate pilot은 아래 H12로 분리한다.

| Selector | MATH d=0 | MATH d=100 | MATH d=400 | Public macro d=0 | Public macro d=400 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cached SR | −0.60 | +1.15 | +1.74 | +0.35 | +0.86 |
| On-policy alignment | +0.58 | +0.78 | +0.12 | +1.02 | +0.06 |
| Off-policy alignment | −0.58 | +1.51 | +0.82 | +0.50 | +0.22 |

모두 Random 대비 pp, 3-seed 평균이다. Public macro는 다섯 benchmark의 동등 가중
평균이며 다른 평가 목표다. 초기에는 On-policy, 후반에는 SR이 더 좋은 평균을 보인다.
다만 d100은 seed 3개 중 2개가 On-policy를 선호하면서 평균은 SR을 선호한다.
모든 상태에서 같은 순서라는 결과는 아니다. [평균 표][S01], [seed별 결과][S02].

## 3. H02/H04 — 고정 step 전환, Pair 결과와 목표 비용

개발 seeds 0–2, branch steps 25/50/100에서 On-policy를 유지하거나 SR로 바꾼
**9쌍/18개 실행**이다. 각 곡선에는 평가 11개가 기록되어 있다.

| Branch step | 요약에 포함한 쌍 | On-policy 평균 % | SR 평균 % | SR−On pp | SR 높은 쌍 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25 | 3 | 32.88 | 34.54 | +1.67 | 2/3 |
| 50 | 2 | 35.58 | 36.98 | +1.40 | 2/2 |
| 100 | 3 | 37.14 | 37.51 | +0.37 | 2/3 |
| 합계 | 8 | 35.15 | 36.27 | +1.11 | 6/8 |

endpoint는 301–319 additional updates 범위다. s1/t50은 SR 311, On-policy 407
updates여서 위 요약에서 제외했으며 원자료에서는 보존한다. 정확히 같은 update의
비교로 쓰지 않는다. 별도 held-out seeds 3/4의 6쌍에서는 SR이 4쌍에서 높고
평균 차이는 +1.18 pp다. [고정 전환 요약][S03], [전체 18 endpoint][S04], [held-out][S05].

35% reward를 처음 관측한 평가는 SR 6/9, On-policy 5/9이며, 양쪽 모두 도달한
상태는 4개다. 비용까지 있는 비교는 그중 2개다.

| 상태 | SR 첫 도달 updates / GPU-h | On-policy 첫 도달 updates / GPU-h | H_SR = SR−On 비용 |
| --- | ---: | ---: | ---: |
| s0/t50 | 250 / 19.34 | 285 / 28.57 | −9.23 GPU-h |
| s2/t100 | 285 / 21.95 | 310 / 30.65 | −8.70 GPU-h |

s0/t50 SR 비용은 실행 기록에서 재구성했다. s0/t25는 SR만 목표를 관측했다
(319 updates, 24.17 GPU-h). 미도달·도달했으나 비용 미계측은 0이 아니다.
계산에는 prefix와 보고용 평가를 제외하고 목표 도달까지 selection/setup/training을
포함한다. [목표 도달 전체 표][S06].

Pair 최종 export는 **42/42 endpoints, 42 curves, 462 evaluations**를 보존한다:
개발 18 + held-out 고정 대조 18 + single-check 6. 이는 별도 실험 462개가 아니다.
`branch_results_complete=true`와 전체 paired certification은 별개이며, export에는
후자가 미실행으로 기록되어 있다. 두 saved-final endpoint의 예산 적격성도 별도다.
[export 설명][S07], [원본 TXT][S08]. 원본 SHA-256:
`972d391809e3585666ea82a8b283fef3eff7dc968074ca73ed4cc65e9cf01d23`.

## 4. H03 — seed 3/4 전환, total step 275

공통 step-25 출발 이후 250 additional updates의 보상이다.

| Seed | 전환 step | Random % | On-policy % | SR % | Switch % | Switch−On pp | Switch−SR pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 125 | 29.33 | 32.12 | 32.33 | 33.04 | +0.92 | +0.71 |
| 4 | 100 | 31.08 | 32.96 | 35.79 | 37.00 | +4.04 | +1.21 |
| 평균 | — | 30.21 | 32.54 | 34.06 | 35.02 | +2.48 | +0.96 |

보존 JSON의 반올림 전 값으로 차이를 다시 계산했다. [step-275 원자료][S09],
[보상 표·그림 설명][S10].

현재 V7의 [실행 대응 기록][S11]은 과거 A/B checks가 On-policy 대조 경로에서
측정됐고, optimizer 복구를 위한 전환 전 replay 중 seed 4는 원 상태와 다르다고
구분한다. 이 기록을 새 E01/E02의 **자기 경로·25-update refresh 실행 결과**라고
재분류하지 않는다. 기존 관측과 새 구현의 일치 여부는 새 실험으로 확인한다.

보고된 비용 성분의 평균은 아래와 같다. **학습 시간을 곱해 선택 비용을 다시 만들지 않는다.**

| Arm | Training GPU-h | Selection GPU-h | 기존 SR cache 준비 GPU-s | 보고 성분 합 GPU-h |
| --- | ---: | ---: | ---: | ---: |
| Random | 20.18 | 0 | 0 | 20.18 |
| On-policy | 19.69 | 6.53 | 0 | 26.22 |
| SR | 19.20 | 0 | 1.231 | 19.20 |
| Switch | 19.12 | 2.62 | 1.231 | 21.74 |

합계는 반올림 전 값으로 계산한다. Switch selection은 seed별 2.84/2.40 GPU-h의
**전환 전 선별 시간 총합으로 저자가 제공한 값**이다. On-policy와 training은 로그 기반
성분이다. 이 표는 공유 prefix·cache 최초 생성·별도 A/B 진단 획득 비용을 제외한
보고 성분이며, 새 periodic-refresh 구현의 동일 절차 end-to-end 실측으로 쓰지 않는다.
1.231 GPU-s는 기존 cache를 읽고 순위를 정하는 시간이다.
[개별 seed 비용 표][S12], [저자 제공 selection 시간 기록][S13].

## 5. H05 — MATH 고정 예산: 비용 교정 필요

보존 실험은 selection+training에 29,040 GPU-s를 배정했다. held-out 6상태에서
On-policy−Random 평균 −1.74 pp(5/6에서 낮음), SR−자기 Random 대조 평균 +0.83 pp
(5/6에서 높음)으로 기록되어 있다. [원고 결과][S14].

**보존된 문제 값:** On-policy selection 평균 6.84 GPU-h, training 1.17 GPU-h,
평균 14.3 optimizer updates. 저자가 이전 Table 7의 실험 값이 잘못됐다고 지적했고
교정 원자료는 아직 확보되지 않았다. 이 숫자는 오류 추적용으로 남기며 확정된 비용
결론이나 새 실험 예산 산정에 사용하지 않는다. [보존 JSON][S15], [교정 필요 기록][S16].

표 번호가 바뀌므로 `compute_comparison.json`, `tab:switch-current`,
`6.84`로 추적한다. 새 E01/E02 비용을 단위당 가격으로 환산해 과거 값에 덮어쓰지 않는다.

## 6. H06 — MBPP 학습과 비용

| 비교 범위 | 결과 | 해석 범위 |
| --- | --- | --- |
| 개발 9쌍, diagnosis-adjusted | On-policy−Random 평균 +1.26 pp | 진단비 차감 후 학습 배정 |
| held-out 6상태, Full | 평균 reward On-policy 33.58%, Random 32.78%; 차이 +0.79 pp | 4양수·1동률·1음수; 평균 updates 89.83/87.17 |
| held-out 6상태, diagnosis-adjusted | 평균 34.22% vs 32.65%; +1.57 pp | Full과 별도 조건 |
| 보상+비용이 있는 Full 4상태 | reward 차이 +0.76 pp; score+train 16.690 vs 7.785 GPU-h | 6상태 전체 reward 평균과 구분 |

paired 95% 질문 bootstrap 구간은 개발 8쌍, Full held-out 4쌍, diagnosis-adjusted
held-out 4쌍에 있다. 4상태 비용 평균은 scoring 8.900, On-policy training 7.790,
Random training 7.785 GPU-h다. 평가·기타 setup·diagnosis·공유 prefix는 제외하며
계측 phase 내 실패 시도는 포함한다. [MBPP 전체 표][S17], [해석·별도 조건][S18].

별도로 scoring을 28,380 GPU-s 배정 **안에** 넣은 조건에서는 selected 21개 arm에
최종 reward가 없고, selection 시간 평균은 28,359.80 GPU-s로 기록됐다.
미측정 reward를 0으로 계산하지 않는다. cached-SR 대조 s0/t25는 32.08% vs 32.00%
(+0.08 pp, CI [−1.46,+1.58]), scoring 1.693 GPU-s다. s0/t50은 SR 33.46%만 있어
paired Random 결과로 쓰지 않는다.

## 7. H07 — RLOO objective 대조

d=0/100/400 × seeds 0–2 × Random/SR/On-policy = **27개 학습 결과**, 시작 정책
평가 9개다. 이전 목록의 “18 runs”는 d100 보강 전 범위로, 현재 합계와 구분한다.
같은 시작 상태·선별 subset·100 updates·평가 조건을 두고 continuation objective를
RLOO로 바꾼 비교이며 RLOO에서 subset을 다시 선별한 결과가 아니다.

- 평균 SR reward가 세 checkpoint 모두 On-policy보다 높다.
- d0에서 SR−On-policy 평균은 +0.18 pp, 2/3 seeds가 SR을 선호한다.
- d100은 3/3 seeds가 SR을 선호한다. On-policy−Random은 seed 0에서 −3.71 pp다.
- d400 seed 0에서 RLOO−GRPO 차이는 Random +13.83, SR +13.71,
  On-policy +13.88 pp다. 이 상태에서는 selector 하나만의 개선으로 읽지 않는다.

[27개 결과·구간][S19], [d100 원본 export][S20]. d100 export의 `complete=true`와
3개 상태×3개 arm을 확인했다. 질문 bootstrap CI를 seed 간 재현성 구간으로 부르지 않는다.

## 8. H08/H09/H10 — SR-GC 진단, single-check, 반복 점검

| 기록 | 규모 | 결과 | 다른 실험과의 구분 |
| --- | --- | --- | --- |
| 사후 D 부호 vs 학습 보상 순서 | 9상태 | A 5/9, B 7/9, A/B 평균 7/9; A/B 부호 불일치 4상태 | 이미 수행한 비교 결과의 진단 |
| D 크기 vs On-policy−SR 보상 차이 | 같은 9상태 | Pearson 0.058, Spearman 0.033 | 장기 이득 크기를 잘 예측한다는 근거 아님 |
| 결과를 본 stage 요약 | 같은 9상태 | 6/9, 선택된 정책 reward 평균 30.13% | 사후 요약이며 독립 온라인 rule이 아님 |
| 한 번 D를 측정해 전략 선택 후 학습 | 별도 held-out 6상태 | SR 대비 0/6승, 평균 −2.17 pp; On-policy 대비 3/6승, −0.99 pp | 반복 확인의 효과를 단독으로 분리하는 대조가 아님 |
| 반복 checkpoint 진단 | seeds 3/4, 각 11회 | A/B 평균 시계열의 최초 trigger 125/100 | H03과 연결되지만 추가 독립 학습 seed로 세지 않음 |

seed 3의 step 50/75/100은 음→양→음이지만 세 check 평균 +2.15여서 그때 전환하지
않고 step 125에서 신호를 낸다. 별도 준비된 t100 시작 상태의 D와 step-25 경로에서
도달한 step 100의 D는 같은 상태 측정이 아니다. [진단 설명][S21], [9상태 JSON][S22],
[single-check 전체 표][S23].

single-check의 동일 전략 반복도 최종 update가 다르므로 보상 차이를 곧바로 동일
조건 training 표준편차로 해석하지 않는다. 그래서 E03 독립 반복과 E04 고정 시점
대조를 새로 준비한다.

## 9. H11/H12 — score 신뢰도와 correlation gate

On-policy 독립 순위의 top-k overlap 평균은 다음과 같다. 데이터셋별 5 seeds다.

| Pool | d0 | d25 | d100 | d400 | 독립 uniform 기준 k/n |
| --- | ---: | ---: | ---: | ---: | ---: |
| MATH | 0.135 | 0.170 | 0.135 | 0.095 | 0.1000 |
| MBPP | 0.102 | 0.133 | 0.125 | 0.122 | 0.0996 |

선별 score gain은 한 응답 표본으로 고르고 다른 표본으로 평가한 값이다.
Off-policy gain이 Gaussian 예측보다 큰 행은 MATH 22/24, MBPP 13/24다.
MBPP off-policy는 6 source points, 4 estimators의 24행이며 mean measured gain
0.08105, mean prediction 0.04713 score-SD다. 이는 학습 reward가 아니다.
반복 SR 행과 같은 응답을 공유한 estimator 행은 독립 반복 수에 더하지 않는다.
[overlap 표][S24], [전체 calibration 수치][S25], [MBPP 진단 JSON][S26].

별도 correlation gate pilot은 uniform 10 updates 후 SR 또는 Random을 골라 90
updates를 더 학습한다. d0/400의 6개 실행과 d100의 3개 보강 실행이 있다.
d0의 Random 대비 차이는 −0.71/−0.79/−0.96 pp, d400은 −1.21/+2.04/0.00 pp다.
d100에서 고정 SR 대비 −1.00/−0.50/−1.83 pp로 모두 낮다.
Public macro의 Random 대비 평균은 d0 +0.71, d400 +0.06 pp다.
[pilot 및 해석][S27]. SR-GC temporal switching과 같은 방법으로 묶지 않는다.

Gaussian/binomial/t3 synthetic gain 관측도 보존되어 있다([수치][S25]).
이는 CPU 합성 검증이며 실제 LLM 학습 성능 또는 GPU 속도 실험으로 세지 않는다.

## 10. H13 — CPU 산술 및 코드 검증

**CPU SR-GC 단일 모의 측정:** wall 18.009815 ms, process CPU 0.935607 ms.
각 40개 집합의 합집합 72 prompts, validation 25, 4096차원 float64 모의 벡터로
평균·차이·내적·부호 판단만 수행했다. 1회이며 import/벡터 생성/rollout/backward/
디스크 읽기/CPU-GPU 전송은 제외했다. [측정 기록][S28].
원고의 18.32 ms는 [별도 저자 제공 평균 기록][S13]이며 이 단일 관측값과 같다고
표시하지 않는다. 둘 다 gradient 획득 전체의 무료 실행을 입증하지 않는다.

**2026-09-28 Qwen 코드 감사:** 247개 CPU 회귀 검사 통과, 모의 두 노드 프로세스가
60개 queue 작업을 중복 없이 처리, 실제 tokenizer로 10개 bundle/40개 continuation
plan 준비. [감사 기록](QWEN35_SRGC_AUDIT_KO.md).
이 테스트·입력 준비 수를 학습 완료·reward 결과로 세지 않는다.

## 11. 새 실험 E01–E09: 아직 확인된 결과가 없는 범위

아래 표는 **2026-09-28 확인 기록**이다. 2026-09-30 명령 점검에서 E04의
`switch_fixed200` 구현을 확인했다. 최신 실행 가능 범위는
[명령 모음](REBUTTAL_COMMANDS_KO.md)을 따르며, 새 GPU 결과를 확인했다는 뜻은 아니다.

| 새 실험 | 현재 확인 상태 | 결과 MD에 추가할 것 |
| --- | --- | --- |
| E01/E02 OLMo seeds 5–9 | 구현됨; 저자에게 실행 진행을 전달받았으나 최신 완료 export 미수신 | 모든 seed×네 arm endpoint, own-path D, phase 비용 |
| E03 독립 반복 | 미구현 | 사전 고정 반복 수·stream, paired 변동 |
| E04 total-step-200 | 미구현 | 동일 prefix의 고정 전환 vs Switch |
| E05/E06/E07 추가 arm 30개 | 구현됨; GPU 완료 결과 미확인 | seed별 reward, 갱신/복귀 이력, 추가 selection 비용 |
| E08 Qwen 네 arm 40개 | 준비·CPU 검사 완료; 9B GPU 결과 미확인 | 새 Qwen cache/prefix, 모델별 paired 결과와 비용 |
| E09 방향 대조 | 미구현 | 대조 조건 사전 고정 및 matched 결과 |

현재 환경에서 `/group-volume`을 읽을 수 없어 **로컬에서 못 찾은 결과를
서버에서도 미완료라고 단정하지 않는다.** 위 상태는 원격 완료 검증 여부다.
모든 결과를 수집하고 V7까지 완성한 뒤 리뷰를 맞는 것이 작업 목표다.

## 12. 그 밖의 보존 연구와 확인 범위

- **CFCS CPU 합성 탐색:** 로컬 `outputs/cfcs-anchor-study-20260911/summary.json`
  및 `docs/CFCS_RESULTS_2026-09-11.md`에 기록이 있다. 실제 MATH/MBPP/LLM 실험은
  아니다. drift 4, K=8, 20 repeats에서 anchor CFCS−g00는 −1.923 pp,
  paired SE 0.573 pp로 확인했다. 일반적 우위를 입증한 결과가 아니며 V7 주 결과에
  합치지 않는다. 해당 연구 파일은 기존 미추적/로컬 산출물이라 이번 commit에 포함하지 않았다.
- **2026-08-24 readout:** 서로 다른 generation revision을 합친 보고서로 이미
  폐기되어 있다. [폐기 사유](results/2026-08-24/PROVENANCE_STATUS.md)를 유지하며
  그 평균을 새 V7 근거로 사용하지 않는다.
- **이전 모델 확장·MoPPS·low-order·method-choice·기타 gate/E1–E6:**
  [전체 실험 명세](EXPERIMENTS_COMPLETE_GUIDE_KO.md)에 설계·코드가 있다.
  이 문서 작성 시 확인된 최종 수치가 없는 계열을 완료로 승격하지 않는다.
  원시 export를 수집하면 실험 revision과 완료 coverage를 붙여 별도 행으로 추가한다.

## 13. 다음 결과를 받을 때 갱신할 항목

각 결과에 experiment ID, model/revision, dataset/seed/arm, 시작·종료 step,
prefix/plan/input/code hash, reward와 paired 차이, 비용 범위·누락 여부,
원본 파일·export 시각을 붙인다. 기존 행과 같은 정책/평가를 재사용하면 그 관계를
명시한다. 결과가 불리해도 보존하고, 원고 반영 여부와 실험 완료 여부를 따로 기록한다.

<!-- Sources pinned to the inspected manuscript commit. -->

[S01]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/training_table.tex
[S02]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/math_training_table.tex
[S03]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/fixed_switch_summary_table.tex
[S04]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/pair_branch_endpoints.tex
[S05]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/pair_heldout_endpoints.tex
[S06]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/pair_target_observations_table.tex
[S07]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/evidence/2026-09-25/selector-pair-final-results.md
[S08]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/evidence/2026-09-25/selector-pair-final-results.txt
[S09]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/evidence/pair_progress.json
[S10]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/online_results.tex
[S11]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/code_correspondence.tex
[S12]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/pair_common_step_costs.tex
[S13]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/evidence/2026-09-26/switch-prefix-author-timings.json
[S14]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/results.tex
[S15]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/evidence/compute_comparison.json
[S16]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/REBUTTAL_PLAN.md
[S17]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/mbpp_primary_tables.tex
[S18]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/online_comparison_mbpp.tex
[S19]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/rloo_results_table.tex
[S20]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/evidence/2026-09-25/rloo-d100-results.txt
[S21]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/sr_gc_diagnostics.tex
[S22]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/experiments/sr-gc-20260923/analysis.json
[S23]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/pair_single_check_table.tex
[S24]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/repeatability_table.tex
[S25]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/calibration_values.tex
[S26]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/evidence/mbpp_offpolicy_calibration.json
[S27]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/overleaf/sections/appendix_controls.tex
[S28]: https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/344856e3f7d4cdc90cba662bfff6d48b19ad8438/v7/evidence/2026-09-26/srgc-cpu-one-check.md
