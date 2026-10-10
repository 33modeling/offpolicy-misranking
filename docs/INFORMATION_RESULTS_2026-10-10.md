# Information 실험 결과 분석 — 2026-10-10

MATH에서는 On-policy가 SR보다 probe 개선 방향에 더 잘 맞는 업데이트를 만들었다. SR이 정답과 오답이 섞인 문제를 더 많이 골랐는데도 나타난 차이다. 성공률만으로는 업데이트의 방향을 설명하지 못한다는 단서가 있다.

다만 실제 업데이트 후 probe loss는 On-policy가 평균적으로 더 좋지 않았다. MBPP에서는 업데이트 방향 차이도 작다. 이번 결과는 **초기 On-policy 우위의 원인을 입증한 결과가 아니라, 문제 선택과 실제 업데이트 사이의 차이를 보여준 초기 상태의 관측**으로 보는 것이 맞다.

## 1. 분석 대상과 완료 범위

| 항목 | 확인한 내용 |
| --- | --- |
| 모델 | `allenai/Olmo-3-1025-7B` |
| 모델 revision | `a81bae42db3975be1671e27b9c9a56da1a9f980f` |
| Dataset | MATH, MBPP |
| Seed | Dataset별 5, 6, 7, 8, 9 |
| 측정 시점 | 모두 `t0`. 초기 weights와 optimizer state |
| 측정 완료 | 10/10, `complete=true`, `pending=[]`, `errors=[]` |
| 선택 후보 | 측정당 40문제, 총 400개의 dataset·seed별 후보 기록 |
| 선택 비교 | 측정당 On-policy와 SR 각각 4문제 |
| 실제 업데이트 | 방법별 1회. 총 20회 |
| 선택 문제 기록 | 총 80개. 방법·seed 사이에 중복될 수 있으므로 고유 문제 80개라는 뜻은 아님 |
| 실제 학습 응답 | 문제당 8개, 총 640개 |
| 독립 probe | 측정당 8문제, 문제당 고정 응답 8개 |
| 학습 objective | GRPO, 실제 AdamW step |

이 파일은 Qwen3.5, Gemma4, Llama3.1의 실험 결과가 아니다. 초기·중기·말기 비교를 위한 수집 코드로 실행했지만, 이번 파일에 들어 있는 시점은 초기 `t0` 하나뿐이다.

선택은 같은 초기 상태에서 비교했다. On-policy는 scoring rollout A의 projected dense gradient 점수로 문제를 고른다. SR은 입력에 저장된 cached success rate로 고른다. 독립 scoring rollout B는 선택 결과를 다시 측정하는 데 사용한다. 실제 학습 응답은 A/B와 별도로 생성하며, 두 방법의 학습 응답도 서로 다른 seed로 생성한다.

Probe는 선택용 reference 및 최종 evaluation 300문제와 분리되어 있다. 아래 probe loss는 고정 응답에 대한 GRPO surrogate loss이며, 새 응답의 정답률이나 최종 benchmark 점수가 아니다.

## 2. 어떤 문제를 골랐나

아래 값은 dataset별 5개 seed의 평균이다. `mixed`는 한 문제의 8개 응답에 정답과 오답이 모두 있는 경우다. 50%에 가까운 성공률과 mixed 여부는 다른 지표다. 1/8, 7/8 성공도 mixed에 포함된다.

| 독립 rollout B에서 측정 | MATH 후보 전체 | MATH On-policy | MATH SR | MBPP 후보 전체 | MBPP On-policy | MBPP SR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 정답률 | 26.50% | 33.75% | 47.50% | 27.31% | 42.50% | 47.50% |
| Mixed 문제 비율 | 70.5% | 75.0% | 95.0% | 65.0% | 100.0% | 100.0% |
| 선택용 gradient cosine | 0.00393 | 0.00685 | 0.01148 | 0.00341 | 0.01463 | 0.00702 |

MATH에서는 SR이 현재 모델에서도 정답·오답이 섞인 문제를 더 잘 확보했다. On-policy가 더 낮은 성공률의 문제를 고른 것이 관측되지만, 이 사실만으로 어려운 문제의 정보가 더 유용했다고 해석할 수는 없다. 원본의 난도 label은 기록되어 있지 않다.

MBPP에서는 두 방법 모두 독립 B에서 mixed 문제만 골랐다. 다만 실제 학습용 응답을 다시 생성하면 mixed 비율은 On-policy 90%, SR 95%로 떨어진다. 선택 당시의 유효 신호가 학습 rollout에서도 그대로 유지되는 것은 아니다.

| 실제 학습 rollout | MATH On-policy | MATH SR | MBPP On-policy | MBPP SR |
| --- | ---: | ---: | ---: | ---: |
| 정답률 | 31.25% | 43.13% | 41.25% | 45.00% |
| 정답 수 / 응답 수 | 50/160 | 69/160 | 66/160 | 72/160 |
| Mixed 문제 수 / 선택 문제 수 | 15/20 | 17/20 | 18/20 | 19/20 |
| 문제별 GRPO gradient가 0인 문제 수 | 5/20 | 3/20 | 2/20 | 1/20 |

정답만 있거나 오답만 있는 문제는 해당 rollout에서 reward advantage가 0이 되어 직접적인 GRPO gradient를 만들지 않는다. 성공률 기반 선택도 새 rollout에서는 이런 문제를 완전히 피하지 못했다.

두 방법이 동시에 선택한 문제는 MATH에서 평균 0.6/4개, MBPP에서 1.0/4개다. 같은 성공률 범위에 있어도 실제 선택 집합은 상당히 달랐다.

## 3. 선택용 gradient 점수는 재현되는가

같은 checkpoint에서 scoring A/B를 독립적으로 생성했을 때, 후보 40개의 점수 순위 상관은 낮았다.

| Seed | MATH A/B Spearman | MBPP A/B Spearman |
| --- | ---: | ---: |
| 5 | 0.0715 | 0.0851 |
| 6 | -0.0441 | 0.2676 |
| 7 | 0.3560 | 0.1756 |
| 8 | 0.1476 | 0.0419 |
| 9 | -0.4242 | -0.2482 |
| Seed별 평균 | **0.0214** | **0.0644** |

Pearson 상관의 seed별 평균도 MATH 0.0773, MBPP 0.0525다. 이 설정의 한 번의 rollout으로 얻은 순위는 반복 생성했을 때 안정적이지 않았다. A/B에서는 후보 응답과 reference 응답을 모두 다시 생성하므로, 이 값으로 후보 gradient의 잡음과 reference gradient의 잡음을 분리할 수는 없다.

MATH의 독립 B cosine은 오히려 SR이 높았다. MBPP에서는 On-policy가 높았지만, 아래 실제 LoRA/AdamW 업데이트의 정렬에서는 차이가 거의 사라진다. **선택용 점수의 우위를 실제 학습 업데이트의 우위로 바로 옮겨 쓰면 안 된다.**

선택용 점수는 projected dense LOO gradient를 사용한다. 실제 업데이트는 LoRA parameter에 대한 GRPO gradient와 AdamW를 사용한다. Parameter 범위, advantage, token 길이의 가중 방식, optimizer 변환이 다르다. 선택 점수가 측정하는 신호와 실제 학습이 반영하는 신호 사이의 불일치가 이번 분석의 중요한 지점이다.

Cached SR과 현재 B 성공률의 차이도 있다. 선택 집합에서 `현재 B - cached`의 평균은 MATH On-policy +7.5%p, SR +0.625%p, MBPP On-policy +2.5%p, SR -2.5%p다. 이번 시점은 `t0`이므로 이 차이를 학습에 따른 cache 노후화의 증거로 쓸 수 없다. 유한한 응답 수에 따른 표본 변동도 포함된다.

## 4. 실제 업데이트는 어떤 방향으로 움직였나

`g_probe`를 업데이트 전 probe loss의 gradient, `Δθ`를 실제 AdamW step의 parameter 이동량으로 두면 다음과 같다.

```text
업데이트 정렬 = cosine(-g_probe, Δθ)
1차 예측 loss 변화 = g_probe · Δθ
실제 loss 변화 = L_probe(θ + Δθ) - L_probe(θ)
```

정렬은 높을수록 probe loss를 줄이는 국소 방향에 가깝다. Loss 변화는 낮을수록 좋다. 이 셋을 구분해서 봐야 한다.

| 실제 업데이트 지표 | MATH On-policy | MATH SR | MBPP On-policy | MBPP SR |
| --- | ---: | ---: | ---: | ---: |
| Clip 전 batch gradient norm | 0.24960 | 0.20157 | 0.10119 | 0.11306 |
| 실제 parameter update norm | 0.019454 | 0.019405 | 0.019352 | 0.019380 |
| Probe 개선 방향과 실제 update cosine | **0.31109** | **0.23473** | 0.13177 | 0.12572 |
| 1차 예측 probe loss 변화 | -0.00093310 | -0.00072117 | -0.00020910 | -0.00019736 |
| 실제 probe loss 변화 | +0.00035466 | +0.00012350 | +0.00028480 | +0.00027818 |

MATH에서 On-policy의 gradient norm은 더 컸지만 실제 parameter 이동량은 두 방법이 거의 같았다. 실제 이동량의 크기보다 방향에서 차이가 나타났다. MBPP에서는 SR의 gradient norm이 더 컸으며, 실제 이동량과 정렬 차이는 작았다.

20개 batch에서 gradient clipping에 따른 norm 감소는 없었다. 두 방법의 차이를 clip 작동 여부로 설명할 수는 없다. 같은 초기 AdamW state에서 zero-gradient step을 한 이동량도 20개 모두 0이었다. 이번 `t0` 결과에는 이전 optimizer momentum으로 인한 이동이 관측되지 않았고, actual update와 batch incremental update가 같았다.

### Seed별 실제 정렬

| Dataset | Seed | On-policy | SR | On-policy - SR |
| --- | ---: | ---: | ---: | ---: |
| MATH | 5 | 0.42907 | 0.37989 | +0.04919 |
| MATH | 6 | 0.05051 | 0.01868 | +0.03183 |
| MATH | 7 | 0.31196 | 0.02734 | +0.28462 |
| MATH | 8 | 0.38517 | 0.35184 | +0.03333 |
| MATH | 9 | 0.37873 | 0.39592 | -0.01719 |
| MBPP | 5 | 0.03665 | 0.14821 | -0.11156 |
| MBPP | 6 | 0.18630 | 0.15870 | +0.02761 |
| MBPP | 7 | 0.29174 | 0.25865 | +0.03309 |
| MBPP | 8 | 0.11357 | 0.02114 | +0.09243 |
| MBPP | 9 | 0.03056 | 0.04191 | -0.01135 |

MATH에서는 4/5개 seed에서 On-policy가 높다. 다만 seed 7을 제외하면 평균 차이는 +0.07636에서 +0.02429로 줄어든다. MBPP에서는 3/5개 seed에서 높지만 평균 차이는 +0.00604이고, seed 7을 제외하면 -0.00072다.

통계 단위는 응답 640개나 선택 문제 80개가 아니라 **dataset별 paired seed 5개**로 잡았다. 아래는 seed별 차이의 평균과 불확실성이다. 서로 다른 dataset은 합치지 않았다.

| Paired 지표: On-policy - SR | MATH 평균 차이 | MATH 95% t 구간 | MBPP 평균 차이 | MBPP 95% t 구간 |
| --- | ---: | --- | ---: | --- |
| 실제 update 정렬 | +0.07636 | [-0.07147, +0.22418] | +0.00604 | [-0.08768, +0.09977] |
| 실제 probe loss 변화 | +0.00023116 | [-0.00102401, +0.00148633] | +0.00000662 | [-0.00055275, +0.00056599] |

구간은 `mean ± t(0.975, df=4) × sample_sd / sqrt(5)`로 계산했다. Seed 5개뿐이므로 분포 가정에 민감하다. 차이의 부호 대칭을 가정한 탐색적 양측 sign-flip 검정은 실제 update 정렬에서 MATH p=0.125, MBPP p=0.8125였다. 모든 `2^5=32` 부호 조합을 열거했다. 여러 진단 지표를 확인한 뒤의 탐색적 분석이며, 사전에 정한 확증 검정으로 취급하지 않는다.

MATH의 방향 우위는 관측되었지만 불확실성이 크다. MBPP의 작은 평균 차이를 우위라고 주장할 근거는 없다. 반대로 이 표만으로 두 방법이 동등하다고 증명한 것도 아니다.

## 5. 방향이 좋아 보이는데 실제 loss는 왜 줄지 않았나

아래는 실제 업데이트 전후의 probe loss 변화다. 표의 수치는 **실제 변화 × 10,000**이다. 음수는 개선, 양수는 악화다.

| Dataset | Seed | On-policy | SR | 실제 loss가 더 낮은 방법 |
| --- | ---: | ---: | ---: | --- |
| MATH | 5 | +13.76993 | -0.88521 | SR |
| MATH | 6 | -0.05299 | -2.37660 | SR |
| MATH | 7 | -2.79904 | +9.82037 | On-policy |
| MATH | 8 | -1.24219 | -0.95973 | On-policy |
| MATH | 9 | +8.05735 | +0.57622 | SR |
| MBPP | 5 | +1.54781 | +2.04919 | On-policy |
| MBPP | 6 | +7.75136 | +3.53522 | SR |
| MBPP | 7 | +0.30475 | +7.69726 | On-policy |
| MBPP | 8 | +2.97492 | +0.47572 | SR |
| MBPP | 9 | +1.66127 | +0.15155 | SR |

On-policy가 SR보다 실제 loss가 낮았던 경우는 두 dataset 모두 2/5개다. 업데이트 전보다 loss가 줄었던 경우는 MATH에서 두 방법 각각 3/5개이고, MBPP에서는 두 방법 모두 0/5개다.

1차 예측은 20개 업데이트 모두 음수였다. 그러나 실제 변화가 음수인 경우는 6개뿐이다. **국소 gradient 정렬과 실제 finite-step loss 변화가 자주 반대 부호를 보인다.** 이것은 이번 결과를 해석할 때 반드시 남겨야 하는 문제다.

Probe loss의 초기값이 거의 0인 것은 정답률이 0이어서가 아니다. 같은 policy의 old log probability를 기준으로 ratio가 1이고, 문제별로 중심화한 GRPO advantage의 합이 0이기 때문이다. 이 loss는 cross-entropy와도 다르다.

이 JSON만으로 부호 불일치의 원인을 특정할 수는 없다. 곡률과 ratio clipping, 실제 step 크기, forward 계산의 수치 정밀도 등을 구분하려면 저장된 weights·응답을 이용한 재계산이 필요하다. Gradient clipping이 없었다는 사실은 **surrogate objective의 ratio clipping**이 없었다는 뜻은 아니다. Probe loss 악화를 곧바로 새 rollout 정답률의 악화로 바꾸어 해석해서도 안 된다.

## 6. 실제 선택 문제: MATH seed 7

Seed 7은 MATH 정렬 차이에 가장 크게 기여했다. 유리한 사례만 고르지 않고 두 방법의 선택 문제 4개씩을 모두 적었다. 문제 설명은 원문을 줄인 것이며, 공식 난도·분야 label은 아니다. `문제 gradient 정렬`은 개별 문제의 실제 LoRA GRPO gradient와 probe gradient의 cosine이다.

| 방법 | 선택 문제 | Cached SR | 독립 B 성공률 | 학습 성공률 | 문제 gradient norm | 문제 gradient 정렬 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| On-policy | 정수 m,n이 3m+4n=100일 때 최소 \|m-n\| | 12.5% | 75.0% | 50.0% | 0.26527 | +0.36866 |
| On-policy | Amy, Ben, Chris의 나이 조건에서 Chris의 현재 나이 | 37.5% | 62.5% | 50.0% | 0.02447 | -0.03862 |
| On-policy | z=(2x-y)²-2y²-3y의 순차 minimax에서 x 선택 | 0.0% | 0.0% | 25.0% | 0.02452 | -0.02180 |
| On-policy | 복소수 \|(1-i)⁸\| 계산 | 62.5% | 50.0% | 50.0% | 0.83759 | +0.56302 |
| SR | 7,2,x,10의 평균이 9일 때 x | 50.0% | 62.5% | 62.5% | 0.05452 | +0.01419 |
| SR | 8x≡1 mod p가 해를 갖지 않는 소수 p의 합 | 50.0% | 62.5% | 50.0% | 0.16073 | +0.05032 |
| SR | 유리함수 (4x³+2x-4)/(3x³-2x²+5x-1)의 수평 점근선 | 50.0% | 0.0% | 0.0% | 0.00000 | 정의되지 않음 |
| SR | a=8에서 (16∛a²)^(1/3) 계산 | 50.0% | 75.0% | 87.5% | 0.03519 | +0.00143 |

이 seed에서 SR은 cached 성공률이 모두 50%인 문제를 골랐다. 하지만 수평 점근선 문제는 현재 B와 학습 rollout에서 정답이 하나도 없어 gradient가 0이었다. 나머지 문제의 probe 정렬도 낮았다.

On-policy가 고른 네 문제가 모두 좋았던 것은 아니다. 나이 문제와 minimax 문제의 개별 gradient 정렬은 음수였다. 반면 복소수 문제는 큰 gradient norm과 양의 정렬을 함께 보였다. 선택 문제의 성공률 평균만 보면 놓치는 차이다. 다만 이 값은 실제 AdamW 업데이트의 문제별 기여도를 선형 분해한 결과가 아니다. 개별 norm과 cosine만으로 각 문제의 기여 비율을 계산하지 않는다.

반대 사례도 있다. MBPP seed 5에서는 On-policy의 독립 B 선택 점수 평균이 0.02272로 SR의 0.00719보다 높았지만, 실제 업데이트 정렬은 0.03665 대 0.14821로 SR이 높았다. 선택 점수와 실제 업데이트 사이의 불일치는 On-policy에 유리한 방향으로만 나타나지 않았다.

## 7. 초기 adapter에서는 무엇이 바뀌었나

업데이트마다 학습 가능한 LoRA parameter는 8,388,608개, parameter tensor는 128개였다. 총 20개 업데이트의 2,560개 parameter 기록을 확인했다.

| Parameter | 업데이트당 tensor 수 | 실제 변화 |
| --- | ---: | --- |
| LoRA A | 64 | 20회 모두 이동량 0 |
| LoRA B | 64 | 20회 모두 각 tensor에 변화가 있음 |
| 학습 가능한 LoRA dtype | — | `torch.float32` |

측정된 parameter 이동량의 제곱 norm은 100% LoRA B에 있었다. 이는 일반적인 LoRA 초기화에서 B가 0이면 첫 step의 A gradient가 0이 되는 구조와 일치한다. 다만 이번 분석에서 JSON 외의 checkpoint tensor를 직접 열어 초기 B 값을 별도로 검증한 것은 아니다.

따라서 이번 `t0`은 첫 adapter update의 성질을 강하게 반영한다. 이미 여러 번 업데이트한 초기 학습 구간 전체를 대표한다고 보기 어렵다. 특히 원래 SRGC continuation은 shared prefix 25 updates 뒤에 시작한다. **t0의 비교와 continuation이 실제 갈라지는 시점의 비교는 다르다.**

## 8. 정답·오답 응답의 probability 변화

학습 rollout의 응답을 고정하고, 실제 update 전후의 token log probability 변화를 확인했다. 아래는 **응답별 평균 token logp 변화의 응답 단위 평균 × 10,000**이다. 문제별 평균을 다시 평균한 값과 구분했다.

| Dataset·방법 | 정답 응답 수 | 정답 변화 | 오답 응답 수 | 오답 변화 |
| --- | ---: | ---: | ---: | ---: |
| MATH On-policy | 50 | +0.12853 | 110 | -8.27301 |
| MATH SR | 69 | +0.04988 | 91 | -24.31685 |
| MBPP On-policy | 66 | +0.25290 | 94 | -1.41004 |
| MBPP SR | 72 | +0.43655 | 88 | -6.52416 |

응답 단위 평균에서는 네 경우 모두 정답 logp가 소폭 오르고 오답 logp가 내렸다. 오답 감소량은 두 dataset 모두 SR이 컸다. 이 지표도 On-policy가 일관되게 더 좋다는 설명을 주지 않는다.

길이 가중을 바꾸면 해석도 바뀐다. 예를 들어 MATH On-policy 오답의 응답 단위 평균은 -0.00082730이지만, 전체 오답 token에 동일한 가중을 둔 평균은 +0.00005950이다. MATH SR 정답도 응답 단위에서는 양수지만 token 단위에서는 -0.00002075다. 응답 길이와 problem별 정답 개수가 다르므로 가중 방식 없이 “정답 확률을 높였다”라고 한 줄로 쓰면 부정확하다.

평균 completion 길이는 MATH On-policy 892.5 tokens, SR 694.2 tokens, MBPP On-policy 757.1 tokens, SR 840.3 tokens였다. 이 결과에서 추론 길이가 장기 성능 차이의 원인인지까지는 확인되지 않았다. 새 응답을 생성해서 측정한 성공률 변화도 없다.

## 9. 비용과 기록 검증

측정 내용은 10/10 완료했지만 **GPU 비용 기록은 0/10 완료**다. 비용 ledger에 종료되지 않은 phase가 총 55개 있고, 각 측정의 `total_gpu_seconds`가 `null`이다.

완료된 phase에서 확인되는 allocated GPU 시간의 합은 약 33.358 GPU-hours다. 이는 전체 실험 비용이 아니다. 누락된 시간과 방법별 비용을 복원하지 않았으며, 이 숫자로 On-policy와 SR의 비용 효율을 비교하지 않는다. 실패·중단 이력이 비용 기록에 남은 것과, 최종 측정 결과가 완료된 것은 별개다.

이번 파일에서 다음을 직접 대조했다.

- Dataset·seed·stage가 2×5×1의 10개 조합과 일치한다.
- 선택 문제 80개, 실제 batch update 20개, 후보 기록 400개, 학습 응답 640개가 맞는다.
- 문제마다 응답 8개, batch마다 32개가 있고, 원시 reward 평균이 문제 성공률 및 batch `sample_reward`와 일치한다.
- 128개 parameter tensor의 update norm 제곱합이 실제 batch update norm의 제곱과 일치한다.
- 문제별 gradient 재구성 오차는 20개 모두 기록된 허용 범위 안이다. 최대 `오차/허용범위`는 0.72877이고, 최대 `오차/batch gradient norm`은 약 0.00010845다.
- 모든 기록은 동일한 측정 코드 hash를 사용하며, stage는 0, source checkpoint hash는 `null`이다.

이 검증은 다운로드된 JSON 내부 기록의 일관성 확인이다. 클러스터의 원본 phase 파일이나 `.pt` tensor를 이번 장비에서 다시 hash 검증한 결과는 아니다. Export 당시의 원본 검증과 현재 로컬 분석을 구분한다.

## 10. 이 결과로 답할 수 있는 질문

**초반부터 SR을 사용하면 학습 신호가 없나?** 그렇지 않다. SR은 MATH 17/20개, MBPP 19/20개 선택 문제에서 실제 GRPO gradient를 확보했다. MATH에서는 mixed 문제 확보도 On-policy보다 좋았다.

**그런데 On-policy가 잡는 다른 정보가 있나?** MATH에서는 실제 업데이트의 probe 방향 정렬이 더 높았다. 특히 seed 7의 실제 선택 문제를 보면 성공률이 비슷해도 gradient norm과 probe 정렬이 크게 다르다. 성공률 기반 선별과 방향 기반 선별이 같은 정보를 잡지는 않는다는 관측이다.

**그 차이가 초반 학습 성능 우위를 설명하나?** 아직 부족하다. 실제 probe loss에서 일관된 우위가 없고, scoring 순위가 불안정하며, adapter의 첫 update만 측정했다. 장기 학습 결과와 연결하려면 실제 continuation이 갈라지는 checkpoint에서 같은 내용을 확인해야 한다.

**초기·중기·말기에 정보가 어떻게 달라지는가?** 이번 파일로는 답할 수 없다. 기존에 저장된 weights와 optimizer state를 함께 가진 checkpoint의 측정이 필요하다. 우선순위는 shared prefix 직후, 중기, 후기의 비교와 고정 rollout에서의 실제 loss 변화 재확인이다. 이번 분석 과정에서는 새로운 학습이나 측정을 실행하지 않았다.

논문에 현재 쓸 수 있는 문장은 다음 정도다.

> 초기 OLMo-3-7B의 MATH 진단에서 성공률 기반 선택은 정답·오답이 혼재한 문제를 더 많이 확보했지만, On-policy 선택의 실제 GRPO/AdamW 업데이트는 독립 probe의 국소 개선 방향에 평균적으로 더 잘 정렬되었다. 그러나 이 차이는 seed 간 변동이 크고 실제 한 단계 probe loss 개선으로 이어지지 않았으며, MBPP에서는 뚜렷한 차이가 관측되지 않았다.

## 분석 출처

- 입력: `/home/kms/Downloads/cur/information-results.json`
- 입력 크기: 574,668,871 bytes, 약 548.05 MiB.
- 입력 SHA-256: `8c39cbbedaeca0c3b55fcc75a8af5e9ad5bd96bff986f339d5d6a2461dbd5b3f`
- 입력 export schema: `srgc-information-results-v1`
- 측정 protocol: `srgc-selection-information-v1`
- 측정 코드 hash: `c30cc0e85055a32cc4a27b063afc7348cd33286ad74d5bbb98d3438e094722f6`
- 방법·지표의 코드 정의: [선택 정보 측정 설명](SRGC_SELECTION_INFORMATION.md), [측정 코드](../srgc_research/information.py), [probe loss](../srgc_research/backend.py), [요약 계산](../srgc_research/information_report.py).

원본 다운로드 파일은 수정하지 않았다. 큰 token 배열을 전체 메모리에 올리지 않고 streaming으로 읽었다. 결과 파일에는 원시 token 배열과 scalar 요약이 반복되어 있었으므로 파일 크기 자체는 측정 수나 효과의 크기를 뜻하지 않는다. 기본 내보내기는 compact 형식으로 수정되어, 선택 정보·실제 update 지표·응답별 reward와 logp 변화만 남긴다.
