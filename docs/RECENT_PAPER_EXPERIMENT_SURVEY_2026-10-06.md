# 최근 추가 논문들의 추가 실험과 Switch 후속 설계

확인일: 2026-10-06. **문헌 조사와 실험 제안이며, 아래 신규 제안은 구현·실행 결과가 아니다.**

## 조사 범위

저자가 최근 추가한 Obsidian 논문 노트와 Confluence용 로컬 자료에서 논문을 찾고,
각 논문의 arXiv 본문·부록을 대조했다. 실제 Confluence 서버의 최신 페이지를 조회한 것은 아니다.
이번에는 논문의 최고 성능보다 **추가 실험에서 무엇을 고정하고 무엇을 바꿨는지**를 조사했다.
학회 채택 여부를 새로 판정하는 조사는 아니다.

- Obsidian: `../../obsidian/Data-Selection-Five-Papers-2026-10-06-MOC-ko.md`,
  `../../obsidian/Learner-Aware-Data-Four-Papers-2026-10-06-MOC-ko.md`, 개별 논문 노트.
- Confluence 자료: `../../confluence/explanations/`의 `lesser`, `arcus`, `lev`,
  `saturated-data`, `mifs`, `tessy`, `quality-utility-paradox`, `fac-synthesis`,
  `echo-grpo`, `datamaster`, `finetuning-with-sampling` 자료.
- MOPD 분석 논문은 Obsidian의 `papers/2610.02179-*.md`와 원문으로 확인했다.

## 1. 논문들이 실제로 수행한 추가 실험

다음 표는 원문에 있는 실험이다. Switch에 적용할 아이디어는 다음 절에서 별도로 구분한다.

| 논문·원문 위치 | 실제 추가 실험 | 검증하려는 질문 |
| --- | --- | --- |
| [LESSER](https://arxiv.org/html/2610.03702v1), Sec. 5, App. C/E/F | 점수 10분위별로 별도 SFT; 선택 집합 겹침과 개별 gradient 순위 비교; 무작위 배치 평균을 제거한 batch-gradient alignment; 학습 전 구간 query-loss 추적; RL reward 오염 대조 | 높은 점수가 실제 학습 효과와 연결되는가? 서로 다른 문제를 골라도 같은 방향으로 배우는가? 이득이 언제 발생하는가? |
| [ARCUS](https://arxiv.org/html/2609.38018v1), Sec. 4, App. E/F | 불확실성·drift·dead-zone·pacing 등을 각각 제거; 고정 난도와 이동 난도 비교; 응답 그룹 크기 4/8/16; 후보 생성 여유분, batch·LR·pool 크기 변화; cold-start prior/probe; 모델·loss 확장 | 난도 선택, 무신호 회피, 상태 추정 중 무엇이 기여하는가? 더 생성해서 얻은 성능인가? |
| [Quality-Utility Paradox](https://arxiv.org/html/2606.16152v1), Sec. 4-6, App. C/H | 같은 문제 집합에서 학생 답변·교사 수정·문체 보존 수정을 비교; token/PPL 분석; reward model 교체; 학습률과 batch 조합, 모델 크기·계열 변경 | 좋은 품질 점수가 나쁜 학습 결과로 이어지는 이유가 무엇인가? 내용 수정과 표현 변화의 효과를 구분할 수 있는가? |
| [TESSY](https://arxiv.org/html/2604.14164v2), Sec. 4.1-4.4 | 모델 A용으로 합성한 데이터를 B에, B용 데이터를 A에 교차 학습; teacher 교체; 합성 데이터 자체의 정답률과 학습 후 성능을 별도 평가; OOD 평가 | 데이터가 일반적으로 좋은 것인가, 해당 학습자와 잘 맞는 것인가? |
| [Echo-GRPO](https://arxiv.org/html/2608.26684v1), Table 1/4, App. D/E | 같은 수의 random token과 semantic token만 각각 unclipping; 두 reference를 하나씩 제거; top-k 1/5/10; 수학 도메인·데이터 규모 확장; 일회성 재작성 시간 측정 | gradient를 더 주는 것과 필요한 위치에 주는 것 중 무엇이 중요한가? 각 구성요소가 실제로 필요한가? |
| [FAC / Less is Enough](https://arxiv.org/html/2602.10388v4), Tables 3/6/12, App. L | dense coverage, SAE+Random, SAE+FAC 비교; 누락 feature 비율 변경; annotation 오염; threshold·사전 크기·sparsity 변화; feature 추출 모델·생성 모델·학습 모델 분리 | representation과 선택 규칙의 기여를 분리할 수 있는가? 지표의 오차와 모델 변경을 견디는가? |
| [Save Your Saturated Data](https://arxiv.org/html/2609.33126v1), Tables 2/3/4, Figs. 2/3, App. C | 답만 오답으로 교체, 오답 조건부 이어쓰기, 전체 negative 생성 비교; zero-padding 개수 변화; 포화 문제 비율 변화; 새 포화 문제 반복 탐색; 모델·seed 확장 | 보상 분산이나 advantage 크기만 회복하면 충분한가, 유용한 응답 대비가 필요한가? |
| [Lev](https://arxiv.org/html/2609.32484v1), Table 2, Sec. 4.3, App. D.4/E.2 | code 데이터·혼합 비율을 고정하고 web 선별법만 교체; 다른 code subset에서도 반복; code 문서 잔존량 분석; 선별 비율·batch·선택 단위 변화 | 전체 평균에 가려진 도메인 손실이 있는가? 데이터 구성과 선별 기준 중 무엇이 영향을 주는가? |
| [MIFS](https://arxiv.org/html/2609.16059v1), Figs. 3/6, App. F.6/F.7 | density/difficulty/trajectory 필터 누적 적용; raw와 curated 학습곡선 비교; 일반 시각 능력 평가; 추가 baseline; verifier-human 일치 검사 | 어느 정제 단계가 유용한가? 빠른 수렴과 높은 학습 reward가 test 성능으로 이어지는가? 검증기가 맞는가? |
| [DataMaster](https://arxiv.org/html/2608.10579v1), Sec. 4 Ablation, Figs. 2/3, App. C | Domain Agent 출력만 기존 SuperFiltering에 결합; Stage 2, 2+3, 2+3+4 비교; 같은 10K 예산으로 여러 데이터 원천·모델·도메인 비교 | 전체 시스템이 아니라 어느 단계가 개선을 만드는가? 기존 방법에 그 단계만 넣어도 유효한가? |
| [Finetuning with Sampling](https://arxiv.org/html/2610.02140v1), Sec. 5.3, Figs. 3-5, App. B/C | sampling 횟수 변화에 따른 분포 차이와 정확도; pass@k 곡선; 다른 모델용으로 가공한 데이터 교차 사용; 변환된 trace 정답률; 기존 능력 보존 평가 | proxy가 좋아지면 성능도 좋아지는가? 반복 시도 성능과 기존 능력도 유지되는가? |
| [From Gradients to Capabilities / MOPD](https://arxiv.org/html/2610.02179v1), Sec. 3-5, App. C/F | 동일 응답·동일 Adam 상태에서 loss 가중 방식만 변경; moment 초기화 대조; 현재 gradient를 0으로 둔 update 제거; FP32/BF16 비교; gradient 근사와 최종 과제 성능 비교 | 높은 update cosine이 현재 데이터 때문인가, 공통 optimizer 이력 때문인가? gradient 충실도가 실제 성능을 보장하는가? |

### 원문 결과를 가져올 때 지킬 구분

- LESSER의 초기 효과 집중과 centered batch 분석은 주로 SFT다. 이를 online RL 후반의
  gradient 선별 무용성이나 특정 전환 시점의 증명으로 쓰지 않는다.
- Echo의 unclipping은 teacher trace가 포함된 mixed-policy 조건이다. Switch의 fresh-rollout
  학습에서 같은 원인이 발생한다고 가정하지 않는다.
- Saturated Data의 zero-padding은 magnitude-only 대조지만, 서로 다른 negative 생성법의
  gradient norm까지 정확히 맞춘 실험은 아니다.
- MIFS의 누적 필터 비교는 데이터 양도 변한다. 각 필터의 독립 효과를 모두 분리한 실험이 아니다.
  DataMaster의 단계 누적 비교 역시 모든 조합의 factorial ablation은 아니다.
- 논문의 추가 모델·seed 실험은 일반화와 변동성을 확인하는 데 필요하다. 다만 이미 받은
  MATH/MBPP 다섯 seed 결과가 있는 상태에서 동일-prefix replicate를 가장 먼저 늘릴 이유는 약하다.

## 2. Switch에서 이미 하는 것과 실제로 비어 있는 것

| 검증 항목 | 현재 구현 | 이번 조사에 따른 판단 |
| --- | --- | --- |
| 초반/중반/후반 selector 효과와 점수 정보 제거 | E14-E16: On-policy, direction shuffle, SR, SR shuffle, fresh-SR, Random | 의미 있는 개입 실험이다. 실행 중인 것을 유지하고 결과를 먼저 분석한다. |
| 포화·무신호 group과 오래된 SR 캐시 | mechanism의 실제 학습 reward 기록, fresh-SR; 기존 support/cache 분석 | 새 실험처럼 중복 생성하지 않는다. 현재 성공률·유효 group·held-out 개선을 연결해 해석한다. |
| 다른 backbone과 cache 갱신 | Qwen, matched refresh 등 기존 runner | 이미 구현된 범위다. 추가 논문을 이유로 같은 runner를 다시 만들지 않는다. |
| 싼 gradient 선별법과 강한 비-gradient online 선별법 | 현재 SR-GC runner에서 LESSER·ARCUS 구현 미확인 | 외부 비교군이 비어 있다. 기존 비싼 On-policy에만 대비한 효율 주장을 점검할 가치가 크다. |
| 실제 선택 batch의 centered gradient와 optimizer 효과 | mechanism은 선택 문제별 B cosine/dot/norm의 평균을 저장 | 평균 cosine은 합산 batch gradient의 cosine이 아니다. centered batch·history-subtracted update는 별도 진단이 필요하다. |
| 점수 순위 전체와 실제 학습 이득의 대응 | mechanism은 선택된 여섯 집합의 25-update 효과를 비교 | 상·중·하 점수 구간별 calibration이나 1-step 대 25-step 연결은 별도 확장이다. |
| reference 집합 교체의 영향 | 같은 ranking-validation 문제에 독립 A/B 응답을 생성 | 새 응답의 변동과 reference 문제 구성의 변동은 다르다. 실제 reference 교체 실험은 아직 필요하다. |

근거 코드: [mechanism](../scripts/srgc_stage_mechanism.py)의 `acquire`와
`finalize_measurement`, [실행 목록](../scripts/srgc_experiments.py),
[기존 설계](STAGE_MECHANISM_EXPERIMENTS_2026-10-06.md).
현재 mechanism 측정은 cosine/dot/norm을 저장하지만 개별 gradient vector 전체를 반환하지 않는다.
따라서 JSON만으로 centered batch나 Adam update를 나중에 정확히 복원할 수 있다고 약속하지 않는다.

## 3. 추가 실험 추천: 해결할 질문 기준

아래 순서는 **신규 설계 제안의 우선순위**다. 기존 실행 queue나 E01-E16 조건을 바꾸지 않는다.
SR-GC는 보조 지표라는 위치를 유지한다. 단순 점수 상관만으로 실제 성능 차이를 대체하지 않는다.

### 우선 1: 더 싼 gradient 선별법과도 비교

**질문:** 현재 On-policy의 gradient 계산이 비싸기 때문에만 Switch가 유리한가?

LESSER의 출력층 gradient를 사용해 선별을 계속하는 조건을 추가한다. 같은 시작 상태,
후보·query 문제, 문제당 응답 수, 선택량, 재선별 간격, 학습법을 맞추고
기존 On-policy와 Switch에 비교한다. LESSER의 원문 runtime 배수를 기존 로그에 곱하지 않고 실측한다.
출력층 scoring으로 바꾸어도 실제 GRPO training의 학습 대상 parameter는 그대로 유지한다.

**측정:** held-out learning curve, 동일 목표 도달 시간, 동일 GPU 예산 성능;
candidate/query rollout, scoring forward/backward, projection/ranking, training,
SR 일회성 cache 비용을 구분한다. 목표 미도달은 누락하거나 도달시간 0으로 적지 않는다.

**왜 replicate보다 먼저인가:** 같은 방법의 반복은 비교군의 계산을 낮춰도 전환이 유용한지 답하지 못한다.
성능과 비용 모두를 보는 직접적인 외부 대조다. LESSER가 더 유리한 결과도 그대로 보고한다.

**범위:** step-25 공통 prefix를 쓰면 그 이후의 조건부 비교다. 처음부터의 end-to-end 주장에는
각 방법을 t0부터 비교해야 한다. 두 결과를 섞지 않는다.

### 우선 2: 점수가 실제로 어떤 업데이트를 만들었는지 확인

**질문:** 후반에 줄어드는 것이 점수의 신뢰성인가, 유효한 gradient의 크기인가,
optimizer를 거친 증분 효과인가, 독립 평가 성능인가?

LESSER와 MOPD의 진단 구조를 참고한다. 보존된 동일 모델·optimizer 상태에서
selector별 선택 batch를 만들고 raw batch gradient, random 평균 제거 후 gradient,
실제 optimizer update, `U(g; H) - U(0; H)`를 구분한다.
`H`는 optimizer 상태이며 원고의 H 진단과 같은 기호라는 뜻이 아니다.
필요하면 문서·출력에서는 `optimizer_state`로 표기한다.

**통제:** 동일 loss reduction·응답 길이 처리·학습 batch 크기·정밀도를 사용한다.
독립 random batch로 centering 기준을 추정하고 같은 표본을 양쪽 기준에 중복 사용하지 않는다.
AdamW의 weight decay·step counter·moments를 복제하며, zero-gradient는 `grad=None`과 구별한다.
현재 학습의 optimizer를 SGD로 바꾸는 것이 아니라 복제 상태에서 원인을 측정한다.

**측정:** 단계별 raw/centered alignment, gradient norm, 실제 update norm,
독립 평가의 전후 차이. 점수 구간별 학습을 추가한다면 같은 크기의 상·중·하 구간을
동일 상태에서 분기해 1-step과 짧은 학습 후 개선을 비교한다. 이는 기존 E16의 추가 확장이다.

**기존 실험과 관계:** shuffle·fresh-SR 개입은 이미 진행 중이다. 이를 재실행하지 않는다.
gradient vector나 optimizer snapshot이 보존되지 않았다면 별도 진단 실행이 필요하다.
센터링 수치가 달라졌다는 것만으로 기존 SR-GC가 틀렸다고 결론내리지 않는다.

### 우선 3: 실제 reference 집합과 학습 규모의 민감도

**질문:** 특정 reference 문제 구성이나 작은 batch 조건에서만 관측되는 현상인가?

먼저 같은 크기의 reference 집합을 학습/평가와 분리해 미리 정하고, 동일 checkpoint에서
각 reference의 ranking·SR-GC 부호·top-k를 비교한다. 이는 A/B rollout 재추출과 다른 개입이다.
실제 Switch 성능의 안정성까지 주장하려면 각 reference에 따른 자체 학습 경로도 실행해야 한다.
동일 경로에서 재계산한 신호만으로 다른 경로의 최종 보상을 추정하지 않는다.

그다음 ARCUS·LESSER·Lev의 규모 검증을 참고해 후보 수와 선택 수를 분리해서 바꾼다.
후보 수만 바꾸는 대조와, 선택 비율을 유지하면서 학습 batch도 늘리는 대조는 별개다.
모델·pool·evaluation은 고정하고 생성 응답·token·누적 학습 노출량·GPU 시간을 모두 보고한다.
동일 update 수가 동일 학습량이나 동일 계산량을 뜻하지 않는다.

**우선 이유:** V6의 reference 의존성과 작은 실험 범위에 직접 대응한다.
넓은 hyperparameter 격자를 한꺼번에 추가하지 말고 주요 비교가 유지되는 최소 확장부터 설계한다.

### 그다음: ARCUS와 데이터 구성 대조

ARCUS는 처음부터 성공률 이력을 갱신하는 강한 비-gradient 비교군 후보다.
공식 방법을 옮길 때 belief·pacing·post-rollout filtering을 유지해야 한다.
그 방법을 임의로 기존 후보 40개 안의 top-4 규칙으로 줄여 놓고 원래 ARCUS라고 부르지 않는다.
현재 학습 구성에 맞춘 변형이면 그 차이를 명시하고, 전체 비용을 맞춘 비교를 함께 둔다.

Lev/FAC를 참고한 topic·난도·재노출·유효 group 분포 분석도 가능하다.
그러나 MATH와 MBPP의 평균 차이만으로 데이터 다양성이나 포화를 원인이라고 정하지 않는다.
관측으로 차이가 확인된 뒤 동일 크기·난도 등의 조건을 맞춘 추가 선택 대조를 설계한다.
새 SAE 학습이나 teacher trace 재작성 전체 pipeline은 현재 Switch 검증에 바로 필요한 것은 아니다.

## 4. 비용과 실행 상태

- 이번 변경은 조사 문서와 안내 링크뿐이다. 신규 실험 runner·실행 명령·GPU 결과는 없다.
- 실행 중인 mechanism, 기존 frozen runtime·plan·cache·checkpoint를 수정하지 않았다.
- fixed200 채택 제외와 replicate 보조 위치는 유지한다. 기존 결과는 보존한다.
- deployment 비용에는 선택에 쓰는 단일 reference만 포함한다. 별도 진단 B·centering·
  optimizer probe 비용은 연구 진단 비용으로 분리한다.
- SR cache의 최초 rollout·채점·저장과 cache load/sort를 구별한다.
  cold-start 비용과 재사용 이후 비용을 각각 보고하고, 누락된 계측을 0으로 채우지 않는다.
- LESSER의 scoring 시간은 rollout 생성 이후의 추가 pass이며, Echo의 재작성은 초기 처리다.
  ARCUS는 rollout·token·wall-clock을 구분한다. 서로 다른 범위의 비용을 직접 나누지 않는다.
- 논문 본문·PDF와 Confluence/Obsidian 원노트는 변경하지 않았다.
