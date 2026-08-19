# Off-Policy Misranking 논문·실험 전수 점검

- 점검일: 2026-08-19
- 기준 커밋: `291381c9ba9ab3e6ff5ee21bb40ddc1a96e8e8ca`
- 논문 기준 파일: `paper/main.tex` (`sha256: 6e36542bfab975d6d0d74179ba881818c1e01ed7ceaa5fdae45610a38b3d2753`)
- 범위: 본문 전 문장, 수식·증명, 실험 코드, 외부 결과표, 통계 처리, 재현성 자료, 참고문헌
- 판정 용어:
  - **확정 결함**: 코드·수식·결과표 사이의 직접 모순이나 계산 오류가 확인됨
  - **검증 필요**: 현재 산출물만으로 주장을 지지할 수 없으며 추가 실험 또는 증명이 필요함
  - **표현 과장**: 제한된 결과보다 본문 결론의 범위가 넓음

## 1. 총평

현재 원고의 핵심 문제의식, 즉 한쪽만 보정한 off-policy gradient가 정책 순위까지 뒤집을 수 있다는 방향은 유효하다. 그러나 현재 LLM 실험은 **실제 샘플링 정책과 likelihood 계산 정책이 다르고**, **종료 뒤 패딩 토큰을 gradient·ratio·학습에 포함하며**, **중요도비를 거의 전부 clip**하고 있다. 이 세 문제는 논문의 핵심 독립변수와 oracle을 동시에 오염시키므로, 현 수치로 실증 결론을 확정해서는 안 된다.

또한 top-k 정의가 분석 스크립트마다 `25`와 `26`으로 갈리고, corrected floor 구현·통계 검정·CertaGrad 평가에도 서로 다른 수준의 결함이 있다. 따라서 현재 상태는 **이론 construction은 보강 후 유지 가능하지만, LLM 표·그림·FIRST 프로토콜·CertaGrad 결론은 수정 코드로 재실험하기 전까지 보류**가 맞다.

## 2. 제출 차단 항목(P0)

### P0-1. 샘플링 분포와 importance-ratio 분포가 다름: 확정 결함

**증거**

- `src/rollout.py:27-30`은 `temperature=1.0`, `top_p=1.0`만 지정하고 이를 full-softmax sampling이라고 설명한다.
- `src/rollout.py:163-170`의 `generate()` 호출은 `top_k`와 `repetition_penalty`를 해제하지 않는다.
- Qwen2.5-7B-Instruct와 14B-Instruct의 배포 `generation_config.json`은 둘 다 `top_k=20`, `repetition_penalty=1.05`, `top_p=0.8`, `temperature=0.7`이다. 코드가 덮어쓰는 것은 뒤의 두 항목뿐이다.
  - 7B: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/generation_config.json
  - 14B: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/generation_config.json
- `src/grads.py:177-185`는 generation processor를 거치지 않은 raw-softmax teacher-forced log-probability를 사용한다.

따라서 rollout을 만든 실제 `beta`/`pi`와 ratio 계산에 사용한 `beta`/`pi`가 서로 다르다. 현재의 `g11`은 full importance sampling도 아니고 fresh on-policy oracle도 아니다. `paper/main.tex:282-307`의 population identity와 직접 충돌한다.

**필수 수정**

1. 생성 시 `top_k=0`, `repetition_penalty=1.0` 등 모든 processor를 명시적으로 고정하여 raw-softmax와 일치시킨다. 또는 processor 적용 후의 정확한 sampling log-probability를 저장·사용한다.
2. 실제 해석된 generation config 전체를 manifest에 저장한다.
3. 수정 전 LLM rollout과 그로부터 계산한 모든 gradient·표·그림을 폐기하고 다시 생성한다.

### P0-2. 종료 뒤 EOS padding을 응답 토큰으로 계산함: 확정 결함

**증거**

- `src/rollout.py:163-184`는 배치 생성 결과를 그대로 `seq.tolist()`로 저장한다. 일찍 끝난 행은 배치 최대 길이까지 pad/EOS가 붙지만 실제 종료 위치나 response mask를 저장하지 않는다.
- `src/grads.py:170-185`는 저장 tensor 끝까지 gradient와 log-probability를 계산한다.
- `src/experiment.py:188-220`, `src/hybrid.py:52-57,76-110,124-129`, `src/train_downstream.py:61-77`, `src/rollout.py:236-240`도 같은 padded sequence를 사용한다.
- hybrid prefix cut은 실제 응답이 끝난 뒤 EOS padding 내부를 자를 수 있다.

이는 gradient 방향, token KL, ratio, hybrid continuation, drift SFT, downstream 학습을 모두 오염시킨다. P0-4의 음수 KL과도 일관되는 증상이다.

**필수 수정**

- 첫 generated EOS를 포함한 위치에서 응답을 자르거나, 실제 response length와 attention mask를 저장하여 모든 계산 경로에서 동일하게 적용한다.
- multi-EOS 설정을 포함해 tokenizer별 종료 규칙을 테스트한다.
- 기존 LLM 결과를 전부 다시 생성한다.

### P0-3. 논문은 exact ratio를 정의하지만 실험은 거의 전부 clip함: 확정 결함

**증거**

- `src/grads.py:52-66`은 모든 importance weight를 기본 `[0.1, 10]`으로 자른다. 코드 주석도 논문이 clipped variant임을 밝혀야 한다고 적고 있다.
- 기본 CLI는 `src/experiment.py:348`의 `--clip-cap 10`이다.
- 외부 `FRONTIER.md` F4에서 `g11` clip fraction은 `0.9412-0.9993`, `g10`은 `0.5246-0.9866`이다.
- `paper/main.tex:282-307`은 unclipped population quantity를 정의하고 `g11=g_pi`를 사용한다.

clip은 안정화용 미세 조정이 아니라 현재 추정량의 지배적 연산이다. 따라서 `g11`을 exact/full correction 또는 oracle처럼 해석할 수 없다.

**필수 수정**

- 실험 셀을 `clipped-g00` 등으로 명시적으로 구분한다.
- unclipped, 여러 cap, self-normalized variant를 함께 보고 clip sensitivity와 bias를 분리한다.
- 이론 quantity와 구현 estimator를 동일 기호로 쓰지 않는다.

### P0-4. beta-sample KL 추정치가 큰 음수임: 확정 이상 신호

**증거**

- `src/grads.py:69-78`은 `E_beta[log beta - log pi]` plugin을 계산하며, `src/experiment.py:140-149`에서 token-weighted aggregate한다.
- 외부 `FRONTIER.md`의 7B GSM8K 결과는 `-0.4699`, `-0.4751`이다.
- KL은 모집단에서 음수가 될 수 없다. 이 정도 크기의 일관된 음수는 작은 표본 오차로 보기 어렵다.
- `paper/main.tex:474-475,780-784`는 이 진단량이 정상인 것으로 해석한다.

P0-1의 sampling/log-prob mismatch와 P0-2의 padding 문제가 우선 원인 후보이다. 원인을 확정하고 rerun하기 전에는 KL·drift 건전성 주장을 삭제해야 한다.

### P0-5. top-k가 스크립트마다 25와 26으로 다름: 확정 결함

`n=256`, `alpha=0.1`에서 다음 구현이 공존한다.

- `int(0.1*n)=25`: `src/experiment.py:246-252`, `src/frontier.py:61`, `src/judge.py:21-23`, `src/make_tables.py:72`
- `round(0.1*n)=26`: `src/kcurve_floor.py:65`, `src/stats_extra.py:55`, `src/readout_summary.py:32`, `src/reversal_freq.py:97`

그 결과 본문 표가 precision/chance는 `k=25`, corrected floor와 일부 통계는 `k=26`인 값을 섞는다.

예를 들어 14B MATH의 `k=26` 외부 집계는 floor `0.221`, `g00/g10/g01/g11 = .192/.231/.154/.192`이다. chance `26/256=.1016`로 retention을 다시 계산하면 약 `.76/1.08/.44/.76`이며, 본문의 `g10=1.00`은 이전 `k=25` 계열 값이다. 7B MATH도 통일된 `k=26` 값에서는 retention이 약 `.28/.69/.90/.90`으로, 본문 `.28-.64`와 맞지 않는다.

**필수 수정**

- top-k 규칙을 한 함수로 통합한다. 권장: 논문에서 `k=floor(alpha*n)` 또는 명시적 정수 `k=25`를 고정한다.
- 모든 표, floor, retention, 검정, 그림을 같은 candidate set과 같은 k로 다시 계산한다.
- retention은 floor보다 높을 때 1을 넘을 수 있으므로 비율의 의미와 clipping 여부를 명시한다.

### P0-6. corrected floor가 논문 설명과 다르고 공유 validation noise를 제거하지 못함: 확정 결함/증명 공백

**증거**

- `paper/main.tex:371-380`은 20개 독립 jitter pair 평균을 정의한다.
- 기본 report 경로 `src/experiment.py:264-268`은 한 pair만 계산한다.
- `src/precheck_hard.py:61-71`은 20회, `src/kcurve_floor.py`는 30개 random partition을 사용한다. canonical 구현이 없다.
- 두 split-half gradient는 동일한 `val_gradient`를 사용한다(`src/experiment.py:227-232`; k-curve도 동일). 따라서 두 반쪽 score는 validation 방향 추정오차를 공유한다.
- estimator와 oracle도 같은 validation gradient를 공유한다.

현재 floor는 독립적인 oracle reliability가 아니라 **주어진 validation vector에 조건부인 rollout split agreement**이다. 공유오차 때문에 overlap이 부풀 수 있고, `paper/main.tex:385-397`의 독립 additive-noise ceiling model을 그대로 적용할 수 없다.

**필수 수정**

- rollout뿐 아니라 validation-gradient 데이터도 독립 split한다.
- jitter 횟수, seed, 분산·CI를 manifest와 표에 기록한다.
- floor를 lower bound나 절대 ceiling으로 부르지 말고, 정확한 조건부 측정량으로 정의한다.

### P0-7. CertaGrad가 논문에 기술된 방법이 아니며 truth leakage가 있음: 확정 결함

**증거**

- `paper/main.tex:985` 이후는 off-policy score를 prior로 쓰는 two-stage method를 설명한다.
- 실제 `src/certagrad.py:73`의 `certagrad()`는 fresh group pool, validation vector, k만 받고 stale score를 전혀 사용하지 않는다.
- `src/experiment.py:255-312`는 같은 `oracle_micro_groups.pt`로 CertaGrad를 실행하고, 그 group 전체로 만든 `o_top`에 대해 선택 결과를 평가한다.
- 외부 `READOUT.md`에서 v2 CertaGrad precision `1.000`은 이 자기평가 구조의 영향을 받는다.
- confidence radius `src/certagrad.py:16-40`은 CountSketch 좌표의 isotropic Gaussian성을 전제하지만 coverage 검증이 없다.

따라서 현재 precision/cost 비교와 judge C2는 유효하지 않다. `certified=False` 자체는 관찰값으로 남길 수 있으나, 이를 일반적인 certification 불가능성으로 확대할 수 없다.

**필수 수정**

- 실제 stale prior를 사용하는 알고리즘으로 구현하거나 원고 설명을 현재 알고리즘에 맞춘다.
- 선택·중단에 쓰는 sample과 최종 truth 평가 sample을 완전히 분리한다.
- empirical coverage를 검증하고, 보장되지 않는 동안 `certificate` 대신 `heuristic interval`로 명명한다.
- 비용 결론은 “이 CI, 이 pool, 이 margin” 범위로 제한한다.

### P0-8. 27B DAPO가 전부 맞는 현상은 가설 검증 데이터가 아님: 검증 필요

27B behavior model이 현재 DAPO candidate를 전부 맞히면 reward variance와 hard boundary가 사라진다. 이는 모델이 좋다는 사실을 보여주지만 off-policy misranking 가설을 검증하거나 반증하지 않는다. pass-rate가 1인 pool에서는 selector 비교가 tie-breaking 비교로 퇴화한다.

**27B 검증 통과 조건**

1. 27B가 실제로 틀리는 문제에서 만든 capability-matched pool 또는 더 어려운 dataset을 사용한다.
2. 본실험 시작 전 live prompt 수, reward variance, top-k boundary의 비영 마진을 gate로 확인한다.
3. corrected floor가 chance보다 유의하게 높고, full/fresh 대비 one-sided estimator의 loss 또는 sign/rank reversal이 seed 반복에서 재현되어야 한다.
4. 동시에 KL, ESS, cosine 등 통상 진단량이 양호해야 “benign diagnostics에도 실패” 가설이 검증된다.
5. 조건을 못 맞추면 결론은 “현 DAPO pool이 27B에 포화됨”으로만 제한한다.

이 항목은 `docs/BACKLOG.md`의 B12/B17에도 반영되어 있다.

## 3. 이론·수식 문제

### T1. Bernoulli parameter 범위 누락: 확정 결함

`paper/main.tex:313`의 “모든 `epsilon>0`”은 `1/2 +/- epsilon`이 확률이 되려면 `0<epsilon<1/2`로 제한되어야 한다.

### T2. importance-sampling identity의 성립 조건 누락: 검증 필요

`paper/main.tex:282-307`에 최소한 다음 조건이 필요하다.

- 유한 종료 horizon 또는 적절한 적분가능성
- `pi`가 양의 trajectory에 대해 `beta`의 absolute continuity
- unclipped exact likelihood ratio
- `z`가 target policy 하에서 정의된 objective와 일치함
- 환경 transition은 정책 간 동일하며 ratio에서 action policy만 바뀜

현재 구현은 clipping과 sampling mismatch 때문에 이 조건을 만족하지 않는다.

### T3. 모든 group size에 대한 construction 주장이 증명되지 않음: 증명 공백

`paper/main.tex:317`은 모든 `K>=2`에서 group-normalized failure가 가능하다고 주장한다. 그러나 본문/appendix의 machine check는 `K={2,4,8}`만 열거한다(`paper/main.tex:323-325,957-962`). 일반 K 증명을 추가하거나 주장을 검증한 K로 좁혀야 한다.

### T4. “metric blindness” corollary의 범위가 정의되지 않음: 표현 과장

`paper/main.tex:327-336`의 “norm-blind, scale-invariant single-estimator diagnostic”가 수학적으로 정의되지 않았다. construction에서 추정 gradient와 current true gradient의 pointwise cosine을 계산하면 `-1`이므로, 모든 cosine류 진단이 blind하다는 결론은 성립하지 않는다.

실험에서 실제로 계산한 “stale 자료만으로 가능한 aggregate diagnostic”으로 범위를 좁히고, 관측할 수 없는 true-gradient cosine과 구분해야 한다.

### T5. “double failure” 명칭이 theorem 내용과 어긋남: 문구 오류

`paper/main.tex:339-343`의 continuation example에서는 `g10`이 틀리고 `g01`은 exact이다. 여기서 double failure는 `g00`과 `g10`을 뜻하지만, 독자는 두 one-sided correction이 동시에 실패한다고 읽기 쉽다. random search의 double one-sided reversal과 별개임을 명시해야 한다.

### T6. 한 prompt의 gradient sign에서 top-k misranking으로 가는 연결이 약함: 검증 필요

핵심 theorem은 한 prompt의 gradient sign reversal이다. 논문 제목과 실험 결론은 prompt top-k ranking failure다. 두 prompt 이상을 사용한 explicit top-1/top-k construction을 본문 또는 appendix에 넣고, `epsilon -> 0`에서 margin도 0으로 가는 점을 함께 다뤄야 한다.

### T7. measurement ceiling 명제가 증명보다 강함: 증명 공백

`paper/main.tex:385-397`은 “임의 estimator”에 대한 ceiling처럼 쓰였지만 appendix `967-974`는 estimator와 oracle score의 Gaussian correlation 및 top-k overlap 단조성을 사용한다. 이는 임의 estimator에 자동으로 성립하지 않는다.

- iid Gaussian additive-noise model에서 Bayes-optimal selector를 별도로 증명하거나 명제 범위를 joint-Gaussian score로 좁혀야 한다.
- correlation equality 조건을 “monotone transform”이라 한 것은 틀리며, Pearson correlation equality에는 positive affine relation이 필요하다.
- 유한 split-half overlap을 `Ovl(rho_h)`와 동일시하지 말고 추정 CI를 제공해야 한다.
- ceiling은 noisy oracle과의 agreement ceiling이지 latent true utility recovery의 절대 ceiling이 아니다.

### T8. P4와 heavy-tail 위반을 혼동함: 문구 오류

`paper/main.tex:978-980`은 heavy-tail/noise-model violation과 P4를 연결한다. P4는 reward saturation, liveness 부족, tie 증가로 인한 structural absence에 가깝고 heavy-tail은 별도 failure mode다. 분리해야 한다.

### T9. “floor”와 “retention” 해석이 과도함: 표현 과장

현재 floor는 lower bound가 아니며 estimator가 이를 넘을 수 있다. 실제 14B MATH의 통일 k 계산에서는 retention이 1을 넘는다. 따라서 “available signal의 보존 비율”이라는 해석은 지정한 noise model 아래에서만 가능하다.

### T10. empirical group normalization 정의가 이론과 다름: 재현성 문제

실제 code의 group leave-one-out 값은 unnormalized 형태인 반면 theorem은 standardized group normalization을 논한다. scale-invariant ranking에서는 일부 상쇄될 수 있지만 gradient magnitude, clipping, confidence radius에는 영향을 준다. 식과 코드의 정확한 normalization convention을 맞추거나 차이를 명시해야 한다.

## 4. 실험 구현·데이터 문제

### E1. CountSketch 오차가 작은 angular margin보다 클 수 있음: 검증 필요

`src/grads.py:85-123`은 gradient를 4096차 CountSketch로 투영하고, `126-142`는 마지막 4 decoder layer와 norm만 사용한다. 보고된 angular margin은 `0.00-0.24`도 수준이라 projection distortion과 layer truncation에 매우 민감할 수 있다.

- boundary prompt 일부에 exact/full-parameter gradient를 계산해 순위 보존율을 측정한다.
- projection seed·hash·dimension을 manifest에 저장한다.
- JL 오차 bound 또는 empirical distortion CI를 보고한다.

### E2. 수학 reward judge가 동치 답을 제대로 처리하지 못함: 확정 한계

`src/data.py:423-432,504-520`은 문자열 exact match 또는 float 변환 중심이다. prediction의 boxed regex는 nested brace를 처리하지 못하고 symbolic equivalence, fraction/decimal equivalence, set·interval·LaTeX normalization을 충분히 다루지 않는다. gold extractor와 prediction extractor의 기능도 비대칭이다.

MATH/DAPO reward variance, 27B saturation, top-k boundary가 judge artifact일 수 있다. 표준 math verifier를 사용하거나 stratified manual audit로 false-positive/false-negative rate를 보고해야 한다.

### E3. drift 학습과 selector 평가 prompt가 겹침: measurement circularity

`train_drift_lora`는 이후 ranking할 candidate prompt의 correct behavior rollout으로 `pi`를 학습한다. 이 prompt overlap은 drift가 candidate 특성에 맞춰지는 경로를 만든다. disjoint drift-training prompt pool로 재실험하고 기존 결과와 비교해야 한다. `docs/BACKLOG.md` B16과 같은 문제다.

### E4. downstream 비교의 실제 update 수와 총비용이 같지 않음: 확정 결함

- `src/train_downstream.py:52-79`는 reward가 모두 같은 step에서 optimizer update를 건너뛴다. selector마다 실제 update 수가 달라질 수 있다.
- 기본 `budget_rollouts=0`은 oracle selector의 selection cost를 downstream budget에 포함하지 않는다.
- 같은 iteration 수를 사용해도 토큰 수, 유효 update 수, fresh rollout 비용이 다르다.
- 작은 validation과 단일 finetune 경로만으로 ordering을 판정하며 반복 seed/CI가 없다.

실제 optimizer update, 학습 token, rollout, wall-clock/GPU cost를 모두 기록하고 동일 예산 비교 또는 Pareto curve로 바꿔야 한다.

### E5. DAPO live-fraction 계산이 사실상 항상 1임: 확정 결함

`src/frontier.py:306-308`은 8개 rollout에서 pass-rate를 `(1+sum)/(2+n)`으로 smoothing한 뒤 `(0.05,0.95)` 안인지 검사한다. 그러면 전부 실패한 prompt도 `0.1`, 전부 성공한 prompt도 `0.9`라서 모두 live로 분류된다. 외부 F4의 `live_frac_beta=1.0`은 실제 mixed-reward fraction이 아니다.

raw pass count 기준 `0 < successes < n`을 사용하고, smoothing은 별도 uncertainty 추정에만 써야 한다.

### E6. k-curve가 “exact”가 아니며 simulation이 약함: 확정 불일치

- `src/kcurve_floor.py:62-82`는 30개 random partition을 사용한다. exact recomputation이라는 표현은 틀리다.
- `S_SIM=40`뿐이고 Monte Carlo CI가 없다.
- `predicted_floor`는 음수 correlation을 0으로 자르며, simulation 결과 `.094`는 analytic chance `.100`보다도 낮다.
- 관측 `.156`과 예측 `.094`의 calibration이 좋지 않은데 이를 structural verdict에 사용한다.
- 직접 관측하지 않은 K64/K128을 작은 simulation으로 외삽한다.

partition·simulation 수를 크게 늘리고 CI를 제공하며, K64/K128을 직접 관측하기 전에는 P4의 증거를 보조적 결과로 제한해야 한다. chance 아래 prediction은 analytic floor 또는 충분한 MC precision과 함께 해석해야 한다.

### E7. “100% fresh budget” 표현이 실제 pool과 다름: 문구 오류

`paper/main.tex:773-785`의 100%는 K32 전체 truth를 모두 쓰는 의미가 아니다. frontier의 가장 큰 지점은 4개 even microgroup, 즉 prompt당 16 rollout을 selection에 쓰고 다른 16개를 truth로 보류한다. 정확한 분모와 withheld truth 구조를 써야 한다.

### E8. random-search 음성 결과의 범위가 과도함: 통계 보강 필요

`paper/main.tex:345-349`의 50,000회 무발견은 지정한 proposal distribution과 threshold 안의 결과다. “존재하지 않는다”가 아니라 탐색 분포 아래 관측되지 않았다고 써야 하며, binomial upper bound와 parameter coverage를 제시해야 한다. `docs/BACKLOG.md` B15의 50k→200k 문구도 현재 본문과 동기화해야 한다.

### E9. theory verifier와 결과 bundle이 repository에 없음: 재현성 차단

- 원고 `paper/main.tex:323-325,998-1003`은 machine-readable witnesses, verification script, manifests, readouts를 배포한다고 말한다.
- 실제 theory script는 다른 repository의 `/home/kms/dev/new-paper-ideas/68-one-sided-offpolicy-misranking/verify_theory.py`에만 있다.
- 이 repository에는 `results/`와 run manifest가 없다. `README.md`는 results가 있다고 설명한다.
- 외부 `/home/kms/Downloads/0818` bundle에는 수정 전 DAPO floor `.804/.745`와 중복 run 이름 `v2-s1-dapo-math-dapo-math`가 남아 있고, 본문은 일부 수치를 수동 교정했다.

제출 artifact에는 exact script, immutable run manifest, model revision, raw/derived hash, canonical table-generation 명령을 포함해야 한다. 수정 전/후 bundle을 분리하고 중복 suffix가 데이터 매핑 오류를 만들지 않았는지 확인해야 한다.

## 5. 통계·결과 해석 문제

### S1. hypergeometric/Fisher 검정이 실제 귀무가설을 충분히 반영하지 않음

`src/stats_extra.py:24-77`의 hypergeometric test는 두 top-k 집합이 exchangeable random set이라는 귀무가설을 쓴다. 그러나 selector와 oracle은 같은 prompt difficulty와 validation vector를 공유한다. Fisher 결합도 rollout RNG만 다르다고 독립이 보장되지 않는다.

- prompt-paired permutation 또는 method-label randomization을 우선 사용한다.
- pool/seed 단위 clustered bootstrap을 사용한다.
- 두 seed p-value 결합보다 seed별 effect size와 uncertainty를 먼저 보고한다.

### S2. upper-tail 유의성이 “oracle과 구별되지 않음”을 뜻하지 않음: 해석 오류

`paper/main.tex:481`의 “not distinguishable from the oracle; they track it”은 random-overlap보다 높다는 upper-tail p-value로 지지되지 않는다. 이는 “무작위보다 overlap이 높음”만 말한다. oracle-equivalence를 주장하려면 equivalence margin과 양측/비열등 검정이 필요하다.

### S3. v2 개선 결론은 seed 2개뿐임: 검증 필요

GSM8K pass-rate가 `0.292/0.219`, best stale이 `0.275/0.216`인 관찰은 방향성은 있으나 seed 2개로 일반 결론을 내리기 어렵다. paired candidate-level CI, 여러 model seed, fresh-rollout seed를 추가해야 한다.

### S4. DAPO는 signal prompt가 2-4개뿐이라 tie-breaking 지배적임

chance 수준이라는 결론은 현재 pool의 saturation을 보여주지만 estimator 일반 성능을 보여주지 않는다. top-k precision보다 live subset conditional precision과 실제 reward variance를 먼저 보고해야 한다.

### S5. “5-8 orders of magnitude”와 “never certify”가 일반 lower bound가 아님

`src/c2_diagnose.py:73-79`의 필요량은 특정 Gaussian CI와 근사식을 사용한다. exact top-k certification 자체의 정보이론 lower bound가 아니다. “이 CI extrapolation에서”로 제한하고 margin uncertainty를 포함해야 한다.

### S6. downstream 결과는 인과적 실용 효과를 지지하지 못함

현재 paper가 ordering 부재를 인정하는 것은 맞다. 다만 README와 일부 문구에는 anti-selection/causal recovery처럼 더 강한 이전 결론이 남아 있다. 실험 비용과 업데이트 수를 맞추기 전까지 practical impact의 근거로 사용하면 안 된다.

## 6. 원고 내부 모순과 과장

### M1. abstract의 “one measurable quantity가 실패를 결정”: 표현 과장

`paper/main.tex:48-50`의 regime map은 seed 2개와 제한된 pool에서 관측된 상관관계다. 결정 규칙이나 검증된 classifier가 아니다. “organizes the observed regimes” 정도로 낮춰야 한다.

### M2. “every pool had no recoverable signal”이 strong-signal 결과와 모순

`paper/main.tex:55-57`은 every pool이라고 하지만 7B MATH와 drift400 등은 strong-signal regime로 보고되어 있다. weak-signal v2 GSM8K/DAPO subset으로 범위를 제한해야 한다.

### M3. “zero-cost heuristic matched every gradient method”의 범위가 좁음

이는 v2 weak-signal GSM8K/DAPO에서의 관찰이다. 전체 12개 configuration에 대한 보편 결과처럼 쓰면 안 된다.

### M4. “no new rollouts”와 optional fresh probe가 모순

원고는 무-rollout artifact-only protocol을 강조하지만 48-prompt fresh probe와 branch-c audit을 사용한다. core artifact-only gate와 optional paid validation을 명확히 분리해야 한다.

### M5. FIRST를 철회하면서 계속 처방함: 내부 모순

`paper/main.tex:773-777`은 floor-gated policy가 out-of-sample에서 졌고 prescriptive use에서 철회됐다고 한다. 그러나 abstract와 `796-841`은 FIRST를 운영 프로토콜로 제안한다. 현재 증거로는 diagnostic checklist/hypothesis로만 제시해야 한다.

### M6. “artifact-only”라는 비용 표현이 불완전함

필요 artifact 자체가 rollout, gradient, microgroup 저장을 요구한다. 이미 artifact가 존재할 때의 marginal cost가 낮다는 뜻이지 전체 비용이 0은 아니다.

### M7. “48 prompts면 충분”에 power 분석이 없음

`paper/main.tex:848`의 48은 검증된 sample-complexity가 아니다. false-positive/false-negative, expected effect, confidence target을 바탕으로 정해야 한다.

### M8. “never budget for boundary certification”은 과도한 정책 결론

`paper/main.tex:853-855`는 특정 CI와 weak-margin pool의 관찰을 일반 운영 규칙으로 확대한다. certification value가 높은 안전·규제 상황과 larger-margin pool을 배제하지 못한다.

### M9. “no method and no certificate can have an effect”는 식별과 효용을 혼동

`paper/main.tex:593-600`에서 현재 noisy oracle로 차이를 측정하지 못하는 것과 실제 method effect가 없는 것은 다르다. “cannot be reliably distinguished under this evaluation design”으로 바꿔야 한다.

### M10. safety extension이 검증되지 않음

`paper/main.tex:875-884`은 verifiable reward 결과가 safety preference/risk ranking에 unchanged로 적용된다고 말한다. 안전 metric은 noisy, partial, adversarial일 수 있어 reward judge와 oracle 조건이 다르다. 별도 가설·한계로 표시해야 한다.

### M11. 실험 설정이 재현에 부족함

`paper/main.tex:436-444`에 다음이 빠져 있다.

- 정확한 model revision과 tokenizer/chat template
- 해석된 generation config 전체
- LoRA rank, alpha, learning rate, batch, target modules
- response cap과 EOS/padding 규칙
- gradient 대상 layer와 projection dimension/hash
- ratio clip 범위
- reward verifier 버전
- validation set 구성
- framework/CUDA/GPU 및 manifest hash

현재 `constraints/h100-cu126.txt`만으로 실행을 재현할 수 없다.

### M12. 현재 원고는 submission-ready 상태가 아님

- `paper/main.tex:28-31`은 내부 draft/version metadata를 노출한다.
- `paper/main.tex:515-519`에는 pending seed box가 남아 있다.
- 수정 전 결과 bundle과 수동 교정 수치가 혼재한다.

## 7. 참고문헌·관련연구 문제

### R1. 참고문헌 제목 오류: 확정 결함

다음 arXiv 원제와 manuscript entry가 다르다.

- TIC-GRPO, arXiv:2508.02833: **TIC-GRPO: Provable and Efficient Optimization for Reinforcement Learning from Human Feedback**  
  https://arxiv.org/abs/2508.02833
- ACE, arXiv:2601.20989: **Top-k on a Budget: Adaptive Ranking with Weak and Strong Oracles**  
  https://arxiv.org/abs/2601.20989
- Floor/Ceiling, arXiv:2608.01704: **Floor, Ceiling, and the Fusion Gap: How Much of Crowd Reading Attention Can Machines Predict?**  
  https://arxiv.org/abs/2608.01704
- Multi-Step Off-Policy, arXiv:2605.20865는 제목 끝의 **for Reinforcement Learning with Verifiable Rewards**까지 포함해야 한다.  
  https://arxiv.org/abs/2605.20865

저자, 연도, venue도 정식 BibTeX로 복원해야 한다.

### R2. related-work 분류가 과도하게 단순화됨

- “기존 estimator는 정확히 한 축만 보정한다”는 표현은 원고 후반에 언급하는 full-trajectory 방법을 누락한다. “여러 practical estimator”로 제한한다.
- full correction이 selector로 비실용적이라는 평가는 조건부 비용 주장으로 써야 한다.
- ACE는 weak score가 단순 noise까지 유효하다고 일반 가정하는 논문이 아니라 weak/strong oracle과 CI 조건이 명시된 adaptive ranking이다. systematic bias가 가정 밖임을 정확히 서술한다.
- Floor/Ceiling 논문의 floor는 lead baseline, ceiling은 split-half oracle이라는 원 맥락을 밝혀 이 원고의 용어 전용과 구분한다.

### R3. 일부 인용 수치는 맞지만 표현을 정밀화해야 함

- CROPI의 “40/50 cosine > 0.6”, “28.80% consistency”는 원문과 일치한다.  
  https://arxiv.org/abs/2510.26491
- TIDE의 1% token 결과는 “가장 negative한 1% token이 gradient의 거의 절반”이다. 이를 단순 top-1%나 ESS 현상으로 동일시하지 말고 analogy라고 표시해야 한다.  
  https://arxiv.org/abs/2608.09836

## 8. 문서·레이아웃·저장소 문제

### D1. README가 철회된 결론을 유지함

`README.md:50-58`의 anti-selection, 21/21 hybrid causal recovery, downstream loss 서술은 현재 원고가 leakage/재검증 문제로 낮춘 결론과 맞지 않는다. paper와 README의 claim level을 동기화해야 한다.

### D2. LaTeX overfull box

현재 build log 기준:

- `paper/main.tex:174-183`: 약 `82.6pt`
- `paper/main.tex:305`: 약 `67.1pt`
- `paper/main.tex:939-948`: 약 `15.7pt`

수식 줄바꿈, `aligned`/`split`, 표 column 폭을 조정해야 한다.

### D3. 생성 보조파일이 추적되지 않음

worktree에 `paper/main.aux`, `paper/main.out`이 untracked로 남는다. build output 경로 또는 `.gitignore` 정책을 정리하되, review 과정에서는 기존 파일을 삭제하지 않았다.

### D4. 테스트 파일이 pytest 수집과 호환되지 않음

`tests/test_core.py`, `tests/test_judge.py`, `tests/test_reversal_freq.py`는 import 시점에 테스트를 실행하고 최상위에서 `sys.exit()`한다. 따라서 `python3 -m pytest -q`는 `test_judge.py` 수집 중 `SystemExit: 0`으로 internal error가 난다. 각 파일을 script로 직접 실행하면 통과하지만 CI 표준 수집은 실패한다. 테스트를 함수형 `test_*` case로 바꾸고 종료 코드를 pytest에 맡겨야 한다.

## 9. 현재 유지 가능한 부분

다음은 위 P0 수정과 표현 축소를 전제로 유지할 가치가 있다.

- two-axis decomposition 자체는 필요한 가정 아래 algebraically 타당하다.
- one-sided correction의 sign reversal construction은 `epsilon` 범위와 일반 K 증명을 고치면 핵심 이론 기여가 될 수 있다.
- oracle reliability를 selector 평가 전에 측정해야 한다는 문제 제기는 타당하다.
- DAPO에서 27B가 전부 맞는 현상은 “pool saturation과 liveness gate의 필요성”을 보여주는 운영상 사례로는 유효하다.
- 현재 결과가 null/negative여도 숨기지 않고 evaluation breakdown과 분리하려는 방향은 유지할 수 있다.

## 10. 권장 수정·재실험 순서

1. **rollout 계약 수정**: generation processor, EOS length/mask, exact token log-probability를 하나의 tested API로 통합한다.
2. **metric 정의 통합**: top-k integer rule, reward verifier, group normalization, clipping variant, KL/ESS 단위를 공통 모듈로 만든다.
3. **oracle 재구성**: validation과 rollout을 독립 split하고, projection distortion 및 reward judge error를 audit한다.
4. **소규모 smoke rerun**: KL 비음수, unclipped `g11`과 fresh gradient 일치, EOS trim, ratio replay를 unit/integration test로 확인한다.
5. **7B/14B 전면 rerun**: 기존 LLM 표·그림을 새 manifest에서 재생성한다.
6. **27B capability-matched 검증**: B17 gate를 통과한 hard pool에서 핵심 가설을 검증한다.
7. **CertaGrad 재설계**: stale prior 사용 여부를 명확히 하고 disjoint truth와 calibrated coverage로 다시 평가한다.
8. **통계 재작성**: paired/clustered uncertainty, seed 반복, effect size 중심으로 표를 갱신한다.
9. **원고 claim 축소**: FIRST는 validated protocol이 아니라 diagnostic hypothesis로, certification 결론은 해당 CI/pool 범위로 제한한다.
10. **artifact 동결**: theory script, raw result hash, manifest, table generator, exact environment를 repository release에 포함한 뒤 PDF를 다시 빌드한다.

## 11. 재실험 완료 판정 체크리스트

- [ ] sampled-token log-probability replay가 저장값과 tolerance 내 일치
- [ ] 모든 response에서 first-EOS 이후 token이 gradient·ratio·학습에서 제외됨
- [ ] beta-sample KL이 seed 반복에서 비음수이며 estimator CI가 제시됨
- [ ] unclipped `g11`과 independent fresh-gradient oracle의 방향·순위 일치가 smoke test로 확인됨
- [ ] clip fraction과 cap sensitivity가 표에 포함됨
- [ ] 모든 분석이 동일한 정수 k를 사용함
- [ ] split-half floor가 rollout과 validation 양쪽 독립 split 및 반복 CI로 계산됨
- [ ] CountSketch/partial-layer approximation의 boundary ranking distortion이 보고됨
- [ ] math reward verifier manual audit가 통과함
- [ ] drift training prompt와 selector evaluation prompt가 분리됨
- [ ] CertaGrad selection sample과 final truth sample이 분리됨
- [ ] 실제 update/token/rollout 비용이 downstream 비교에서 정렬됨
- [ ] 27B hard pool이 liveness gate를 통과하고 핵심 reversal/loss가 여러 seed에서 재현됨
- [ ] manuscript 표·그림이 immutable manifest에서 한 명령으로 재생성됨
- [ ] README, backlog, paper, bibliography가 같은 결론과 artifact를 가리킴

## 12. 점검에 사용한 외부 결과 파일

이 파일들은 현재 repository 밖에 있으므로 재현 artifact로는 불충분하다.

- `/home/kms/Downloads/0818/TABLES.md` (`sha256: 6a70540df12ddc1956595d34530407b4387f5664a5e66dfdc105c9051cd25d60`)
- `/home/kms/Downloads/0818/FRONTIER.md` (`sha256: 1d9efb4fe98123873ec5c7b272feac133800bf5436714c7bda3624dcf24e7152`)
- `/home/kms/Downloads/0818/KCURVE.md` (`sha256: 26cb4bf3d62f9c7e8acfe1a0a0fa1e2e2e78f1194ad9b7467e964d091fd7c607`)
- `/home/kms/Downloads/0818/READOUT.md` (`sha256: 4c76a5acb94f9e41da30356320ee30181dcafa34c98b6d226313afbe7169d4f1`)
- `/home/kms/Downloads/0818/STATS.md` (`sha256: bdfbcad522297ed0377da01bbecba745070f7ecdaf3abce21db57cd8b7c67bec`)
- `/home/kms/Downloads/0818/REVERSAL.md` (`sha256: f558a3bde0f7b45989906e87e5c17311f988a8d8ac79e4405f9fa5eafff52971`)

## 13. 점검 검증 결과

- `PYTHONPATH=src .work/.venv-cu126/bin/python tests/test_core.py`: 통과
- `PYTHONPATH=src python3 tests/test_judge.py`: 통과
- `PYTHONPATH=src python3 tests/test_reversal_freq.py`: 통과
- `python3 /home/kms/dev/new-paper-ideas/68-one-sided-offpolicy-misranking/verify_theory.py`: 통과. 다만 verifier가 현재 repository 밖에 있다는 재현성 문제는 E9에 기록했다.
- `python3 -m pytest -q`: 실패. assertion 실패가 아니라 `tests/test_judge.py`의 import-time `sys.exit(0)`로 인한 collection internal error이며 D4에 기록했다.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`: 15페이지 PDF 생성 성공. overfull box 3개와 hyperref PDF-string warning은 D2 및 build log에 남아 있다.

---

## 14. 독립 재검증 (2차 검증, 2026-08-19)

위 점검(§1–§13)과 별개로, P0 전 항목과 주요 하위 항목을 코드·`0818` 수확
수치에 직접 대조해 재검증했다. 원 점검을 읽지 않은 상태의 백지 검증은 아니고,
각 주장에 대해 인용된 파일·행을 직접 열어 확인하는 반박 지향 검증이다.

### 14.1 항목별 판정

| 항목 | 판정 | 독립 확인 근거 |
|---|---|---|
| P0-1 샘플링·ratio 분포 불일치 | **재확인 — 확정** | `src/rollout.py`의 `SAMPLING`은 `top_p`만 통제하고 `generate()`는 `do_sample·temperature·top_p`만 넘긴다. HF `generate`는 미지정 인자를 모델 `generation_config`에서 병합하므로 Qwen2.5-Instruct 기본값 `top_k=20`, `repetition_penalty=1.05`가 그대로 적용된다. 채점 측 `grads.py`는 processor 없는 raw-softmax teacher forcing. 주석 자체가 "top-p만 고치면 estimand가 맞는다"고 믿은 흔적이다 — top_k·repetition_penalty를 놓쳤다. |
| P0-2 EOS padding 포함 | **재확인 — 확정 (단, 결론 일부 정정: §14.2)** | `rollout.py`는 `seq.tolist()` 전체와 `resp_start`만 저장하고 종료 위치·mask를 저장하지 않는다(`pad_token_id=eos`). `grads.py`는 `tok_logp[resp_start-1:]`로 저장 텐서 끝까지 계산한다. |
| P0-3 clip 지배 | **재확인 — 확정** | `grads.py` `clip_cap=10.0` 기본, `token_weights`가 `clamp(±log 10)`. 실측(F4): `clipfrac_g11` 0.9412–0.9993, `clipfrac_g10` 0.5246–0.9866. clip이 지배 연산이라는 판단 그대로다. |
| P0-4 음수 KL | **재확인 — 확정 + 추가 관찰** | F4 실측 `token_kl_beta_pi`: gsm8k 계열 −0.4699/−0.4751, **dapo 계열은 +0.6772/+0.419**. 음수는 gsm8k에서만 나타난다. 원인 후보를 좁히는 단서다: 두 풀은 같은 코드 경로를 타므로, 차이는 응답 길이 분포(padding 비율)와 생성 분포 왜곡의 상호작용에서 나올 가능성이 높다. rerun 전 원인 규명 실험(§14.3의 스모크)에서 gsm8k/dapo를 모두 포함할 것. |
| P0-5 top-k 25/26 혼재 | **재확인 — 확정** | `experiment.py:251`·`judge.py:22` = `int()`, `stats_extra.py:55`·`kcurve_floor.py:65` = `round()`. n=256, frac=0.1에서 25 vs 26 실재. |
| P0-6 floor 구현 상이·공유 val | **재확인 — 확정** | report 경로(`experiment.py`)는 프롬프트당 split-half **1쌍**(스택 앞/뒤 반분)만 계산하고, 양쪽 half와 oracle이 **같은 `val_grad`** 를 쓴다. 본문 "20 jitter pair" 서술과 불일치, 공유 validation 오차 미제거 지적 모두 맞다. |
| P0-7 CertaGrad 서술·구현 불일치 | **재확인 — 확정** | `certagrad()` 시그니처가 `(cand_pools, val_pool, k, …)` — stale prior 입력 자체가 없다. 원고 후반 two-stage 서술과 다른 알고리즘이다. |
| P0-8 27B 포화 풀 | **동의** | 기존 B12/B17 게이트와 동일 결론. 추가로 §14.4의 27B 재시작 권고 참조. |
| E5 live-frac 항상 1 | **재확인 — 확정** | `frontier.py:81` `passrate=(1+s)/(2+n)` 스무딩 후 `0.05<p<0.95` 판정 → n=8이면 전패=0.1, 전승=0.9로 전부 live. F4 전 행 `live_frac_beta=1.0`이 그 증거다. |
| T1 ε 범위 누락 | **부분 동의 — 경계값 정정** | 범위 누락 지적은 맞다. 다만 본문 구성은 `1/2±ε/2`(gradient ±ε/2)이므로 필요한 제약은 **0<ε<1**이다. 원 점검의 `0<ε<1/2`는 `1/2±ε` 가정으로, 본문 표기와 다르다. |
| T3 일반 K 증명 공백 | **동의** | 정리는 "any K≥2", 기계 검증은 K∈{2,4,8} 열거. 일반 K 대수 증명을 추가하거나 주장을 좁혀야 한다. |
| M5 FIRST 철회 vs 처방 모순 | **동의 (정리 방향 제안)** | `main.tex:775-777` "withdrawn from prescriptive use" vs abstract "prescribes one of three policies" 공존 확인. 다만 철회된 것은 downstream **floor-gated 선택 정책**이고 FIRST의 1차 산출은 신뢰도 진단이다. "진단은 유지, 3분기 처방은 가설로 강등"으로 양쪽 문장을 통일하면 모순 없이 수습된다. |
| R1 서지 제목 오류 | **판정 보류 (온라인 재확인 필요)** | 이 재검증에서는 arXiv 원문 대조를 하지 않았다. A3(서지 확정) 작업에서 4건 모두 원문 대조로 처리할 것. |
| 나머지 (T4–T10, E1–E4, E6–E9, S1–S6, M1–M4, M6–M12, D1–D4) | **대체로 동의** | 표현·범위 축소 및 재현성 요구로, 원고·분석 수정으로 처리 가능하다. 개별 반박 없음. |

### 14.2 원 점검에 대한 정정 2건

1. **P0-2의 "기존 LLM 결과 전부 재생성" 은 P0-2 단독으로는 과하다.**
   저장된 `input_ids`에 EOS 위치가 그대로 있으므로, first-EOS 절단 mask는
   기존 산출물에서 **사후 복원 가능**하다. gradient·ratio 재계산(GPU 필요)으로
   충분하고 rollout 재생성은 불필요하다. 다만 P0-1이 어차피 재생성을 강제하므로
   실무 결론(재생성)은 달라지지 않는다 — 귀속만 정확히 해 둔다: 재생성의
   근거는 P0-1이고, P0-2는 mask 저장 계약 추가로 해결된다.
2. **T1 경계값**: 위 표 참조 (0<ε<1).

### 14.3 실무 트리아지 (수정 비용 기준)

- **(A) rollout 재생성 필요 — P0-1**: 생성 시 `top_k=0`(해제),
  `repetition_penalty=1.0`을 명시하고, 해석된 generation config 전체와
  response mask를 manifest에 저장(P0-2 계약 포함). 이후 7B/14B 재실행.
- **(B) GPU 재계산만 — P0-6, P0-3**: val rollout 독립 split 후 val_grad 2벌
  재계산, clip 민감도(unclipped·cap 스윕)는 저장 rollout에서 재계산 가능.
- **(C) CPU 재계산만 — P0-5, E5, E6, S1**: k 규칙 한 함수 통일 후 전 표
  재생성, live-frac은 raw count 기준으로 재계산, k-curve partition/simulation
  수 증대.
- **(D) 원고·문서 수정만 — P0-7, T1, T3~T10, M1~M12, R1~R3, D1**: 재실험 없이
  처리 가능.

### 14.4 일정 함의 (결정점 9/3, 초록 9/11, 본문 9/16)

- **진행 중인 27B 본실행(B12)도 같은 `rollout.py` 계약을 타므로 P0-1·P0-2의
  영향을 동일하게 받는다.** 지금 산출되는 27B rollout은 수정 후 폐기 대상이
  된다. 권고: (A) 계약 수정을 최우선으로 커밋하고, 27B는 수정 커밋 기준으로
  재시작한다. 수정 전 완주분은 진단(포화 게이트 B17 판정)에만 쓴다.
- 최소 경로: (A) 수정 → §11 체크리스트 1–4를 스모크로 통과(0.5B/7B 소규모)
  → 7B 전면 재실행 → 표·그림 재생성 → 그 수치로 9/3 결정점 판정.
- (C)·(D)는 재실행과 병렬로 지금 진행 가능하다.
