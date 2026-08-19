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
| T1 ε 범위 누락 | **재확인 — 확정** | 부록 `paper/main.tex:941-943,949-953`은 성공확률·방문확률을 `1/2±ε`로 정의하고, 그 결과 gradient가 `±ε/2`가 된다. 따라서 필요한 범위는 **`0<ε<1/2`**이다. `ε=1/2`에서는 support가 깨지고 KL이 무한대이며, 그보다 크면 확률 자체가 아니다. |
| T3 일반 K 증명 공백 | **동의** | 정리는 "any K≥2", 기계 검증은 K∈{2,4,8} 열거. 일반 K 대수 증명을 추가하거나 주장을 좁혀야 한다. |
| M5 FIRST 철회 vs 처방 모순 | **동의 (정리 방향 제안)** | `main.tex:775-777` "withdrawn from prescriptive use" vs abstract "prescribes one of three policies" 공존 확인. 다만 철회된 것은 downstream **floor-gated 선택 정책**이고 FIRST의 1차 산출은 신뢰도 진단이다. "진단은 유지, 3분기 처방은 가설로 강등"으로 양쪽 문장을 통일하면 모순 없이 수습된다. |
| R1 서지 제목 오류 | **재확인 — 확정** | arXiv 공식 레코드와 다시 대조했다. TIC-GRPO(2508.02833), Multi-Step(2605.20865), ACE(2601.20989), Floor/Ceiling(2608.01704) 네 항목 모두 §7 R1에 적은 원제가 맞고 현재 manuscript entry가 축약·변형되어 있다. |
| 나머지 (T4–T10, E1–E4, E6–E9, S1–S6, M1–M4, M6–M12, D1–D4) | **대체로 동의** | 표현·범위 축소 및 재현성 요구로, 원고·분석 수정으로 처리 가능하다. 개별 반박 없음. |

### 14.2 원 점검에 대한 정정 2건

1. **P0-2의 "기존 LLM 결과 전부 재생성" 은 P0-2 단독으로는 과하다.**
   저장된 `input_ids`에 EOS 위치가 그대로 있으므로, first-EOS 절단 mask는
   기존 산출물에서 **사후 복원 가능**하다. gradient·ratio 재계산(GPU 필요)으로
   충분하고 rollout 재생성은 불필요하다. 다만 P0-1이 어차피 재생성을 강제하므로
   실무 결론(재생성)은 달라지지 않는다 — 귀속만 정확히 해 둔다: 재생성의
   근거는 P0-1이고, P0-2는 mask 저장 계약 추가로 해결된다.
2. **T1 경계값**: 위 표 참조 (`0<ε<1/2`).

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

## 15. 추가 전수조사 (3차, 2026-08-19)

이번 절은 §14 이후 repository의 Python 전 파일, shell 실행기, 결과 수확기,
원고의 수치 생성 경로를 다시 훑어 새로 확인한 사항이다. 단순 스타일 경고는 제외하고,
논문 결론·실험 비용·재현성 또는 실행 안전성에 영향을 주는 항목만 기록한다.

### 15.1 추가 제출 차단 항목(P0)

#### P0-9. 정본 estimator precision도 shared-jitter로 부풀려짐: 확정 결함

- `src/experiment.py:246-252`의 `topk()`는 seed마다 prompt ID에 고정 jitter를
  배정한다. `stage_report()`는 oracle과 모든 estimator에 **동일한 seed**를 넘긴다
  (`src/experiment.py:262,285-290`).
- 따라서 oracle와 estimator가 같은 tie block을 가지면 서로 독립인 임의 top-k가
  아니라 같은 prompt를 고른다. 전 점수가 0인 `n=256, k=25` 재현에서 chance는
  `0.0977`인데 동일 seed precision은 정확히 `1.0`, 독립 seed 한 번은 `0.08`이었다.
- P2 방어는 split-half floor의 두 seed만 분리했을 뿐, estimator-vs-oracle
  precision에는 적용되지 않았다. DAPO처럼 유신호 prompt가 2--4개이고 나머지가
  0점인 조건은 이 결함의 영향을 크게 받는다.
- `src/judge.py:21-23`와 `src/make_tables.py:56-82`는 jitter 없이 동일한 dict/index
  순서를 써 같은 인공 overlap을 다시 만든다. 분석 도구마다 결과도 달라진다.

**영향:** DAPO·포화 조건의 stale precision, ceiling 도달, zero-cost baseline 우위,
regime-map 수치를 현재 값으로 인용할 수 없다. raw score에서 독립 tie randomization을
여러 번 반복해 평균·구간을 다시 계산하고 모든 표·판정을 한 canonical 함수로 통일해야
한다. GPU rerun은 필요 없지만 현재 repository에는 해당 대형 run raw score가 없다.

#### P0-10. hybrid mixed cell의 생성 horizon이 더 길어 equal-budget가 아님: 확정 결함

- `bb`와 `pp`는 최대 `max_new_tokens` 응답이다(`src/hybrid.py:88-98`).
- `bp`와 `pb`는 이미 보존한 response prefix 뒤에 다시 **전체**
  `max_new_tokens`를 생성한다(`src/hybrid.py:100-110`). cut=0.75면 mixed response는
  pure cell보다 이론상 약 1.75배 길 수 있다.
- rollout 수 K만 같을 뿐 생성 token, horizon, 정답 기회, gradient token 수, KV
  compute가 다르다. `paper/main.tex:638-646,658-661`의 disjoint equal-budget decisive
  experiment와 인과 해석을 직접 깨뜨린다.

**조치:** 각 sequence별 남은 budget을 `max_new_tokens-prefix_response_tokens`로 제한하고,
EOS mask를 적용한 실제 response token 수·종료율·보상률을 cell별로 맞춰야 한다.
기존 hybrid 결과는 P0-1/P0-2와 별개로 폐기 대상이다.

#### P0-11. K-curve가 문자열 manifest를 못 읽어 14B K를 32로 오표기할 수 있음: 확정 결함

- `scripts/run_14b.sh:131-139`는 `fresh_k`를 환경변수 문자열로 저장한다.
- `src/kcurve_floor.py:107-123`은 값이 Python `int`일 때만 읽고 아니면 32를
  반환한다. 최소 재현에서 `{"fresh_k":"16"}`은 32, `{"fresh_k":16}`은 16으로
  해석됐다. `src/kcurve_all.py:53-66`도 이 함수로 micro-group당 rollout 수와 K' 축을
  만든다.
- `run_14b.sh`의 단독 기본은 실제 `FRESH_K=16`이다(`scripts/run_14b.sh:134,143`).
  반면 원고는 모든 oracle이 K=32라고 쓰고(`paper/main.tex:436-439`), 확장표도 14B
  행을 K'=32로 표시한다(`paper/main.tex:729-755`). v2는 wrapper가 32를 명시해서
  우연히 fallback과 일치하지만, 기본 실행으로 만든 v1 14B GSM8K에는 적용되지 않는다.
- 현재 repository/`Downloads/0818`에는 대형 run manifest·raw rollout이 없어 실제
  14B GSM8K가 K16인지 K32인지 사후 확정할 수 없다.

**영향:** 14B floor 지점과 K=64/128/256 외삽의 가로축이 최대 2배 틀릴 수 있다.
manifest 문자열을 느슨하게 변환하는 수준이 아니라, 실제 prompt별 rollout 개수와
micro-group size에서 K를 검증해 기록해야 한다. 원자료 확인 전 표의 14B K 표기를
보류한다.

#### P0-12. manifest가 artifact를 구속하지 않아 서로 다른 실행이 조용히 섞임: 확정 결함

- manifest는 매 실행 덮어쓰지만(`scripts/run_14b.sh:122-140`), stage는 output 존재만
  보고 건너뛴다(`src/experiment.py:68-70,158-160,389-415,438-442`).
- `prep`은 prompts를 다시 쓰고(`src/experiment.py:376-379`), 기존 merged behavior
  rollout은 그대로 재사용한다. model, revision, dataset pool, seed, N, K, generation
  config, adapter, projection, clip cap, code가 바뀌어도 content hash 검사가 없다.
- mtime 기반 stale 이동(`scripts/run_14b.sh:160-224`)은 최신 adapter 한 파일과 shard
  index만 볼 뿐 설정·내용 동일성을 확인하지 않는다.
- `merge-grads`는 기대 shard 수, prompt ID 전집합, 정책/config hash를 검사하지 않고
  발견한 shard만 합쳐 final file을 만든다(`src/experiment.py:491-535`).

**영향:** 성공한 것처럼 보이는 한 run 안에 서로 다른 prompt/model/policy/K의 artifact가
섞일 수 있고, 덮어쓴 manifest는 이를 오히려 현재 설정으로 위장한다. immutable run ID와
config digest를 모든 artifact에 넣고, model/tokenizer revision·prompt hash·adapter hash·
generation config·projection spec을 검증한 뒤에만 재사용해야 한다.

#### P0-13. CertaGrad의 보증·비용·평가 대상이 서로 다름: 확정 결함

1. `certagrad()`와 `certagrad_scalar()`는 매 adaptive draw 뒤 fixed-n interval을 반복
   검사한다(`src/certagrad.py:104-169,233-262`). candidate Bonferroni만 있고 시간축
   보정이 없어 optional stopping 하에서 advertised delta coverage가 성립하지 않는다.
2. scalar 관측치는 group별 cosine의 평균이다(`src/certagrad.py:205-227`). oracle은
   `cos(mean(group gradients), mean validation gradient)`이다
   (`src/experiment.py:226-232`). 두 값은 일반적으로 다르며, 재현 예에서는 각각
   `0.4502`와 `0.0`이었다.
3. scalar는 `val_pool.mean()`으로 validation group 전체를 처음부터 사용하면서
   `fresh_groups`에는 candidate draw만 센다(`src/certagrad.py:205-215,233-262`). vector
   CertaGrad와 uniform은 validation cost를 센다(`src/certagrad.py:132,173-184`).
4. sweep은 이 서로 다른 estimand와 비용을 그대로 oracle/uniform에 비교한다
   (`src/c2_sweep.py:92-107`). selection과 truth가 같은 micro-group pool을 쓰는 기존
   P0-7 leakage도 남아 있다.

**영향:** `paper/main.tex:603-627,985-993`의 “certification measuring instrument”,
delta 보증, 5--8 orders 결론은 현재 구현으로 뒷받침되지 않는다. confidence sequence
또는 time-uniform union bound, 하나의 명시적 estimand, validation 관측 비용 포함,
selection/truth 독립 split으로 재설계해야 한다.

#### P0-14. 27B 실행기는 유효하지 않은 smoke와 구조적 OOM stage를 실행함: 확정 결함

- `scripts/go_27b.sh:30-31`은 `FRESH_K=4`로 smoke를 돌린다. 기본
  `micro_group=4`라 prompt당 group이 하나뿐이고, oracle split은 `h=0`이 되어
  empty mean/NaN을 만든다(`src/experiment.py:213-232,526-533`). smoke report 존재는
  floor/report 경로의 유효성을 검증하지 못한다.
- 같은 script는 smoke와 본실행 모두 `OM_SKIP_HYBRID=1`을 설정하지 않는다
  (`scripts/go_27b.sh:30-31,61-62`). 그러나 runner 자체가 27B hybrid의 pi+beta 동시
  상주가 80GB에서 불가능하다고 명시한다(`scripts/run_14b.sh:258-263`).
- `run_14b.sh`는 `set -uo pipefail`만 사용하고 merge/report `run_stage` 실패를 확인하지
  않은 채(`scripts/run_14b.sh:256-257`) 마지막에 `DONE`을 만든다
  (`scripts/run_14b.sh:273-274`). hybrid를 skip하는 대형 모델 경로에서는 부분 결과가
  완주로 표시될 수 있다.

**조치:** smoke는 `fresh_k >= 2*micro_group` 및 나눗셈 조건을 강제하고 27B에서는
hybrid를 명시적으로 제외한다. 모든 필수 stage를 `|| exit 1`로 묶고 최종 artifact
schema·coverage 검증을 통과한 경우에만 원자적으로 DONE을 써야 한다. `go_new.sh`의
`OM_SKIP_HYBRID=1` 경로를 정본으로 합치는 편이 안전하다.

### 15.2 추가 주요 구현·재현성 문제(P1)

#### E10. rollout/shard 완결성 검사가 정확한 K를 보장하지 않음

- `merge_rollouts()`는 prompt 존재와 중복 rollout ID만 검사하고, prompt마다 정확히
  K개인지와 ID가 `0..K-1`인지 검사하지 않는다(`scripts/run_14b.sh:47-75`).
- `make_hard_pool.py:59-67`도 prompt coverage가 불완전해도 live row가 하나라도 있으면
  partial pool을 기록한다. prompt별 prescreen K는 확인하지 않는다.
- behavior K가 `k_cell`보다 작아도 hybrid는 `b_rows[:k_cell]`을 그대로 사용해 bb/bp와
  pp/pb의 K가 달라질 수 있다(`src/hybrid.py:84-110`).

각 artifact에 기대 prompt ID와 정확한 K·rollout ID 집합을 검증하고, 하나라도 다르면
merge/DONE을 거부해야 한다.

#### E11. hard pool provenance가 모델 간 충돌하고 선택 불확실성을 버림

- prescreen run 디렉터리는 model tag를 포함하지만 output은 공통
  `$OM_WORK/pools/$DS-hard.jsonl`이다(`scripts/prescreen_pool.sh:14-16`). 다른 모델·revision이
  이전 모델의 pool을 그대로 사용할 수 있다(`scripts/go_27b.sh:42-47`,
  `scripts/go_new.sh:33-38`).
- pool row에는 원본 dataset ID, prescreen model/revision, K, seed, pass-rate, 설정 hash가
  없다(`src/make_hard_pool.py:49-76`).
- K=8에서 우연히 mixed reward였던 prompt를 고르는 절차는 winner's curse/regression-to-
  mean을 만든다. 본실행 liveness는 별도 표본으로 다시 qualification해야 한다.
- `go_new.sh:34`는 0-byte 포화 pool을 “미실행”으로 보고 호출 때마다 prescreen을 반복한다.
  이 문제를 고친 `go_27b.sh:43-53`과 동작이 다르다.

#### E12. 결과 수확이 run별 divergence 파일을 덮어씀

`scripts/go_v2.sh:125-133`은 모든 run의 `divergence_stats.shard*.json`을 같은 결과
폴더·같은 basename으로 복사한다. 뒤 run이 앞 run을 덮어써 artifact bundle에서 KL,
ESS, clip fraction의 run provenance가 사라진다. run별 하위 폴더 또는 tag prefix와
checksum이 필요하다.

#### E13. judge/readout의 인과 gate가 하나의 일관된 조건을 요구하지 않음

- C1은 g10과 g01 실패가 서로 다른 run에서 나와도 통과한다
  (`src/judge.py:53-96`).
- C1'은 각 축이 실패한 run들 중 **어느 한 cut**에서 한 번만 strict improvement를
  보이면 OR로 누적한다(`src/judge.py:98-140`). 두 축의 회복이 같은 run·같은 cut에서
  일어날 필요가 없고 세 cut 탐색의 다중비교 보정도 없다.
- `readout_summary.py:74-94`는 두 one-sided cell이 같은 run에서 모두 나빠야 하고,
  pp가 두 mixed cell을 같은 cut에서 이겨야 한다. judge와 의미가 다르다.
- C3 문서는 oracle/인증 선택을 말하지만 코드는 oracle과 random만 비교한다
  (`src/judge.py:147-160`).

사전 등록 gate는 run×cut 단위의 하나의 사건으로 정의하고 seed 반복 및 다중비교를
처리해야 한다. judge, readout, table generator가 같은 판정 함수를 호출해야 한다.

#### E14. downstream budget·seed·tie 처리에 추가 결함이 있음

- selector top-k는 jitter 없이 index/dict order로 tie를 자른다
  (`src/train_downstream.py:132-141`).
- random selector와 학습 prompt RNG가 모든 run에서 `Random(0)`으로 고정되고
  (`src/train_downstream.py:46,133-134`), `scripts/go_full.sh:61-63`도 run seed를 넘기지
  않는다. random baseline의 seed 분산을 잴 수 없다.
- `max(1, (budget-selection_cost)//k)` 때문에 selection cost가 총예산을 이미 넘겨도
  학습 한 step을 실행한다(`src/train_downstream.py:142-145`).
- `go_full.sh:56-70`은 source 네 개를 동시에 띄워 GPU가 4장보다 적으면 같은 GPU에
  모델 여러 벌을 올린다.

#### E15. frontier의 비용과 요약 통계가 비교 기준과 다름

- behavior rollout이 없으면 모든 prompt를 smoothed pass-rate 0.5로 채운다
  (`src/frontier.py:74-82`). 누락 artifact가 가장 informative한 predictor처럼 보일 수 있다.
- fresh 비용은 micro-group 수로 세지만 원고와 일부 출력은 rollout 수로 읽힌다
  (`src/frontier.py:206-223`). 기본 group size 4만큼 절대 비용 단위가 다르다.
- divergence shard를 token/rollout 수로 가중하지 않고 단순 평균한다
  (`src/frontier.py:309-318`).
- F3는 budget이 다른 정책 중 family별 최고 precision만 비교한다
  (`src/frontier.py:353-368`). budget-matched 우위가 아니다.
- 48-prompt floor gate는 top-k가 약 4개이고 한 진단 split에 의존해 임계값 분산이 매우
  크다(`src/frontier.py:225-250`). “48이면 충분”에는 별도 power/calibration이 필요하다.

#### E16. code-domain loader는 확정 crash와 host code-execution 위험이 있음

- `src/data.py:258-272`의 `_maybe_json_list()`는 `json`을 import하지 않는다. 문자열로
  저장된 MBPP tests 또는 KK names/solution에서 실제 호출하면 `NameError`가 재현된다.
- MBPP/APPS reward는 모델 생성 코드를 host Python으로 실행한다
  (`src/data.py:443-489`). `python -I`는 Python path 격리일 뿐 filesystem, network,
  subprocess, fork, CPU/memory를 격리하지 않는다. APPS `capture_output=True`에는 output
  크기 제한도 없다.

code-domain 실험은 container/seccomp, non-root UID, read-only filesystem, network off,
cgroup CPU/memory/PID/output 제한 안에서 실행해야 한다.

#### E17. reset·harvest가 실패 또는 정리 누락을 숨김

- soft reset은 `$OUT_ROOT/drift*` 자식만 순회해 v2처럼 artifact가 run root에 있는 구조를
  정리하지 못한다(`scripts/reset_run.sh:43-61`). 완료 메시지 뒤에도 score/floor가 남는다.
- `pkill -f "src/experiment.py"`와 전역 `gpu_keepalive` kill은 다른 run까지 종료한다
  (`scripts/reset_run.sh:21-24`).
- `harvest.sh`는 kcurve/reversal/stats 실패를 `|| true`와 stderr 폐기로 숨긴다
  (`scripts/harvest.sh:12-20`). 빈·부분 보고서도 전달 폴더에 들어갈 수 있다.
- `make_tables.py`는 표 생성 예외를 문장 하나로 넣고 exit 0을 반환한다
  (`src/make_tables.py:305-322`). 자동 수확기가 partial table을 성공으로 오인한다.

#### E18. 통계 도구의 tie·zero·K 규칙이 통일되지 않음

- `experiment/judge/frontier/make_tables`는 `int`, `stats/readout/reversal/kcurve`는
  `round`를 쓴다. P0-5가 분석 도구 전체에 퍼져 있다.
- report는 ID-correlated same-seed jitter, stats/readout은 한 RNG stream을 순차 소비,
  kcurve/precheck는 독립 seed, judge/make_tables/downstream은 deterministic order다.
- `stats_extra.py:37-40`의 sign interface는 loss 수를 받지 않고 `0.5**wins`를 출력한다.
  모든 non-tie가 win일 때만 맞는 식이다.
- `reversal_freq.py:103-128`은 exact `score==0`으로 signal을 판정하지만 다른 도구는
  gradient norm `<1e-6`을 쓴다. 수치상 작은 noise가 signal로 분류될 수 있다.
- `tag_of()`는 dapo/math500/hard/27b가 아닌 MBPP·KK·APPS·신규 모델 run을 GSM8K로
  분류한다(`src/kcurve_floor.py:126-133`, `src/precheck_hard.py:118-125`). 사전 등록
  majority vote가 다른 task run으로 오염될 수 있다.

### 15.3 원고에 직접 미치는 추가 영향

1. `paper/main.tex:371-377`의 “20 jitter pairs”는 정본 report의 floor 1쌍 및
   estimator precision same-jitter와 모두 다르다.
2. `paper/main.tex:436-441,729-755`의 전 조건 K=32와 K-curve 표는 P0-11 원자료
   확인 전 확정 수치로 둘 수 없다.
3. `paper/main.tex:638-661`의 equal-budget hybrid guard는 P0-10 때문에 구현되지 않았다.
4. `paper/main.tex:603-627,985-993`의 certification 보증·비용 결론은 P0-13을 해결한
   뒤 다시 산출해야 한다.
5. `paper/main.tex:680-682`의 “every RLVR run already produces”는 micro-group gradient와
   split artifacts가 일반 run의 기본 산출물이 아니므로 사실과 다르다.
6. `paper/main.tex:441,661,998-1003`의 manifest 검증·배포 주장은 현재 manifest가
   artifact hash를 구속하지 않고 repository에 대형 raw result가 없다는 사실과 충돌한다.
7. 서지 4건의 제목 오류는 arXiv 공식 레코드로 재확인했다. §14.1 R1의 보류 판정을
   확정으로 갱신했다.

### 15.4 수정 우선순위 갱신

1. **즉시 중단/격리:** 현재 generation 계약으로 돌고 있는 27B 결과를 제출용 수치로
   승격하지 않는다. 기존 run directory를 수정된 설정과 재사용하지 않는다.
2. **CPU 우선 복구:** raw score가 있는 서버에서 P0-9 tie-independent precision과
   P0-5 동일 k를 먼저 재계산한다. 이 결과가 핵심 regime 결론을 바꾸는지 확인한다.
3. **runner 계약 수정:** P0-12/P0-14/E10을 먼저 고쳐 실패·혼합 run이 DONE이 되는
   경로를 막는다. 이후에만 비싼 GPU rerun을 시작한다.
4. **rollout/hybrid 수정:** P0-1/P0-2 generation·mask 계약과 P0-10 remaining-horizon를
   함께 고치고 unit test를 만든다.
5. **인증 주장 보류:** P0-13이 해결될 때까지 CertaGrad를 certified procedure나
   5--8-order lower-bound instrument로 부르지 않는다.
6. **27B 가설 검증:** model-specific hard-pool manifest와 독립 qualification gate를
   통과한 pool에서만 B17을 수행한다. 포화 pool은 포화 사례로만 기록한다.

### 15.5 추가 완료 판정 체크리스트

- [ ] all-tie synthetic에서 estimator-vs-oracle expected precision이 chance로 수렴함
- [ ] hybrid 4 cell의 실제 response-token 분포와 최대 horizon이 일치함
- [ ] manifest/config를 하나라도 바꾸면 stale artifact 재사용이 거부됨
- [ ] 모든 rollout artifact가 prompt별 정확 K와 `rollout_idx=0..K-1`을 만족함
- [ ] K-curve 축이 manifest 문자열이 아니라 raw artifact count와 일치함
- [ ] CertaGrad CI가 time-uniform coverage simulation을 통과하고 validation 비용을 포함함
- [ ] scalar/vector/oracle이 같은 estimand를 사용하고 selection/truth가 독립임
- [ ] 27B smoke가 최소 2 micro-group/half를 가지며 hybrid 없이 완료됨
- [ ] 필수 stage 실패 시 DONE이 생성되지 않음
- [ ] hard pool 파일명이 model/revision/config hash를 포함하고 독립 liveness 재검증을 통과함
- [ ] MBPP/KK string-field loader test와 sandboxed code-reward integration test가 통과함
- [ ] table/harvest 도구가 partial artifact에서 nonzero exit하며 run별 파일을 덮어쓰지 않음

### 15.6 이번 추가 점검의 실행 검증

- `.work/.venv-cu126/bin/python -m compileall -q src tests scripts`: 통과.
- 전체 `scripts/*.sh`에 대한 `bash -n`: 통과.
- `ruff check src tests`: 90 diagnostics. 대부분 스타일/유지보수 항목이며, 실제 실행
  결함 `src/data.py:265 F821 Undefined name json`은 E16으로 승격했다.
- `find_fresh_k()` 문자열/int manifest 최소 재현: 각각 32/16 반환(P0-11).
- all-tie top-k 최소 재현: same-seed precision 1.0, independent-seed 0.08,
  chance 0.0977(P0-9).
- scalar estimand 최소 재현: mean group cosine 0.4502, cosine of mean gradient 0.0(P0-13).
- repository와 `/home/kms/Downloads` 검색: 대형 run manifest/raw rollout 없음. local smoke
  artifact만 존재해 14B K와 대형 precision을 이 머신에서 재산출할 수 없었다.

---

## 16. 실험을 제외한 수식·본문·인용 전수 점검 (4차, 2026-08-19)

이번 절은 실험 수치의 진위와 구현 결함을 판정 대상에서 빼고, `paper/main.tex`의
population 수식, theorem/corollary/proposition, measurement ceiling, certification 수식,
관련연구 설명과 참고문헌만 다시 검토한 결과다. 결론부터 말하면 **2x2 change-of-measure와
두 반례의 부호는 맞지만, 현재 정리의 양화 범위, metric-blindness corollary, measurement
ceiling 증명은 그대로 제출할 수 없다.**

### 16.1 수식별 최종 판정

| 대상 | 판정 | 핵심 이유 |
|---|---|---|
| `g00/g10/g01/g11` change-of-measure (`282-307`) | **조건부 타당** | exact unclipped ratio, support, 동일 환경, 적분가능성이 있으면 항등식과 두 error decomposition이 맞다. |
| continuation/occupancy 반례 (`939-955`) | **타당** | 직접 재유도와 verifier가 모두 `g_pi=-epsilon/2`, 실패 cell `=+epsilon/2`를 확인한다. |
| arbitrarily small KL | **타당하나 문장 오류** | 정확한 KL은 `2 epsilon log((1/2+epsilon)/(1/2-epsilon)) = 8 epsilon^2 + O(epsilon^4)`이다. 단 `0<epsilon<1/2`, `epsilon -> 0`인 family로 써야 한다. |
| 모든 `K>=2` group normalization | **결론은 타당, 증명 누락** | 이진 보상과 현재 두 construction에는 일반 K 폐형식 증명이 가능하다. K=2,4,8 열거만으로 현재 문장의 일반성은 증명되지 않는다. |
| Metric blindness corollary (`327-337`) | **현재 형태는 수학적으로 성립하지 않음** | diagnostic class가 정의되지 않았고, pointwise true-gradient cosine은 오히려 `-1`로 실패를 탐지한다. norm/margin lower bound가 필요하다는 결론도 단독으로는 나오지 않는다. |
| Disagreement proposition (`339-360`) | **예제 명제만 타당, 제목·후속 설명 과장** | continuation 예제에서 `cos(g10,g01)=-1`은 맞지만 두 one-sided cell의 동시 실패 정리가 아니다. |
| Measurement ceiling (`385-407,967-980`) | **증명 결함 및 가정 누락** | Spearman-Brown 조건, estimator의 Gaussian/Markov 조건, iid prompt 조건이 빠졌고 correlation에서 top-k overlap으로 넘어가는 단계가 임의 estimator에는 성립하지 않는다. |
| angular radius `arcsin(r/||mu_hat||)` (`617`) | **deterministic ball 아래 타당** | `r<||mu_hat||`일 때 ball 안 vector의 최대 방향각이다. 다만 그 ball 자체의 coverage와 순차 재사용은 별도 문제다. |
| 5-8 orders certification 결론 | **일반 정리 아님** | `alpha(m)=arcsin(c/sqrt(m))` 외삽과 특정 CI가 맞다는 조건 아래의 산수일 뿐, 모든 top-k 인증 알고리즘의 lower bound가 아니다. |

### 16.2 2x2 항등식에서 빠진 가정과 정의

`paper/main.tex:282-307`의 식은 다음 조건을 명시하면 맞다.

1. 유한 horizon이거나 score-weighted return이 적분가능하고 미분과 기대값 교환이 가능하다.
2. target trajectory measure가 behavior measure에 절대연속이다. 즉 `pi(a|h)>0`인
   target-reachable `(h,a)`에서는 `beta(a|h)>0`이어야 한다.
3. 두 정책은 같은 initial-state distribution, transition dynamics, reward function을 쓴다.
4. `r_j`는 clipping, truncation, self-normalization이 없는 정확한 likelihood ratio다.
5. `Q_t^beta(h,a)=E[R|H_t=h,A_t=a, A_{t+1:T}~beta]`, `d_pi`, `d_beta`를
   명시해야 한다. 현재 원고는 이 기호들을 정의하지 않는다.
6. variable-length response에서는 EOS를 trajectory action으로 포함할지와 ratio product의
   종료점을 정의해야 한다.

이 조건 아래 `P_t r_t`는 prefix와 현재 action을 target measure로 바꾸므로
`g10=Sum_t E_{d_pi,pi}[Q_t^beta z_t]`이고, `r_t S_t`는 action과 continuation만
바꾸므로 `g01=Sum_t E_{d_beta,pi}[Q_t^pi z_t]`이다. 따라서 본문의 두 error
decomposition은 대수적으로 맞다. `P_t r_t S_t`는 전체 trajectory ratio이므로
`g11=g_pi`도 맞다.

다만 `g_pi`는 앞에서 vector로 정의했는데 Theorem 1에서는 갑자기
`g_pi=-epsilon/2`인 scalar처럼 쓴다. 반례는 “첫 logit에 대한 한 좌표” 또는 “고정
방향으로 투영한 directional gradient”에 대한 결과임을 theorem statement에서 밝혀야 한다.

### 16.3 반례 정리의 정확한 수식

부록은 확률을 `1/2 +/- epsilon`으로 정의한다. 따라서 Theorem 1의 “for every
`epsilon>0`”은 틀리고 정확한 범위는 `0<epsilon<1/2`다. 두 construction에서 정책을
바꾼 Bernoulli의 trajectory KL은 정확히

```text
D_KL(pi || beta)
  = 2 epsilon log((1/2 + epsilon)/(1/2 - epsilon))
  = 8 epsilon^2 + O(epsilon^4),  epsilon -> 0.
```

그러므로 고정된 하나의 `epsilon`에 “KL is `O(epsilon^2)`”라고 쓰기보다,
`epsilon downarrow 0`으로 가는 policy family와 상수 범위를 명시해야 한다. KL 방향도
`D_KL(pi || beta)`로 고정한다.

일반 group size는 열거 대신 다음 한 줄로 닫을 수 있다. `M`을 K개 rollout 중 성공 수,
`Delta=E[z|R=1]-E[z|R=0]`라 하자. 두 construction 모두 `P(R=1)=1/2`이고
`E[z]=0`이므로 raw gradient는 `g=Delta/4`다. population-denominator group std를 쓰고
동일보상 group의 update를 0으로 정의하면

```text
E[g_group | M]
  = sqrt((M/K)(1-M/K)) Delta,
E[g_group]
  = 4 E[sqrt((M/K)(1-M/K))] g.
```

마지막 계수는 모든 `K>=2`에서 양수이므로 부호가 보존된다. sample-std나 denominator
epsilon을 쓰더라도 계수만 양수로 바뀐다. 원고에는 zero-variance group의 `0/0` 처리
convention과 이 일반 K 증명을 넣어야 한다.

### 16.4 Metric-blindness와 disagreement의 논리 오류

1. `norm-blind, scale-invariant single-estimator diagnostic`가 함수의 정의역, 관측 정보,
   연속성까지 전혀 정의되지 않았다. 이 상태에서는 “모든 그러한 diagnostic”이라는
   보편명제를 증명할 수 없다.
2. 원고가 넓게 말하는 “gradient cosine은 blind”는 거짓이다. 반례에서 실패 estimator와
   true current gradient의 cosine은 `-1`이므로, fresh oracle과 비교하는 cosine은 즉시
   탐지한다. 성립 가능한 주장은 “stale estimator의 checkpoint self-cosine만으로는
   배제할 수 없다”처럼 실제 관측 가능한 특정 metric으로 좁혀야 한다.
3. KL upper bound와 ESS lower bound만으로 실패를 배제할 수 없다는 결론은 살릴 수 있다.
   continuation 반례의 retained prefix weight는 정확히 1이라 one-sided ESS가 최대인데도
   부호가 뒤집히고, full ratio ESS도 `epsilon -> 0`에서 최대값으로 간다.
4. construction은 신호 norm과 두-prompt ranking margin도 `epsilon`과 함께 0으로 보낸다.
   따라서 “어떤 signal-strength/error-control 가정이 필요하다”까지는 말할 수 있지만,
   **gradient-norm lower bound나 margin lower bound 자체가 필요하고 충분하다는 결론은
   나오지 않는다.** 양의 norm도 큰 bias로 뒤집힐 수 있고, margin은 estimator error의
   upper bound와 함께 있어야 순위를 보장한다.
5. Proposition 제목의 “double failure”는 continuation 예제의 `g00`과 `g10`을 뜻할 뿐,
   `g10`과 `g01`의 동시 실패가 아니다. 제목을 “Cell disagreement in the continuation
   construction” 정도로 바꿔야 한다.
6. `358-360`의 “both one-sided가 크게 실패하려면 두 bias term이 agree하므로
   non-generic”도 일반 vector에서는 틀리다. 두 bias는 true gradient에 대해 각각 adverse
   projection만 가지면 되고 서로 같은 방향일 필요가 없다. 50k random search는 지정한
   proposal에서의 음성 관찰이지 이 문장의 증명이 아니다.
7. theorem은 한 prompt의 scalar directional gradient reversal을 보인다. 논문 제목의
   top-k misranking으로 연결하려면 verifier에만 있는 two-prompt top-1 construction을
   원고에 넣고, 그 ranking margin도 `epsilon`과 함께 사라진다는 한계를 밝혀야 한다.

안전한 corollary는 다음 정도다: “임의의 `delta>0`에 대해 `D_KL(pi||beta)<delta`이고
retained-weight ESS가 최대에 임의로 가까우면서도 각 one-sided estimator의 한 방향 성분이
true gradient와 반대인 예제가 존재한다. 따라서 KL과 ESS만으로 sign safety를 보장할 수
없다.” 이 문장은 현재 construction으로 직접 증명된다.

### 16.5 Measurement ceiling의 수학적 결함

Spearman-Brown 식 자체는 parallel-test model에서 맞다. 즉
`a=t+e_a`, `b=t+e_b`, `e_a,e_b`가 iid이고 `t`와 독립이며 full oracle이
`o=(a+b)/2`일 때

```text
rho_h = Corr(a,b) = Var(t)/(Var(t)+Var(e)),
rho_f = Corr(o,t)^2 = 2 rho_h/(1+rho_h).
```

현재 Proposition 2의 “exchangeable Gaussians”와 `Corr(a,b)=rho_h`만으로는 이 구조가
나오지 않는다. 최소한 다음을 고쳐야 한다.

1. `t_i`와 half errors가 prompt별 iid Gaussian이고, halves가 equal-variance parallel
   measurements이며, `o_i=(a_i+b_i)/2`라고 명시한다. “exchangeable”만으로는 부족하다.
2. reliability가 `Corr(o,t)^2`인지 correlation 자체인지 정의한다. 현재 식은 reliability를
   squared correlation으로 쓰고 다음 줄에서 `sqrt(rho_f)`를 correlation으로 쓴다.
3. `0<=rho_h<=1`을 가정한다. 유한표본에서 관측되는 음수 half-correlation에는 이
   reliability model과 역변환을 그대로 적용할 수 없다.
4. `Ovl(c)`가 오직 c의 함수가 되려면 prompt 간 pair가 iid이고 tie가 없는 연속분포여야
   한다. 일반 exchangeability나 cross-prompt dependence에서는 전체 covariance가 필요하다.

부록의 proof도 현재 형태로는 닫히지 않는다. `s`가 standardized이고 oracle noise와
독립이면 `Corr(s,o)=Corr(s,t)sqrt(rho_f)`라는 correlation bound는 맞다. 그러나 다음
문장의 Gaussian top-k overlap 함수를 적용하려면 `(s,o)`가 jointly Gaussian이어야 한다.
Proposition은 `s`를 “any estimator”로 두므로 이 조건이 없다. 예를 들어 `s=f(t)`인
strictly increasing nonlinear transform은 `t`와 top-k가 완전히 같지만 Pearson
correlation은 1보다 작고 `(s,o)`도 jointly Gaussian이 아니다. 이 예는 correlation 기반
proof step이 임의 estimator에 적용되지 않음을 바로 보여준다.

또한 `Corr(s,t)=1`의 equality condition은 “monotone transform”이 아니라 거의 확실하게
**positive affine transform**일 때다. 임의 estimator까지 포함하는 ceiling을 유지하려면
`s <- t -> o` Markov/no-extra-information 조건 아래 top-k의 Bayes-optimal selector가
`TopK(t)`임을 별도로 증명해야 한다. 더 간단한 수습은 proposition을 jointly Gaussian
`(s,t,o)`에 한정하는 것이다.

마지막으로 `floor = Ovl(rho_h)`와 ceiling은 population **expected overlap** 식이다. 한
번 관측한 유한 `n` split-half overlap은 그 기대값과 같지 않고, discrete하며 불확실성이
있다. 따라서 “observed floor를 invert하면 well-posed”, “ceiling에 도달했다”, “ceiling
위 결과는 artifact”라고 단정할 수 없다. CI를 통해 `rho_h`와 ceiling의 구간을 전파해야
한다. 낮은 ceiling은 noisy oracle과의 기대 agreement를 제한할 뿐 latent utility recovery나
두 방법 간 통계적 비교를 자동으로 무효화하지도 않는다.

`978-980`의 heavy-tail caveat와 P4 연결도 잘못됐다. P4는 saturation, zero/tied score,
shared liveness가 만드는 overlap artifact이고 heavy-tail Gaussian-model violation은 별도
failure mode다.

### 16.6 Certification 수식에서 유지할 것과 버릴 것

`||mu_hat-mu||<=r`, `r<||mu_hat||`일 때 최대 방향각
`alpha=arcsin(r/||mu_hat||)`은 맞다. 두 vector ball의 cosine interval에
`alpha_i+alpha_v`를 쓰는 것도 deterministic simultaneous balls가 참이면 보수적으로
유효하다.

문제는 ball을 만드는 통계와 반복 사용이다.

- 구현의 Gaussian radius는 covariance를 isotropic으로 치환한다. anisotropic Gaussian의
  norm bound에는 trace뿐 아니라 Frobenius/operator norm 항이 필요하다.
- adaptive draw마다 fixed-n interval을 다시 보고 멈추므로 time-uniform confidence
  sequence나 시간축 union bound 없이는 advertised delta coverage가 아니다.
- `alpha(m)=arcsin(c/sqrt(m))`은 covariance와 c가 sample size에 따라 고정되고 iid
  `1/sqrt(m)` scaling이 유지된다는 가정이다. candidate radius와 validation radius의 합,
  estimated variance의 변화도 별도로 반영해야 한다.
- 따라서 `4.1e5`에서 `1.0e8` 배라는 비율은 해당 extrapolation의 계산값일 수는 있어도
  “structural”, “proved impossibility”, “never certify”의 근거는 아니다. bandit lower
  bound와 연결하려면 gap-dependent instance-wise lower bound를 별도로 제시해야 한다.

### 16.7 인용 원문 대조

**원문과 맞는 부분**

- CROPI의 40/50 cosine `>0.6`과 top-10% consistency `28.80%`는 공식 논문 부록 수치와
  일치한다. CROPI의 practical estimator가 current-token ratio와 behavior group reward를
  쓰므로 axis 표에서 `g00` 계열로 분류하는 것도 타당하다.
- Cumulative Token/CTPO는 exact cumulative prefix ratio의 occupancy correction을
  이론적으로 보이고, practical objective에서는 group outcome reward surrogate와 clipping을
  쓴다고 명시한다. 원고 `223-226`의 핵심 문제 제기는 이 구분을 유지하면 맞다.
- TIC-GRPO가 trajectory-level ratio로 current-policy gradient를 겨냥한다는 설명,
  Data Shapley와 NASH의 random-selection 문제 및 재설계, ACE의 jointly valid weak-CI
  조건, crowd-reading 논문의 split-half ceiling 설명은 원문 방향과 맞다.

**고쳐야 하는 인용 설명**

1. Multi-Step/NFPO는 다음 `N-1`개 token ratio를 쓰는 연속적인 partial correction이다.
   `N`이 남은 horizon 전체를 덮을 때만 정확한 `g01`이다. `227-229`의 “restore the
   outcome axis”는 “interpolates toward `g01`”로 낮춰야 한다.
2. A Step Back의 theorem은 exact prefix ratio를 쓰지만 실제 제안법 MinPRO는 minimum
   prefix proxy다. CTPO도 practical clipping 이후에는 population `g10`과 같지 않다.
   관련연구 표에서 theorem quantity와 implemented algorithm을 같은 cell로 단정하지 않는다.
3. TIDE 원문은 **가장 negative한 1% token**이 gradient의 거의 절반을 차지한다고 한다.
   `261-263`의 일반 “top 1%”는 방향 정보를 잃었고, 이를 본 논문의 `0.1% ESS`와 “same
   behavior”라고 동일시할 근거도 없다. “analogous tail concentration” 정도가 정확하다.
4. Kaufmann의 top-k bandit lower bound는 gap-dependent identification 배경이다. 현재
   CertaGrad 측정치를 그 lower bound의 “empirical instance”라고 부를 수는 있어도,
   본문의 5-8 orders를 그 논문이 보증하는 lower bound처럼 읽히게 하면 안 된다.

**참고문헌 자체의 오류·누락**

- TIC-GRPO, Multi-Step, ACE, Floor/Ceiling 네 제목 오류는 §7 R1과 §15.3에서 이미
  확정했다.
- `GradAlign`의 정식 제목은 *GradAlign: Gradient-Aligned Data Selection for LLM
  Reinforcement Learning*이고 COLM 2026이다. `LearnAlign`의 정식 제목은
  *LearnAlign: Data Selection for LLM Reinforcement Learning with Improved Gradient
  Alignment*이고 ACL 2026 Findings다. 현재 항목은 제목과 venue가 불완전하다.
- Liu et al. 항목은 PMLR 정식 레코드상 2020 출판물이다. UAI 행사 연도 2019를 쓸 경우
  proceedings year와 혼동되지 않게 정식 BibTeX를 사용해야 한다.
- arXiv 항목 대부분에 저자와 연도가 없고 Spearman/Brown에는 journal, volume, pages가
  없다. 현재 manual bibliography는 검색 가능한 최소 서지정보를 충족하지 못한다.
- setup의 Qwen2.5, GSM8K, MATH-500, DAPO-Math-17k, LoRA, GRPO, JL projection에 원 출처
  인용이 없다. 특히 `MATH-500`은 원 MATH dataset과 어떤 curated subset을 썼는지
  dataset ID와 revision까지 밝혀야 한다.
- `272-274`의 “prior work가 oracle reliability를 보고하지 않았다”는 novelty absence
  claim은 문헌검색 범위와 날짜를 제시하지 않으면 검증할 수 없다. “To our knowledge”가
  있어도 systematic-search appendix가 안전하다.

공식 대조 출처: [CROPI](https://aclanthology.org/2026.acl-long.2141/),
[A Step Back](https://arxiv.org/abs/2601.22718),
[Cumulative Token](https://arxiv.org/abs/2605.07331),
[Multi-Step](https://arxiv.org/abs/2605.20865),
[TIC-GRPO](https://arxiv.org/abs/2508.02833),
[GradAlign](https://arxiv.org/abs/2602.21492),
[LearnAlign](https://arxiv.org/abs/2506.11480),
[TIDE](https://arxiv.org/abs/2608.09836),
[ACE](https://arxiv.org/abs/2601.20989),
[Floor/Ceiling](https://arxiv.org/abs/2608.01704),
[Data Shapley](https://proceedings.mlr.press/v235/wang24cg.html),
[NASH](https://arxiv.org/abs/2605.10684),
[Liu et al.](https://proceedings.mlr.press/v115/liu20a.html),
[Huang and Jiang](https://proceedings.mlr.press/v119/huang20b.html),
[Kaufmann and Kalyanakrishnan](https://proceedings.mlr.press/v30/Kaufmann13.html).

### 16.8 본문 내부의 비실험적 오류·과장

1. `40-48,95-101,139-141,175-180`의 “any one-sided”, “every monitored metric”,
   “KL/ESS/cosine blind”는 theorem보다 넓다. “each of the two one-sided population
   quantities has a counterexample; KL and retained-weight ESS alone do not rule it out”로
   제한한다.
2. `67`의 “Exact scores need fresh rollouts”는 finite rollout도 exact score가 아니라
   on-policy Monte Carlo estimate다. “Estimating the target score requires current-policy
   rollouts”가 정확하다.
3. `72-74`의 “Existing estimators correct exactly one axis”는 full trajectory TIC와
   finite-step partial corrections를 스스로 뒤에서 인용하므로 내부적으로 거짓이다.
   “Several practical estimators correct at most one axis”로 바꾼다.
4. split-half overlap을 정의만으로 “recoverable ranking signal”, “conservative floor”라고
   부를 수 없다. 그것은 지정된 split, budget, tie rule, shared validation target에 대한
   oracle self-agreement statistic이며, signal interpretation은 추가 noise model에
   조건부다.
5. `401-407`의 low ceiling에서 “comparative evaluation is void”와 “further method
   development cannot show gains”는 Proposition 2가 주지 않는 결론이다. 기대 overlap의
   상한과 방법 간 검정력은 다른 문제다.
6. `835-837`은 ceiling 위 결과를 곧바로 artifact라고 하지만 `857-859`는 assumption
   violation이나 dependence일 수도 있다고 쓴다. 뒤 문장이 맞고 protocol box를 그에
   맞춰야 한다.
7. `882-884`의 safety target에 protocol이 “unchanged” 적용된다는 말은 검증되지 않았다.
   safety score의 noise, partial observability, adversarial reward 조건을 별도 가정으로
   두어야 한다.

### 16.9 제출 전 수식·인용 수정 우선순위

1. Theorem 1을 scalar/directional statement, `0<epsilon<1/2`, `epsilon -> 0`, exact KL,
   support 조건으로 다시 쓴다.
2. 일반 K group-normalization 폐형식 증명과 zero-variance convention을 부록에 넣는다.
3. Corollary 1을 KL+ESS 단독 불충분 명제로 축소하고 cosine 보편명제 및 norm/margin
   necessity 문장을 삭제한다.
4. Proposition 1 제목과 `non-generic` 설명을 고치고 two-prompt ranking construction을
   원고에 포함한다.
5. Proposition 2를 parallel iid Gaussian model로 다시 정의하고, jointly Gaussian
   estimator로 좁히거나 Bayes-optimal top-k 증명을 새로 쓴다. finite-floor uncertainty를
   ceiling까지 전파한다.
6. Certification을 theorem/lower bound가 아니라 특정 CI extrapolation으로 명명한다.
7. 관련연구의 partial/exact/practical correction을 구분하고 정식 BibTeX와 setup 인용을
   복원한다.

### 16.10 이번 수식 검산

- 외부 theory verifier 재실행: 통과. 두 반례의 전 cell, `KL/epsilon^2 -> 8`,
  K=2/4/8 group normalization, two-prompt top-1 reversal을 재확인했다.
- verifier의 50k search는 theorem 증명이 아니라 proposal-dependent 음성 관찰로만 판정했다.
- 일반 K 식과 Spearman-Brown 식은 위와 같이 손으로 재유도했다.
- `sympy`는 현재 system Python에 설치되어 있지 않아 symbolic CAS 출력은 만들지 못했다.
  이 점은 폐형식 유도 자체의 판정에는 영향을 주지 않는다.

## 17. 강한 주장 유지형 이론 개정 (2026-08-19)

§16의 판정 뒤 `paper/main.tex`을 직접 개정했다. 이번 개정은 문제 명제를 단순 삭제한 것이
아니라, 실제 배포에서 one-sided selector가 사용하는 정보 집합을 정의하고 그 정보 전체에
대한 no-free-lunch로 강화한 것이다. 아래 상태가 §16의 해당 항목을 대체한다.

### 17.1 One-sided impossibility

- `c in {10,01}` 각각에 대해 `c`-only selector를 정의했다. 임의의 후처리, calibration,
  threshold, randomized top-k rule, KL, nonzero directional summand의 retained-weight
  moments, ESS, directional norm, stale self-similarity를 모두 허용하고 omitted ratio
  axis만 금지한다.
- 임의의 rollout 수 `N`과 `delta>0`에 대해, 전체 `c`-only 관측 law와 정확한 KL이 같지만
  true top-1이 반대인 두 pool `W0,W1`을 구성했다.
- 따라서 randomized selector의 worst-case top-1 error는 최소 `1/2`이고 deterministic
  selector는 두 pool 중 하나에서 반드시 틀린다. 고정 prompt를 추가하면 임의 top-k
  boundary에도 그대로 embedding된다.
- canonical sign reversal은 별도 특수 경우로 유지했다. 정확한 범위 `0<epsilon<1/2`,
  exact KL, scalar directional statement를 본문에 반영했다.
- binary group normalization은 모든 `K>=2`에 대해 양의 상수배가 된다는 폐형식 lemma를
  추가했다.

### 17.2 Measurement ceiling

- estimator 자체가 Gaussian이라는 잘못된 proof를 제거했다.
- iid latent Gaussian과 equal-variance parallel oracle noise 아래, estimator 정보 `Z`가
  held-out oracle noise와 latent truth 조건부 독립이면 임의의 nonlinear/randomized
  selector `S(Z)`를 허용한다.
- fixed latent vector에서 `TopK(t)`가 oracle top-k inclusion probability를 최대화한다는
  exchange coupling을 사용해 Bayes-optimality를 증명했다.
- floor는 population expected overlap으로 정의하고 finite observed floor의 uncertainty를
  ceiling까지 전파하도록 본문, 표 caption, protocol을 수정했다.
- ceiling 초과를 곧바로 artifact로 단정하지 않고 parallel-noise model, held-out-noise
  independence, evaluation pipeline 중 최소 하나의 premise가 깨졌다는 falsification으로
  바꿨다.

### 17.3 Certification lower bound

- 두 Gaussian boundary prompt의 mean을 swap하는 change-of-measure proposition을 추가했다.
- 모든 uniformly delta-correct adaptive exact top-k algorithm에 대해
  `E[N_k+N_{k+1}] >= 2 sigma^2 Delta^{-2} kl(1-delta,delta)`를 증명했다.
- 기존 5--8 orders 수치는 CI instrument의 plug-in factor로 구분하되, inverse-square
  dependence 자체는 algorithm-independent minimax obstruction으로 유지했다.

### 17.4 인용·본문 동기화

- Multi-Step/NFPO를 full outcome restoration이 아니라 remaining horizon 길이에 따라
  `g01`로 수렴하는 partial correction으로 수정했다.
- TIDE를 일반 top 1%가 아니라 most-negative 1%로 수정하고 ESS와는 analogy라고 구분했다.
- TIC-GRPO, Multi-Step, ACE, Floor/Ceiling, GradAlign, LearnAlign의 정식 제목과 Liu의
  proceedings year 표기를 수정했다.
- true-gradient cosine은 stale-only information이 아니라는 경계를 명시하고, 안전 target은
  partial observability와 adversarial/misspecified reward를 별도 점검하도록 수정했다.

### 17.5 검증

- `python3 scripts/verify_theory.py`: 통과. 기존 sign reversal, exact IS cells, KL limit,
  ranking flip, group normalization과 새 indistinguishable-pool construction을 검산했다.
- `python3 tests/test_reversal_freq.py`: 통과.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`: 통과. undefined reference와
  overfull box를 제거했고, 본문 핵심식과 부록 증명을 분리한 17-page `paper/main.pdf`를
  재생성했다.
- `tests/test_core.py`는 현재 system Python에 `torch`가 없어 실행하지 못했다. 이번 변경은
  paper와 standalone theory verifier에 한정되므로 이 미실행은 이론 검산 결과와 분리한다.
