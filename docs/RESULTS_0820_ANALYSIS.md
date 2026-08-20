# 2026-08-20 결과 번들 분석 로그

분석 대상은 `/home/kms/0820`에 모인 Markdown 8개다. 이 문서는 원본을 수정하지
않고 파일 해시, 생성 계보, 수치 해석, 현재 프로토콜에서의 사용 가능 범위를 기록한다.

## 27B 식별 판정

이 번들은 **27B 실험 결과로 식별할 수 없다**. 파일 본문 전체에 `27B`,
`Qwen3.8-27B`, `Qwen3.6-27B`, `v2-27b-*` 모델 식별자나 모델 revision이 없고,
run 이름은 `gate-7b`, `gate-14b`, `v2-*`, `v3-*`뿐이다. 여기서 `v3`는 모델
크기가 아니라 당시 실행/계약 버전 이름이다.

현재 실행 스크립트가 만드는 27B run은 `qwen3.8-27b-...` 또는
`v2-27b-<dataset>-s<seed>` 계열이며, 각 run의 generation manifest,
`score_protocol.json`, `oracle_protocol.json`, source commit이 함께 있어야 모델과
프로토콜을 확정할 수 있다. 이 조건을 만족하는 run이나 수확 보고서는
`/home/kms/0820`에 없다.

따라서 아래 수치 분석은 전달 번들의 historical 상태를 설명할 뿐이다. 특히 DAPO의
포화/무신호 관측을 27B 기본 모델의 성능으로 귀속하거나, 27B에서 one-sided
misranking 가설이 검증 또는 반증됐다고 쓰면 안 된다. 27B 판정은 실제 27B run
디렉터리와 교정된 수확물이 도착한 뒤 별도로 수행한다.

## 결론

이 번들은 **historical exploratory snapshot**으로만 보존한다. 현재 논문의
confirmatory evidence나 최종 표 수치로 사용할 수 없다.

- `TABLES-v2.md`와 `TABLES-v3.md` 내부 생성 시각은 각각 2026-08-19 14:07:25,
  2026-08-19 21:03:44다. 모든 파일의 동일한 2026-08-20 10:18:53 수정 시각은
  결과 재계산 시각이 아니라 번들을 복사하거나 모은 시각으로 보인다.
- 두 표는 generation/EOS 계약 수정 `c6ca013`의 2026-08-19 22:57:54 KST보다 먼저
  생성됐다. 적어도 이 표들은 수정된 생성 계약의 결과일 수 없다.
- 어느 파일에도 generation manifest, `score_protocol.json`,
  `oracle_protocol.json`, source commit 또는 corrected protocol marker가 없다.
- 현 감사에서 확인한 score/oracle validation 공유 오류와 K-curve validation 재사용
  오류의 영향을 배제할 근거가 없다. 자세한 판정 기준은
  [FULL_AUDIT_2026-08-20.md](FULL_AUDIT_2026-08-20.md)를 따른다.
- `READOUT.md`는 0 byte이고 `STATS.md`의 `v2-s0-math500-math500` 섹션은 결과 행이
  하나도 없다. 번들 생성 파이프라인도 완결되지 않았다.

따라서 이 로그의 수치 해석은 무엇이 관측됐는지 설명하는 데만 사용한다. 교정된
프로토콜로 재점수화하고 모든 산출물을 다시 만든 뒤에만 논문 주장을 판정한다.

## 파일 인벤토리

| 파일 | 크기 | SHA-256 |
|---|---:|---|
| `FRONTIER-v2.md` | 4,190 | `9482a2ae4182b59c3861e0bf3bb3c6bb88410e72c95d081036af3a7359d7c9db` |
| `FRONTIER-v3.md` | 4,683 | `2e9253d601fd74b45148a6666f022c9a04e10af2e3025a16f627d793f04aeead` |
| `KCURVE.md` | 5,624 | `63588b4c15cf4c9a3a354ae3134ff30189d126853bb29866b3fa234348c3f534` |
| `READOUT.md` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `REVERSAL.md` | 28,518 | `2514e5982ad51b59f5e31a9848c100245bbe878f9ac384f267940b8cc0d76136` |
| `STATS.md` | 7,831 | `695afe097e6a925b2b1d2b8eb7179dd1831550c90eda29247282f5f2a727b9c5` |
| `TABLES-v2.md` | 2,839 | `f8a0394872b793db9f813d046345d88509fb940211adb37649561cdb60026e2e` |
| `TABLES-v3.md` | 2,919 | `e6da14bed35681d267d3586afbfc86f84702663710e0b50b6bde542022fe84ef` |

## 주장별 판정

| 주장 | 이 번들의 관측 | 판정 |
|---|---|---|
| C1: one-sided correction 실패 | v2와 v3 모두 `g10`, `g01`이 각 floor보다 높다 | 지지하지 않음 |
| C1': 두 축 복구의 인과 증거 | v3 cut=0.5에서 두 회복량은 양수지만 C1의 양쪽 실패 전제가 없다 | 확립되지 않음 |
| C2: 저비용 fresh audit 인증 | v2와 v3 모두 `certified=False`; margin이 각도 오차보다 작다 | 실패 |
| fresh 예산 증가의 이득 | 두 frontier 모두 단조 개선이 없고 2D refresh가 기준선을 넘지 못한다 | 지지하지 않음 |
| `FRESH_K=128` 권고 | Spearman--Brown 외삽과 관측치의 보정 오차가 크고 구 validation 재사용 영향이 남는다 | 채택 불가 |
| one-sided 반전 경보 | 일부 run의 `g10` 불일치 경보는 유의하지만 pooled 반전율이 oracle 자기 불일치와 비슷하다 | 탐색적 신호만 있음 |
| DAPO에서의 estimator 비교 | oracle 무신호가 91--95%이고 유효 분모가 2 또는 4인 run이 있다 | 해석 불가 |

## 게이트 결과

### v2-s0

- floor는 0.176이고 precision은 `g00=0.196`, `g10=0.314`, `g01=0.255`,
  `g11=0.216`이다. `g10`이 가장 높으며 어느 one-sided 조건도 floor 아래로
  떨어지지 않는다.
- floor 곡선은 관측 그룹 증가에 따라 0.255, 0.255, 0.255, 0.176이다. 더 많은
  관측이 안정적으로 floor를 높인다는 패턴이 없다.
- fresh live fraction은 258/512 (50%), behavior는 185/512 (36%)다. behavior에서
  전부 오답이 317/512 (62%)라 선택 신호가 제한된다.
- hybrid cut=0.5에서 `g10` 회복은 +1.00, `g01` 회복은 +0.00이다. 두 축 동시
  회복의 증거가 아니다.
- live margin은 0.12도이고 `alpha_v=73.12`도라 C2는 인증되지 않는다.

### v3-s2-math500

- floor는 0.225이고 precision은 `g00=0.325`, `g10=0.350`, `g01=0.275`,
  `g11=0.375`다. full correction인 `g11`이 가장 높고 역시 one-sided failure가 없다.
- floor 곡선은 0.325, 0.275, 0.275, 0.225로 관측 그룹이 늘수록 감소했다.
- fresh live fraction은 164/400 (41%), behavior는 109/400 (27%)다. behavior에서
  전부 오답이 287/400 (72%)다.
- hybrid cut=0.5의 회복량은 `g10=+0.06`, `g01=+0.19`다. 다만 C1 실패 조건이
  성립하지 않고 cut별 부호도 바뀌므로 canonical causal witness가 아니다.
- live margin은 0.01도이고 `alpha_v=180.00`도라 C2는 인증되지 않는다.

두 표의 `signal retention > 1`은 estimator가 oracle보다 우수하다는 뜻으로 쓰면
안 된다. 분모인 split-half floor가 noisy한 참조값이라 precision이 floor보다 높으면
기계적으로 1을 넘는다.

## Fresh-audit frontier

### v2-s0

- family 최고 precision은 `passrate_beta=0.340`이다. gradient 0.255, 2D refresh
  0.226, random audit 0.227, boundary audit 0.235, fresh 0.235보다 높다.
- fresh 관측은 `m1=0.235`, `m2=0.196`, `m4=0.216`으로 예산 증가 이득이 없다.
- 2D refresh는 p5에서 최대 0.241이지만 p10과 p25에서 0.139--0.164까지 악화된다.
- `clipfrac_g10=0.56`, `clipfrac_g11=0.959`, `traj_ess_frac_g11=0.001`이다.
  importance weight가 사실상 붕괴한 상태다.
- `token_kl_beta_pi=-0.5677`은 이론적 KL의 비음수성과 강하게 충돌한다. 유한 표본
  추정량이 음수가 될 수 있다는 점을 감안해도 크기가 커서 likelihood/generation
  계약 또는 KL 구현을 확인하기 전에는 estimator 결과를 신뢰할 수 없다.

### v3-s2-math500

- gradient, 2D refresh, boundary audit, fresh 최고값은 모두 약 0.275이고 random audit
  최고값도 0.276이다. 2D refresh의 우위가 없다.
- fresh 관측은 `m1=0.230`, `m2=0.275`, `m4=0.225`로 역시 단조 개선이 없다.
- `clipfrac_g11=0.3113`, `traj_ess_frac_g11=0.0079`다. v2보다는 낫지만 ESS는
  여전히 극단적으로 낮다.
- `truth_margin_k=0.0013`, `truth_reliability=0.325`라 정책 간 0.001--0.02 수준의
  precision 차이를 안정적 우위로 해석할 근거가 약하다.

각 frontier의 dataset 평균은 seed가 1개뿐이다. 표의 `seed-sd=0.000`은 재현성이
높다는 뜻이 아니라 seed 간 분산을 추정할 수 없다는 뜻이다.

## K-curve

`KCURVE.md`는 GSM8K 계열 4개 중 3개가 K<=256에서 2x chance에 도달할 것으로
예측하고 `FRESH_K=128`을 권고한다. 현재는 이 권고를 실행 근거로 쓰지 않는다.

- 9개 run 중 다수에서 관측 floor가 K 증가와 함께 감소하거나 비단조적이다.
- split-half 상관은 대부분 0에 가깝고 여러 run에서 음수다. 이 영역에서
  Spearman--Brown 외삽은 작은 상관 추정 오차에 매우 민감하다.
- `v2-s1-math500-math500`은 K=32 예측 0.109와 관측 0.260으로 보정 오차가 0.151이다.
  다른 run도 예측과 관측의 차이가 작지 않다.
- K=64--256 값은 관측값이 아니라 모형 외삽이다. 이를 실제 확장 성공률처럼
  보고하면 안 된다.
- 현재 교정 코드에서는 매 반복 candidate와 validation을 함께 독립 분할한다. 이
  번들은 구 validation 방향을 재사용한 결과일 가능성을 배제할 protocol marker가 없다.

## 통계와 반전 분석

`STATS.md`에는 15개 섹션이 있지만 `v2-s0-math500-math500`은 헤더만 있고 네
estimator 행이 모두 누락됐다. 나머지 bootstrap interval도 top-k 크기 25--51에서
대체로 넓다.

`P(<=x|rand)`는 null CDF이지 우측 단측 p-value 자체가 아니다. chance보다 우수함을
검정하려면 inclusive 경계를 명확히 한 `P(X>=x)`를 계산하고, run과 estimator를 반복
검정한 데 대한 다중비교 보정을 적용해야 한다. 현재 값이 1에 가깝다는 이유만으로
그 숫자를 p-value처럼 인용하면 방향을 잘못 읽게 된다.

`REVERSAL.md`의 pooled 결과는 다음과 같다.

| estimator | 전체 반전 | 경계 대역 반전 |
|---|---:|---:|
| g00 | 636/1454 (44%) | 109/307 (36%) |
| g10 | 645/1454 (44%) | 116/307 (38%) |
| g01 | 616/1454 (42%) | 106/307 (35%) |
| g11 | 610/1454 (42%) | 102/307 (33%) |
| oracle 자기 불일치 | 756/1678 (45%) | 89/350 (25%) |

전체 반전율이 42--44%로 높지만 oracle 자기 불일치도 45%다. 현재 데이터에서는 높은
반전율을 off-policy correction 축에만 귀속할 수 없고 oracle 측 측정 잡음이 주된
교란이다. `g11`의 pooled 반전율도 one-sided estimator보다 0--2%p 낮을 뿐이다.

`g10`의 불일치 경보는 gate-7b-math500 (p=0.0121), v2-s0 (p=0.0342), v2-s1
(p=0.0146), v2-s2 (p=0.00691)에서만 유의하고 다른 run에서는 재현되지 않는다.
여러 run과 estimator를 탐색한 보정 전 p-value이므로 이 네 건도 confirmatory
증거가 아니다. pooled 자료 역시 동일 prompt pool의 seed 반복을 포함할 수 있어
독립 표본으로 합친 p-value를 만들 수 없다.

DAPO run은 더 심각하다. oracle 무신호가 v2-s1에서 487/512 (95.1%), v2-s2에서
467/512 (91.2%)이고 estimator 반전의 유효 분모가 각각 2와 4뿐이다. 이 조건의
precision, 반전율, 경보 통계는 비교 자료로 사용하지 않는다. 이는 기본 모델의 높은
정답률만으로 설명할 수도 있고 전부 오답 또는 다른 score 퇴화일 수도 있다. 이 파일만으로
원인을 구분할 수 없고, 확인 가능한 사실은 대부분의 oracle gradient score가 0이라는
것뿐이다. pass-rate 분포와 raw reward를 함께 진단해야 한다.

## 산출물 품질 문제

1. `READOUT.md`가 비어 있어 최종 수확 단계가 실패했거나 stdout을 기록하지 못했다.
2. `STATS.md` 한 run이 빈 섹션이다. 빈 입력을 성공으로 처리한 것으로 보인다.
3. `-math500-math500`, `-dapo-math-dapo-math`처럼 dataset suffix가 중복된 run 이름이
   남아 있다. 같은 run의 alias 또는 잘못된 경로 탐색으로 중복 집계될 위험이 있다.
4. 생성 commit, dirty diff hash, 모델 revision, generation kwargs, prompt coverage,
   exact K, EOS/`resp_end` 검증 정보가 결과 문서에 없다.
5. frontier와 table의 숫자는 서로 다른 split을 사용하므로 직접 비교할 수 있지만,
   문서만 보면 같은 precision의 재계산처럼 오해할 수 있다. 각 표에 truth split과
   policy split ID를 명시해야 한다.

## 재생성 기준

현재 실행 중인 구버전 GPU 작업은 중단할 필요가 없다. 다만 완료 후 아래 순서로
별도 corrected 산출물을 만들고, 이 번들 위에 덮어쓰지 않는다.

1. generation manifest와 rollout의 `resp_end`, prompt coverage, prompt별 exact K를
   검증한다. 생성 계약을 통과하지 못하면 새 `OUT_ROOT`에서 generation부터 다시 한다.
2. corrected commit으로 `python3 src/rescore_completed_run.py RUN_DIR`를 실행해 scalar
   score와 oracle을 다시 만든다. 구 score는 재사용하지 않는다.
3. `score_protocol.json`과 `oracle_protocol.json`의 독립 validation split, source
   commit, artifact hash를 확인한다.
4. floor, gate, K-curve, frontier, reversal, stats, readout을 모두 재생성한다.
5. nonempty readout, 모든 run의 정확히 4개 stats 행, 고유 canonical run ID, finite
   likelihood diagnostics를 자동 검사한다. 낮은 ESS와 큰 음수 KL 추정치는 hard warning으로
   남긴다.
6. seed별 결과를 먼저 보고한 뒤 사전 정의된 집계만 계산한다. corrected 결과가 나온
   후에 이 문서의 주장별 판정을 새 로그에서 갱신한다.
