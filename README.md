# offpolicy-misranking

**Half-corrected importance ratios can reverse RLVR data influence.** This repo
contains the pilot/gate experiments for that claim: when stale (behavior-policy)
rollouts are reused to rank training prompts for critic-free RLVR, correcting
only the prefix occupancy *or* only the continuation outcome can flip the
per-prompt gradient direction — at arbitrarily small policy KL — and therefore
flip top-k data selection. We also test **CertaGrad**, a sequential procedure
that certifies the top-k decision by spending fresh on-policy rollouts only near
the ranking boundary.

> 상태: 연구 진행 중 (게이트 실험 전). 결과·주장은 실측 전이며 언제든 바뀔 수 있다.

## 1. 문제와 주장

RLVR 데이터 선택은 비용 때문에 예전 정책 `β`의 rollout을 현재 정책 `π`의 데이터
가치 추정에 재사용한다. 프롬프트 `x`의 현재-정책 gradient는

```
g_π(x) = Σ_t E_{H_t, A_t ~ π} [ Q_t^π(H_t, A_t) · ∇log π(A_t|H_t) ]
```

인데, β의 궤적으로 이걸 추정하려면 두 분포를 모두 복원해야 한다 —
**prefix occupancy** (그 토큰까지 π가 어떻게 오는가)와 **continuation outcome**
(그 토큰 뒤를 π가 어떻게 마무리하는가). 토큰 비율 `r_j = π(A_j|H_j)/β(A_j|H_j)`,
`P_t = Π_{j<t} r_j`, `S_t = Π_{j>t} r_j`로 쓰면 네 추정량이 2×2 사각형을 이룬다:

| | continuation = β | continuation = π |
|---|---|---|
| **occupancy = β** | `g00 = Σ E[r_t R z_t]` (CROPI류 token-ratio) | `g01 = Σ E[r_t S_t R z_t]` (suffix만 교정, NFPO류) |
| **occupancy = π** | `g10 = Σ E[P_t r_t R z_t]` (prefix만 교정, CTPO/MinPRO류) | `g11 = Σ E[P_t r_t S_t R z_t] = g_π` (full IS) |

`g10`과 `g01`은 서로 대체재가 아니라 **다른 편향**을 없앤다. 두 토큰짜리 반례에서
one-sided 추정량은 trajectory KL이 `O(ε²)`로 0에 가면서도 gradient 부호가 정확히
반대가 되며, binary group normalization(GRPO식)으로도 구제되지 않는다
(개념 문서의 `verify_theory.py`가 표준 라이브러리만으로 열거 검산).

실무적 증상은 CROPI(2510.26491) 부록에 이미 있다: 점별 gradient cosine은 0.6+인데
**top-10% 이웃 보존율은 28.8%**. 평균 방향이 아니라 실제 채택되는 top-k 결정이
estimand여야 한다는 것이 이 레포의 관점이다.

## 2. CertaGrad — 순위 경계만 현재 정책으로 확인

새 importance ratio가 아니라 배분 절차다: 모든 후보에 작은 fresh micro-group을
주고, projected gradient의 confidence ball로 score 구간 `[L_i, U_i]`를 만들어
`min_{i∈S} L_i > max_{j∉S} U_j`가 될 때까지 **경계에 걸린 후보에만** fresh를
추가한다. 공유 validation 방향의 오차 `α_v`는 모든 후보에 함께 더한다(독립 취급
금지). 인증에 실패하면 실패했다고 보고한다 — 저렴한 점수로 고른 결과와 현재
정책 기준으로 확인된 결과를 구분하는 것이 목적이다.

기하: `||μ̂-μ|| ≤ r`, `r < ||μ̂||`이면 방향 오차는 `α = arcsin(r/||μ̂||)`,
score 구간은 `[cos(φ+α_i+α_v), cos(φ-α_i-α_v)]`.

## 3. 실험 설계 (게이트)

- 모델: Qwen2.5-**7B**-Instruct 기본 (`MODEL`로 변경). β = base, π = 정답 rollout
  LoRA RFT 50/100/200 step (drift 축).
- 데이터: GSM8K 256 train + 50 val (`--dataset math500 | dapo-math` 지원).
  binary verifiable reward (`####` 추출·수치 동등).
- 판정: oracle(π fresh rollout) top-10% 대비 각 추정량의 precision/Jaccard,
  split-half noise floor 병기. **처치축**은 hybrid 2×2 — prefix 절단
  25/50/75%에서 β/π-prefix × β/π-continuation cell을 직접 생성.
- downstream: 선택 소스별(oracle/g10/g01/random) 200-step GRPO-lite
  (LOO-baseline REINFORCE) 학습 후 val 정확도.

통과/사망 조건은 `new-paper-ideas` 컨셉 문서 10절에 고정되어 있다 (예: one-sided
추정량 각각이 noise floor 대비 −0.15, CertaGrad fresh ≤50%·precision 차 ≤0.02).

## 4. 코드 구성

```
src/data.py             데이터 로딩·정답 추출·binary reward
src/rollout.py          rollout 수집(β), drift 생성(LoRA RFT), 정책 로드
src/grads.py            2×2 토큰 가중치, LOO advantage, JL 투영 prompt gradient
src/certagrad.py        confidence-ball 순차 top-k 인증 + uniform baseline
src/hybrid.py           2×2 hybrid rollout 생성·채점 (처치축)
src/train_downstream.py GRPO-lite 학습·greedy 평가
src/experiment.py       stage orchestrator (아래 실행 절차)
tests/test_core.py      모델 없는 핵심 로직 테스트 (2×2 항등식·CertaGrad 동작)
```

구현 노트:
- prompt gradient는 `(1/K) Σ_j Σ_t w_{j,t} ∇log π`를 estimator별 가중치로 backward
  한 뒤 고정 시드 가우시안 JL 투영(float32, 기본 4096차원). gradient 대상은
  마지막 `--grad-layers`개 decoder block + final norm (모든 방법 동일 범위).
- 가중치는 log-공간 누적 후 `clip_cap`으로 양측 클리핑.
- CertaGrad 반경 기본값은 χ² 근사(`--radius-mode gaussian`) — coverage는 게이트의
  반복실험 항목으로 실측하고, 보수적 Hoeffding 판본(`hoeffding`)을 비교 보고.

## 5. 실행

```bash
pip install -r requirements.txt
python3 tests/test_core.py                 # 모델 없이 로직 검증

bash scripts/run_smoke.sh                  # 0.5B, 8 프롬프트 — 파이프라인 완주 확인
bash scripts/run_gate.sh outputs/pilot 100 # 단일 drift 게이트 (report까지)

# 4×H100 병렬 일괄 (drift 3수준 + hybrid + downstream):
export HF_ENDPOINT=<HF mirror>             # 폐쇄망이면 필수
export OUT_ROOT=<산출 경로>
bash scripts/run_h100_all.sh
```

`run_h100_all.sh`의 병렬 배치: phase 0에서 β rollout을 GPU0에서 1회 수집(공유),
phase 1에서 drift 50/100/200 파이프라인(drift SFT→oracle→score→report→hybrid×3)을
GPU 0/1/2에 병렬, phase 2에서 downstream 4소스를 GPU 0~3에 병렬. 스테이지별
로그는 `$OUT_ROOT/logs/`에 분리 저장되고 `main.log`가 타임스탬프 타임라인이다.
실패한 잡은 해당 로그 tail이 main.log에 남는다. drift 수준은 `DRIFTS="50 100"`처럼
환경변수로 바꿀 수 있다.

stage를 개별 실행할 수도 있다 (`--stage prep|rollout-behavior|drift|oracle|score|report|hybrid|downstream`).
모든 stage는 산출물이 있으면 스킵하는 재개 방식이다. 결과는
`outputs/<run>/report.md`(추정량 표·CertaGrad 비교), `scores_*.json`,
`downstream_*.json`.

## 6. 직접 선행 (전체 목록은 컨셉 문서)

CROPI [2510.26491](https://arxiv.org/abs/2510.26491) ·
GradAlign [2602.21492](https://arxiv.org/abs/2602.21492) ·
MinPRO [2601.22718](https://arxiv.org/abs/2601.22718) ·
CTPO [2605.07331](https://arxiv.org/abs/2605.07331) ·
NFPO [2605.20865](https://arxiv.org/abs/2605.20865) ·
TIC-GRPO [2508.02833](https://arxiv.org/abs/2508.02833) ·
M2PO [2510.01161](https://arxiv.org/abs/2510.01161) ·
VIP [2606.05606](https://arxiv.org/abs/2606.05606)

이 레포는 위 방법들의 재구현이 목적이 아니라, one-sided 교정이 남기는 순위 오류의
실측과 top-k 결정 인증이 목적이다.
