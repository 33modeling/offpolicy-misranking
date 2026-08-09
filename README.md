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
src/data.py             데이터 로딩(GSM8K·MATH-500·DAPO-Math)·정답 추출·binary reward
                        — provision이 받아둔 로컬 jsonl($OM_DATA) 우선 (오프라인 노드)
src/rollout.py          rollout 수집(β, 원자적 .tmp→rename 쓰기), drift 생성(LoRA RFT
                        + gradient checkpointing), 정책 로드(디바이스 로그)
src/grads.py            2×2 토큰 가중치, LOO advantage, 위치해시 CountSketch 투영
src/certagrad.py        confidence-ball 순차 top-k 인증 + uniform baseline
src/hybrid.py           2×2 hybrid rollout — left-padding 배치 이어쓰기 생성·채점
src/train_downstream.py GRPO-lite 학습(checkpointing)·greedy 평가
src/experiment.py       stage orchestrator — analyze가 oracle→score→report→hybrid를
                        단일 프로세스로 묶어 7B 재로드 제거
src/judge.py            게이트 5조건(C1·C1'·C2·C3) 자동 PASS/FAIL 판정
src/show_selection.py   방법별 top-k 선택 내용·β정답률·겹침 행렬
tests/test_core.py      모델 없는 핵심 로직 테스트 (2×2 항등식·CertaGrad 동작)
tests/test_judge.py     drift 집계·hybrid 축별 회복 판정 회귀 테스트
```

구현 노트:
- **투영**: 전역 원소 위치의 정수 해시(splitmix형) 기반 CountSketch — 밀집 JL의
  `chunk×dim` 행렬(수십 GB, OOM 원인)을 제거. 같은 grad→같은 투영, 청크 크기
  불변, cosine·norm 보존을 테스트로 고정 (실측: cosine 0.9191→0.9198).
- prompt gradient는 `(1/K) Σ_j Σ_t w_{j,t} ∇log π`를 estimator별 가중치로 backward.
  대상은 마지막 `--grad-layers`개 decoder block + final norm — `grad_params()`가
  나머지를 동결해 LoRA `merge_and_unload()` 후 requires_grad 소실도 함께 복구.
- 가중치는 log-공간 누적 후 `clip_cap`으로 양측 클리핑.
- CertaGrad 반경 기본값은 χ² 근사(`--radius-mode gaussian`) — coverage는 게이트의
  반복실험 항목으로 실측하고, 보수적 Hoeffding 판본(`hoeffding`)을 비교 보고.
- 7B 학습(단계 drift·downstream)은 gradient checkpointing + 시퀀스 1280 상한.

## 5. 클러스터 셋업 (1회)

무거운 것(venv·모델·데이터·산출물)은 전부 group-volume, 체크아웃에는 코드만.
group-volume이 없는 머신은 레포 옆 `.work/`로 자동 폴백된다.

```bash
source scripts/setup_env.sh     # 경로·오프라인 HF·캐시 (셸마다 source)
bash scripts/provision.sh       # venv(torch 2.7.1+cu126 constraints 고정)
                                # + 모델 고정 revision 스냅샷(0.5B·7B, hf-mirror 폴백)
                                # + GSM8K 로컬 jsonl + 로직 테스트까지 멱등 실행
```

- 미러 수동 설정 불필요 — provision에 hf-mirror.com 폴백 내장, pip/curl/HF 전부
  타임아웃이 있어 폐쇄망에서 무한 대기 대신 명확한 에러로 실패한다
  (`PIP_INDEX_URL=<사내 미러>`로 우회).
- 컴퓨트 노드는 기본 오프라인(HF_HUB_OFFLINE=1); 다운로드 머신에서만
  `OM_ONLINE=1 source scripts/setup_env.sh`.

## 6. 실행 — 일상 명령 4개

```bash
bash scripts/go.sh          # 리셋→백그라운드 실행→로그 tail (원샷)
bash scripts/go.sh fast     # 빠른 판정: drift 100만 + fresh 절반 + downstream 절반 (~1/4 시간)
bash scripts/status.sh      # 진행 위치·ETA 자동계산·산출물 체크리스트·학습 건전성(loss/보상 추세)
bash scripts/result.sh      # report 원문 + judge.py 게이트 5조건 자동 PASS/FAIL
bash scripts/selection.sh   # 방법별로 실제 뽑힌 문제·난이도·겹침 행렬
```

보조: `reset_run.sh [--hard|--dry-run]`(프로세스 종료+GPU 확인+산출물 정리 —
soft는 투영 산출물만 지우고 rollout·adapter 보존, 미완성 fresh 파일은 프롬프트
수 대조로 자동 삭제), `run_smoke.sh`(0.5B 완주 확인), `run_gate.sh RUN DRIFT`
(단일 파이프라인 수동 실행).

### run_h100_all.sh 병렬 배치 (4×H100)

- **phase 0**: β rollout을 **4-GPU 샤딩**(`--shard i:4`, 전역 prompt_idx 유지)으로
  수집 후 병합 — 최장 직렬 구간 ~1/4
- **phase 1**: drift 50/100/200 파이프라인(drift SFT → **analyze**)을 GPU 0/1/2에
  병렬. analyze는 oracle·score·report·hybrid(절단 25/50/75%)를 한 프로세스에서
  실행해 모델 로드를 파이프라인당 7회→2회로 줄였다. 동시에 GPU 3에서
  downstream-random(점수 불필요)을 선실행
- **phase 2**: 남은 downstream(oracle/g10/g01)을 GPU 병렬, 완료분 스킵
- **keepalive 상주**: 클러스터의 "GPU 유휴 3시간 → 잡 킬" 정책 대응 — 모든 GPU에
  소형 커널을 연속 발사해 사용률이 상시 36%+ 로 찍힘 (실측; 실연산 미미, 본
  작업 커널에 자연 양보). `scripts/gpu_keepalive.py`
- 손잡이(환경변수): `DRIFTS="50 100"`, `DOWNSTREAM_DRIFT=100`, `FRESH_K=16`, `VAL_K`, `HYBRID_PROMPTS`,
  `DOWNSTREAM_STEPS`, `MODEL`, `OUT_ROOT`, `OM_SKIP_GPU_CHECK=1`
- 시작 전 **GPU 점유 검사**(2GB+ 잡 발견 시 PID 출력 후 중단 — 좀비 위 재시작 OOM 방지)

### 로그

- `logs/main.log` — 타임라인: 스테이지 시작/완료(소요초)/실패(tail 자동 첨부) +
  **10분 하트비트**(GPU 사용률·각 파이프라인 마지막 줄)
- 스테이지별 로그 — 장기 루프(rollout·score·oracle·downstream)는 매/5건마다
  `[HH:MM:SS] rollout 34/256 (13%, 38s/개, 정답 5/8, ETA 2h20m)` 형식으로
  진행률·ETA 내장. HF 경고·프로그레스 바는 억제
- 모든 stage는 산출물 존재 시 스킵(재개); rollout 파일은 원자적 쓰기라 중단
  잔재(.tmp)가 완성본으로 오인되지 않는다

산출물: `report.md/json`(추정량 표·CertaGrad), `scores_*.json`,
`scores_hybrid_*.json`, `downstream_*.json`, `oracle_micro_groups.pt` 등 —
읽는 법은 `result.sh`가 대신한다.

## 7. 트러블슈팅

구축 중 잡은 주요 에러(투영 OOM, 유휴 킬, 다중 감시자 폭풍, 재개 오염 등)와
교훈은 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)에 정본으로 기록했다.

## 8. 직접 선행 (전체 목록은 컨셉 문서)

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
