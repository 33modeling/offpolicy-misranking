# offpolicy-misranking

**Stale Rollouts Can Pick the Wrong Prompts** (ICLR 2027 target) — 실행 레포.
stale(behavior-policy) rollout을 재사용해 RLVR 학습 프롬프트를 top-k 선택할 때,
prefix occupancy *또는* continuation outcome 중 **한쪽만 교정하면** 임의로 작은
policy KL에서도 per-prompt gradient 부호가 뒤집힐 수 있고, 그 오류는 KL·ESS·
cosine 대시보드로 탐지되지 않으며, 신뢰구간으로 top-k 결정을 인증하는 것도
구조적으로 불가능하다 — 이 주장과 그 경계 조건(언제 stale로 골라도 되는가,
fresh 감사에 얼마를 내야 하는가)의 게이트/본실험 코드.

> **2026-08-20 감사 상태:** 8월 18일 수치와 아래 v1/v2 결과는 생성 계약 및
> 독립 validation 교정 이전의 역사적 탐색 결과다. 제출용 근거로 사용하지 않는다.
> 교정 내용, 결과 계보, 재실행 조건은
> [`docs/FULL_AUDIT_2026-08-20.md`](docs/FULL_AUDIT_2026-08-20.md)에 있다.
> confirmatory 실행은 `go_v2.sh`/`run_14b.sh`만 사용하며 legacy
> `run_h100_all.sh`/`babysit.sh`는 기본 비활성화했다.
> 교정 전 코드로 완주한 run은 공유 checkout을 갱신한 뒤 GPU 환경에서
> `python3 src/rescore_completed_run.py RUN_DIR`로 score와 oracle/report를 함께 재생성한다.

> 역사적 상태 (2026-08-11): **7B 게이트 종결** — C1(관찰)·C1′(개입)·C3(downstream)
> PASS, C2(인증)는 구조적 FAIL → 부정 결과로 수록. 14B GSM8K는 포화 퇴화,
> MATH-500 2종은 관찰 순위 역전 확인. 당시 **감사 P0 교정판 v2 본실행**이
> 진행 중으로 기록됐으나, 2026-08-20 감사 기준으로 그 결과도 계약 확인과
> 후처리 재생성이 필요하다. 원고는 별도 비공개 레포
> [offpolicy-misranking-paper](https://github.com/33modeling/offpolicy-misranking-paper)에서
> 관리하며, 이 레포에는 실행 코드와 검증 기록만 둔다.

## 1. 문제와 주장

RLVR 데이터 선택은 비용 때문에 예전 정책 `β`의 rollout을 현재 정책 `π`의 데이터
가치 추정에 재사용한다. 프롬프트 `x`의 현재-정책 gradient는

```
g_π(x) = Σ_t E_{H_t, A_t ~ π} [ Q_t^π(H_t, A_t) · ∇log π(A_t|H_t) ]
```

인데, β의 궤적으로 이걸 추정하려면 두 분포를 모두 복원해야 한다 —
**prefix occupancy** (그 토큰까지 π가 어떻게 오는가)와 **continuation outcome**
(그 토큰 뒤를 π가 어떻게 마무리하는가). 토큰 비율 `r_j = π(A_j|H_j)/β(A_j|H_j)`,
`P_t = Π_{j<t} r_j`, `S_t = Π_{j>t} r_j`로 쓰면 importance correction이 두 인자로
인수분해되고, 각 인자를 켜고 끄는 **2×2 추정량 가족(estimator family)**이 나온다.
기존 방법들이 네 칸을 하나씩 점유한다:

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

## 2. 역사적 게이트 결과 요약 (v1, 단일 seed; 제출 증거 아님)

- **C1 관찰 (7B GSM8K)**: one-sided가 chance 아래로 붕괴 — `g10` precision
  0.000(drift 50)·0.040(drift 400) vs chance 0.098, 정규화 retention은 음수
  (−0.26 등) = **역선택**. 무보정 `g00`·full `g11`은 살아남는다.
- **C1′ 개입 (hybrid rollout)**: β 궤적을 절단점(25/50/75%)에서 자르고 나머지
  반쪽을 π가 이어 쓰게 해 네 칸을 **데이터로 직접 생성**. 잘린 반쪽을 복원하면
  순위가 회복 — 21/21 비교에서 비열등, 평균 회복 +0.46/+0.49. 반쪽 구조
  자체가 원인이라는 인과 증거.
- **C3 downstream**: 잘못 뽑힌 명단으로 GRPO-lite 학습 시 val 정확도 손실
  (drift 100에서 0.84/0.84/0.88/0.82 = oracle/g10/g01/random, base 0.74).
- **C2 인증: 구조적 FAIL** — 3절 참조. 부정 결과로 논문에 수록.
- **체제 의존**: MATH-500에서는 관찰 순위가 **역전** — 무보정 `g00`이 최악.
  어떤 추정량이 최악인지는 태스크에 따라 뒤집히고, 불변인 것은 C1′의 개입
  결과와 "혼합 셀이 순수 셀보다 낮다"(mixed-cell dip)이다.
- **14B GSM8K 포화 퇴화**: 정답률 ~95%에서 oracle split-half floor 0.080 <
  chance 0.098 — 고를 것 자체가 없다. 선택 파이프라인은 모델 성장에 맞춰
  후보 난이도를 올려야 한다는 실무 교훈.

## 3. top-k 인증 — 시도와 구조적 실패 (C2)

stale 점수를 prior로 두고, top-k **경계에 걸린 후보에만** fresh rollout을 부어
confidence 구간 `[L_i, U_i]`가 `min_{i∈S} L_i > max_{j∉S} U_j`로 분리될 때까지
반복하는 순차 인증 절차를 구현·시험했다(`src/certagrad.py` — 코드명이며 논문
본문에서는 서술형으로만 표기). 기하:
`||μ̂-μ|| ≤ r`, `r < ||μ̂||`이면 방향 오차 `α = arcsin(r/||μ̂||)`,
score 구간 `[cos(φ+α_i+α_v), cos(φ-α_i-α_v)]`.

결과는 **일관된 불가능**: 경계 margin이 0.00–0.24°인데 도달 가능한 validation
방향 반경 `α_v`는 51–83°(14B/MATH-500에서는 180°로 퇴화) — 두세 자릿수 격차.
drift 8배·선택 비율 5–25%·val 심화·(ε,δ)-PAC 완화 전부에서 유지된다. 경계
근처 프롬프트 가치는 본질적으로 밀집해 있어 **선택이 중요한 그 지점에서 선택을
인증할 수 없다.** 후속 CPU 재판정: `scripts/c2_sweep.sh`, `fix_c2.sh`,
`retry_c2.sh`.

## 4. confirmatory 신규 실행 경로 (2026-08-20 계약)

아래 명령은 2026-08-20 감사 branch를 병합한 뒤 새 `OUT_ROOT`에서 실행한다.
기존 v2 폴더를 이 계약으로 완료된 결과라고 간주하면 안 된다. raw-softmax sampling
(`temperature=1.0`, `top_p=1.0`, `top_k=0`) 통일, seed
관통, hybrid 4셀 **독립 equal-K 재설계**, seeded tie-break, divergence/clipfrac
통계 자동 실측, 인증 ε 실반영, run manifest 기록.

```bash
bash scripts/go_v2.sh    # GPU 건강검사 → 30분 스모크 게이트(실패 시 본실행 미진입)
                         # → 3-seed × {gsm8k, dapo-math}, n=512·val 100·fresh K=32
                         #   ·hybrid 64 (drift 100 단일, 기본 7B)
                         # 재시도 2회·DONE 스킵·죽어도 재개, 끝나면 results/v2 수집
```

- sweep 축: **seed × dataset** (v2의 목적 = 주결과 오차대 + DAPO-Math 확보).
  drift 스윕·14B·MATH-500은 v1 결과를 인용한다. 필요 시
  `DATASETS="gsm8k dapo-math math500" SEEDS="0 1 2"`로 확장 — DONE 스킵 덕에
  완주 후 재실행하면 추가분만 돈다.
- 산출: run별 `report.json`·`manifest.json`·judge 판정 +
  `results/v2/TABLES.md`(표 생성기)·`FRONTIER.md`(아래 5절).
- `score_protocol.json`과 `oracle_protocol.json`이 모두 없는 run은 모든 판정·표 생성기가
  거부한다. 두 마커를 수동으로 만들지 말고 교정 코드로 해당 단계를 실행한다.

## 5. fresh-audit frontier — 비용–품질 사후 분석

v2 산출물만으로 selection 정책 스펙트럼을 CPU에서 replay한다 (GPU 불필요,
go_v2 말미 자동 실행, 수동은 `bash scripts/frontier.sh`):

- **정책**: stale 4셀 · pass-rate 난이도(Beta posterior) · random ·
  fresh(m∈{1,2,4}) 전수 재채점 · **audit** — stale 순위 유지 + p∈{1,5,10,25}%
  만 fresh로 교체 (uniform random vs top-k 경계 근접) · sequential 변형
  (margin 획득 + 추정량 셀 불일치 + ranker switch; 코드명 `2dref`, ablation 2종)
- **누수 차단**: oracle micro-group을 짝/홀 반으로 분리 — 짝수 = 정책 관측,
  홀수 = truth 전용. fresh-heavy 정책이 자기 채점하는 것을 방지.
- **출력** (`results/v2/FRONTIER.md` + `frontier.json`): F1 run별 정책×예산
  precision/regret · F2 dataset 집계(seed 평균±sd) · F3 predictor-family vs
  gradient-family · F4 조건 지표(KL̂·ESS·clipfrac·live·floor·margin — "언제
  stale로 골라도 되는가" 위상도 재료)

## 6. 코드 구성

```
src/data.py             데이터 로딩(GSM8K·MATH-500·DAPO-Math·mbpp·kk)·정답 추출·binary reward
                        — provision이 받아둔 로컬 jsonl($OM_DATA) 우선 (오프라인 노드)
src/rollout.py          rollout 수집(β, 원자적 .tmp→rename 쓰기), drift 생성(LoRA RFT
                        + gradient checkpointing), 정책 로드(디바이스 로그)
src/grads.py            2×2 토큰 가중치, LOO advantage, 위치해시 CountSketch 투영
src/certagrad.py        confidence-ball 순차 top-k 인증 + uniform baseline (C2, 부정 결과)
src/hybrid.py           2×2 hybrid rollout — left-padding 배치 이어쓰기 생성·채점
src/frontier.py         fresh-audit 비용–품질 replay (5절) — 정책 스펙트럼·짝/홀
                        누수 차단·F1~F4 생성
src/train_downstream.py GRPO-lite 학습(checkpointing)·greedy 평가
src/experiment.py       stage orchestrator — analyze가 oracle→score→report→hybrid를
                        단일 프로세스로 묶어 7B 재로드 제거
src/select_rules.py     top-k 크기·seeded tie-break·독립 tie overlap 정본
src/score_artifacts.py  oracle·4 estimator·split-half의 schema·finite 값·ID coverage 검증
src/judge.py            게이트 5조건(C1·C1'·C2·C3) 자동 PASS/FAIL 판정
src/make_tables.py      논문용 표 생성 → results/TABLES.md
src/show_selection.py   방법별 top-k 선택 내용·β정답률·겹침 행렬
tests/test_core.py      모델 없는 핵심 로직 테스트 (2×2 항등식·인증 동작)
tests/test_protocol.py  tie·oracle split·hybrid horizon·artifact metadata 회귀 테스트
tests/test_judge.py     drift 집계·hybrid 축별 회복 판정 회귀 테스트
tests/test_readout_summary.py  corrected run 탐색·부분 artifact 거부 회귀 테스트
tests/test_harvest.py   원자적 수확·stderr 보존·부분 산출물 격리 shell 회귀 테스트
```

구현 노트:
- **투영**: 전역 원소 위치의 정수 해시(splitmix형) 기반 CountSketch — 밀집 JL의
  `chunk×dim` 행렬(수십 GB, OOM 원인)을 제거. 같은 grad→같은 투영, 청크 크기
  불변, cosine·norm 보존을 테스트로 고정 (실측: cosine 0.9191→0.9198).
- prompt gradient는 `(1/K) Σ_j Σ_t w_{j,t} ∇log π`를 estimator별 가중치로 backward.
  대상은 마지막 `--grad-layers`개 decoder block + final norm — `grad_params()`가
  나머지를 동결해 LoRA `merge_and_unload()` 후 requires_grad 소실도 함께 복구.
- 가중치는 log-공간 누적 후 `clip_cap`으로 양측 클리핑.
- 인증 반경 기본값은 χ² 근사(`--radius-mode gaussian`), 보수적 Hoeffding 판본
  (`hoeffding`) 비교 보고.
- 7B 학습(단계 drift·downstream)은 gradient checkpointing + 시퀀스 1280 상한.

## 7. 클러스터 셋업 (1회)

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
- 노드별 GPU 진단: `scripts/gpu_check.sh` (matmul/SDPA 분리 판정 — fused SDPA
  커널이 병든 노드는 `OM_ATTN=eager`로 우회).

## 8. 실행

**현재 주 진입점은 4절의 `go_v2.sh`.** 보조 명령:

```bash
bash scripts/status.sh      # 진행 위치·ETA·산출물 체크리스트·학습 건전성
bash scripts/result.sh      # report 원문 + judge.py 게이트 자동 PASS/FAIL
bash scripts/selection.sh   # 방법별로 실제 뽑힌 문제·난이도·겹침 행렬
bash scripts/tables.sh      # 논문용 결과 테이블 → results/TABLES.md
bash scripts/frontier.sh    # frontier replay 수동 실행 (5절)
```

v1 게이트용 스크립트(`go.sh`, `run_h100_all.sh` 4×H100 병렬 배치, `go7_14.sh`
7B+14B 원샷, `run_14b.sh` 단일 run — go_v2가 내부에서 재사용)와 리셋·스모크
(`reset_run.sh`, `run_smoke.sh`)는 그대로 유지. 클러스터의 "GPU 유휴 3시간 →
잡 킬" 정책은 `scripts/gpu_keepalive.py` 상주로 대응.

### 로그

- `logs/main.log` — 타임라인: 스테이지 시작/완료(소요초)/실패(tail 자동 첨부) +
  **10분 하트비트**(GPU 사용률·각 파이프라인 마지막 줄)
- 스테이지별 로그 — 장기 루프(rollout·score·oracle·downstream)는 매/5건마다
  `[HH:MM:SS] rollout 34/256 (13%, 38s/개, 정답 5/8, ETA 2h20m)` 형식으로
  진행률·ETA 내장. HF 경고·프로그레스 바는 억제
- 모든 stage는 산출물 존재 시 스킵(재개); rollout 파일은 원자적 쓰기라 중단
  잔재(.tmp)가 완성본으로 오인되지 않는다

## 9. 트러블슈팅

구축 중 잡은 주요 에러(투영 OOM, 유휴 킬, 다중 감시자 폭풍, 재개 오염, cuDNN/
SDPA 커널 계열 등)와 교훈은
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)에 정본으로 기록했다.

## 10. 직접 선행 (전체 목록은 컨셉 문서)

CROPI [2510.26491](https://arxiv.org/abs/2510.26491) ·
GradAlign [2602.21492](https://arxiv.org/abs/2602.21492) ·
MinPRO [2601.22718](https://arxiv.org/abs/2601.22718) ·
CTPO [2605.07331](https://arxiv.org/abs/2605.07331) ·
NFPO [2605.20865](https://arxiv.org/abs/2605.20865) ·
TIC-GRPO [2508.02833](https://arxiv.org/abs/2508.02833) ·
M2PO [2510.01161](https://arxiv.org/abs/2510.01161) ·
VIP [2606.05606](https://arxiv.org/abs/2606.05606)

이 레포는 위 방법들의 재구현이 목적이 아니라, one-sided 교정이 남기는 순위
오류의 실측, 그 인증 불가능성의 입증, 그리고 fresh 감사의 비용–품질 경계 측정이
목적이다.
