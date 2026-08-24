# ARCHITECTURE.md — 설계 구조

> 이 문서는 **구조와 이유**를 다룬다. 모듈별 한 줄 설명은
> [`CODE.md`](CODE.md), 장애 사례는 [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md),
> 실행·복구 경로는 [`USAGE.md`](USAGE.md), 감사 판정은
> [`FULL_AUDIT_2026-08-20.md`](FULL_AUDIT_2026-08-20.md)·
> [`PAPER_REVIEW_2026-08-19.md`](PAPER_REVIEW_2026-08-19.md), 실행 절차는
> [`USAGE.md`](USAGE.md)가 정본이다. 중복 서술은 요약하고 링크한다.
>
> 작성 기준 커밋: `2244e89` (2026-08-24). 아래 수치·상수·경로는 전부 그
> 커밋의 `src/`·`scripts/`를 직접 읽어 확인한 값이다. 문서와 구현이 어긋나는
> 곳은 12절에 따로 모았다.

---

## 1. 규모와 구성

| 구분 | 수 | 비고 |
|---|---:|---|
| 추적 파일 | 144 | `git ls-files` |
| `src/*.py` | 33개 · 7,356줄 | 최대 `experiment.py` 871줄, `data.py` 554줄, `frontier.py` 531줄 |
| `scripts/` | 70개 (`.sh` 66 · `.py` 4) | 최대 `run_14b.sh` 21,704 B, `verify_theory.py` 20,218 B |
| `tests/*.py` | 29개 · 3,576줄 | pytest 회귀 테스트 + 직접 실행하는 스크립트형 테스트 |
| `docs/*.md` | 9개(이 문서 포함) | 별도로 `docs/results/2026-08-24/` 아래 결과 번들 11개 |

계층은 네 개다.

```
셸 오케스트레이터        go_v4.sh · go_v4_27b.sh · resume_v4.sh · go_v2.sh
        │                (배정·재개·감시·재시도·집계)
        ▼
단일 run 실행기          run_14b.sh
        │                (config lock · GPU 배정 · 샤딩 · 병합 · DONE)
        ▼
스테이지 오케스트레이터   src/experiment.py --stage <이름>
        │                (산출물 단위 스킵 · 원자적 저장 · 계약 검증)
        ▼
계산 커널                rollout.py · grads.py · hybrid.py · certagrad.py · data.py
```

판정·분석(`judge.py`, `make_tables.py`, `frontier.py`, `kcurve_floor.py`,
`reversal_freq.py`, `stats_extra.py`, `readout_summary.py`)은 이 네 계층과
분리된 **CPU 전용 재집계 계층**으로, GPU 산출물을 읽기만 한다.

---

## 2. 이 실험이 무엇을 재는가

### 2.1 estimand — 2×2 추정량 가족

토큰 비율 `r_j = π(A_j|H_j)/β(A_j|H_j)`, prefix `P_t = Π_{j<t} r_j`,
suffix `S_t = Π_{j>t} r_j`로 두면 off-policy gradient 보정은 두 인자로
분해된다. `src/grads.py:36` `log_weights()`가 네 칸을 **log 공간에서** 구현한다.

| 칸 | log 가중치 구현 | 의미 |
|---|---|---|
| `g00` | `log_r` | 무보정 stale (token-ratio) |
| `g10` | `cumsum(log_r)` | prefix occupancy만 보정 |
| `g01` | `log_r.sum() - cumsum(log_r) + log_r` | continuation outcome만 보정 |
| `g11` | `log_r.sum()` 브로드캐스트 | full trajectory IS |

`token_weights()`(`grads.py:59`)가 이 log 가중치를 `[-log(cap), +log(cap)]`로
clamp한 뒤 `exp`를 취하고 LOO advantage를 곱한다. 기본 `clip_cap=10.0`이므로
**실험값은 전부 clipped variant**다. 이 사실은 `grads.py` 모듈 docstring과
`token_weights` docstring에 명시돼 있고, PAPER_REVIEW P0-3의 지적 그대로다.
unclipped population 식과 같은 기호를 쓰되 구현이 clipped임을 표기하는 것이
현재 규약이다.

advantage는 전부 leave-one-out 비정규화 값
`A_j = R_j - mean_{l≠j} R_l` (`loo_advantages()`, K=1이면 0)이다.

### 2.2 프롬프트 점수

`prompt_gradient()`(`grads.py:156`)가
`ĝ = (1/K) Σ_j Σ_t w_{j,t} ∇log π(a_t|h_t)`를 계산한다. 대상 파라미터는
`grad_params()`가 고르는 **마지막 `--grad-layers`개 decoder block + final
norm**이며, 나머지는 `requires_grad_(False)`로 동결한다. lm_head/embedding은
제외한다. 요구 조건은 "모든 estimator와 oracle이 같은 파라미터 목록을 쓴다"
하나뿐이다.

투영은 고정 시드 **CountSketch**(`project_grads()`)다. 전역 원소 위치의
splitmix형 정수 해시로 좌표당 버킷 인덱스와 부호만 만들어
`scatter_add_` 한다. 상수는 곱수 `6364136223846793005`,
가산 `1442695040888963407`, 이후 `x ^= x>>33`, 곱수 `-7046029254386353131`,
`x ^= x>>29`, 인덱스 `(x & 0x7FFFFFFF) % dim`, 부호는 비트 31이다.
`ProjectionSpec`의 기본값은 `dim=4096`, `seed=20260807`, `chunk=8_000_000`.
해시가 **전역 위치**에만 의존하므로 청크 크기나 호출 순서가 결과를 바꾸지
않는다. 밀집 JL 행렬(`chunk × dim`)이 131GB급 OOM을 냈던 것이 교체 이유다
(TROUBLESHOOTING A1).

프롬프트 점수는 `cosine(ĝ, v)` 하나의 스칼라다. `v`가 무엇인지가 2026-08-20
감사의 핵심 쟁점이었다(2.4절).

### 2.3 판정량

- **top-k precision** — oracle top-k와 estimator top-k의 겹침 비율.
  `k = min(n, max(1, int(n·frac)))` (`select_rules.topk_count`, `frac=0.10`).
  n=512이면 k=51, n=400이면 k=40이다. `int()`/`round()` 혼재로 k가 25와 26으로
  갈리던 P0-5 결함을 이 함수 하나로 통일했다.
- **floor (split-half reliability)** — oracle micro-group을 짝/홀로 나눠 만든
  두 반쪽 점수의 top-k 겹침. `report.json`에는 `noise_floor`와
  `split_half_reliability` 두 키로 같은 값이 들어가고, 보고서 첫 줄에
  "참조치이며 상한 아님"을 박아 둔다.
- **chance** — `k/n`. 무작위 선택의 기대 precision.
- **닻(anchor)** — oracle 자기 부호 불일치율. `reversal_freq.py`가
  `scores_splithalf.json`의 a·b 반부호 비율로 계산한다. 원시 반전율은 이 값
  이하로 내려갈 이유가 없으므로, **닻과의 차이만** estimator 결함으로 읽는다.

### 2.4 동점(tie) 처리 — 왜 20개 독립 스트림인가

DAPO처럼 대부분의 프롬프트 점수가 0인 체제에서는 top-k 경계가 동점으로
가득 찬다. 이때 인덱스 순 절단은 임의의 우열을 만들고, 양쪽에 **같은** jitter를
쓰면 겹침이 인위적으로 부풀려진다.

`select_rules.py`가 규약을 하나로 고정한다.

- `jittered_topk(scores, k, seed)` — seed 고정 난수를 2차 키로 쓰는 결정적 draw.
- `overlap_under_independent_ties(left, right, k, seed=0, pairs=20)` —
  좌우에 **다른 스트림**을 주고(오프셋 `_RIGHT_STREAM_OFFSET = 104_729`,
  쌍 간격 `_PAIR_STRIDE = 7_919`) 20쌍의 평균·최소·최대·표준편차를 낸다.
- 동점 점수가 전부 같으면 기대 precision은 1이 아니라 chance다.

`report.json`은 이 요약을 `split_half_jitter_range`,
`split_half_jitter_sd`, estimator별 `precision_jitter_range`,
`precision_jitter_sd`로 함께 저장한다.

### 2.5 게이트 — 무엇이 통과·실패인가

`src/gate_rules.py`가 술어(predicate)의 정본이고 `judge.py`는 그것을 출력만
한다. 상수는 `ONE_SIDED_DROP = 0.15`, `CAUSAL_CUT = "0.5"`.

| 게이트 | 판정 규칙 (코드 기준) |
|---|---|
| **C1** | 동일 run에서 `g10`·`g01`의 precision이 모두 `floor − 0.15` 이하 |
| **C1′** | C1을 만족한 **같은 run**의 **사전고정 cut 0.5**에서 `pp > pb` **와** `pp > bp` 동시 성립 |
| **C2** | `certified=True` **와** fresh ≤ 0.5 × uniform **와** precision ≥ uniform precision − 0.02 |
| **C3** | `downstream_oracle.val_acc ≥ downstream_random.val_acc − 0.02` |

`_complete_verdict()`는 필수 결과가 다 있을 때만 PASS를 주고, 하나라도
False면 즉시 FAIL로 내린다. 산출물이 모자라면 `None`(미판정)이다.

**C1과 C1′를 서로 다른 run이나 cut에서 합치지 못하게 한 것**이 2026-08-20
감사의 High 항목 수정이다. `evaluate_causal_run()`이 한 run 안에서
`joint_failure`와 `eligible(cut == "0.5")`와 `joint_recovery`를 모두 만족하는
결과만 `witnesses`로 인정한다.

---

## 3. 파이프라인 — 단계별 책임

`run_14b.sh`가 호출하는 순서는 다음과 같다. NGPU는 자동 감지값
(`OM_GPUS` 지정 시 그 목록), NM은 `NGPU ≥ 2`이면 `NGPU-1`, 아니면 1이다.

| # | stage | 병렬 | 입력 | 출력 | 스킵 조건 |
|---|---|---|---|---|---|
| 0 | `prep` | 1 | dataset | `prompts.json` | 파일 존재 + 내용 일치 (불일치면 abort) |
| 1 | `rollout-behavior` | NGPU 샤드 | `prompts.json` | `rollouts_behavior_train[.shardI].jsonl` + `.manifest.json` | 병합본 존재 또는 샤드 존재 |
| 2 | `drift` | 1 | behavior rollout | `drift_<steps>/` LoRA adapter | `adapter_config.json` 존재 |
| 3 | `rollout-fresh` | NGPU 샤드 | `prompts.json`, adapter | `rollouts_fresh_train[.shardI].jsonl`, shard 0은 `rollouts_fresh_val.jsonl`도 | 산출물 **각각** 독립 검사 |
| 4 | `val-grads` | 1 (마지막 GPU) | `rollouts_fresh_val.jsonl` | `val_groups.pt`, `val_gradient.pt` | 두 파일 **모두** 존재 |
| 5 | `oracle-grads` | NM 샤드 | fresh train rollout | `oracle_micro_groups.shardI.pt` | 샤드 파일 존재 |
| 6 | `score-shard` | NGPU 샤드 | behavior rollout, `val_groups.pt` | `scores_offpolicy.shardI.json`, `divergence_stats.shardI.json`, `score_protocol.shardI.json` | 세 파일 **모두** 존재 + 프로토콜 스키마 일치 |
| 7 | `merge-grads` | 1 | 샤드 전부 | `oracle_micro_groups.pt`, `scores_offpolicy.json`, `score_protocol.json`, `scores_oracle.json`, `scores_splithalf.json`, `oracle_protocol.json` | 점수 2종 + oracle 프로토콜 존재 |
| 8 | `report` | 1 | 위 전부 | `report.md`, `report.json` | 없음(항상 재계산) |
| 9 | `hybrid` ×3 cut | 라운드로빈 | behavior rollout, β·π | `rollouts_hybrid_<cut>.jsonl`(+manifest), `scores_hybrid_<cut>.json`, `hybrid_protocol_<cut>.json` | 점수 + 프로토콜 존재 (`OM_SKIP_HYBRID=1`이면 전체 생략) |

마지막에 필수 산출물 12종을 다시 확인하고 `DONE.tmp` → `DONE` rename으로
완료를 표시한다.

### 3.1 rollout (β·π 생성)

`collect_rollouts()`(`rollout.py:161`)가 프롬프트마다 K개 응답을 생성해
`{prompt_idx, rollout_idx, input_ids, resp_start, resp_end, reward}` 행을
쓴다. logp는 저장하지 않는다 — score 단계에서 재계산한다. 생성 배치는
`OM_GEN_BATCH`(미설정이면 K 전체 한 배치)로 쪼갠다. 27B처럼 가중치가 GPU를
거의 채우는 모델에서 KV 캐시·프리필 활성값 OOM을 막기 위한 장치다.

모델 로드(`load_model()`)는 **device_map을 쓰지 않고** CPU 로드 후
`.to("cuda")` 2단계다. 신아키텍처가 meta-init을 못 타면 device_map 경로가
GPU에 스켈레톤과 체크포인트를 이중 상주시켜(27B에서 약 52+49GB) OOM이
나기 때문이다. dtype은 로드 후 `model.to(want)`로 한 번 더 통일해
일부 모듈만 fp32로 남는 혼합 로드를 막는다. `AutoModelForCausalLM`이
`ValueError`/`KeyError`로 실패하면 `AutoModelForMultimodalLM`으로 폴백한다
(Qwen3.6/3.8 계열은 멀티모달 automap이라 CausalLM 매핑에 없다).

cuDNN SDPA 비활성화(`torch.backends.cuda.enable_cudnn_sdp(False)`)는
`load_model` 안이 아니라 **`rollout.py` 모듈 import 시점**에 있다. drift 학습과
downstream이 `load_model`을 거치지 않던 시절 설정이 누락돼 같은 버그가
재발했기 때문이다(TROUBLESHOOTING C5).

### 3.2 drift (β → π)

`train_drift_lora()`가 정답 rollout만으로 LoRA SFT(rejection FT)를 돌린다.
LoRA `r=16, α=32, dropout=0`, 대상 모듈은 기본 `["q_proj", "v_proj"]`,
`OM_LORA_TARGETS=all-linear`이면 전 linear. optimizer는 AdamW `lr=1e-4`,
`batch_size=4` 누적, gradient checkpointing(`use_reentrant=False`) +
`enable_input_require_grads()`.

학습 예제는 `response_only_training_example()`이 만든다. `[0, resp_start)`는
`-100`으로 마스킹하고, `resp_end` 이후 padding은 잘라낸다. `max_length=1280`
안에 응답 시작이 들어오지 않으면 **예외를 던진다** — prompt 토큰만 학습하는
사고를 조용히 넘기지 않는다.

`select_drift_training_rows()`는 정답 rollout이 하나도 없으면
`ValueError`를 던진다. 예전에는 전체 rollout으로 폴백했는데, 오답만으로
SFT하면 π의 정의 자체가 달라지므로 2026-08-21 재감사에서 제거했다.

**drift는 재개 시 재학습하지 않는다.** LoRA 초기화가 랜덤이라 재학습하면
π가 매번 다른 정책이 되고, 이전에 계산된 점수와 새 oracle이 다른 π를
기준으로 삼게 된다. 다시 학습하려면 adapter 폴더를 지워야 한다
(TROUBLESHOOTING B6).

### 3.3 score (2×2 채점)

`stage_score()`는 프롬프트마다 π·β 양쪽 로그확률을 구해 네 estimator의
토큰 가중치를 만들고, estimator마다 `prompt_gradient`를 한 번씩 backward한다.

대형 모델에서는 **2-pass**로 돈다. β 로그확률을 전부 먼저 계산해 두고 β를
언로드한 뒤 π를 올린다. 두 모델 동시 상주가 14B에서 attention OOM의 원인이었다.
모델이 주입된 경우(`analyze` 스테이지)에는 기존 동시 경로를 유지한다.

같은 루프에서 진단 통계를 모은다: 토큰 KL̂(β‖π), 궤적 ESS 비율,
estimator별 clip 비율, `traj_logw_logsumexp`, `traj_logw2_logsumexp`.
마지막 두 값은 frontier가 샤드를 **가중·log-sum으로** 합치기 위한 것이다
(샤드별 평균의 단순 평균은 틀린 집계다).

### 3.4 oracle (π fresh 채점)과 floor

`stage_oracle()`은 프롬프트별 fresh rollout을 `micro_group` 크기로 잘라
micro-group마다 LOO advantage 가중 gradient를 만든다. 요구 조건은 두 가지다.

- `len(rows) % micro_group == 0`
- micro-group 수가 **2 이상의 짝수** (split-half가 성립해야 한다)

기본값 `fresh_k=32`, `micro_group=4`이면 프롬프트당 8 micro-group이다.

`score_oracle_microgroups()`가 세 값을 만든다.

```
oracle score = cos(stack.mean(0),        evaluation_val)   # 전체 평균 × 평가용 방향
half a       = cos(stack[0::2].mean(0),  val_half_a)       # 짝수 그룹 × 선택용 방향
half b       = cos(stack[1::2].mean(0),  val_half_b)       # 홀수 그룹 × 평가용 방향
```

**핵심은 두 반쪽이 서로 다른 validation 방향을 쓴다는 것**이다. 예전에는 두
반쪽이 같은 `val_gradient`를 공유했고, 그 결과 floor는 "독립적인 oracle
reliability"가 아니라 "주어진 validation 벡터에 조건부인 rollout split
agreement"였다(PAPER_REVIEW P0-6). 공유 오차 때문에 겹침이 부풀 수 있어
독립 additive-noise ceiling 모델을 그대로 적용할 수 없었다.

### 3.5 selection/evaluation validation 분리

`split_validation_directions(val_groups)`(`experiment.py:62`)가 한 줄로 정의한다.

```python
return val_groups[0::2].mean(dim=0), val_groups[1::2].mean(dim=0)
#      selection (stale 점수용)        evaluation (oracle 정답용)
```

- `stage_score`는 `selection_val`로만 채점한다.
- oracle 전체 점수는 `evaluation_val`로만 채점한다.
- hybrid도 `selection_val`을 쓴다.
- `report`의 CertaGrad 평가는 pool을 `micro[i][0::2]`(선택용) /
  `micro[i][1::2]`(진실용)로, validation을 `val_groups[0::2]` /
  `val_groups[1::2]`로 각각 나눈다.
- `frontier.py`도 같은 규약을 쓰되 한 단계 더 나눈다: 관측용 절반을 다시
  `obs_val_a`/`obs_val_b`로, 진실용 절반을 다시 `val_a`/`val_b`로 나눠
  truth reliability까지 표본을 공유하지 않게 한다.

`val_k=8`, `n_val=100`이면 `val_groups`는 (100, 4096)이고 선택 50 / 평가 50으로
갈린다.

### 3.6 hybrid (C1′ 개입)

`hybrid.py`가 네 셀을 **데이터로 직접 생성**한다. 셀별 K는 `k_cell`(기본 8)로
전부 같고, 모두 oracle fresh 표본과 독립이다.

| 셀 | 생성 방법 |
|---|---|
| `bb` | 기존 β rollout에서 앞 `k_cell`개 |
| `pp` | π가 프롬프트부터 새로 생성 (oracle fresh 재사용 금지) |
| `bp` | β rollout을 `cut_frac`에서 자르고 π가 이어 씀 |
| `pb` | **새로 만든 `pp`의** π-prefix를 자르고 β가 이어 씀 |

`_cut_prefixes()`는 `max_prefix_tokens = resp_len - 1`을 상한으로 둔다. 원본의
마지막 토큰이 EOS일 수 있고(한 토큰짜리 응답은 EOS만인 경우가 흔하다) 그
토큰을 보존한 채 이어 쓰면 즉시 종료되기 때문이다.

`continue_rollouts_batch()`는 보존한 prefix 길이를 원래 horizon에서 차감해
**셀 간 총 생성 토큰 예산을 맞춘다**. 같은 잔여 예산끼리 묶어 left-padding
배치로 생성한다.

`validate_hybrid_cells()`가 네 셀 집합·프롬프트 커버리지·셀별 정확 K·셀 라벨을
전부 검사하고 하나라도 어긋나면 예외를 던진다.

프롬프트 subset은 호출측(`experiment.run_hybrid`)이 고른다. β 보상이 섞인
live 프롬프트에서 `Random(seed*104_729 + 11)`로 뽑으므로 **cut 세 개가 같은
subset**을 쓴다. live가 `hybrid_prompts`보다 적으면 전체 풀에서 뽑는다.

**hybrid는 π와 β를 한 GPU에 동시 상주시키는 유일한 스테이지**다. 27B급
(약 57GB × 2 > 80GB)은 구조적으로 불가능해서 v4의 27B는 `OM_SKIP_HYBRID=1`로
생략한다. 그래서 27B에는 C1′ 판정이 없다.

### 3.7 downstream (C3)

`train_downstream.grpo_lite_train()`이 선택된 프롬프트로 LoRA +
LOO-baseline REINFORCE(clip 없는 GRPO-lite)를 돌리고 greedy val 정확도를 잰다.
`selection_rollout_cost()`가 선택 방식별 rollout 예산을 집계한다.

**v4 파이프라인(`run_14b.sh`)은 downstream 스테이지를 호출하지 않는다.**
`--stage downstream`은 `run_gate.sh`(v1), `run_h100_all.sh`(비활성),
`go_full.sh`에만 있다. 따라서 v4 run에는 `downstream_*.json`이 없고
judge의 C3는 항상 "미판정"으로 나온다. 이는 결함이 아니라 v4 범위 밖이라는
뜻이다.

---

## 4. 데이터·산출물 흐름과 경로 규약

### 4.1 저장 위치

무거운 것은 전부 `$OM_WORK`(group-volume), 체크아웃에는 코드만 둔다.
`scripts/setup_env.sh`가 경로를 정한다.

```
$OM_WORK/                                    # group-volume/<user>/offpolicy-misranking
  .venv-cu126/                               # $VENV_DIR — torch 2.7.1+cu126
  cache/{huggingface,pip,pycache}/           # $HF_HOME, $PIP_CACHE_DIR
  tmp/                                       # $TMPDIR
  data/                                      # $OM_DATA — provision이 받은 로컬 jsonl
  models/                                    # group-volume/models 없을 때만
  pools/<ds>-hard-<model>-<confighash>-ps<seed>.jsonl   # prescreen 산출 hard pool
  runs/<이름>/                               # run 디렉터리 (아래 4.2)
  results/<세대>/                            # TABLES.md · FRONTIER.md · report-*.json · judge-*.txt
  results/v4/V4_COMPLETE                     # 20-run 집계 완료 표식
  readouts/<타임스탬프>-<종류>.XXXXXX/       # harvest·read_now의 원자적 publish 폴더
  progress/<MMDD-HHMM>-cluster<라벨>/        # progress_snapshot 산출
  quarantine/v4/, quarantine/v4-27b-rerun/   # 계약 불일치 run 보존 이동
  code-snapshots/offpolicy-misranking-<12자리>/  # 재개용 detached worktree
  locks/, runs/v4-finalize.lock              # flock 큐·집계 단일화
  console-logs/                              # 워커 콘솔 로그 (레포에 로그 금지)
```

group-volume이 없는 머신에서는 `OM_WORK`가 `$OM_REPO/.work`로 자동 폴백한다.
로컬에서도 같은 코드가 도는 것이 목적이며, `.work/`는 `.gitignore`에 있다.

### 4.2 run 디렉터리

이름 규약은 `v4-<모델>-s<seed>[-<dataset>]`이다 (gsm8k는 접미사 없음).
`run_14b.sh`는 데이터셋 접미사를 **멱등하게** 붙인다 — 호출자가 이미
데이터셋명이 든 경로를 넘기면 그대로 쓴다. 무조건 덧붙이면
`v2-s0-dapo-math-dapo-math` 같은 이중 접미사 폴더에 산출물이 쌓이고
호출자의 DONE 검사가 완주를 영영 인식하지 못해 매 루프 전체가 재실행된다.
`tables.sh`·`frontier.sh`에는 아직도 이중 접미사 run을 걸러내는
`_legacy_dup()`가 남아 있다.

| 파일 | 의미 |
|---|---|
| `prompts.json` | train/val 분할. 샤드 병합 검증의 기준 |
| `run_config.json` | **변경 불가** 실행 설정 + digest. 다르면 재사용 거부 |
| `manifest.json` | run_config + torch/transformers/cuda 버전 + 시각 |
| `rollouts_*.jsonl` (+`.manifest.json`) | rollout 원본과 생성 설정 스냅샷 |
| `rollouts_*.partial` | 진행 중 내구 저장분 (병합·청소 대상 아님) |
| `drift_<steps>/` | π LoRA adapter |
| `val_groups.pt` / `val_gradient.pt` | validation prompt gradient 스택 / 평균 |
| `oracle_micro_groups.pt` | 프롬프트별 micro-group gradient 스택 (K-curve 재조합 재료) |
| `scores_oracle.json` / `scores_splithalf.json` / `scores_offpolicy.json` | 점수 정본 |
| `divergence_stats[.shardI].json` | KL̂·ESS·clip 비율 |
| `score_protocol.json` / `oracle_protocol.json` | **프로토콜 마커** (없으면 전 판정기가 거부) |
| `scores_hybrid_<cut>.json` / `hybrid_protocol_<cut>.json` | hybrid 점수와 마커 |
| `report.md` / `report.json` | 게이트 수치 정본 |
| `postprocess_manifest.json` | 재점수화 경로가 남기는 입력/출력 해시 |
| `DONE` | 완료 표식 (`completed <ISO시각>`) |
| `stale-*/`, `stale-shards-*/` | 격리된 옛 산출물 (삭제 아님) |
| `FAILURE_DIAGNOSTIC.txt` | 실패 시 자동 진단 |
| `logs/` | 스테이지별 로그 + `main.log` |
| `keepalive.pid` | GPU keepalive 프로세스 PID |

### 4.3 프로토콜 마커 (schema 문자열)

| 상수 | 값 | 위치 |
|---|---|---|
| `SCORE_PROTOCOL_SCHEMA` | `offpolicy-score-validation-split/v1` | `experiment.py`, `gate_rules.py` |
| `ORACLE_PROTOCOL_SCHEMA` | `offpolicy-oracle-validation-split/v1` | 〃 |
| `HYBRID_PROTOCOL_SCHEMA` | `offpolicy-hybrid-validation-split/v2` | 〃 |
| `HYBRID_ROLLOUT_SCHEMA` | `offpolicy-hybrid-rollouts/v2` | `hybrid.py` |
| 후처리 manifest | `offpolicy-corrected-postprocess/v1` | `recompute_oracle_scores.py` |

`_has_protocol()`은 스키마 문자열이 맞고 **`generation_validation.validated_rows > 0`**
일 때만 통과시킨다. `gate_rules.has_valid_*_protocol()`은 여기에 더해 기록된
`manifest_sha256`·`artifact_sha256`이 **현재 파일 해시와 일치**하는지까지
확인한다. 즉 채점 후 rollout이 바뀌면 그 run은 자동으로 판정 대상에서 빠진다.

### 4.4 데이터셋 탐색

`load_prompts()`의 순서는 다음과 같다.

1. `OM_POOL_FILE`이 있으면 그 jsonl이 풀을 통째로 대체한다. dataset 이름은
   reward 분기용으로만 남는다 (27B hard-slice 경로).
2. gsm8k·math500은 `$OM_DATA/<파일>`, `$DATASETS_DIR/<dataset>/<파일>` 순으로
   로컬 사본을 찾는다. **판독까지 성공해야 채택**한다 — 손상·빈 사본이 정상
   사본을 가리는 것을 막기 위해서다.
3. 그 뒤 `_candidate_roots()`가 `<DS>_DIR` 환경변수 → `_dataset_bases()`
   (= `$DATASETS_DIR`, `$GROUP_VOLUME/datasets`, `$GROUP_VOLUME/<user>/datasets`,
   `$OM_WORK/data`) 아래 이름·fuzzy 후보를 훑는다.
4. `_load_rows_any()`가 jsonl → parquet → json 트리 → `load_from_disk` 순으로
   전부 수용한다.
5. 마지막으로 HF 허브. **실패하면 찾아본 위치 전체를 나열하는 에러**를 낸다
   (조용한 폴백은 에러를 원인에서 먼 곳에 찍는다 — TROUBLESHOOTING E3).

`_split()`은 `Random(seed=0)`로 셔플한 뒤 앞에서 train, 그다음 val을 잘라낸다.
**프롬프트 분할은 실행 seed와 무관하게 고정**이다(`load_prompts`의 `seed`
인자는 기본 0으로만 호출된다). 실행 seed는 생성 샘플링·LoRA 초기화·tie-break에만
관여한다.

reward는 `gold` 문자열의 접두사로 분기한다: `assert`로 시작하면 mbpp
실행 채점, `APPS:`면 stdin/stdout 채점, `KK:`면 knight/knave 전원 매치,
`ARC:`면 ARC-Challenge option label exact match,
그 외는 `####`/`\boxed{}` 수치 매칭이다. 코드 채점은 **bubblewrap 샌드박스**
(`--unshare-all --die-with-parent --new-session --clearenv`, `/usr`·`/lib`만
읽기 전용 바인드, `/tmp`는 tmpfs, 출력 1 MiB 상한, 기본 8초 타임아웃)에서
돌고, `bwrap`이 없으면 **실행하지 않고 `None`을 반환**한다(fail-closed).

---

## 5. v3 → v4에서 바뀐 것

v3는 생성 계약(P0-1·P0-2) 수정 커밋 `c6ca013` 직후 7B로 돌린 세대이고,
v4는 2026-08-20 전수 감사의 **독립 validation 교정까지** 들어간 confirmatory
세대다. 산출물 폴더를 재사용하지 않고 새 `RUN_BASE`에서 시작한다.

| 축 | v3 | v4 |
|---|---|---|
| 메인 모델 | Qwen2.5-7B-Instruct | **Qwen3.8-27B-BF16** (7B는 동일조건 재현 축으로 강등) |
| 행렬 | seed × {gsm8k, dapo-math, math500, mbpp, kk} | **2 모델 × 5 seed × 2 데이터셋 = 20 run** |
| dapo-math | 포함 | **제외** — 27B가 후보를 전부 맞혀 live prompt가 0. 포화 진단은 되지만 가설 검정이 성립하지 않는다 |
| validation 분리 | 없음 (selection·evaluation 공유) | `val_groups[0::2]` / `[1::2]` 분리 + 프로토콜 마커 fail-closed |
| oracle split-half | 두 반쪽이 같은 validation 방향 | 반쪽마다 다른 validation 방향 |
| K-curve / frontier | candidate만 재분할 | candidate·validation 매 반복 독립 분할, 예산은 rollout 단위 |
| 실행 배정 | 단일 진입점 | 3-cluster slot 배정 또는 flock 공유 큐 |
| 재개 | 경로 재사용 | run별 **generation commit** 격리 worktree 재개 |
| 27B 커널 | 없음 | FLA 0.5.2 fused recurrent/chunk 강제, fallback이면 시작 거부 |
| hybrid | 전부 실행 | 27B는 `OM_SKIP_HYBRID=1` (π+β 동시 상주 불가) |
| 감시 | 로그 무변화 = 사망 | 로그·GPU·CPU가 **모두** 멈춰야 재시작 |
| 워커 종료 | child exit 신뢰 | 필수 artifact 재검사 (false-success 차단) |
| legacy runner | 사용 가능 | `run_h100_all.sh`·`babysit.sh` 기본 비활성 |

v4 고정 설정(`validate_v4_27b.FIXED_CONFIG` 기준):
`behavior_k=8, fresh_k=32, val_k=8, micro_group=4, hybrid_prompts=64, k_cell=8,
drift=100, max_new_tokens=512, proj_dim=4096, grad_layers=4, clip_cap=10.0,
temperature=1.0, topk_frac=0.10, radius_mode=gaussian, top_p=1.0, thinking=off,
attn=eager, n_val=100`, `n_train`은 gsm8k 512 / math500 400.

---

## 6. rollout contract — P0-1·P0-2

`src/rollout_contract.py`가 생성과 소비를 한 파일에서 정의한다. 이 계약이
없으면 뒤의 모든 수치가 무의미하다.

### 6.1 P0-1 — 샘플링 분포와 ratio 분포의 불일치

HF `generate()`는 **미지정 인자를 모델 `generation_config`에서 병합**한다.
Qwen2.5-Instruct 배포 기본값은 `top_k=20, repetition_penalty=1.05, top_p=0.8,
temperature=0.7`이다. 예전 코드는 `temperature`와 `top_p`만 덮었으므로 실제
표본은 top-k 절단 + repetition penalty가 적용된 분포에서 나왔는데,
`sequence_logprobs()`는 processor를 거치지 않은 **raw softmax** teacher-forced
로그확률을 쓴다. 즉 rollout을 만든 β/π와 ratio를 계산한 β/π가 서로 달랐고,
그 상태의 `g11`은 full IS도 fresh on-policy oracle도 아니었다.

수정은 `gen_kwargs()` 하나다.

```python
dict(do_sample=True, temperature=…, top_p=…,
     top_k=0, repetition_penalty=1.0, no_repeat_ngram_size=0,
     max_new_tokens=…, pad_token_id=…)
```

`top_k=0`·`repetition_penalty=1.0`·`no_repeat_ngram_size=0`은 "기본값이라 생략"이
아니라 **generation_config 병합 차단용 명시**다. 지우면 P0-1이 재발한다.
`resolved_manifest()`가 명시 인자와 모델 `generation_config` 원본을 함께
sidecar에 남겨, 사후에 무엇이 실제로 적용됐는지 확인할 수 있게 한다.

`experiment.main()`은 `--temperature != 1.0`이거나 `OM_TOP_P != 1.0`이면
아예 시작을 거부하고, `artifact_contract.validate_generation_contract()`는
manifest의 `explicit_kwargs`가 기대값과 다르면 채점을 거부한다.

### 6.2 P0-2 — 종료 뒤 EOS padding

배치 생성은 일찍 끝난 행을 배치 최대 길이까지 pad(=eos)로 채운다. 예전에는
`seq.tolist()`를 그대로 저장했고, gradient·로그확률·token KL·hybrid 절단·
drift SFT·downstream이 전부 그 padding 구간까지 계산했다.

수정은 **생성 시점 절단**이다.

- `eos_ids_of(model, tok, pad_id)` — tokenizer eos + `generation_config.eos_token_id`
  + `config.eos_token_id`(리스트 가능, Qwen2.5는 `[im_end, endoftext]`) + pad를
  모두 모은다.
- `resp_end_index(ids, resp_start, eos_ids)` — `resp_start` 이후 첫 EOS를
  **포함한** 위치 + 1. 없으면 `len(seq)`.
- 응답 구간은 `[resp_start, resp_end)`이고 저장 시 `input_ids`를 `resp_end`에서
  자른다. 따라서 신형 산출물은 항상 `resp_end == len(input_ids)`다.
- `trim_row()`는 구버전 행에 `OM_EOS_IDS="151645,151643"`처럼 EOS id를 주면
  재유도해 절단하고, 미지정이면 원본 그대로 두되 경고를 찍는다.

`validate_generation_contract()`가 모든 primary rollout 행에 대해
`resp_end` 존재, `resp_end == len(input_ids)`, `0 ≤ resp_start < resp_end`를
검사한다. 하나라도 어긋나면 예외다.

### 6.3 계약 검증기가 실제로 보는 것

`artifact_contract.validate_generation_contract(run, source_names)`는
torch 없이 도는 순수 파일 검증기다. 검사 항목:

1. manifest 존재 — 병합본 sidecar 또는 샤드 sidecar들.
2. `artifact_file` 이름 일치, 바인딩된 jsonl 존재, `artifact_sha256` 일치.
3. manifest의 `k`가 config의 `behavior_k`/`fresh_k`/`val_k`와 일치.
4. 샤드 manifest들의 `idx_offset..idx_offset+n_prompts` 범위가 **겹치지 않고**
   합집합이 정확히 `range(n_prompts)`.
5. `explicit_kwargs`가 기대 생성 설정과 일치, `eos_token_ids` 비어 있지 않음,
   `model_name_or_path`의 basename이 config의 모델과 일치.
6. 병합본의 행 집합이 샤드가 바인딩한 행 집합과 **키·내용까지 동일**.
   커버리지만 보면 reward나 토큰이 조작된 행을 못 잡는다.
7. 프롬프트별 rollout_idx 집합이 정확히 `range(K)` — 중복·누락·초과 전부 거부.

반환값(`validated_rows`, `manifest_sha256`, `artifact_sha256`,
`generation_hash_missing`)은 그대로 프로토콜 마커에 박힌다. `stage_report`는
저장된 해시와 현재 해시를 다시 대조해 "채점 후 provenance가 바뀌었다"를
탐지한다.

---

## 7. atomic publish 대 durable progress

두 요구가 충돌하는 것처럼 보이지만 층을 나누면 양립한다. **발행 원자성은
rename 한 번으로 유지하고, 내구성은 부분 파일 + 구제 규칙으로 얻는다.**

### 7.1 원자적 발행

- `_atomic_text()` / `_atomic_save()`(`experiment.py:49,56`) — `.tmp`에 쓰고
  `replace()`. 중단 시 부분 파일이 `exists()` 재개 검사를 통과하는 오염을 막는다.
- `collect_rollouts`의 최종 `part_path.rename(out_path)`.
- `make_hybrid_cells`의 `tmp_path.replace(out_path)`.
- `DONE.tmp` → `DONE`, `manifest.json.tmp` → `manifest.json`,
  `V4_COMPLETE.tmp` → `V4_COMPLETE`.
- 보고서 publish(`harvest.sh`의 `publish_markdown`, `_report_io.sh`의
  `publish_report`) — 종료 코드와 비어 있지 않음을 확인한 뒤에만 최종 이름을 준다.

### 7.2 내구 진행 (`.partial`)

2026-08-21의 C7이 계기다. 27B fresh rollout이 128개 중 46번째에서
CUDA `unspecified launch failure`로 죽었는데, 당시 `collect_rollouts`는 shard
전체를 `.tmp`에 쓰고 마지막에 rename했다. 크래시하면 수 시간 진행분이 통째로
사라지고 재시도가 프롬프트 0부터 다시 돌았다. 간헐 ULF 노드에서 2.5시간짜리
shard는 영원히 완주할 수 없는 구조였다.

현재 흐름은 이렇다.

```
part_path = out_path.with_suffix(".partial")
legacy    = out_path.with_suffix(".tmp")      # 구버전 진행분 승계
if not part_path.exists() and legacy.exists(): legacy.rename(part_path)

done = salvage_partial(part_path, k)          # K개 완주 프롬프트만 남긴다
with part_path.open("a") as f:
    for i, item in enumerate(prompts):
        if idx_offset + i in done: continue
        ... 생성 ...
        f.flush()                             # 프롬프트 단위 내구 지점

n_rows = 행 수
if n_rows != len(prompts) * k: raise RuntimeError(...)   # fail-closed
part_path.rename(out_path)                    # 여기서 처음 최종 이름을 얻는다
manifest 최종본 write → replace
```

`salvage_partial()`(`rollout.py:133`)의 규칙:

- 줄 단위 JSON 파싱 실패(강제 종료로 찢긴 꼬리 줄) → **그 지점부터 전부 폐기**.
- 프롬프트별 행이 정확히 K개인 것만 남긴다. K 미달 프롬프트는 통째로 제거해
  중복 행·부분 행이 최종 산출물에 들어갈 수 없게 한다.
- 남긴 것만 다시 써서 `.partial`을 교체하고, 완주 프롬프트 집합을 반환한다.

`.partial` 확장자를 고른 이유는 merge·stale 청소기의 `*.jsonl` 글롭에 걸리지
않기 위해서다. `progress_snapshot.sh`는 반대로 `.partial` 행 수를 진행률로
읽는다.

manifest는 생성 **시작 시점에 `.tmp`로만** 써 둔다. 이미 유효한 sidecar를
덮어쓰지 않기 위해서다. 최종 manifest는 jsonl이 발행되고 해시가 계산된 뒤에만
`replace()`로 게시된다. 따라서 **manifest 존재는 완료 판정이 아니다** —
BACKLOG 2026-08-21 항목에 같은 취지가 기록돼 있다.

### 7.3 셸 계층의 fail-closed

`run_14b.sh`의 `merge_rollouts()`는 `cat` 대신 커버리지 검증 병합이다.
누락·범위 밖·중복·exact-K 실패를 각각 세어 하나라도 있으면 **정리 명령까지
출력하고 중단**한다. 병합본이 이미 있으면 검증만 하고 종료한다.
GPU 수가 바뀐 재시작에서 샤드 분할이 어긋나 조용히 누락·중복되는 것을 막는
장치다(TROUBLESHOOTING B5 전수 점검, B7).

---

## 8. provenance gate

### 8.1 run 단위 — immutable `run_config.json`

`run_14b.sh`의 manifest 블록(180~264행)이 실행 설정 전부를 모아 정렬 JSON의
SHA-256을 `digest`로 넣는다. 포함 항목은 git HEAD, `git status --porcelain -- src scripts`,
`git diff HEAD -- src scripts`의 SHA-256, 모델 경로와 `config.json`·
`tokenizer_config.json`·`generation_config.json` 해시, dataset과 pool 해시,
n_train/n_val/K들/drift/proj_dim/grad_layers/clip_cap/temperature/topk_frac/
radius_mode/top_p/thinking/attn/gen_batch/lora_targets/skip_hybrid, 그리고
`linear_attention_backend`(`fla-core` 설치 여부)와 `fla_core_version`이다.

- `git_status`가 비어 있지 않으면 **`OM_ALLOW_DIRTY=1`이 아닌 한 exit 2**.
- 같은 경로에 이미 `run_config.json`이 있고 digest가 다르면 달라진 키 목록을
  찍고 exit 2. 이것이 `existing artifacts use a different run config: ['git']`
  메시지의 출처다.
- `verify_code_snapshot()`이 **매 스테이지 전에** HEAD·status·diff 해시를 다시
  대조한다. 실행 중 공유 checkout이 바뀌어 stage마다 다른 코드 버전이 섞이는
  것을 막는다.

### 8.2 20-run 집계 게이트

`go_v4.sh`의 `collect_targets()`가 모델별로 10개 run을 모아 검사한다.

1. 6종 필수 artifact(`DONE`, `run_config.json`, `manifest.json`,
   `score_protocol.json`, `oracle_protocol.json`, `report.json`)가 비어 있지 않을 것.
2. 경로에서 유도한 seed·dataset이 config와 일치, `n_train`이 512(gsm8k)/400(math500),
   `n_val=100`.
3. `model_config_sha256`이 그 모델의 기대 해시와 일치.
4. 다음 27개 키가 10개 run에서 **모두 동일**할 것:
   `git, git_diff_sha256, git_status, model_config_sha256, tokenizer_config_sha256,
   generation_config_sha256, behavior_k, fresh_k, val_k, micro_group, hybrid_prompts,
   k_cell, drift, max_new_tokens, proj_dim, grad_layers, clip_cap, temperature,
   topk_frac, radius_mode, top_p, thinking, attn, lora_targets, skip_hybrid,
   linear_attention_backend, fla_core_version`.
5. `git_status`가 비어 있을 것(dirty tree에서 초기화된 run 거부).

**왜 필요한가.** 20개 run은 하나의 사전등록 행렬이다. 어느 run 하나가 다른
commit·다른 모델 스냅샷·다른 K로 돌았다면, seed 간 차이라고 부른 것이 실은
코드 차이일 수 있다. seed 평균±sd는 "같은 절차를 seed만 바꿔 반복했다"를
전제로만 의미가 있다. 그 전제를 파일 해시로 강제하는 것이 이 게이트다.
`skip_hybrid`·`lora_targets`처럼 27B와 7B가 애초에 다른 값을 쓰는 키가 있으므로
게이트는 **모델별 10 run 단위**로 돈다.

27B 전용 게이트는 더 엄격하다. `validate_v4_27b.py`의 `FIXED_CONFIG`는 값을
**리터럴로 못 박고**, 추가로 `git_status == ""`, `git_diff_sha256 == sha256(b"")`,
그리고 저장된 `digest`가 나머지 필드로 재계산한 값과 일치하는지까지 본다.
`fla_core_version == "0.5.2"`, `linear_attention_backend == "fla"`를 요구하므로
PyTorch recurrent fallback으로 돈 27B run은 자동으로 거부된다.

### 8.3 commit 상속 — `v4_resume_commit.py`

중단된 run을 재개할 때 **어느 코드로 계산을 이어갈지**를 정하는 규칙이다.
`resume_plan(runs_root, slot, current)`가 slot에 배정된 (모델, seed) × 2
데이터셋을 돌면서 판단한다.

```
complete(run) == True                      → skip (DONE + 5종 artifact)
run_config.json 있음                        → 그 파일의 git            (source="recorded run_config")
없음 · 같은 (모델, seed)의 반대 데이터셋 존재 → 그 run의 git            (source=반대 run 이름)
없음 · 같은 모델의 다른 config 존재          → 그 모델 config들의 최빈 commit
없음 · 아무 v4 config든 존재                → 전체 config의 최빈 commit
아무것도 없음                               → 현재 checkout HEAD
```

`shell_environment(config_path)`는 `ENV_KEYS`(22개)를 config에서 읽어
`export`/`unset` 문자열을 만든다. `None`인 키는 `unset`으로 나가므로,
"기록 당시 설정되지 않았던 변수"까지 정확히 복원된다.

`resume_v4.sh`가 이 계획을 받아 run마다 `ensure_snapshot()`으로 detached
worktree(`$OM_WORK/code-snapshots/offpolicy-misranking-<12자리>`)를 확보하고,
없는 commit은 `git fetch --no-tags origin <sha>`로 받아 온다. worktree 생성은
`flock`으로 직렬화한다.

**계산 코드와 감시 코드를 분리한 것이 핵심 설계**다.

```
supervisor  = 현재 checkout의 go_v2.sh          (watchdog·재시도·완료 검사)
compute     = OM_PIPELINE_SCRIPT = <snapshot>/scripts/run_14b.sh
              OM_REPO = PYTHONPATH = <snapshot>/src
```

이렇게 하지 않으면 `git pull` 뒤에도 옛 watchdog 버그가 그대로 재현되거나
(감시까지 옛 코드), immutable run_config의 `git`이 달라 부분 산출물이 전부
버려진다(계산까지 새 코드). 자식 셸에서 `OM_REPO`·`PYTHONPATH`를 먼저 `unset`한
뒤 `setup_env.sh`를 다시 source하는 이유도 같다 — Git은 과거 revision인데
Python import만 최신 revision인 혼합 실행을 차단한다.

pass는 기본 3회(`OM_V4_SUPERVISOR_PASSES`)다. 한 run이 실패해도 나머지를 계속
돌리고, pass가 끝나면 완료 목록을 **다시 계산해** 미완료만 재시도한다.
마지막 pass 뒤에도 미완료가 있으면 run 이름을 남기고 exit 1이다.

### 8.4 격리 이동 — `prepare_run_path.py`

계약이 안 맞는 run을 **삭제하지 않고** `quarantine_root`로 rename한다.
목적지 이름은 `<run이름>-git-<12자리 또는 사유>-<타임스탬프>[-N]`이다.

- 기본: `run_config.json`의 `git`이 기대 commit과 다르면 이동.
- `--quarantine-unconfigured`: `run_config.json`이 없는 비어 있지 않은 디렉터리도 이동.
- `--force-quarantine`: 기록된 git과 무관하게 비어 있지 않으면 이동.

이 덕분에 `['git']` 충돌을 사용자가 파일을 직접 확인하거나 지우지 않아도
자동 해소한다.

---

## 9. 프로세스·GPU 관리

### 9.1 `cleanup_run_processes.py`

전역 `pkill`은 다른 사람의 run이나 무관한 GPU 작업까지 죽인다. 그래서
`/proc`을 직접 훑어 **소유자가 나이고**, 다음 중 하나를 만족하는 프로세스만
대상으로 한다.

- cmdline에 `--run <prefix>`가 있다
- `OUT_ROOT`/`RUN_BASE`/`RUN_BASE_SMOKE` 환경변수가 prefix로 시작한다
- `RUN_LABEL`이 `v4-`로 시작한다
- cmdline에 `scripts/go_v4.sh`가 있다

여기에 **자손을 재귀로 추가**해 launcher가 CUDA 자식을 남기지 못하게 하고,
자기 자신과 조상 체인은 `_protected_ancestors()`로 제외한다. TERM → 최대
`--timeout`(기본 15초) 대기 → 남은 것에 KILL.

### 9.2 GPU 점유·회전

- `run_14b.sh`: 자기가 쓸 GPU만(`OM_GPUS` 지정 시 그 목록) 2000 MiB 초과
  사용 여부를 검사해 점유 중이면 abort (`OM_SKIP_GPU_CHECK=1`로 무시 가능).
- `OM_RETRY_INDEX`로 **shard의 물리 GPU 배정을 회전**시킨다
  (`rotation = (retry_index - 1) % NGPU`). 특정 GPU에서 ULF가 반복될 때 같은
  shard를 같은 GPU에 계속 재투입하지 않기 위해서다.
- `wait_all_stages()`는 모든 샤드가 끝날 때까지 기다린 뒤 실패 여부를 반환한다.
  예전에는 첫 shard가 실패하면 즉시 exit → EXIT trap의 `pkill`이 **정상
  계산 중인 형제 shard에 SIGTERM**을 보내 `rc=143`(128+15)을 만들었다.
- `gpu_keepalive.py`가 GPU마다 듀티 사이클 커널을 돌려 "GPU 유휴 3시간 → 잡 킬"
  정책을 피한다. PID는 `keepalive.pid`에 남기고 EXIT trap에서 정리한다.

### 9.3 watchdog — 무엇이 "멈춤"인가

`go_v2.sh`의 백그라운드 워처가 15초마다 돈다.

1. 활성 run 정보는 `$TMPDIR/go-v2-<label>-<pid>.active` 파일(PID·콘솔 로그·run
   디렉터리 3줄)에서 읽는다. 공유 볼륨의 **다른 클러스터 로그**를 진행 신호로
   오인하지 않기 위해 감시 대상 로그를 이 run으로 한정한다.
2. 가장 최근 로그의 `stat`(이름:mtime:크기)이 바뀌면 정상 진행으로 보고
   CPU 기준선을 갱신한다.
3. `OM_STALL_MINUTES × 4` 틱(= 그 분 수) 동안 변화가 없으면 **GPU 최대
   사용률**(3회 샘플)과 **process group 누적 CPU 시간 증가분**을 확인한다.
   - GPU 사용률 > 0 이거나 CPU가 2초 넘게 늘었으면 "계속 실행"으로 판정하고
     기준선만 갱신한다.
   - 로그·GPU·CPU가 **모두** 정지했을 때만 `kill -TERM -- -<pgid>` →
     5초 후 `kill -KILL`.

임계값은 7B 5분, 27B 20분이다(27B는 4장 동시 스냅샷 로드가 정당하게 5분 넘게
조용할 수 있다). 이 3중 조건이 없으면 정상 장시간 계산을 죽이고, 조건이
전혀 없으면 진짜 hang을 못 잡는다.

`cleanup_strays()`는 활성 process group에 TERM → 최대 10초 대기 → KILL,
그리고 `--run $BASE` 패턴 자식 정리, 30분 넘은 HF datasets lock 삭제까지 한다.

---

## 10. 판정·분석 계층 (CPU 전용)

전부 기존 산출물만 읽어 재집계한다. GPU가 필요 없다.

| 모듈 | 출력 | 하는 일 |
|---|---|---|
| `gate_rules.py` | — | 술어 정본. `canonical_gate_report()`가 **저장된 report를 믿지 않고** 원시 점수에서 floor·precision을 재계산한다 |
| `score_artifacts.py` | — | oracle·4 estimator·split-half의 스키마, finite 값, prompt ID 커버리지 일치를 강제 |
| `run_select.py` | — | 세대(`v\d+-`)·legacy(`gate-`, `drift`)·protocol-only run의 공통 탐색과 **미선택 사유 진단**(`describe_skips`) |
| `judge.py` | `judge-*.txt` | C1·C1′·C2·C3 자동 판정 출력 |
| `readout_summary.py` | `READOUT.md` | 한눈 표 + 자동 결론 + 용어 설명 + 원시 judge 출력. judge를 subprocess로 부르고 exit≠0이면 오류로 기록 |
| `make_tables.py` | `TABLES.md` | T1 게이트요약 · T2 정규화 재판정 · T3 floor-vs-관측 · T4 live fraction · T5 hybrid · T6 C2·margin · T7 downstream. 표마다 독립 생성 — 하나가 깨져도 나머지는 만든다 |
| `frontier.py` | `FRONTIER.md`, `frontier.json` | 정책 스펙트럼 replay (아래 10.1) |
| `kcurve_floor.py` | `KCURVE.md` | 사전등록 K-curve 판정 |
| `kcurve_all.py` | `KCURVE_ALL.md` | 전 세대 확장 K-curve |
| `reversal_freq.py` | `REVERSAL.md` | 부호반전율 + 닻 + 경계 대역 + McNemar + 불일치 경보 Fisher |
| `stats_extra.py` | `STATS.md` | run별 초기하 정확 p, 프롬프트 bootstrap CI, 부호검정 |
| `c2_sweep.py` / `c2_diagnose.py` | — | C2 재판정 스윕·진단 |
| `show_selection.py` | — | 방법별 top-k 선택 내역·겹침 행렬 |

### 10.1 frontier — 비용–품질 replay

정책 스펙트럼: stale 4셀, pass-rate(Beta posterior 난이도), random,
fresh(m ∈ {1,2,4}), audit_random(p), audit_boundary(p), 그리고 sequential 변형.
`AUDIT_FRACS = (0.01, 0.05, 0.10, 0.25)`, `REPEATS = 20`.

누수 차단은 이중이다. micro-group을 짝(정책 관측)/홀(진실)로 나누고,
validation prompt gradient도 짝/홀로 나눈다. 진실 쪽은 다시 절반씩 갈라
truth reliability를 독립 jitter로 잰다. 짝수 절반이 4그룹이므로
`FRESH_MS`의 상한이 4다.

live 판정은 **스무딩 없는 원시 성공 수**(`0 < sum < len`)로 한다. Beta
스무딩한 pass-rate로 live를 판정하면 사실상 항상 live가 되던 것이
PAPER_REVIEW E5의 지적이다. 스무딩 값은 난이도 점수에만 쓴다.

예산 비교는 **rollout 단위 envelope**로 바꿨다. 서로 다른 fresh budget의
최고값을 직접 비교하면 예산이 큰 정책이 자동으로 이긴다.

### 10.2 K-curve

`oracle_micro_groups.pt`를 재조합해 절반 크기 m(그룹 수)마다 split-half floor를
다시 잰다. 매 반복 candidate와 validation을 **함께 독립 분할**한다.
`T_REP = 30`회 재표집, `S_SIM = 40`회 이변량 정규 시뮬, Spearman–Brown
`r_m = m·r1 / (1 + (m-1)·r1)`로 `K_MAX_PRED = 256`까지 외삽하고,
관측 가능한 최대 m에서 예측-관측 보정 오차를 함께 출력한다.

사전등록 판정은 exit code로 나온다: 0 = 확장 권고(과반 run에서 예측 floor ≥
`GO_MULT = 2.0` × chance가 되는 최소 K′ ≤ 256 존재), 3 = 구조적 부재,
4 = 대상 run 0개. `harvest.sh`는 이 세 값을 "정상"으로 취급한다
(`publish_markdown kcurve ... "0 3 4"`) — 과학적 판정과 크래시를 구분하기
위해서다.

### 10.3 재점수화 경로

교정 전 코드로 완주한 run을 되살리는 두 진입점이 있다.

- `rescore_completed_run.py RUN_DIR` — dirty tree 거부 → 생성 계약 검증 →
  GPU에서 `stage_score` 재실행 → `recompute()`. rollout과 저장된 micro-group은
  계약을 통과할 때만 재사용한다.
- `recompute_oracle_scores.py RUN_DIR` — **이미 새 score 프로토콜이 있는 run**의
  oracle/split-half/report만 다시 만든다. GPU 없이 저장된 gradient로만 돈다.
  `score_protocol.json`의 스키마가 구형이면 "rerun the score stage"로 거부한다.

두 경로 모두 `postprocess_manifest.json`에 `source_run_git`, `postprocess_git`,
입력·출력 SHA-256을 남긴다. `score_protocol.json`과 `oracle_protocol.json`은
**이 경로와 정규 파이프라인에서만** 생성된다. 수동으로 만들면 안 된다.

---

## 11. 설계 결정과 이유

| 결정 | 이유 | 근거 |
|---|---|---|
| 스테이지 단위 CLI + 산출물 스킵 | 장시간 파이프라인에서 "처음부터 다시"를 없앤다 | RECOVERY 전제 |
| **스킵 판단은 산출물 단위로 독립** | 한 스테이지가 산출물 2개를 만들면 스킵 판단도 두 번 해야 한다. val fresh가 train 샤드 스킵에 딸려 증발하던 사고 | TROUBLESHOOTING B5 |
| 랜덤 스테이지(drift)는 반드시 스킵 가드 | 결정적 스테이지의 재실행은 낭비지만 랜덤 스테이지의 재실행은 오염이다 | B6 |
| 원자적 저장 전면화 | 깨진 파일이 `exists()`를 통과하는 것을 차단 | B1, B5 |
| 발행 전 행수 = n×K 검증 | 부분 파일이 완성본 이름을 얻는 유일한 경로를 막는다 | C7 |
| CountSketch (위치 해시) | 밀집 JL이 131GB OOM. RNG 스트림 방식은 청크 경계가 결과를 바꿔 기각 | A1 |
| 2-pass score (β→언로드→π) | 두 모델 동시 상주가 14B attention OOM | A2 계열 |
| CPU 경유 단일 사본 로드 | device_map이 27B에서 스켈레톤+체크포인트 이중 상주 | `rollout.py:63` 주석 |
| cuDNN SDPA 전역 비활성을 import 시점에 | 로더 함수 안에 두면 drift·downstream이 빠진다 | C5 |
| `OM_ATTN` 환경 스위치 | 노드별 fused SDPA 커널 병을 코드 수정 없이 우회 | C6 |
| 가드 대신 로더 자체를 preflight로 실행 | 같은 판단을 두 곳에서 구현하면 반드시 어긋난다 | E4 |
| 폴백 실패 시 탐색 위치 전체 출력 | 조용한 폴백은 에러를 원인에서 먼 곳에 찍는다 | E3 |
| top-k 규칙을 함수 하나로 | 스크립트마다 k=25/26이 갈려 표가 섞였다 | P0-5 |
| 독립 tie stream 20쌍 | 공유 jitter는 동점 체제에서 floor를 부풀린다 | P0-6, CODE.md 함정 3 |
| selection/evaluation validation 분리 + 마커 fail-closed | 마커 없는 산출물이 판정에 흘러드는 것을 구조적으로 차단 | FULL_AUDIT Critical |
| C1·C1′를 동일 run·사전고정 cut으로 묶음 | 서로 다른 run/cut의 조합은 사후 선택이다 | FULL_AUDIT High |
| 코드 채점 bubblewrap 격리 | 생성 코드가 host Python subprocess에서 돌고 있었다 | FULL_AUDIT High |
| run-scoped 프로세스 정리 | 전역 pkill이 무관한 GPU 작업까지 종료 | FULL_AUDIT Medium |
| 감시자 싱글턴 | 다중 babysit이 서로의 run을 죽이며 무한 재시작 | A4 |
| 로그·GPU·CPU 3중 정지 판정 | 무출력 정상 구간(score β-pass 등)을 hang으로 오인해 무한 재시작 | A6 |
| 워커 완료를 artifact로 재검사 | `OM_SKIP_POSTPROCESS=1` 경로가 실패 run을 남기고 exit 0을 반환 | 2026-08-24 false-success |
| supervisor / compute 코드 분리 | 감시 버그만 즉시 교체하면서 부분 산출물을 보존 | 2026-08-22 외부 중단 복구 |
| 삭제 대신 quarantine | 되돌릴 수 없는 손실을 만들지 않는다 | `prepare_run_path.py` |
| legacy runner 기본 비활성 | manifest·code lock 없는 runner가 confirmatory 실행에 쓰이는 것 차단 | FULL_AUDIT Medium |

---

## 12. 실측 — 구현 제약과 역사적 문서 불일치

아래는 코드를 직접 읽어 확인한 구현 제약과 2026-08-24 당시 문서 불일치다.
현재 정본인 `README.md`·`docs/README.md`·`USAGE.md`·`CODE.md`에는 실행에 영향을
주는 항목을 반영했다. 날짜가 붙은 감사 스냅샷은 provenance 보존을 위해 수정하지
않는다.

### 12.1 `go_v4.sh` 본문 대부분이 실행되지 않는다 (영향 큼)

`scripts/go_v4.sh:23-25`:

```bash
if [ "${OM_V4_RESUME_WRAPPED:-0}" != "1" ]; then
  exec bash scripts/resume_v4.sh "$@"
fi
```

`OM_V4_RESUME_WRAPPED`는 **레포 전체에서 이 한 줄에만 등장한다**
(`grep -rn OM_V4_RESUME_WRAPPED` 결과 1건). 설정하는 코드가 없으므로
`bash scripts/go_v4.sh <slot>`은 **항상** `resume_v4.sh <slot>`으로 exec되고,
27행 이후는 사용자가 수동으로 `OM_V4_RESUME_WRAPPED=1`을 export하지 않는 한
도달하지 않는다. 도달하지 않는 부분에는 다음이 포함된다.

- GPU 메모리 해제 확인 루프(최대 30 × 2초). `resume_v4.sh`는
  `cleanup_run_processes.py` + `pkill` TERM/KILL만 하고 **메모리 해제를 기다리지
  않는다**.
- `prepare_worker_paths()` — commit 불일치 run의 자동 quarantine.
- `collect_targets()` — 8.2절의 27키 provenance 게이트.
- `matrix_complete()` / `finalize_v4_once()` / `V4_COMPLETE` 표식 —
  마지막 워커의 자동 집계.
- 4-GPU 요구 검사, 모델 스냅샷 존재 검사.

실질적 귀결: **v4 집계는 자동으로 일어나지 않으며, provenance 게이트도
자동으로는 돌지 않는다.** 20 run이 모인 뒤 `collect_v4.sh`를 수동 실행해야
하는데(README 4절도 그렇게 안내한다) `collect_v4.sh`는 artifact 존재만 보고
**commit·모델 해시·설정 동일성을 검사하지 않는다**. 27B 쪽만
`go_v4_27b.sh`가 `validate_v4_27b.py`로 별도 검증한다.

### 12.2 `go_v4.sh` 경로의 27B run은 `validate_v4_27b.py`를 통과할 수 없다

| 키 | `go_v4.sh`/`resume_v4.sh`가 설정하는 값 | `FIXED_CONFIG` 기대값 |
|---|---|---|
| `gen_batch` | `OM_GEN_BATCH=8` | `"4"` |
| `linear_attention_backend` | FLA 미설치 시 `"torch"` | `"fla"` |
| `fla_core_version` | FLA 미설치 시 `null` | `"0.5.2"` |

`go_v4.sh`/`resume_v4.sh`에는 FLA 설치 단계가 없다(설치는 `go_v4_27b.sh`에만
있다). 따라서 `go_v4_27b.sh`를 나중에 돌리면 `config_matches_27b()`가 실패해
그 27B run들을 `--force-quarantine`으로 전부 격리하고 처음부터 다시 시작한다.
**두 27B 진입점은 산출물이 호환되지 않는다.** README 4절의 "7B가 완료된 뒤
27B만 새 코드로 재실행할 때는 아래 전용 진입점을 쓴다"는 서술은 맞지만,
`go_v4.sh`로 만든 27B 부분 진행분이 버려진다는 점은 명시돼 있지 않다.

### 12.3 재개는 mixed commit을 허용하는데 집계 게이트는 단일 commit을 요구한다

`v4_resume_commit.py` docstring: "Plan interrupted v4 runs **without requiring
one Git commit for the matrix**." 반면 `collect_targets()`의 `same_keys`에는
`git`이 들어 있어 10개 run이 commit을 공유하지 않으면
`mixed v4 provenance/config for git`으로 중단한다. 설계상 모순은 아니다
(재개는 진행 보존, 집계는 동질성 요구) — 다만 **여러 commit으로 재개해 완주한
행렬은 집계 단계에서 거부되므로**, 그 경우 남는 선택지는 `collect_v4.sh`
(provenance 미검사)로 표를 만들거나 단일 commit으로 재실행하는 것뿐이다.
현재 12.1 때문에 실제로는 `collect_v4.sh` 경로만 돌고 있다.

### 12.4 완료 판정 기준이 세 가지다

| 위치 | 요구 artifact |
|---|---|
| `v4_resume_commit.complete()`, `go_v4.sh`, `harvest.sh` | 6종 |
| `go_v2.sh run_complete()`, `collect_v4.sh`, `go_v4_27b.sh run_complete_27b()` | 11종 (6종 + `scores_oracle.json`, `scores_offpolicy.json`, `scores_splithalf.json`, `oracle_micro_groups.pt`, `val_groups.pt`) |
| `run_14b.sh` 종료 검사 | 12종 (11종 중 `DONE`·`manifest.json` 대신 `prompts.json`·`rollouts_*.jsonl`·`val_gradient.pt` 포함) |

6종만 보는 경로는 점수 파일이 없어도 "완료"로 판단할 수 있다. 실제로는
`report.json`이 점수 파일 없이 만들어지지 않으므로 현재 위험은 낮지만, 기준이
한 곳에 모여 있지 않다.

### 12.5 `CODE.md`의 줄 번호·제거 동작 서술 (2026-08-25 해소)

모듈 줄 수와 함수 줄 번호는 코드 변경 때마다 무효가 되어 `CODE.md`에서 제거했다.
정답 rollout이 없을 때 drift SFT를 중단하는 현재 동작, APPS 실행 하네스와 ARC
reward까지 현재 모듈 책임 표에 반영했다. 이 항목은 재발 방지 기록으로만 남긴다.

### 12.7 `check.sh`가 비활성화된 스크립트를 재시작 방법으로 안내한다

`scripts/check.sh` 마지막 줄이 `nohup bash scripts/babysit.sh > babysit.log 2>&1 &`를
출력하는데, `babysit.sh`는 `OM_ENABLE_LEGACY_RUNNER=1` 없이는 exit 2로 거부한다.
또 `check.sh`·`status.sh`·`result.sh`는 `_find_root.sh`를 통해 기본 대상을
`$OM_WORK/runs/gate`(v1 경로)로 잡으므로, v4 run을 보려면 경로를 인자로 넘겨야
한다.

### 12.8 pytest 전체 수집 문제 (2026-08-25 해소)

`tests/test_rollout_resume.py`의 모듈 최상위 `sys.exit`가 pytest 수집 중
`INTERNALERROR`를 일으켰다. `91025ca`에서 종료를 `__main__` 아래로 옮기고
회귀 case를 추가했으며, 전체 `pytest tests/ -q`가 51건을 통과한다.

### 12.9 그 밖의 작은 것

- `src/c2_diagnose.py:62`가 `topk_count`를 쓰지 않고 `max(1, int(m*frac))`를
  인라인으로 갖고 있다. 식은 같지만 P0-5 단일화의 예외다.
- `readout_summary.precisions()`는 `topk_frac`을 `run_config.json`에서 읽지 않고
  `0.10`으로 고정한다. `make_tables.py`(`FRAC = 0.10`), `frontier.py`
  (`FRAC = 0.10`)도 같다. v4는 `topk_frac=0.10`이라 현재는 일치한다.
- `run_14b.sh`의 종료 검사 목록에 `divergence_stats*.json`이 없다. 반면
  `go_v2.sh`의 결과 수집과 `collect_v4.sh`는 그 파일이 없으면 abort한다.
- `experiment.run_hybrid()`에 `if True:` 블록(817행)이 남아 있다 — 이전 조건이
  제거된 흔적이다. 동작에는 영향이 없다.
- `grads.token_weights()`에 `clip_cap < 1.0` 검사가 두 번 있다(72행, 75행).

---

## 13. 미사용 코드·죽은 경로

| 대상 | 상태 |
|---|---|
| `go_v4.sh:27` 이후 전부 | 12.1 — `OM_V4_RESUME_WRAPPED` 미설정으로 도달 불가 |
| `experiment.py --stage val-deepen` | 항상 `ValueError`. in-place val 확장은 immutable `val_k` 계약을 깨서 비활성 |
| `scripts/deepen_val.sh` | 같은 이유로 비활성 (`--stage val-deepen` 호출) |
| `scripts/fix_c2.sh` | val-deepen 의존이라 즉시 abort |
| `scripts/run_h100_all.sh`, `scripts/babysit.sh` | `OM_ENABLE_LEGACY_RUNNER=1` 없으면 exit 2 |
| `scripts/go_hard.sh` | precheck NO-GO로 폐기. `FORCE_HARD=1`로만 강행 |
| `scripts/go_full.sh`, `go_boost.sh`, `go_27b.sh` | BACKLOG 폐기절 — 수확만 받고 신규 착수 금지 |
| `experiment.py --stage analyze` | v4 경로에서는 안 쓴다. `run_smoke.sh`, `run_h100_all.sh`(비활성), `retry_c2.sh`만 호출 |
| `experiment.py --stage score` / `--stage oracle` (비샤드) | `run_gate.sh`(v1)만 호출. v4는 `score-shard`/`oracle-grads`+`merge-grads` 경로 |
| `experiment.py --stage downstream` | v4 미호출 → C3 항상 미판정 |
| `stage_oracle`의 비샤드 val 방향 계산 블록 | `val-grads` 스테이지가 담당하므로 v4에서는 도달하지 않는다 |
| `_find_root.sh`의 `7b`/`14b`/`7bm`/`14bm`/`fast` 별칭 | v1 게이트 경로 전용 |
| `tables.sh`/`frontier.sh`의 `_legacy_dup()` | v2 시절 이중 접미사 폴더 대응. v4 이름 규약에서는 발생하지 않는다 |
| `judge.py`의 CertaGrad legacy 분기 | `uniform_precision_vs_oracle`이 없는 옛 report에 oracle=1.0을 보수적으로 가정 |
| `read_rollouts`의 `OM_EOS_IDS` 재유도 | 계약 이전 산출물 전용. v4 산출물은 이미 절단돼 있어 멱등 |
| `data._apps_reward` / apps 로더 | 구현돼 있으나 v4 행렬에는 포함되지 않는다 |
| `outputs/` (비추적) | v1 시절 로컬 산출물 잔재 (`local-gate`, `shard-test`, `smoke`) |

---

## 14. 검증 상태

이 문서를 쓰면서 **CPU에서 안전한 테스트만** 실행했다. GPU가 필요한 테스트는
실행하지 않았고, 실행이 필요한 것도 없었다 — `tests/` CPU 회귀는 전부 모델·GPU 없이
도는 설계다(GPU를 쓰는 스크립트는 `nvidia-smi` 셸 shim으로 대체한다).

실행 환경: `.work/.venv-cu126`(torch 2.7.1+cu126, transformers 5.14.1),
`CUDA_VISIBLE_DEVICES=""`, `OM_WORK`는 임시 스크래치로 지정.

```
pytest tests/                                          →  51 passed
python tests/test_rollout_resume.py                    →  PASS 18 / FAIL 0
python tests/test_cleanup_run_processes.py             →  PASS
python tests/test_data_sandbox.py                      →  PASS
python tests/test_failure_diagnostic.py                →  PASS
python tests/test_frontier.py                          →  PASS
python tests/test_pool_qualification.py                →  PASS
python tests/test_prepare_run_path.py                  →  PASS
```

2026-08-25 `91025ca` 뒤 같은 격리 환경에서 전체 pytest **51 passed**와
스크립트형 7개(rollout resume **PASS 18 / FAIL 0** 포함)를 재확인했다. 개별
스크립트 실행은 상세 PASS 로그가 필요할 때만 사용한다.

`tests/test_frontier.py`는 torch가 아직 import되지 않았을 때만
`sys.modules["torch"]`를 빈 모듈로 채운다. 알파벳 순서상 `test_contract`·
`test_core`가 먼저 실제 torch를 import하므로 현재는 문제가 없지만, 파일 이름이
바뀌면 깨질 수 있는 순서 의존이다.

GPU가 필요해 **실행하지 않은 것**: 실제 rollout 생성, drift 학습, score/oracle
gradient 계산, hybrid 생성, downstream 학습, `check_27b_fla.py`의 FLA 커널
스모크, `gpu_check.sh`의 matmul/SDPA 판정. FULL_AUDIT의 "GPU 환경에서 반드시
재실행" 항목(`test_contract.py`, `test_core.py`, `test_protocol.py`)은 이번에
CPU에서 통과했고, GPU 환경 재실행은 여전히 BACKLOG 항목으로 남는다.
