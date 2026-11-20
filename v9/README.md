# V9 실험 코드 (ICML 2027)

계획: 원고 저장소 `v9/PLAN.md`. 이 폴더는 그 계획의 P0·X1–X5 실행 코드다. V8 코드(`v8/experiments/switching`, `cluster_child.py`)와 학습기(`srgc_rebuttal/`)는 import만 하고 고치지 않는다. V8 결과(`$WORK/v8`)는 읽기만 하고, V9는 `$WORK/v9`에만 쓴다.

## 노드에서 칠 명령

모든 4×H100 노드에서 같은 한 줄을 친다. 인자는 없다.

```sh
sh v9/run.sh
```

순서: `freeze-tau → x1 → x2 → x5 → x3 → x4 → collect`. 준비된 작업만 하나씩 가져가고, 작업이 끝날 때마다 앞 단계부터 다시 본다. 준비 안 된 단계는 건너뛴다. 한 단계나 작업이 실패하면 그것만 기록하고 나머지를 계속한다. 가져갈 작업이 없으면 collect를 돌리고 상태표를 찍고 끝난다. 끝난 작업은 다시 하지 않고, 멈춘 학습은 `latest.pt`에서 이어 간다. 같은 작업을 두 노드가 동시에 돌리지 않는다(작업별 flock lease, run 폴더 실행 잠금).

| 명령 | 하는 일 |
|---|---|
| `sh v9/run.sh status` | 단계·cohort별 done/ready/blocked/failed 수와 막힌 이유(영어 한 줄씩). CPU. V8은 읽기만 하고, V9 쪽에는 V8 locator 사본(`$WORK/v9/<cohort>/locator.json`)만 처음 한 번 쓴다 |
| `sh v9/run.sh freeze-tau` | τ 고정 (CPU, write-once) |
| `sh v9/run.sh x1` … `x5` | 한 단계만 |
| `sh v9/run.sh collect` | 원고용 결과 파일 (CPU) |
| `sh v9/run.sh retry-failed` | 두 번 실패해 멈춘 작업의 실패 횟수를 0으로 (이력은 남김). 원인을 고친 뒤에만 |

환경 변수는 V8 런처와 같다: `OM_WORK`/`GROUP_VOLUME`/`OM_USER`(기본 `/group-volume/minsoo3.kim/offpolicy-misranking`), 인터프리터 `V8_PYTHON_<COHORT>`·`QWEN_PYTHON`·`PAIR_PYTHON`(MATH)·`SWITCH_PYTHON`(MBPP)·`V8_PYTHON`, 기본은 `$WORK/.venv-cu126/bin/python`. 학습기 checkout은 `V8_CODE_ROOT`, 없으면 V8 locator에 기록된 경로, 없으면 이 checkout.

## 단계

- **freeze-tau** (CPU, write-once, `$WORK/v9/frozen/tau.json`). OLMo MATH V8 Branch Bank(seed 5–9)만 읽는다. 후보 τ는 bank grid에서 25를 뺀 {50, 75, …, 250}(τ=25 분기는 SR 자체다). 두 기준을 모두 계산해 기록한다. (a) 같은 예산: V8 olmo_math의 고정 예산(평가 전 비용만으로 고정됨, 모든 τ에 공통) 아래 예산 endpoint의 보상. (b) matched update: update 275 보상. 사전 규칙상 (a)가 모든 τ·seed에서 계산되면 (a)를, 아니면 (b)를 쓴다. 고른 기준에서 최고 평균의 0.5 pp 이내인 가장 이른 τ를 고른다. 규칙 문장, τ별 평균, seed별 값, τ별 평균 Staged 비용, 다른 기준의 평균과 그 기준이 고를 τ, 입력 파일 sha256을 남긴다. V9 집계 파일(`$WORK/v9/paper`, `*/analysis`)이 이미 있으면 고정을 거부한다. collect와 x1–x4는 tau.json이 없으면 막힌다. 이후 모든 단계는 tau.json의 τ로 `bank_sr_t{τ}`를 쓴다(코드에 τ 값이 박혀 있지 않다).
- **x1**. OLMo MATH·OLMo MBPP·Qwen MATH·Qwen MBPP: Staged(τ)=V8 `bank_sr_t{τ}`, SR=`bank_sr_t025`, Alignment=V8 On 궤적, 모두 update 275의 V8 eval-final을 읽기만 한다. 평가가 없으면 `waiting for V8 eval-final`, 분기가 없으면 `waiting for V8 branch …`. Llama MATH·MBPP: 원래 실행(seed 5–9)의 25-update prefix에서 On 궤적(275, 25마다 스냅샷), 그 τ 스냅샷에서 SR 분기(275), prefix에서 SR(275)을 새로 학습하고 275에서 평가한다. 아래 "Llama" 참고.
- **x2** (OLMo MATH, OLMo MBPP). 예산 B = Staged(τ) 연속 비용의 seed 평균(V8 `freeze_budget`과 같은 범위·규칙, 정책만 `fixed_sr_t{τ}`, 평가 파일은 읽지 않음, `$WORK/v9/<cohort>/frozen/budget.json`, write-once). SR(`bank_sr_t025`, τ보다 이른 다른 분기가 있으면 그것도)의 275 상태를 B까지 연장(V8 budget-extension 로직, hard cap 400). 평가: SR@B endpoint, Alignment@B(On 궤적에서 B 안의 마지막 스냅샷), Staged@B(비용이 B를 넘는 seed는 B 안의 마지막 스냅샷). V8에 같은 스냅샷·같은 spec 평가가 있으면 다시 하지 않고 읽는다.
- **x5** (측정, 학습 없음). OLMo MATH seed 5–9, 다른 V8 cohort는 첫 seed. On 궤적 스냅샷 {25(prefix), 50, 100, 175, 275}마다 후보 400개를 기존 scoring 경로(`backend.score_gradients`, group 4 LOO)로 4응답씩 두 번 점수 매긴다. 같은 rollout으로 (a) ρ_on(split-half Pearson·Spearman, Spearman-Brown), σ_on, ‖v‖(검증 50문항 8응답), (b) ρ_SR(p 추정치 split-half, 캐시 4/4 split-half도), (c) p_t(8 rollout)와 캐시 p_β의 차, (d) KL(π_t‖β) = π_t 표본에서 log π_t − log β의 수열 합 평균과 토큰 평균. β는 SR 캐시를 만든 정책 = adapter 없는 사전학습 기반 모델(`srgc_rebuttal/build_cache.py` 설명, `import_pair_inputs.py`의 "initial-policy behavior rollouts")이고, log β는 같은 forward를 PEFT `disable_adapter()`로 계산한다. 번들의 캐시 출처를 요약에 함께 적는다. 40문항 묶음마다 `rows.jsonl`에 쓰고 재개 시 건너뛴다.
- **x3** (OLMo MATH). 275 스냅샷에서 Alignment(On), SR(`bank_sr_t025`), Staged(`bank_sr_t{τ}`)를 고정 횟수로 400까지 연장(25마다 스냅샷), 400에서 평가.
- **x4** (OLMo MATH seed 10–14). V8 prospective 경로를 V9 root에서 쓴다: V8 `prepare-seeds` 로직(같은 split·초기 캐시, 학습 RNG만 seed)으로 V9 locator 사본에 seed를 추가 → 25-update prefix → Random(`uniform_early`)·SR(`sr_early`)·Alignment(`on_continued`)·Staged(`fixed_sr_t{τ}`) → 275 평가. τ 튜닝 없음(tau.json의 τ 고정). seed 10–14는 seed registry가 고른 fresh test seed와 같은지 확인하고 다르면 막는다.
- **collect** (CPU). 아래 파일.

## 결과 파일 (`$WORK/v9/paper/`)

| 파일 | 내용 |
|---|---|
| `results.json` | 모든 칸과 상태(measured/pending+이유), tau.json 전체, 열린 항목 목록 |
| `x1_staged.tex` | cohort별 Staged·SR·Alignment 평균, Staged−SR, Staged−Alignment, Staged−best: 평균 pp [95% paired-t] (승/무/패). best = 그 cohort에서 평균이 높은 단일 선택기(SR 또는 Alignment) |
| `x1_per_seed.csv` | cohort, seed, arm, reward_fraction, status, reason |
| `tau_sweep.tex` | OLMo MATH τ별 평균(선택 기준 행 + 다른 기준 행, 고른 τ 굵게) |
| `x2_budget.tex` | B(GPU-h), Staged@275, SR@B, Alignment@B, Staged@B의 보상·평균 endpoint update·평균 비용, Staged@275 − 각 arm |
| `x3_horizon.tex` | 275와 400에서 Alignment·SR 평균과 Staged−SR, Staged−Alignment |
| `x4_seeds.tex` | seed 10–14 네 arm 평균, Staged−SR/Alignment(10–14), 합친 seed 5–14 |
| `x5_summary.tex`, `x5_summary.csv`, `x5_summary.json` | cohort·seed·update별 ρ_on, ρ_SR, σ_on, ‖v‖, \|p_t−p_β\|, KL(수열·토큰), Pinsker 범위 안 비율 (그림 입력은 csv) |

보상은 [0,1] 비율, 차이는 pp, 비용은 GPU-seconds(표는 GPU-h). 칸은 그 칸에 필요한 모든 seed가 측정됐을 때만 숫자이고, 아니면 `\pending{Xk}`다(조각마다 `\providecommand{\pending}`가 들어 있다). 추정·보간·다른 cohort 값으로 채우지 않는다.

## Llama

원래 Llama 실행(`scripts/run_srgc_llama31.sh` → `srgc_research/dispatch/llama31`, `meta-llama/Llama-3.1-8B-Instruct` rev `0e9e39f…`, run root `$WORK/srgc-rebuttal/llama31-8b-v1`)은 같은 Engine·체크포인트 형식(`prefix.pt`, optimizer 포함 25-update prefix)이다. V8 child로는 그대로 못 돈다: `switching/cluster.py`에 llama plan resolver가 없고, `switching/coderepo.py::source_runtime`에 llama adapter digest가 없고, `cluster_child.py`는 OLMo 모델 등록부로만 모델을 연다. V9는 V8을 고치지 않고 `v9lib/llama.py`(plan 위치·`validate_extension`·신원 확인: run.json `implementation_sha256` = sha256(srgc_rebuttal digest + adapter digest))와 `v9lib/llama_child.py`(Llama `runtime_adapter()`·`training_adapter()`·`setup_storage`·Llama `load_model`·LlamaBackend)로 돌린다. 클러스터의 코드 바이트가 Llama 실행 당시와 다르면 status에 `Llama runtime not runnable: … digest …`로 뜨고 학습하지 않는다. Llama 25-update마다의 optimizer 포함 체크포인트는 없으므로 On 궤적을 prefix에서 다시 만든다.

## 예상 GPU-h (4×H100 device-hours, 계획값)

근거: `v8/experiments/configs/cohorts/olmo_math.json` cost_basis(SR update 0.0768, On 학습 update 0.0788, On 선택 갱신 0.653/25 update, v7 Fig.4·seed 5–9 기록). 평가 비용은 OLMo 기록이 없어 빠져 있다(Qwen은 endpoint당 약 15.5). τ=75 기준, τ=50이면 조금 작다.

| 단계 | 계산 | GPU-h |
|---|---|---|
| X1 OLMo/Qwen | 학습 없음 (V8 결과 읽기) | 0 |
| X1 Llama | seed당 On 250 update(19.7+선택 6.5) + Staged 200 SR(15.4) + SR 250(19.2) ≈ 61, ×5 seed ×2 데이터셋. Llama 기록 비용이 없어 OLMo 값을 대신 씀(8B라 더 클 수 있음) | ≈ 610 + 평가 30회 |
| X2 | seed당 SR 연장 (B − SR 비용)/0.0768 update ≈ 20–70 update ≈ 1.5–5, ×5 ×2 cohort | ≈ 15–50 + 평가 약 40회 |
| X5 | 스냅샷당 450문항×8응답 scoring(On 갱신 90문항 0.653 기준 ≈ 3.3) + log-prob forward 2회 ≈ 4; OLMo MATH 25회 + 다른 cohort 15회(Qwen은 수 배) | ≈ 100 (OLMo) + 60–200 |
| X3 | (On 125 update 13.1 + SR 9.6 + Staged 9.6) ×5 seed | ≈ 160 + 평가 15회 |
| X4 | seed당 prefix 2.6 + Random 19.2 + SR 19.2 + On 26.2 + Staged(On 50 + SR 200) 20.6 ≈ 88, ×5 | ≈ 440 + 평가 20회 |

## 로컬 검증 (CPU)

```sh
sh v9/fake/e2e_cpu.sh            # 가짜 $WORK(V8 bank 포함) 생성 → sh v9/run.sh 기본 실행 → 재실행 → 중단·재개
CUDA_VISIBLE_DEVICES= python -m pytest v9/tests
```

가짜 환경: OLMo MATH·OLMo MBPP·Qwen MBPP·Llama MATH는 `srgc_rebuttal` ToyBackend, Qwen MATH는 무작위 tiny OLMo-3 + LoRA(TorchBackend, 실제 autograd scoring·`disable_adapter` log-prob). V8 산출물은 V8 코드(`matrix.build_stage`, `runner.execute_train/execute_eval`, `cluster.cmd_freeze_budget`)로 만든다. GPU child 대신 같은 task 함수를 CPU에서 부르는 `fake/fake_child.py`가 torchrun(nproc 1) 안에서 돈다. 가짜 locator는 `V9_ALLOW_FAKE=1`일 때만 받아들인다. 소프트웨어 검사일 뿐 논문 수치가 아니다.

## 클러스터에서 확인할 점 (GPU에서만 확인 가능)

1. OLMo/Qwen 학습·평가 task가 `cluster_child.py`를 `--v8-root $WORK/v9`로 부를 때 모델·LoRA·verifier·attention 상속이 V8과 같은지. 첫 x3 작업 로그의 `attention=…`, `model+LoRA ready`를 본다.
2. X5: 실제 모델에서 `disable_adapter()` 아래 log-prob(β)과 생성 설정(T=1, top_p=1, top_k=0, generation_config의 다른 processor 유무). θ₀에서 KL=0은 tiny OLMo-3에서만 확인했다. 스냅샷 하나의 GPU 시간과 메모리(4응답 두 번 + forward 두 번).
3. X5 Qwen: `training_adapter()` 안에서 PEFT `disable_adapter`가 Qwen adapter를 실제로 끄는지.
4. Llama: 클러스터 checkout의 srgc_rebuttal·Llama adapter 바이트가 run.json 신원과 같은지(`sh v9/run.sh status`의 x1 llama 줄), Llama 가중치 경로(`SRGC_LLAMA_MODEL_PATH`/models), `llama_child.py` 전체 경로(로컬에서 못 돌림).
5. X4: fresh seed bundle이 seed 5 bundle의 split·캐시를 재사용한다는 V8 가정(V8 README §8.5).
6. X2: V8 평가 파일과 같은 snapshot·spec이면 재사용하므로 verifier 신원이 V8 평가와 같아야 한다(다르면 child가 거부).
7. 여러 노드가 동시에 돌 때 group volume에서 flock lease가 동작하는지(V8 run 잠금과 같은 방식).
8. V8 코드에서 찾은 두 가지(현황판 V8-160·V8-161, 11-20 소스 수정). 클러스터 V8 큐는 두 경우 모두 이미 피했다: 큐 코디네이터(`scripts/srgc_v8_nodes.py`)는 선택된 endpoint의 run으로 평가를 만들고, 한 번도 update할 수 없는 extension은 `v8_task_entry.repair_budget_fork_checkpoint`(192f533)가 처리한다. 진행 중인 큐는 묶인 runner 그대로 돈다.
   - `switching/cluster.py::budget_eval_tasks`(수동 `eval-budget` 행렬)는 정책 행의 분기 `run_id`로 평가 task를 만들어, budget extension run(update > 275) 안의 endpoint를 빠뜨렸다. 이제 `budget_endpoint_run_id`를 쓴다.
   - `switching/runner.py`는 한 번도 update할 수 없는 budget extension을 `_finalize`에서 "retained snapshots missing … [275]"로 실패시켰다(진입 보정 없이 돌릴 때). 이제 시작 스냅샷을 자기 기록으로 남긴다. V9 x2는 그런 경우 extension을 만들지 않고 분기에서 B 안의 endpoint를 고르는 방식을 그대로 쓴다(결과는 같다).
