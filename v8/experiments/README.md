# V8 추가 실험 코드 / V8 additional experiments (E0–E10)

리뷰 대응 실험(E0–E10)을 기존 학습 코드 위에 얹은 실행 코드다. 학습 알고리즘은 새로 쓰지 않았다.
GRPO 학습기, On-policy(gradient alignment) 선택기, cached-SR 선택기, SR-GC D 계산, 검증기(verifier)는
코드 저장소 `33modeling/offpolicy-misranking`의 `srgc_rebuttal/`에 있는 것을 그대로 import 해서 쓴다
(`V8_CODE_ROOT`). 이 폴더는 그 위에 selector / transition policy / cache policy 분리, checkpoint branching,
비용 ledger, budget endpoint, seed·matrix, tuning·freeze, 분석을 더한다. 아직 어떤 GPU 결과도 없다.

Execution code for the review-response experiments, layered over the existing trainer. Nothing in the
algorithm is rewritten: the GRPO trainer, On-policy gradient-alignment selector, cached-SR selector,
SR-GC contrast and verifiers are imported from `srgc_rebuttal/` of the code repository
(`V8_CODE_ROOT`). No GPU result exists yet; every number below is a planning estimate.

> 저장소 위치에 관한 주의 / Placement note: 이 원고 저장소의 `AGENTS.md`(2026-09-27)는 실행 코드를 코드 저장소에만
> 두라고 한다. 이번에는 사용자 지시("offpolicy repo에 v8 폴더 … 모든 추가 실험 코드")에 따라 `v8/experiments/`에 두되,
> 알고리즘 코드는 복사하지 않고 코드 저장소를 import 한다. 코드 저장소로 옮기려면 이 폴더를 통째로 옮기면 된다.

## 1. 무엇을 답하나 / What each experiment answers

| ID | 질문 (Korean) | Question (English) | 구현 / implementation |
|---|---|---|---|
| E0 | 원고 표의 수치가 어느 run·checkpoint·평가·비용 기록에서 나왔고 다시 계산하면 같은가 | Does every reported cell trace to a run, checkpoint, raw evaluation and cost receipt, and recompute to the same value? | `audit.py` (v7 evidence CSV → endpoint JSON 재집계, split overlap, Gemma/Llama 별도 case) |
| E1 | SR-GC 전환이 개발 seed로 고른 고정 전환 시점보다 같은 예산에서 나은가 | Does SR-GC switching beat a development-tuned fixed switch time at the same budget? | Branch Bank F_τ (τ=50/100/150/200[,25,75…250]) + LOSO-CV tuned fixed(재분석) + fresh-seed dev/test (`fixed_sr_tuned`) |
| E2 | gradient 신호 대신 학습 보상 plateau로 바꿔도 되는가 | Is a free training-reward plateau trigger as good? | `PlateauPolicy` (W=25, ε∈{0,0.005,0.01}, 2 checks), On 궤적 replay + prospective `plateau_sr_*` |
| E3 | 같은 전환 상태에서 SR 대신 Uniform으로 가면 어떤가 | Same trigger state, SR vs Uniform destination? | trigger 상태 snapshot에서 `bank_sr_tτ` / `bank_uniform_tτ` 형제 분기 (동일 model·optimizer·RNG) |
| E4 | 정확한 확인 규칙(1회/2회 연속/현재/3회)과 check 간격이 중요한가 | Do the temporal rule and check cadence matter? | 모든 check에서 D를 기록한 On 궤적에 규칙 replay → Branch Bank 조회; 50 간격 subsample; prospective `srgc_*` |
| E5 | SR-GC 부호가 실제 분기 결과를 예측하는가 | Does the sign of D predict the branch outcome (decision utility)? | 부모 τ=50/100/150에서 D vs (SR 분기 − On) at τ+25/τ+50/275, selected-batch contrast 별도 version |
| E6 | 모든 비용을 넣으면 어떻게 되나 | Full cost ledger and matched budget | `ledger.py` 4 views (continuation / cold-start / physical / amortized), GPU별 union, budget endpoint |
| E7 | 275 시점 선택이 결론을 바꾸는가 | Horizon/budget sensitivity | 25 update마다 snapshot, offline dev-split curve, budget-extension(최대 400) |
| E8 | 반복·역전환 (선택) | Repeated/reverse switching (optional) | `BidirectionalPolicy` scaffold(dwell/margin) + 코드 저장소 `switch_repeat` arm 호출 |
| E9 | 상태를 읽는 정보가 가치 있는가 | Is reading the state worth it vs a pre-drawn time? | dev τ 분포(무전환 질량 포함) freeze → seed마다 1회 추출(재추출 없음) → 조회/실행 |
| E10 | stale cache 덕분인가 | Is the gain from a stale SR cache? | `sr_early_refreshed_k{050,100}` (sr_entry 기준 K마다 전체 pool 재측정, staging→atomic commit, 실패 시 이전 유지) |

## 2. 실제 학습 코드와 연결 / Integration with the real trainer

검증한 통합 지점 (master 0937526 기준, 로컬에서 import·실행 확인):

- `srgc_rebuttal/srgc.py` `Engine`: 40 후보 추출, scoring gradient, cosine 순위, 40-vs-40 D, SR 순위, 학습 호출. v8의
  `switching/engine.py` `PolicyEngine`은 이 클래스를 상속해 update마다 base arm 이름만 고르고
  (`switch`/`on_policy`/`sr`/`random`), `TemporalRule` 호출을 v8 controller로 넘긴다. **원래 방법과 bit 단위로 같다**:
  toy와 tiny OLMo-3(TorchBackend)에서 Switch/On/SR/Random, 코드 저장소의 `SwitchFixedEngine`, `SwitchRuleEngine`과
  학습 prompt 순서·최종 가중치가 동일함을 테스트한다.
- `srgc_rebuttal/torch_backend.py` `TorchBackend` (GRPO, LoRA AdamW, autograd scoring, `evaluate`), Qwen은
  `scripts/srgc_qwen35.py` `training_adapter()`/`load_model`.
- `srgc_rebuttal/timing.py` `CostMeter`/`torch_meter`: v8 ledger가 phase event를 받아 시작·끝 시각, GPU id, stage별
  배분을 기록한다. SR 비교 집합 scoring은 별도 stage로 떼어 계측한다(위치 기반 rank 배정이 같아 gradient 동일:
  `40 % world == 0`).
- 실행 경로: `scripts/srgc_pair_inputs.default_plan` + `scripts/srgc_shared_storage.route_plan` +
  `scripts/srgc_extra_plan.select_plan` (extra-arm launcher와 같은 plan 해석), `srgc_rebuttal/existing_runtime.load_model`.

## 3. 핵심 설계 / Design

- **Branch Bank**: seed마다 prefix(25)에서 On-policy를 275까지 다시 돌리며 25 update마다 snapshot(가중치+AdamW+selector
  상태+프로세스 RNG)을 보존하고, 모든 check(25…250)에서 SR-GC D를 **진단용으로** 계산한다(학습 RNG·경로 불변, 비용은
  `sr_comparison` 성분으로 분리). τ에서 SR 분기를 만들면 어떤 one-way 규칙이든 τ에서 발사했을 때의 결과는 표 조회다.
  원래 Switch(τ_s) = F_τ_s가 성립함을 테스트한다.
- **정확한 τ만 조회**: 분기가 없는 τ(예: 75)는 `unavailable`. 보간·가장 가까운 grid 대체 없음. floor/ceil은 별도 이름의
  민감도 정책이다.
- **관측 중단(censoring)**: current 규칙은 bridge 때문에 R2보다 먼저 발사할 수 있다(D=[−1,+.5,−1,+.5,−1,−1] → current 75,
  R2 150). 로그가 끝난 뒤에 발사할 규칙은 `censored`(관측 중단), `not_triggered`가 아니다. 빠진 check는 리셋이 아니라
  미확인이며, 같은 check의 다른 값은 오류다. 분석은 censored를 따로 센다. On 궤적은 마지막 check까지 D를 기록하므로
  Branch Bank 쪽에는 censoring이 없다.
- **비용**: 성분(on_selection, sr_comparison, training, cache_build, cache_refresh, prefix …)과 method별 필요 성분으로
  view를 만든다. 측정되지 않은 비용은 0이 아니라 unknown이며 해당 view는 비거나 하한(lower bound)으로 따로 표시한다.
  GPU별 interval union 후 합산(예: 30 GPU-s, 40·60 아님). 중단 attempt의 재실행 구간은 physical에만 들어간다.
- **Budget endpoint**: metadata만으로 B 이하 최신 snapshot 선택, B 이후 모델 사용 금지. 비용이 모르는 snapshot이 선택을
  바꿀 수 있으면 `undetermined`, 알려진 비용이 모두 B를 넘으면 `infeasible_budget`. B는 평가 파일을 읽지 않고 비용만으로
  freeze(write-once).
- **기록 규칙(2026-10-28 사용자 규칙)**: 학습 보상(`training_stats.jsonl`)과 optimizer·RNG 포함 snapshot을 25 update마다
  남긴다(`latest.pt`는 매 update). 학습 중에는 held-out 평가를 하지 않는다. 최종 평가는 별도 offline task(`eval-final`)로
  돌고, 중간 곡선 평가(`eval-dev`)는 선택적이며 자체 비용 ledger(`evaluations/events/`)를 가진다. 평가하지 않은 지점은
  `unmeasured`로 표시한다. 평가 보상은 정수 정답 수로만 기록하고, 정수가 아닌 평균(예: 0.37/8)은 거부한다.
- **Seed**: seed 0–9는 결과를 이미 봤다. registry가 이를 제외하고 test 10–14, development 15–17을 고른다
  (`registry/seed_registry.csv`). seed 5–9 Branch Bank 분석은 "same seeds, hindsight sweep + LOSO tuned fixed"로 부른다.
- **LOSO**: fold들이 tuning seed를 공유하므로 IID paired-t 구간을 1차 결과로 내지 않는다(seed별 차이, 평균, 승패, τ̂ 안정성만).
  새 seed의 dev-tuned fixed는 일반 paired-t.

## 4. H100 실행 순서 / Run order on the cluster

```sh
export V8_CODE_ROOT=/path/on/user-volume/offpolicy-misranking   # seed 5-9를 만든 동결 checkout
# 선택: PAIR_PYTHON / SWITCH_PYTHON / V8_PYTHON, OM_WORK, V8_ROOT
sh v8/experiments/run_h100.sh olmo_math check
sh v8/experiments/run_h100.sh olmo_math rebuttal        # locate→audit→profile (CPU) →trajectory→branches→extended
                                                        # →destination→freeze-budget (평가 전, 비용만)→budget-extension
                                                        # →eval-final→eval-budget→collect→analyze
sh v8/experiments/run_h100.sh olmo_math eval-dev        # 선택: E5/E7 곡선 (offline, 자체 비용)
sh v8/experiments/run_h100.sh qwen_mbpp rebuttal        # replication cohort
sh v8/experiments/harvest.sh                            # 결과 전체를 파일 하나로
```

GPU stage는 비어 있는 4×H100 노드마다 같은 명령을 실행하면 된다. task마다 lease를 잡아 중복 실행이 없고, 끝난 task는
건너뛰며, 중단된 task는 마지막으로 저장된 update에서 이어 간다(Ctrl-C → 안전 경계에서 저장 후 exit 75). 로그는 화면에 그대로
나오고 `$V8_ROOT/<cohort>/logs/`에도 남는다. 결과 위치: `/group-volume/minsoo3.kim/offpolicy-misranking/v8/<cohort>/`
(`runs/`, `evaluations/`, `frozen/`, `analysis/`), E0는 `v8/audit/<cohort>/`. user volume에는 코드만 둔다.

Prospective(새 seed) 확증 실험: `prepare-seeds` → `prefix` → `dev` → **`tune-freeze`(클러스터 stage로는 미구현, exit 2;
순서는 `switching/smoke.py`의 `toy()`에만 있다)** → `test` → `eval-final`. 이번에 지원하는 경로는 위의 rebuttal Branch Bank다.

## 5. GPU 시간 추정 / GPU-hour estimates (planning only)

`python -m switching.cli estimate`가 만드는 값이다. 근거는 v7 기록 비용: OLMo는 Fig. 4(seed 3/4: SR 19.20, On 학습 19.69 +
scoring 6.53 GPU-h / 250 update)와 seed 5–9 후보측 scoring, Qwen MBPP는 `v7/evidence/2026-10-19/qwen-mbpp-gpu-times.*`.
평가 비용은 Qwen만 기록값이 있고(endpoint당 약 15.5 GPU-h), OLMo는 `profile` stage가 기존 endpoint 영수증에서 채운다.
학습·선택 GPU-h(4×H100 device-hours), 5 seeds:

| stage | OLMo MATH (primary) | Qwen MBPP (replication) | OLMo MBPP |
|---|---:|---:|---:|
| trajectory (On + D monitor, snapshot/25) | 143 | 830 | 143 |
| branches τ=50/100/150/200 (X1) | 230 | 950 | 230 |
| destination E3 (기록된 τ_s에서 Uniform) | 79 | 198 | 86 |
| extended τ=25,75…250 (모든 25-update 발사 시점 정확 조회; τ=25 = Early SR 재생성) | 298 | 1,227 | 298 |
| **rebuttal 합계 (trajectory+branches+extended+destination)** | **750** | **3,205** | **758** |
| budget-extension (상한, hard cap 400) | ≤240 | ≤989 | ≤240 |
| final 평가 60회 / dev 곡선 325회 (dev 곡선은 선택) | profile로 측정 | ≈930 / ≈840 | profile로 측정 |
| prospective dev (3 seeds) / test (5 seeds) | ≈646 / ≈1,213 | ≈3,032 / ≈5,845 | ≈646 / ≈1,213 |

OLMo MATH core 230 GPU-h와 E3 79 GPU-h는 rebuttal plan v2 §5.2·§5.6 추정과 같다. 측정값이 나오면 `profile`/ledger로 대체한다.

## 6. 산출물 / Outputs

`analysis/`: `policy_outcomes_{bank,development,test}.csv/json`(정책 × seed: status triggered/not_triggered/censored/
unavailable, switch update, reward, 비용 view, budget endpoint), `analysis.json`(LOSO primary, 2차 비교, sign test,
E3/E4/E5/E7/E9/E10), `pareto_bank.png`, `report.md`(과장 표현 차단). `frozen/`: `budget.json`, `frozen_tuning.json`
(test_access_status), `sampled_schedules.json`, `frozen_study.json`. `runs/<run_id>/`: manifest, attempts/*/events.jsonl
(ledger), history·decisions·signals·training_stats, snapshots.jsonl(hash·identity), `snapshots/step-*.pt`.

`registry/` (커밋됨): `seed_registry.csv`(seed × cohort 사용 이력, fresh test 자격), `e0_local/`(로컬에서 가능한 E0 부분:
v7 evidence 표 cell 정리와 코드 저장소 `srgc_rebuttal/inputs/*seed-*.json` 10개 bundle의 split 검사 — `validate_inputs` 통과,
ranking reference·dev 평가·final 평가 간 겹침 0, dev 평가 50문항. 이 로컬 bundle은 prepared plan용이라 cache가 비어 있고,
실제 seed 5–9 run의 bundle은 클러스터 plan에 있다. run root가 없으므로 cell 상태는 모두 `missing_raw_evaluation`).

### 6.1 학습 이력 export / History export (code repo, Codex)

Exporter: 코드 저장소 `scripts/srgc_v8_history.py`, commit `6cfe1920f2be5f46c039b4d74bd151769cff712a` (master).
실행 중·중단된 v8 run의 학습 보상 곡선, 별도 held-out 평가, 25 update checkpoint를 attempt lineage와 함께 내보낸다.
이 폴더는 그 exporter를 복제하지 않는다.

```sh
python "$EXPORT_CODE_ROOT/scripts/srgc_v8_history.py" --root "$V8_ROOT" --output "${V8_ROOT}-history-exports"
# 최종 확인: 같은 명령에 --verify-checkpoints --strict
sh v8/experiments/run_h100.sh olmo_math history-export      # EXPORT_CODE_ROOT가 있을 때만 실행, 없으면 건너뜀
```

두 code root를 구분한다. `V8_CODE_ROOT`는 seed 5–9 원래 run을 만든 **동결 checkout**이어야 한다
(`srgc_rebuttal` digest = `run.json`의 `implementation_sha256`; 학습·분기·평가에 쓰임). `EXPORT_CODE_ROOT`만 최신 master를
가리킬 수 있다(exporter는 추가된 파일이고 `srgc_rebuttal/*.py`는 바뀌지 않음). 분석은 exporter 출력을 선택 입력으로만 취급하고,
없으면 이 폴더의 원시 기록(`runs/<run_id>/…`, `evaluations/*.json`)만 쓴다.

## 7. 로컬 검증 / Local verification (CPU only)

```sh
sh v8/experiments/smoke_cpu.sh          # unit tests + end-to-end smoke (toy + tiny OLMo-3), GPU 미사용
```

toy smoke는 코드 저장소 `ToyBackend`(400 후보, 40/4/8, prefix 25, horizon 275)로 prefix, 원래 4 arm(E0 입력·censoring 시연),
Branch Bank, E3, budget freeze·extension, dev matrix·tuning·τ 분포·freeze, test matrix(11 arm, E9·E10 포함), E8 외부 arm,
중단·재개, 평가, 분석, 보고서까지 모두 실행한다. tiny smoke는 무작위 4층 OLMo-3 + LoRA로 실제 GRPO·AdamW·autograd 상태의
분기 동일성을 확인한다. 이 결과는 소프트웨어 검사이며 논문 결과가 아니다.

## 8. 클러스터에서 확인해야 할 통합 지점 / Integration points to verify on the cluster

1. `V8_CODE_ROOT`의 `srgc_rebuttal` digest가 seed 5–9 `run.json`의 `implementation_sha256`과 같아야 한다(`locate`가 검사,
   다르면 중단). 현재 master의 digest가 동결 checkout과 다를 수 있다.
2. `cluster_child.py`의 모델 적재·LoRA·verifier runtime(`prepare_verifier_runtime`)·attention 상속은 코드 저장소의 extra-arm
   runner와 같은 함수를 부르지만 GPU에서는 아직 실행해 보지 않았다. 첫 실행은 한 seed로 `trajectory`를 짧게 돌려 원래 On-policy
   arm의 update별 `train_ids`와 비교하는 것을 권한다(재현성 확인: bit-exact가 아니면 그 범위를 기록).
3. Qwen plan 경로(`$SRGC_QWEN_ROOT/experiments/qwen35-9b-mbpp.json`)와 `training_adapter()` 적용 순서.
4. seed 5–9 SR cache 구축 비용: Pair 실행에서 가져온 cache라 측정값이 없을 수 있다 → cold-start view는 하한으로만 나온다.
5. 새 seed(`prepare-seeds`)는 seed 5 bundle의 split·초기 cache를 재사용한다고 가정한다(같은 split seed 0). 저자 확인 필요.
6. On-policy 원래 실행의 25 update 단위 optimizer 포함 checkpoint는 보존되지 않았다(`*-latest.pt`만 남음) → On 궤적을
   prefix에서 다시 만든다(rebuttal plan v2 §5.8 (i)).
