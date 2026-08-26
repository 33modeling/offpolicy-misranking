# CODE.md — 코드 상세 설명서

> **유지 규약**: 코드(src/·scripts/)를 바꾸면 이 문서의 해당 절을 갱신한다.
> 새 모듈·스테이지·환경변수·산출물이 생기면 여기에 추가하고, 삭제되면 여기서도
> 지운다. **단, 급한 픽스는 코드만 먼저 커밋·push하고 문서는 후속 커밋으로**
> — 문서 작업이 배포를 지연시키지 않게. (사용자 지시 2026-08-18)

## 1. 한눈 개요

**질문**: 옛 정책 β가 만든 stale rollout으로 프롬프트 순위를 매길 때,
어느 drift·prompt-pool·oracle reliability 영역에서 current-policy utility를
보존하고 어느 영역에서 실패하는가.

**파이프라인** (seed × dataset 조합마다 반복):

```
prep ──→ rollout-behavior ──→ drift ──→ oracle ──→ score ──→ report
(데이터)   (β 정책 K개 생성)   (LoRA RFT   (π에서     (2×2 추정량   (판정 요약
                               로 β→π 이동) fresh 생성) g00~g11 채점)  report.json)
                                  │
                                  └─→ hybrid (bb/bp/pb/pp 처치) · downstream (GRPO-lite 검증)
```

- 실행 단위: `src/experiment.py --stage <이름>` 을 셸 스크립트가 순서대로 호출.
- 모든 스테이지는 **완료 산출물이 있으면 스킵**(재개 안전), 저장은 전부
  원자적(`_atomic_text`/`_atomic_save`: 임시 파일에 쓴 뒤 rename).
- rollout 계열 스테이지는 GPU 장수만큼 **샤드 분할** 후 병합
  (`merge_rollouts`가 기존 병합본까지 prompt 전수·범위·정확한 K와 rollout ID
  `0..K-1`를 검증 — GPU 수가 바뀐 재시작도 안전).
- `run_config.json`의 config digest가 model/config, dataset/pool, seed, K, 생성·투영
  설정과 git 상태를 고정한다. 기존 run과 digest가 다르면 산출물을 재사용하지 않는다.

## 2. src/ 모듈 상세

### 실행 코어

| 파일 | 역할 |
|---|---|
| `experiment.py` | 스테이지 오케스트레이터. oracle micro-group은 짝/홀 equal-budget split, report의 floor·precision은 20개 독립 tie stream 평균. CertaGrad 평가는 candidate와 validation 모두 선택용 짝수/평가용 홀수 group으로 분리한다. |
| `rollout.py` | 모델 로드(`load_model` — `OM_ATTN=eager` 지원, ULF 대응), `collect_rollouts`: 프롬프트별 K개 생성 + reward 저장 (logp는 score 단계에서 재계산) — **프롬프트 단위 `.partial` 내구 저장·중간 재개**(`salvage_partial`, ULF 4차 C7 대응, K개 완주 프롬프트만 승계·행수 n×K 검증 후 발행). `train_drift_lora`: **rejection FT** — reward>0.5 rollout만 SFT(LoRA r=16 α=32)하며, 정답 rollout이 없으면 잘못된 drift를 만들지 않고 중단한다. |
| `grads.py` | 추정량 수학. `loo_advantages`: 그룹 leave-one-out advantage. `log_weights`/`token_weights`: prefix·suffix·token IS 가중치 (2×2의 실체). `prompt_gradient`+`project_grads`: CountSketch 투영 프롬프트 gradient. `grad_params`: LoRA merge 후 requires_grad 복원 함정 처리. |
| `data.py` | 로더+보상. `load_prompts`: gsm8k / dapo-math / math500 / **mbpp** / **kk** / **apps** / **arc-challenge**. 명시 경로와 로컬 snapshot을 우선하고 허용된 경우에만 HF를 사용한다. 수학 최종답, MBPP/APPS 격리 코드 실행, KK 신원, ARC option-label reward를 각각 검증한다. |
| `hybrid.py` | C1′ 인과 검증. β/π 경로 × β/π 마무리 4셀을 equal-K로 만들며, 보존 response prefix 길이를 전체 생성 horizon에서 차감해 cell 간 토큰 예산을 맞춘다. |
| `train_downstream.py` | C3 검증. `grpo_lite_train`: 선택된 프롬프트로 GRPO 간이판(LOO advantage) 학습, `eval_accuracy`: val 정확도, `selection_rollout_cost`: 선택 방식별 rollout 예산 집계. |
| `certagrad.py` | C2 인증 진단. 후보·validation 비용을 따로 계상하고, 가능한 모든 adaptive look에 delta를 배분해 fixed-n interval의 시간축 재사용을 막는다. `gaussian` 반경은 여전히 모델 기반 근사이며 분포무관 보증이 아니다. |
| `select_rules.py` | 전 분석기의 `topk_count`, seeded tie-break, 독립 tie stream overlap 정본. 동점 점수가 전부 같을 때 기대 precision은 1이 아니라 chance다. |

### 판정·분석 (GPU 불필요, 기존 산출물 재집계)

| 파일 | 산출물 | 역할 |
|---|---|---|
| `judge.py` | judge-*.txt | 게이트 자동 판정 C1(one-sided 실패)·C1′(hybrid 회복)·C2(인증)·C3(downstream) |
| `reversal_freq.py` | REVERSAL.md | 프롬프트 단위 부호반전율 + **닻**(oracle split-half 자기 불일치) + 경계 대역 + McNemar 짝검정 + 불일치 경보 Fisher. v1(gate-*)·v2 겸용 |
| `kcurve_floor.py` / `kcurve_all.py` | KCURVE.md | micro_groups.pt 재조합으로 K′별 floor 정확 재계산 + Spearman-Brown 외삽 → 확장권고/구조적부재 판정 |
| `stats_extra.py` | STATS.md | run별 초기하 정확 p·bootstrap CI (A8a) |
| `frontier.py` | FRONTIER.md | 비용–품질 frontier: stale/passrate/random/fresh/audit/2dref 정책 비교 (표본 비공유 프로토콜) |
| `regime_map.py` | REGIME.json/CSV, FINAL_REPORT.md | fresh half A로 ranking하고 half B로만 평가해 random 대비 utility gain과 fresh-gain retention을 계산. `all/learnable/saturated`와 seed replication으로 usable/unsafe/unresolved를 집계 |
| `first_interval.py` | REGIME.json 내부 구간 | candidate micro-group과 validation prompt를 half별로 독립 재표집해 FIRST floor, fresh gain, stale gain, retention의 고정-pool 계층 bootstrap 구간을 계산. tie stream은 bootstrap draw 사이에 고정 |
| `reuse_behavior.py` | behavior_reuse.json | drift sweep의 source/target model·dataset·prompt hash·sampling manifest·exact-K를 대조하고 immutable behavior artifact만 복제 |
| `model_matrix.py` | `.om_snapshot.json` | 전이 모델의 immutable HF revision, tokenizer/chat template, 전체 safetensors shard hash와 LoRA target을 검증 |
| `qualify_domain_data.py` | `domain_dataset_qualification.json` | 전이 데이터 revision/hash/행 수/split 비중복과 MBPP·KK·ARC 실제 reward 경로를 검증 |
| `regime_contract.py` | `MATRIX.json`, validation marker | clean Git·config·model·data를 행렬에 고정하고 run 재개/격리, exact-K·score/oracle·최종 분석 hash 계약을 집행 |
| `transfer_smoke.py` | host별 runtime marker | BF16 CUDA와 실제 모델 생성, LoRA backward/save/reload/merge를 본 행렬 전에 실행 |
| `precheck_hard.py` | PRECHECK.md | go_hard GO/NO-GO 선판정 (P3-0에서 NO-GO → go_hard 폐기) |
| `make_tables.py` | TABLES.md | T1~T7 표 생성 (게이트·신호보존·floor 곡선·live fraction·hybrid·C2·downstream) |
| `readout_summary.py` | READOUT.md | 사람용 판독 요약 (한눈 표+자동 결론+원시 출력) |
| `score_artifacts.py` | 내부 계약 | oracle·4 estimator·split-half의 schema, finite 값, prompt ID coverage를 공통 검증 |
| `run_select.py` | 내부 계약 | 전 세대·legacy·protocol-only run의 공통 탐색과 미선택 사유 진단 |
| `show_selection.py` / `make_hard_pool.py` / `c2_diagnose.py` / `c2_sweep.py` | — | 보조 유틸 (선택 내역 출력 / hard 풀 구성 / C2 진단·스윕) |

### 2×2 추정량 표기 (전 코드 공통)

- `g00` 무보정 stale · `g10` prefix만 보정 · `g01` suffix(continuation)만 보정 · `g11` full IS
- hybrid 4셀: `bb`(β경로+β마무리) `bp`(β경로+π마무리) `pb`(π경로+β마무리) `pp`(π/π)
- **floor** = oracle split-half 일치도(독립 jitter 교정판이 정본 — 공유 jitter는 동점
  체제에서 부풀려짐), **닻** = oracle 자기 불일치율(반전율의 기준선)
- likelihood ratio는 raw model softmax와만 일치한다. 실행기는
  `temperature=1`, `top_p=1`, `top_k=0`, repetition penalty 없음만 허용한다.
- empirical `g00`~`g11`은 각 완성된 ratio product를 `[1/clip_cap, clip_cap]`으로
  자른 clipped variant다. LOO baseline은 unclipped 식에서는 소거되지만 clipping 뒤
  남는 항은 clipping bias의 일부다.

## 3. 주요 scripts 카탈로그

### 실행 (GPU)

| 스크립트 | 용도 |
|---|---|
| `go_additional.sh` | 4×H100 discovery 단일 진입점. 모델·전체 행렬을 고정하고 노드 싱글턴, GPU 해제 확인, 최대 12회 worker 자동 재시작을 담당한다 |
| `go_retry.sh` | legacy fixed-drift 복구 경로. 신규 regime/transfer 실험의 진입점으로 쓰지 않음 |
| `go_v4.sh` | 고정 slot v4 호환·재개 진입점. 즉시 `resume_v4.sh`로 위임해 run별 generation commit을 보존한다. GPU 해제 대기와 자동 집계는 하지 않음 |
| `go_v4_27b.sh` | 현재 코드의 27B 전용 공유 큐. FLA/GPU 스모크, current-commit 계약, quarantine, 10-run 검증과 마지막 worker 자동 집계를 수행 |
| `go_regime.sh` | 신규 주실험: 모든 클러스터에서 같은 명령을 실행하면 shared flock queue가 seed×dataset family를 분배. 한 behavior pool을 drift 0/25/100/400에 고정하고 완료 후 regime map을 단일 보고서로 집계 |
| `go_domain_transfer.sh` | 비Qwen·비수학 전이 행렬: Mistral/OLMo2 × MBPP/KK/ARC-Challenge를 고정 설정·host runtime smoke·불변 matrix 계약 아래 3-seed shared queue로 실행 |
| `go_v2.sh` | 모델별 worker: GPU 건강검사 → 스모크 게이트 → `SEEDS`×`DATASETS` 루프. console/stage 로그 무변화 watchdog, process-group 자동 종료·최대 재시도, 실패 원인 콘솔 진단 내장 |
| `run_14b.sh` | 단일 (seed,dataset) 실행기: config digest lock, GPU 자동감지, exact-K 병합 검증, 실패 전파, 최종 필수 artifact 검사 후 원자적 `DONE` 생성 |
| `go_new.sh` | **B11 최신 세대 검증**: 기본 Qwen3.8-27B(REPO27B로 교체 가능) 1-seed, 스냅샷 자동 fetch, `RUN_BASE`/`RESULTS_BASE`로 v2와 폴더 격리. rollout.py의 MM automap 폴백(CausalLM 실패 시 AutoModelForMultimodalLM)과 세트 |
| `go_full.sh`/`go_boost.sh`/`go_27b.sh`/`go_hard.sh` | 확장 스택 — **신규 착수 금지**(BACKLOG 폐기절, go_hard는 NO-GO 폐기) |

### 진단·데이터

| 스크립트 | 용도 |
|---|---|
| `diagnose.sh` | 멈춤 원인 원샷 리포트 |
| `diagnose_run_failure.sh` | smoke/run 실패 시 console 및 nested stage 오류, artifact 누락, 잔류 process, GPU 상태를 자동 출력·보존 |
| `gpu_check.sh` | matmul/SDPA 분리 판정 (ULF 계열) |
| `check_data.sh <dataset>` | 데이터 위치·스키마 자가진단 |
| `fetch_datasets.sh` | 데이터셋 수동 다운로드 |
| `prepare_domain_datasets.sh` | MBPP/KK/ARC-Challenge 고정 revision을 내려받고 SHA·행 수·split 결정성·train/validation 비중복을 검증 |
| `fetch_transfer_models.sh` | Mistral 7B/OLMo2 7B 고정 revision 스냅샷을 받아 config·chat template·shard·LoRA target 검사 |

### 수확·분석 (CPU)

| 스크립트 | 용도 |
|---|---|
| `harvest.sh` | **수확 원스톱**: v4가 있으면 2모델×5-seed×2데이터셋의 20-run 완결성을 먼저 검사한다. 사전등록 `KCURVE.md`와 전 세대 확장 `KCURVE_ALL.md`, READOUT·REVERSAL(닻·McNemar 포함)·STATS·TABLES·FRONTIER를 고유 폴더에 원자적으로 publish. 실패 stdout/stderr는 partial/error로 격리하고 nonzero 종료 |
| `collect_v4.sh` | **v4 결과 자동 취합**: 공유 `runs/`의 20개 run과 필수 산출물을 검사하고 27B·7B TABLES/FRONTIER를 staging에서 따로 생성·게시한 뒤 `harvest.sh` 실행. GPU 및 run 디렉터리는 건드리지 않음 |
| `resume_v4.sh` | **중단된 v4 안전 재개**: 완료 run은 스킵하고 각 미완료 `run_config.json`에 기록된 generation commit의 격리 worktree에서 run별로 재개. 커밋이 여러 개여도 서로 섞지 않으며 기존 shard/`.partial`을 재사용. 없는 commit은 자동 fetch하고 snapshot 진입 시 호출 checkout의 `OM_REPO`/`PYTHONPATH`를 제거 |
| `_report_io.sh` | 개별 보고서 I/O | read_now·K-curve·reversal 실행기의 고유 폴더 생성, nonempty 검사, 원자적 publish |
| `reversal_freq.sh`/`kcurve.sh`/`kcurve_all.sh`/`frontier.sh` | 개별 분석 러너 (harvest가 전부 포함하므로 단독 실행은 조기 확인용) |
| `read_now.sh` | judge 전체 출력 즉석 판독 |

## 4. 환경변수

| 변수 | 기본값 | 의미 |
|---|---|---|
| `SEEDS_ALL` (go_retry) / `SEEDS` (go_v2) | `0 1 2 3 4` | 돌릴 seed 목록. 노드 분산 시 겹치지 않게 배정 |
| `DATASETS` | `gsm8k dapo-math` | go_v2 데이터셋 목록 (mbpp·kk·arc-challenge 가능) |
| `REGIME_SEEDS` / `REGIME_DATASETS` / `REGIME_DRIFTS` | `0 1 2` / `gsm8k math500` / `0 25 100 400` | regime discovery matrix. confirmation은 결과 동결 뒤 별도 값으로 실행 |
| `REGIME_ROOT` / `REGIME_RESULTS` / `REGIME_MODEL_TAG` | `$OM_WORK/runs/...` / `$OM_WORK/results/...` / model basename | regime shared queue·artifact·모델별 결과 경로 |
| `REGIME_FIRST_BOOTSTRAP` / `REGIME_FIRST_CALIBRATION` | `2000` / 없음 | discovery 재표집 수와 coverage calibration JSON. 최종 freeze는 10000회와 passing calibration을 요구하며, 없으면 label은 `provisional_*` |
| `N_TRAIN`/`N_VAL` | 512/100 (math500은 400/100) | 프롬프트 수 |
| `OM_GPUS` | 전 GPU | 사용 GPU 제한 (`"0,1"` — 한 노드 두 실험 분할용) |
| `OM_WORK` | — | 작업 루트 (산출물 `$OM_WORK/results/v2/`, 로그 `$OM_WORK/console-logs/` — 레포에 로그 금지) |
| `OM_ATTN` | — | `eager` = fused SDPA 커널 hang(ULF) 우회 |
| `OM_SKIP_GPU_CHECK` | 0 | 점유 검사 무시 (비권장) |
| `HF_ENDPOINT` | — | H100 클러스터 HF 미러 (GitHub egress 없음) |
| `FORCE_HARD` | 0 | go_hard precheck 우회 (폐기됨) |

## 5. 산출물 구조 (run 디렉토리 = `runs/<이름>` 또는 `$OM_WORK/results/v2/v2-s<seed>[-<dataset>]`)

- `prompts.json` — train/val 분할 (샤드 병합 검증의 기준)
- `<stage>.jsonl`(+`.shard*.jsonl`) — rollout (token id·reward; logp는 score에서 재계산)
- drift adapter 폴더 — `adapter_config.json` 존재 = drift 완료 판정(스킵 기준)
- `micro_groups.pt` — kcurve 재조합용 micro-group gradient
- `report.json` — 게이트 수치 정본 (judge·make_tables가 읽음)
- `run_config.json` — 변경 불가 실행 설정 digest. 설정이 달라지면 같은 run 디렉토리 재사용 거부
- π-의존 산출물은 drift 재실행 시 격리 블록으로 오염 방지 (run_14b.sh)

## 6. 알려진 함정 (상세는 docs/TROUBLESHOOTING.md A~E절 21건+)

1. **ULF(GPU 커널 hang)**: 화면 heartbeat는 워처 생존 신호일 뿐 — **util 0% 지속**이
   진짜 판정. 재발 노드는 버리고 새 노드에서 재실행이 정답 (2026-08-18 실증:
   같은 명령을 새 노드에 나누자 즉시 정상 rollout).
2. 동점 jitter: 스칼라 점수 동점이 많은 체제(DAPO)에서 top-k가 jitter로 채워짐 —
   비영 프롬프트 10개 미만이면 결정 칸 해석 금지 (reversal_freq † 표기).
3. 공유 jitter floor 부풀림: split-half는 반드시 독립 jitter (교정값이 정본).
4. transformers 5.x `apply_chat_template` 반환형 — 텍스트로 뽑아 재토크나이즈.
5. LoRA `merge_and_unload()` 후 전 파라미터 requires_grad=False — `grad_params()`가 복원.
6. 생성 길이 128은 GSM8K `####` 도달 전 절단 — 384 필요.
7. 재시작 시 GPU 수 변경 — 샤드 및 기존 병합본의 prompt/exact-K 검증이 어긋남을 차단.
8. oracle 표본 재사용 누수(v1 hybrid 21/21 사건) — fresh는 항상 표본 비공유.
9. HF 스냅샷의 다중 설정 폴더(mbpp full/·sanitized/처럼 컬럼이 다른 parquet
   혼재) — 통짜 `load_dataset("parquet", ...)`이 CastError로 죽음.
   `_load_rows_any`가 폴더 그룹별 로드(full 우선)→파일별 병합 순으로 방어
   (2026-08-18, 실데이터 재현 검증). 리스트 필드가 문자열로 변형된 사본은
   `_maybe_json_list`가 복원, kk solution의 'knight'/'true' 문자열 변형은
   `_kk_truthy`가 수용. 스키마 실패 에러에 첫 행 필드명 자동 출력.
10. 과거 코드로 만든 `hybrid`, `scores_splithalf`, `report`, `DONE`은 정본이 아니다.
    behavior/fresh rollout과 micro-group 원자료는 유지할 수 있지만 이 네 계열은 수정
    코드를 pull한 뒤 다시 생성한다.
11. CertaGrad 비용은 candidate micro-group과 validation prompt-group의 rollout 크기가
    다르므로 단순 group 합이 아니라 각각 `micro_group`, `val_k`를 곱한 rollout 수로
    비교한다. Gaussian 반경은 모델 기반 근사이며 분포무관 인증으로 읽지 않는다.

## 7. 판정 체계 요약

- 게이트: **C1**(one-sided 실패 실증) · **C1′**(hybrid 축 교체 회복 = 인과) ·
  **C2**(인증 가능성 — 3태스크×2스케일 전패, 부정적 결과로 수록) ·
  **C3**(downstream 비열등)
- 유의선: **5-seed 일관 재현** (2-seed는 잠정). 원시 반전율은 닻 대비로만 서술,
  below-chance 단독 과판매 금지(부호검정이 닻).
