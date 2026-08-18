# CODE.md — 코드 상세 설명서

> **유지 규약**: 코드(src/·scripts/)를 바꾸는 커밋은 이 문서의 해당 절을 함께
> 갱신한다. 새 모듈·스테이지·환경변수·산출물이 생기면 여기에 추가하고,
> 삭제되면 여기서도 지운다. (사용자 지시 2026-08-18)

## 1. 한눈 개요

**질문**: 옛 정책 β가 만든 stale rollout으로 프롬프트 순위를 매길 때,
one-sided importance 보정(g10/g01)이 순위를 얼마나 망치는가.

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
  (`merge_rollouts`가 prompt 전수·무중복 검증 — GPU 수가 바뀐 재시작도 안전).

## 2. src/ 모듈 상세

### 실행 코어

| 파일 | 역할 |
|---|---|
| `experiment.py` (604줄) | 스테이지 오케스트레이터. `stage_score`(58): β rollout에 π/β logprob 재계산 → 2×2 추정량 채점. `stage_oracle`(153): π에서 fresh rollout 생성·oracle 점수. `stage_report`(255): 동점·무신호 집계 포함 판정 요약. `run_hybrid`(566): prefix 절단(cut 0.25/0.5/0.75) 후 축 교체 처치. `topk`(246): 동점 jitter 포함 top-k 선택. |
| `rollout.py` (222줄) | 모델 로드(`load_model` — `OM_ATTN=eager` 지원, ULF 대응), `rollout_prompts`(104): 프롬프트별 K개 생성 + reward 저장 (logp는 score 단계에서 재계산). `train_drift_lora`(154): **rejection FT** — reward>0.5 rollout만 SFT(LoRA r=16 α=32), 정답이 하나도 없으면 전체로 폴백(183~187). |
| `grads.py` (197줄) | 추정량 수학. `loo_advantages`(25): 그룹 leave-one-out advantage. `log_weights`/`token_weights`: prefix·suffix·token IS 가중치 (2×2의 실체). `prompt_gradient`(145)+`project_grads`(93): JL 투영 프롬프트 gradient. `grad_params`(126): LoRA merge 후 requires_grad 복원 함정 처리. |
| `data.py` (413줄) | 로더+보상. `load_prompts`(17): gsm8k / dapo-math / math500 / **mbpp** / **kk** / apps(데이터만). 로컬 우선 탐색(`_dataset_bases` 3곳: `$DATASETS_DIR`·`/group-volume/datasets`·사용자 폴더) → HF 폴백. 보상: 수학=최종 수치 매칭(`extract_answer`/`_boxed`), `_code_reward`(336)=테스트 실행 채점(mbpp), `_kk_reward`(385)=전원 신원 매치, `_apps_reward`(355, 하네스 미구현). |
| `hybrid.py` (127줄) | C1′ 인과 검증. `make_hybrid_cells`(57): β/π 경로 × β/π 마무리 = bb/bp/pb/pp 4셀 equal-K 독립 생성, `continue_rollouts_batch`(26): prefix 이어쓰기. |
| `train_downstream.py` (158줄) | C3 검증. `grpo_lite_train`(21): 선택된 프롬프트로 GRPO 간이판(LOO advantage) 학습, `eval_accuracy`(90): val 정확도, `selection_rollout_cost`(107): 선택 방식별 rollout 예산 집계. |
| `certagrad.py` (262줄) | C2 인증(부정적 결과로 강등됨). `certagrad`(73): 순차 top-k 인증(empirical-Bernstein 반경, `--radius-mode gaussian|hoeffding`), `angle_radius`(40): margin↔α_v 각도 비교. |

### 판정·분석 (GPU 불필요, 기존 산출물 재집계)

| 파일 | 산출물 | 역할 |
|---|---|---|
| `judge.py` | judge-*.txt | 게이트 자동 판정 C1(one-sided 실패)·C1′(hybrid 회복)·C2(인증)·C3(downstream) |
| `reversal_freq.py` | REVERSAL.md | 프롬프트 단위 부호반전율 + **닻**(oracle split-half 자기 불일치) + 경계 대역 + McNemar 짝검정 + 불일치 경보 Fisher. v1(gate-*)·v2 겸용 |
| `kcurve_floor.py` / `kcurve_all.py` | KCURVE.md | micro_groups.pt 재조합으로 K′별 floor 정확 재계산 + Spearman-Brown 외삽 → 확장권고/구조적부재 판정 |
| `stats_extra.py` | STATS.md | run별 초기하 정확 p·bootstrap CI (A8a) |
| `frontier.py` | FRONTIER.md | 비용–품질 frontier: stale/passrate/random/fresh/audit/2dref 정책 비교 (표본 비공유 프로토콜) |
| `precheck_hard.py` | PRECHECK.md | go_hard GO/NO-GO 선판정 (P3-0에서 NO-GO → go_hard 폐기) |
| `make_tables.py` | TABLES.md | T1~T7 표 생성 (게이트·신호보존·floor 곡선·live fraction·hybrid·C2·downstream) |
| `readout_summary.py` | READOUT.md | 사람용 판독 요약 (한눈 표+자동 결론+원시 출력) |
| `show_selection.py` / `make_hard_pool.py` / `c2_diagnose.py` / `c2_sweep.py` | — | 보조 유틸 (선택 내역 출력 / hard 풀 구성 / C2 진단·스윕) |

### 2×2 추정량 표기 (전 코드 공통)

- `g00` 무보정 stale · `g10` prefix만 보정 · `g01` suffix(continuation)만 보정 · `g11` full IS
- hybrid 4셀: `bb`(β경로+β마무리) `bp`(β경로+π마무리) `pb`(π경로+β마무리) `pp`(π/π)
- **floor** = oracle split-half 일치도(독립 jitter 교정판이 정본 — 공유 jitter는 동점
  체제에서 부풀려짐), **닻** = oracle 자기 불일치율(반전율의 기준선)

## 3. scripts/ 카탈로그 (52개 중 현역)

### 실행 (GPU)

| 스크립트 | 용도 |
|---|---|
| `go_retry.sh` | **표준 재시작 진입점**: gsm8k 프로브 → 전 seed·데이터셋 스윕(DONE 스킵). `SEEDS_ALL="3"`으로 seed 지정 |
| `go_v2.sh` | 본실행: GPU 건강검사 → 30분 스모크 게이트 → `SEEDS`×`DATASETS` 루프. **무출력 워처**(15분 단위, util>0이면 정상·0% 지속이면 hang) 내장 |
| `run_14b.sh` | 단일 (seed,dataset) 실행기: GPU 자동감지·`OM_GPUS` 분할·점유 검사·샤드 병합·preflight(HF stale lock 청소 포함)·keepalive |
| `go_full.sh`/`go_boost.sh`/`go_27b.sh`/`go_hard.sh` | 확장 스택 — **신규 착수 금지**(BACKLOG 폐기절, go_hard는 NO-GO 폐기) |

### 진단·데이터

| 스크립트 | 용도 |
|---|---|
| `diagnose.sh` | 멈춤 원인 원샷 리포트 |
| `gpu_check.sh` | matmul/SDPA 분리 판정 (ULF 계열) |
| `check_data.sh <dataset>` | 데이터 위치·스키마 자가진단 |
| `fetch_datasets.sh` | 데이터셋 수동 다운로드 |

### 수확·분석 (CPU)

| 스크립트 | 용도 |
|---|---|
| `harvest.sh` | **수확 원스톱**: KCURVE·READOUT·REVERSAL(닻·McNemar 포함)·STATS·TABLES·FRONTIER를 한 폴더에 동봉 → 그 폴더 하나만 전달 |
| `reversal_freq.sh`/`kcurve.sh`/`kcurve_all.sh`/`frontier.sh` | 개별 분석 러너 (harvest가 전부 포함하므로 단독 실행은 조기 확인용) |
| `read_now.sh` | judge 전체 출력 즉석 판독 |

## 4. 환경변수

| 변수 | 기본값 | 의미 |
|---|---|---|
| `SEEDS_ALL` (go_retry) / `SEEDS` (go_v2) | `0 1 2 3 4` | 돌릴 seed 목록. 노드 분산 시 겹치지 않게 배정 |
| `DATASETS` | `gsm8k dapo-math` | go_v2 데이터셋 목록 (mbpp·kk 가능) |
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
7. 재시작 시 GPU 수 변경 — 샤드 어긋남은 merge_rollouts 전수 검증이 방어.
8. oracle 표본 재사용 누수(v1 hybrid 21/21 사건) — fresh는 항상 표본 비공유.
9. HF 스냅샷의 다중 설정 폴더(mbpp full/·sanitized/처럼 컬럼이 다른 parquet
   혼재) — 통짜 `load_dataset("parquet", ...)`이 CastError로 죽음.
   `_load_rows_any`가 폴더 그룹별 로드(full 우선)→파일별 병합 순으로 방어
   (2026-08-18, 실데이터 재현 검증). 리스트 필드가 문자열로 변형된 사본은
   `_maybe_json_list`가 복원, kk solution의 'knight'/'true' 문자열 변형은
   `_kk_truthy`가 수용. 스키마 실패 에러에 첫 행 필드명 자동 출력.

## 7. 판정 체계 요약

- 게이트: **C1**(one-sided 실패 실증) · **C1′**(hybrid 축 교체 회복 = 인과) ·
  **C2**(인증 가능성 — 3태스크×2스케일 전패, 부정적 결과로 수록) ·
  **C3**(downstream 비열등)
- 유의선: **5-seed 일관 재현** (2-seed는 잠정). 원시 반전율은 닻 대비로만 서술,
  below-chance 단독 과판매 금지(부호검정이 닻).
