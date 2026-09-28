# 리뷰 대비 실험 명령 모음

확인: 2026-09-28, 실행 코드 기준 `f4da754` / `master`.
[리뷰 일정](REVIEW_SCHEDULE_2027_KO.md) ·
[실험별 목적·우선순위](LIMITATION_EXPERIMENTS_KO.md) ·
[Qwen 감사 결과](QWEN35_SRGC_AUDIT_KO.md).
아래는 실행 명령 목록이며, 원격 GPU 작업의 완료/진행을 확인한 기록은 아니다.

## 1. 무엇을 실행하는지

| ID / 순위 | 실험 | 예정 continuation | 배정 방식 / 선행 조건 |
| --- | --- | --- | --- |
| E01 / P0 | OLMo MATH, seeds 5–9 × 네 arm | 20 | 자동 queue; cache → prefix → arm |
| E02 / P0 | OLMo MBPP, seeds 5–9 × 네 arm | 20 | 자동 queue; cache → prefix → arm |
| E03 / P1 | 동일 prefix의 SR/Switch 독립 반복 | 반복 수 사전 확정 필요 | **미구현, 실행 명령 없음** |
| E04 / P1 | total-step-200 고정 전환 | 전용 plan 확정 필요 | **미구현, 실행 명령 없음** |
| E05 / P2 | 후보 40개 SR 갱신 | MATH 5 + MBPP 5 | seed별 수동 배정; 해당 prefix 완료 |
| E06 / P2 | 반복 전환 | MATH 5 + MBPP 5 | seed별 수동 배정; 해당 prefix 완료 |
| E07 / P3 | pool 400개 SR 갱신 | MATH 5 + MBPP 5 | seed별 수동 배정; 해당 prefix 완료 |
| E08 / P3 | Qwen3.5-9B 온라인 네 arm | MATH 20 + MBPP 20 | 별도 환경·v2 root; 실제 GPU admission 통과 |
| E09 / P3 | 초기 gradient 방향 matched ablation | 전용 설계 필요 | **미구현, 실행 명령 없음** |

구현된 신규 온라인 continuation은 전체 범위 기준 **110개**다
(OLMo 40 + 추가 arm 30 + Qwen 40). cache/prefix 작업과 과거 진단은 이 수에
포함하지 않으며, 남은 작업 수라는 뜻도 아니다. E03/E04가 준비되면 P3보다
먼저 자원을 배정한다. 이미 정상 실행 중인 작업은 유지한다.

## 2. 모든 명령의 공통 준비

코드 레포 루트에서 실행한다. 현재 PC의 위치는
`/home/kms/dev/offpolicy-misranking`; 노드에서는 해당 노드의 checkout을 사용한다.
**실행 중인 checkout이나 공유 환경을 업데이트하지 않는다.** 아래 갱신은
worker가 사용하지 않는 checkout에서만 한다.

```sh
git status --short
git pull --ff-only origin master
git rev-parse HEAD
```

각 노드에서 실제 group mount와 기존 Python 경로에 맞춰 설정한다.
기존 cohort를 이어받을 때는 이전 `OM_WORK`/root 설정을 유지한다.

```sh
export GROUP_VOLUME=/group-volume
export OM_USER=minsoo3.kim
export OM_WORK="$GROUP_VOLUME/$OM_USER/offpolicy-misranking"
export PAIR_PYTHON="$OM_WORK/.venv-cu126/bin/python"
export SWITCH_PYTHON="$OM_WORK/.venv-cu126/bin/python"
export QWEN_PYTHON="$OM_WORK/.venv-qwen35/bin/python"

test -d "$GROUP_VOLUME"
df -h "$GROUP_VOLUME"
nvidia-smi
```

한 GPU 작업은 **4×H100 80GB 노드 하나**를 사용한다. 여러 노드는 독립 작업을
나눠 받는다. 같은 노드에 worker를 여러 개 띄우지 않는다. 특히 기존
`run_srgc.sh ... run`에는 같은 사용자의 GPU 프로세스를 정리하는 동작이 있으므로
**다른 학습이 없는 할당 노드에서만** 실행한다. 상태 조회용 터미널에서 `run`을
다시 입력하지 않는다. 잠금 파일 삭제, `--fresh`, `SRGC_RUN_NAME` 변경은 재개 방법이 아니다.

## 3. E01/E02 — OLMo 기본 네 arm

### 시작 / checkpoint에서 계속

MATH 노드에서는 첫 번째, MBPP 노드에서는 두 번째 줄을 사용한다.
같은 데이터셋의 다른 빈 노드에서도 동일 명령을 쓰면 queue가 서로 다른 작업을 배정한다.

```sh
sh scripts/run_srgc.sh math run
sh scripts/run_srgc.sh mbpp run
```

두 데이터셋을 한 worker가 순서대로 처리하게 하려면 아래 명령 **하나**를 쓴다.
MATH에서 배정할 작업이 없으면 MBPP를 확인한다.

```sh
sh scripts/run_srgc.sh all run
```

### 상태 / 결과 / 비용 / 백업

```sh
sh scripts/run_srgc.sh all status
sh scripts/run_srgc.sh all results
sh scripts/run_srgc.sh all costs
sh scripts/run_srgc.sh all backup
```

`all`을 `math` 또는 `mbpp`로 바꾸면 하나만 조회한다. `backup`은 저장된
checkpoint를 한 번 복사한다. 지속 백업 감시가 따로 필요하면 데이터셋별로
실행한다. `backup-watch` 자체는 학습을 진행시키지 않는다.

```sh
sh scripts/run_srgc.sh math backup-watch
sh scripts/run_srgc.sh mbpp backup-watch
```

실제 학습·cache·receipt는 status가 출력하는 group-volume active root를 따른다.
OLMo TXT 요약은 기존 동작상 홈에도 복사되므로 홈 TXT만 결과 원본으로 보관하지 않는다.
`all backup-watch`는 첫 watch가 계속 실행되는 동안 둘째로 넘어가지 않으므로 사용하지 않는다.

### 정상 중단 / 중단 표시 해제 / 실패 작업 재개

`stop`은 해당 데이터셋의 현재 작업을 마친 뒤 신규 배정을 멈춘다.
즉시 중단하려면 해당 줄 끝에 `--now`를 추가한다.

```sh
"$PAIR_PYTHON" scripts/run_srgc_rebuttal.py stop --dataset math
"$SWITCH_PYTHON" scripts/run_srgc_rebuttal.py stop --dataset mbpp
```

`resume`은 공유 중단 표시만 해제한다. 이후 빈 노드에서 위 `run` 명령을 다시 실행한다.

```sh
"$PAIR_PYTHON" scripts/run_srgc_rebuttal.py resume --dataset math
"$SWITCH_PYTHON" scripts/run_srgc_rebuttal.py resume --dataset mbpp
```

일반 shell `run`은 실패 작업을 120초 간격, 기본 최대 50회까지 재시도한다.
같은 오류가 반복되면 중단 후 로그를 확인한다. 새 실행에서 상한을 3회로 두려면:

```sh
SRGC_MAX_ATTEMPTS=3 sh scripts/run_srgc.sh math run
```

상한 변경은 누적 attempt 기록을 초기화하지 않는다. 코드를 고치거나 원인을 해결한
다음 재개하며, hash mismatch를 무시해 기존 실험에 다른 설정을 섞지 않는다.

## 4. E05/E06/E07 — 추가 arm 30개

기본 네-arm queue와 별도다. 각 seed의 **공통 prefix가 완료되어야** 한다.
노드별로 서로 다른 `(dataset, seed, arm)`을 배정한다. 아래는 seed 5 예시이며
5를 6, 7, 8, 9로 바꾼다. 한 줄이 한 작업이다.

```sh
# E05: 후보 40문제 SR 갱신
sh scripts/run_srgc_sr_refresh.sh math 5 candidates
sh scripts/run_srgc_sr_refresh.sh mbpp 5 candidates

# E06: SR 상태에서도 점검을 계속하는 반복 전환
sh scripts/run_srgc_sr_refresh.sh math 5 switch_repeat
sh scripts/run_srgc_sr_refresh.sh mbpp 5 switch_repeat

# E07: 전체 400문제 SR 갱신
sh scripts/run_srgc_sr_refresh.sh math 5 pool
sh scripts/run_srgc_sr_refresh.sh mbpp 5 pool
```

**30개 전체 명령과 노드 배정 칸:** [추가 arm 배정표](REBUTTAL_EXTRA_TASKS.tsv).
이 표는 자동 실행 파일이 아니다. `pending_prefix_check`는 prefix 미확인 상태다.
서로 다른 노드가 같은 줄을 선택하지 않도록 node/상태/로그를 기록한다.

같은 명령으로 마지막 저장에서 재개하며 저장 간격은 **25 updates**다.
완료된 동일 실험은 재학습하지 않는다. 별도 `stop/resume/status/costs` 하위 명령은 없다.
중단이 필요하면 해당 실행 터미널의 Ctrl-C를 사용하고 마지막 저장 이후 재수행을 예상한다.

```sh
sh scripts/run_srgc_sr_refresh.sh math results
sh scripts/run_srgc_sr_refresh.sh mbpp results
```

상태는 콘솔과 active root의 `seed-N/<arm>-progress.json`,
`<arm>-run.json`; 완료는 `<arm>-endpoint.json`을 확인한다.
arm 파일명은 `sr_refresh`, `switch_repeat`, `sr_refresh-pool`이다.
상세 비용은 `seed-N/cost-receipts/<arm>/`, `invocations/<arm>/`에 있다.
위 results의 selection 요약만으로 전체 GPU 비용을 계산하지 않는다.

## 5. E08 — Qwen3.5-9B, 별도 v2 실험

OLMo Python을 업그레이드하지 않는다. 별도 Qwen CUDA 환경 준비는
[Qwen 실행 안내](QWEN35_SRGC_KO.md)와
[패키지 목록](../configs/srgc_qwen35/requirements.txt)을 따른다.
대상은 **post-trained `Qwen/Qwen3.5-9B`**, Base 모델이 아니다.

```sh
export SRGC_QWEN_ROOT="$OM_WORK/srgc-rebuttal/qwen35-9b-v2"
sh scripts/run_srgc_qwen35.sh all download
sh scripts/run_srgc_qwen35.sh all doctor
sh scripts/run_srgc_qwen35.sh all prepare
```

다운로드·입력 준비 후 각 빈 4-H100 노드에서 동일 명령을 실행한다.
GPU admission을 통과하면 cache → prefix → 네 arm을 자동 배정한다.
MATH/MBPP 각각 seeds 5–9이며 OLMo reward cache/prefix를 재사용하지 않는다.

```sh
sh scripts/run_srgc_qwen35.sh all run
```

```sh
sh scripts/run_srgc_qwen35.sh all status
sh scripts/run_srgc_qwen35.sh all results
sh scripts/run_srgc_qwen35.sh all stop
```

중단 표시를 해제한 뒤 다시 실행한다. 즉시 중단은 `all stop --now`다.
일반 `run`이 실패 작업을 자동 재시도하도록 설정되어 있지는 않다.
실패 원인을 수정한 뒤에만 `--retry-failed`를 사용한다.

```sh
sh scripts/run_srgc_qwen35.sh all resume
sh scripts/run_srgc_qwen35.sh all run --retry-failed --max-attempts 3
```

모든 명령의 `all`을 `math`/`mbpp`로 바꿀 수 있다. 별도 `costs` 명령은 없고
`results`에 비용 보고가 포함된다. plan은 `$SRGC_QWEN_ROOT/experiments/`,
실험 산출물은 `runs/{math,mbpp}/seed-N/`, TXT는 `reports/`에 저장된다.
checkpoint 백업은 worker가 자동 수행한다. v1 plan/cache/checkpoint는 v2에서
재개하지 않는다. 9B GPU 실행은 현장 admission 검증이 남아 있다.

## 6. 여러 노드 배정과 결과 수집

| 노드 용도 | 입력할 명령 | 같은 명령을 여러 노드에서 실행 |
| --- | --- | --- |
| OLMo MATH | `sh scripts/run_srgc.sh math run` | 가능: 공유 queue |
| OLMo MBPP | `sh scripts/run_srgc.sh mbpp run` | 가능: 공유 queue |
| OLMo 통합 | `sh scripts/run_srgc.sh all run` | 가능: MATH 우선, 이어 MBPP |
| 추가 arm | 배정표의 서로 다른 한 줄 | 자동 배정 아님: tuple별 배정 필요 |
| Qwen | `sh scripts/run_srgc_qwen35.sh all run` | 가능: 별도 Qwen 공유 queue |

동일 모델의 노드들은 같은 group mount, active root, 코드와 실행 환경을 사용한다.
GPU lock이 있으면 실제 owner/heartbeat/task 로그를 확인하며 파일을 지우지 않는다.
필요 노드 수는 고정이 아니다. 1개로 순차 실행할 수 있으며 빈 노드가 늘면
준비된 작업을 병렬 처리한다. 실행 시간은 첫 완료 작업의 실측 후 갱신한다.

결과 수집은 다음 여섯 항목으로 완료 여부를 판단한다:

1. 예정 seed와 모든 비교 arm의 같은 total-step endpoint.
2. 각 Switch의 자기 경로 D/check/전환 기록.
3. seed별 paired reward 차이와 전체 seed 변동. 질문 bootstrap과 seed 변동을 구분.
4. cache/prefix/selection/training/evaluation/저장 비용 receipt; 미계측은 unknown.
5. code commit, plan/input/prefix hash, node, 시작·종료 시각, 실패·재시도 기록.
6. V6 제출 결과와 새 결과를 별도 표로 유지. 모델·프로토콜이 다른 실행을 합산 평균하지 않음.

## 7. 과거 실험·진단 명령 위치

현재 온라인 실험과 과거 진단은 별개다. 아래는 필요 시 과거 결과를 확인하거나
해당 frozen protocol을 보충할 때 찾을 위치이며, E01–E09 완료 수에 합치지 않는다.

| 계열 | 실행 / 조회 진입점 | 상세 명세 |
| --- | --- | --- |
| 과거 seed 3/4 고정 D | `bash scripts/run_srgc_newseeds.sh run`, `status`, `results` | [전체 실험 명세](EXPERIMENTS_COMPLETE_GUIDE_KO.md) |
| 보존 checkpoint의 SR-GC 재측정 | `bash scripts/run_selector_pair_srgc_repeat.sh all-measure`, `all-results` | [스크립트](../scripts/run_selector_pair_srgc_repeat.sh); E03의 학습 반복이 아님 |
| Pair continuation | `bash scripts/run_selector_pair.sh`, `bash scripts/run_selector_pair.sh status`, `bash scripts/run_selector_pair_results.sh` | [전체 실험 명세 11절](EXPERIMENTS_COMPLETE_GUIDE_KO.md#11-selector-pair-실측-h-보강) |
| RLOO | `bash scripts/run_rloo.sh run`, `status`, `results` | [전체 실험 명세 12절](EXPERIMENTS_COMPLETE_GUIDE_KO.md#12-rloo-학습-방식만-바꾼-후속-비교) |
| 과거 MBPP selection/switching | `bash scripts/run_mbpp_experiments.sh`, `status`, `results` | [전체 실험 명세 10절](EXPERIMENTS_COMPLETE_GUIDE_KO.md#10-mbpp-본-조건과-보완-cohort) |
| MBPP off-policy 진단 | `bash scripts/run_mbpp_offpolicy.sh run`, `status`, `results` | [전체 실험 명세 6절](EXPERIMENTS_COMPLETE_GUIDE_KO.md#6-아까-추가한-mbpp-off-policy-calibration) |
| reference axes | `bash scripts/run_reference_axes.sh math500 0 --plan` / `--check` / `--run` | [reference 명세](REFERENCE_AXES_RUN.md); `math500`/`mbpp`, replicate 0–4 |
| reliability budget | `bash scripts/run_reliability_budget.sh math500 64 32 100` | [스크립트](../scripts/run_reliability_budget.sh); dataset, fresh_k, val_k, seed |
| 기존 Qwen selection 매트릭스 | `scripts/run_qwen35_9b.sh` | [과거 Qwen runbook](QWEN35_9B_RUNBOOK.md); E08 온라인 전환과 다름 |
| 그 외 모델·MoPPS·gate·E1–E6·CPU 분석 | [전체 진입점 목록](EXPERIMENTS_COMPLETE_GUIDE_KO.md#18-실행결과-명령과-파일) | 각 실험의 옵션·frozen protocol 유지 |

E03/E04/E09는 기존 실행을 다시 호출해서 만들 수 없다. 구현·검증 후 전용
명령과 출력 root를 이 문서에 추가한다. 이 문서 작성으로 작업을 자동 시작하거나
주기적 실행을 등록하지 않았다.
