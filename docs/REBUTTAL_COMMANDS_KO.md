# 리뷰 대비 실험 명령 모음

확인: 2026-09-30, 실행 코드 기준 `997cab9` / `master`.
[리뷰 일정](REVIEW_SCHEDULE_2027_KO.md) ·
[기존 실험과 결과](EXPERIMENT_RESULTS_LEDGER_KO.md) ·
[실험별 목적·우선순위](LIMITATION_EXPERIMENTS_KO.md) ·
[Qwen 감사 결과](QWEN35_SRGC_AUDIT_KO.md).
아래는 실행 명령 목록이며, 원격 GPU 작업의 완료/진행을 확인한 기록은 아니다.

**완료 목표:** E01–E10의 전체 실험과 비용 검증을 마치고, 결과를 반영한 V7
원고를 **2026-11-04 18:00 KST까지** 준비한 뒤 11월 5일 리뷰 공개를 맞는다.
E10은 실험 목록 MD가 `sr_refresh`의 순수 갱신 효과 분리에 필요하다고 적은 cached-SR 유지 간격 대조다.
P0–P3는 자원 배정 순서이며 P3를 생략한다는 뜻이 아니다.
CPU/GPU 구분과 최대 동시 노드 수는 [6절](#6-여러-노드-배정과-결과-수집)에 있다.

## 1. 무엇을 실행하는지

| ID / 순위 | 실험 | 예정 continuation | 배정 방식 / 선행 조건 |
| --- | --- | --- | --- |
| E01 / P0 | OLMo MATH, seeds 5–9 × 네 arm | 20 | 자동 queue; cache → prefix → arm |
| E02 / P0 | OLMo MBPP, seeds 5–9 × 네 arm | 20 | 자동 queue; cache → prefix → arm |
| E03 / P1 | 동일 prefix의 SR/Switch 독립 반복 | MATH 20 + MBPP 20 (k=1,2 × sr/switch × 5 seeds) | `replicate<k>-sr`, `replicate<k>-switch`; seed별 수동 배정, 해당 prefix 완료 |
| E04 / P1 | total-step-200 고정 전환 | MATH 5 + MBPP 5 | `switch_fixed200`; seed별 수동 배정, 해당 prefix 완료 |
| E05 / P2 | 후보 40개 SR 갱신 | MATH 5 + MBPP 5 | seed별 수동 배정; 해당 prefix 완료 |
| E06 / P2 | 반복 전환 | MATH 5 + MBPP 5 | seed별 수동 배정; 해당 prefix 완료 |
| E07 / P3 | pool 400개 SR 갱신 | MATH 5 + MBPP 5 | seed별 수동 배정; 해당 prefix 완료 |
| E08 / P3 | Qwen3.5-9B 온라인 네 arm | MATH 20 + MBPP 20 | 별도 환경·v2 root; 실제 GPU admission 통과 |
| E09 / P3 | 초기 gradient 방향 matched ablation | MATH 15 + MBPP 15 (3조건 × 5 seeds) | `direction_removed`, `direction_magnitude`, `direction_replaced`; seed별 수동 배정, 해당 prefix 완료 |
| E10 / P3 | 같은 유지 간격의 cached-SR 대조 | MATH 5 + MBPP 5 | `sr_hold`; seed별 수동 배정, 해당 prefix 완료 |

구현된 신규 온라인 continuation은 전체 범위 기준 **200개**다
(OLMo 40 + 추가 arm 40 + Qwen 40 + E03 40 + E09 30 + E10 10). cache/prefix 작업과 과거 진단은 이 수에
포함하지 않으며, 남은 작업 수라는 뜻도 아니다. E04는 prefix 준비 후 P1로
배정한다. E03/E09/E10은 2026-09-30 `997cab9`에 구현했고 GPU 결과는 없다.
저장된 gradient의 방향·크기 분석은 CPU에서 가능하지만 E09의 matched ablation을
대체하지 않는다([6.2절](#62-cpu에서-할-일--gpu가-필요한-일)). 과거 완료 실험을 전부 재실행한다는 뜻은 아니다.
이미 정상 실행 중인 작업은 유지한다.

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
상태·결과 조회와 CPU 분석에는 `nvidia-smi`나 GPU 할당이 필요 없다.

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

기본 plan은 seeds 5–9, 공통 prefix 25 updates, total step 275다.
On-policy는 25 updates마다 후보 40문제를 다시 선정·채점하고 상위 4문제를 다음
갱신까지 학습에 사용한다. 기존 seed-3/4 데이터·캐시를 읽을 수 있으면 이를
가져와 cache 생성 단계를 건너뛰고, 없으면 checkout의 준비된 입력으로 생성한다.

같은 실험 코드이면 위 명령으로 기존 checkpoint를 이어받는다. 실험 코드의
identity가 달라졌으면 이전 결과를 보존하고 **새 cohort를 자동 생성하거나 합류**한다.
`[new-run]`과 `automatic-restart.json`에 이전·새 경로가 남는다. 이 경우 기존
학습 경로의 재개가 아니다. 새 cohort는 자체 생성 캐시를 다시 만들지만 출처가
검증된 과거 Pair 캐시 import는 유지한다. 모든 참여 노드의 코드와 환경을 맞춘다.

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
worker 자체가 checkpoint 백업을 수행하므로 일반 실행에 별도 watch는 필요 없다.
`status`의 `serving mbpp:...` 등은 통합 worker가 다른 데이터셋을 처리 중이라는 뜻이다.

### 정상 중단 / 중단 표시 해제 / 실패 작업 재개

`stop`은 해당 데이터셋의 모든 worker에 적용되며 현재 작업을 마친 뒤 신규 배정을
멈춘다. 통합 worker는 다른 데이터셋을 계속 처리하므로 둘 다 멈추려면 두 줄을
실행한다. 즉시 중단하려면 해당 줄 끝에 `--now`를 추가한다. 실행 중인 작업이
중단되면 그 통합 worker도 종료되지만 다른 데이터셋의 공유 중단 표시는 설정되지 않는다.

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

## 4. E03–E07, E09, E10 — 추가 arm 120개

기본 네-arm queue와 별도다. 각 seed의 **공통 prefix가 완료되어야** 한다.
노드별로 서로 다른 `(dataset, seed, arm)`을 배정한다. 아래는 seed 5 예시이며
5를 6, 7, 8, 9로 바꾼다. 한 줄이 한 작업이다.

```sh
# E04: total step 200까지 On-policy, update 201부터 SR
sh scripts/run_srgc_sr_refresh.sh math 5 switch_fixed200
sh scripts/run_srgc_sr_refresh.sh mbpp 5 switch_fixed200

# E05: 후보 40문제 SR 갱신
sh scripts/run_srgc_sr_refresh.sh math 5 candidates
sh scripts/run_srgc_sr_refresh.sh mbpp 5 candidates

# E06: SR 상태에서도 점검을 계속하는 반복 전환
sh scripts/run_srgc_sr_refresh.sh math 5 switch_repeat
sh scripts/run_srgc_sr_refresh.sh mbpp 5 switch_repeat

# E07: 전체 400문제 SR 갱신
sh scripts/run_srgc_sr_refresh.sh math 5 pool
sh scripts/run_srgc_sr_refresh.sh mbpp 5 pool

# E03: 같은 prefix의 독립 반복. k=1,2; 같은 k의 sr/switch가 한 쌍(공통 sampling stream)
sh scripts/run_srgc_sr_refresh.sh math 5 replicate1-sr
sh scripts/run_srgc_sr_refresh.sh math 5 replicate1-switch
sh scripts/run_srgc_sr_refresh.sh math 5 replicate2-sr
sh scripts/run_srgc_sr_refresh.sh math 5 replicate2-switch
sh scripts/run_srgc_sr_refresh.sh mbpp 5 replicate1-sr
sh scripts/run_srgc_sr_refresh.sh mbpp 5 replicate1-switch

# E09: 방향 정보만 제거/대체한 On-policy 대조 3조건
sh scripts/run_srgc_sr_refresh.sh math 5 direction_removed
sh scripts/run_srgc_sr_refresh.sh math 5 direction_magnitude
sh scripts/run_srgc_sr_refresh.sh math 5 direction_replaced
sh scripts/run_srgc_sr_refresh.sh mbpp 5 direction_removed

# E10: 갱신 없이 유지 간격만 On-policy와 맞춘 cached-SR 대조
sh scripts/run_srgc_sr_refresh.sh math 5 sr_hold
sh scripts/run_srgc_sr_refresh.sh mbpp 5 sr_hold
```

E03의 replicate stream은 `sampling_seed(base seed, k)`로 코드에 고정되어 있고
`seed-N/replicate-<k>/replicate.json`에 기록된다. 기록된 arm은 stream 0이며 `replicate0`은
거부된다. 같은 stream에서 고정 전환도 돌릴 수 있다(`replicate1-switch_fixed200`)만 배정표의
사전 고정 목록에는 넣지 않았다. E09의 세 조건은 On-policy와 후보 추출·scoring·유지 간격·
training stream을 공유하며 선별 규칙만 다르다. `sr_refresh − sr_hold`가 갱신 효과,
`sr_hold − sr`가 유지 간격 효과다.

E04의 200은 prefix 이후 추가 update 수가 아니라 **전체 학습 step**이다.
checkpoint 200의 selection 비용을 포함하고, SR-GC 부호로 전환 시점을 바꾸지 않는다.
구 경계 구현의 checkpoint는 현재 `fixed-boundary-before-training-v2`에서 재개할 수 없다.

**120개 전체 명령과 노드 배정 칸:** [추가 arm 배정표](REBUTTAL_EXTRA_TASKS.tsv).
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
arm 파일명은 `switch_fixed200`, `sr_refresh`, `switch_repeat`, `sr_refresh-pool`, `sr_hold`,
`direction_removed`, `direction_magnitude`, `direction_replaced`이다. 독립 반복은
`seed-N/replicate-<k>/` 아래에 원래 arm 이름(`sr`, `switch`)으로 같은 파일을 둔다.
상세 비용은 `seed-N/cost-receipts/<arm>/`, `invocations/<arm>/`(replicate는 그 폴더 아래)에 있다.
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
`prepare`는 그 시점의 활성 OLMo plan에서 질문 분할을 가져온다. 비교할 OLMo
cohort를 먼저 확정하며, 다른 cohort를 지정하는 `--source-plan` 예시는
[Qwen 실행 안내](QWEN35_SRGC_KO.md)에 있다.

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

### 6.1. 실험별 GPU와 최대 동시 노드

현재 runner는 **작업 하나 = 노드 하나 = GPU 4장**이다.
`torchrun --standalone --nproc_per_node=4`와 `world_size=4`를 사용하므로
한 작업을 여러 노드로 분할하거나 GPU 1장씩으로 쪼개 실행하지 않는다.
기준 할당은 4×H100 80GB이며 Qwen 9B의 실제 GPU 메모리는 admission 검증이 남아 있다.

아래 최대치는 **서로 독립인 미완료 작업 수로 계산한 병렬 상한**이다.
실제 장비 확보량, 공유 볼륨 처리량 또는 해당 규모에서 검증된 실행 성능이 아니다.
prefix 완료·cache 재사용 상태와 이미 끝난 arm 수에 따라 현재 활용 가능한 노드는 줄어든다.

| 실험 | 실행 장치 | 작업당 할당 | cache/prefix만 진행하는 초기 단계 상한 | 모든 해당 prefix 준비 후 continuation 상한 |
| --- | --- | --- | --- | --- |
| E01 OLMo MATH | GPU + 노드 CPU | 1노드 / 4 GPU | 5노드 / 20 GPU | **20노드 / 80 GPU** |
| E02 OLMo MBPP | GPU + 노드 CPU | 1노드 / 4 GPU | 5노드 / 20 GPU | **20노드 / 80 GPU** |
| E04 고정 total-step-200, 두 데이터셋 | GPU + 노드 CPU | 1노드 / 4 GPU | 별도 prefix 생성 없음; E01/E02 prefix 대기 | **10노드 / 40 GPU** |
| E05 후보 SR 갱신, 두 데이터셋 | GPU + 노드 CPU | 1노드 / 4 GPU | 별도 prefix 생성 없음; E01/E02 prefix 대기 | **10노드 / 40 GPU** |
| E06 반복 전환, 두 데이터셋 | GPU + 노드 CPU | 1노드 / 4 GPU | 동일 | **10노드 / 40 GPU** |
| E07 pool SR 갱신, 두 데이터셋 | GPU + 노드 CPU | 1노드 / 4 GPU | 동일 | **10노드 / 40 GPU** |
| E08 Qwen MATH+MBPP | GPU + 노드 CPU | 1노드 / 4 GPU | 10노드 / 40 GPU | **40노드 / 160 GPU**; 데이터셋당 20노드 |
| E03 독립 반복, 두 데이터셋 | GPU + 노드 CPU | 1노드 / 4 GPU | 별도 prefix 생성 없음; E01/E02 prefix 대기 | **20노드 / 80 GPU** (k=1,2 × sr/switch × 5 seeds × 2) |
| E09 방향 대조, 두 데이터셋 | GPU + 노드 CPU | 1노드 / 4 GPU | 동일 | **30노드 / 120 GPU** (3조건 × 5 seeds × 2) |
| E10 cached-SR 유지 간격 대조, 두 데이터셋 | GPU + 노드 CPU | 1노드 / 4 GPU | 동일 | **10노드 / 40 GPU** |

- **현재 구현분:** OLMo 기본 40 + 추가 arm 40 + Qwen 40 + E03 40 + E09 30 + E10 10 = 최대 **200노드 / 800 GPU**.
  전부의 prefix가 준비되고 continuation이 남아 있다는 가정이다. 200노드가 필요하다는 뜻은 아니다.
- 모두 처음부터 시작하여 아직 continuation이 준비되지 않은 경우, cache/prefix 선행 작업은
  OLMo 10 + Qwen 10 = 최대 **20노드 / 80 GPU**다. prefix 완료에 따라 arm으로 병렬성이 늘어난다.
  한 seed의 cache와 prefix를 동시에 별도 노드에 세지 않는다.
- E03의 반복 수 R=2와 E09의 조건 수 C=3은 2026-09-30에 사전 고정했다. 반복 하나를 더하면
  20노드, 조건 하나를 더하면 10노드가 늘어난다.
  이 숫자를 현재 실행 가능 작업 수나 검증된 노드 규모로 쓰지 않는다.

CPU 코어 수와 시스템 RAM의 실측 최소치는 아직 없다. `4 GPU`는 `4 CPU cores`라는
뜻이 아니며, GPU 노드의 CPU는 tokenizer·보상 검증·MBPP 실행·checkpoint 복사를 함께
처리한다. 노드별 첫 실행에서 CPU/RAM, GPU peak, 읽기·쓰기 시간과 실패 여부를 기록한
뒤 추가 노드를 투입한다. 현장 측정 없이 vCPU/RAM 최소치를 확정하지 않는다.

### 6.2. CPU에서 할 일 / GPU가 필요한 일

| 작업 | GPU | CPU 실행 및 병렬 방식 |
| --- | --- | --- |
| 코드 검사·단위 테스트·plan 검사 | 0 | CPU 작업 공간에서 수행; 실제 GPU admission은 별도 |
| Qwen weight/tokenizer 다운로드, 입력 `prepare`, 패키지 `doctor` | 0 | CPU와 네트워크·group 저장소 사용; 공통 모델 다운로드는 한 번 |
| reward cache 새 생성 | 작업당 4 | 모델 rollout이므로 CPU 분석 작업으로 분류하지 않음 |
| prefix·모든 continuation·실행 중 평가 | 작업당 4 | GPU 노드에서 CPU 보상 검증도 함께 수행 |
| On-policy gradient scoring, SR refresh 응답 생성 | 작업당 4 | 캐시 조회나 D 벡터 산술만을 전체 scoring과 혼동하지 않음 |
| `status/results/costs`, 저장된 결과 통계·그림 | 0 | group 접근 가능한 CPU 노드/작업 공간 한 곳에서 수집 가능 |
| checkpoint `backup`/백업 감시 | 0 | 파일 I/O; GPU 작업 수에 추가하지 않음. 모델이 메모리에만 있으면 백업할 수 없음 |
| V7 표·본문·부록 수정, LaTeX build·PDF 점검 | 0 | CPU 작업 공간에서 모든 실험 결과를 순차 반영 |

CPU 작업은 GPU가 모두 찬 동안에도 병행한다. 별도 CPU 노드 1개에서 수집·분석·원고
작성을 모아 처리할 수 있고, 반드시 추가 서버가 필요한 것은 아니다. GPU admission,
실제 새 모델 평가 및 새로운 rollout은 CPU에서 완료한 것으로 표시하지 않는다.

저장된 SR-GC의 방향·크기 분해는 다음 명령으로 수집한다. `math`를 `mbpp`로
바꾸면 다른 데이터셋의 현재 활성 plan을 읽는다. 공통 환경 설정 후 코드 레포에서 실행한다.

```sh
SRGC_ANALYSIS_DATASET=math
SRGC_ANALYSIS_PLAN=$("$PAIR_PYTHON" - "$SRGC_ANALYSIS_DATASET" <<'PY'
import os
from pathlib import Path
import sys
sys.path.insert(0, "scripts")
from srgc_pair_inputs import default_plan
from srgc_shared_storage import route_plan
source = default_plan(Path.cwd(), sys.argv[1], os.environ, writing=False)
print(route_plan(source, writing=False))
PY
)
"$PAIR_PYTHON" scripts/srgc_direction_analysis.py --plan "$SRGC_ANALYSIS_PLAN"
```

출력은 해당 run root의 `analysis/direction/` 아래 `direction.csv`, `summary.txt`,
matplotlib 설치 시 `direction.png`다. 분해 기록이 없는 과거 checkpoint는 새로
계산하지 않는다. `d_source=decision`은 실제 결정 값이고 `reconstructed_diagnostic`은
저장된 norm/cosine으로 재구성한 진단 값이다. 분석 결과를 온라인 전환 기록으로 쓰지 않는다.

### 6.3. 노드 배정과 그룹 볼륨

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

매 배정 시 `min(빈 노드 수, 선행 조건이 충족된 미완료 작업 수)`만큼만 새 worker를
둔다. P0/P1, P2, P3 순으로 배정하되 모든 실험을 완료 대상으로 유지한다.
E04–E07은 [40개 배정표](REBUTTAL_EXTRA_TASKS.tsv)의 서로 다른 tuple을 수동 지정한다.
완료된 노드는 다음 미완료 실험으로 이동하고 모든 작업이 끝났으면 추가 worker를 띄우지 않는다.

| 저장 항목 | 위치 / 사용 방식 |
| --- | --- |
| 모델 weight | group의 고정 snapshot, 같은 모델 노드들이 읽기 공유 |
| cache·입력·prefix·checkpoint·원시 비용 | 해당 모델·데이터셋의 group active root; 로컬 `/tmp`를 실험 원본으로 쓰지 않음 |
| Qwen 라이브러리 캐시·임시 파일 | `$OM_WORK/qwen-runtime-cache` 아래; 노드별 컴파일 캐시 분리 |
| 노드별 배정 기록 | dataset/seed/arm, node, code commit, plan/root, 시작·종료·실패 기록 |
| 수집 결과·표·그림 | group 원본의 hash와 출처를 유지하며 CPU 분석 공간으로 가져옴 |

120노드 동시 실행의 group I/O와 잠금 성능을 실측한 것은 아니다. 많은 노드를
추가할 때 model load·checkpoint 저장·backup이 병목인지 확인한다. 실제 남은 시간은
실험 종류별 완료 작업의 wall-time으로 추정하며 selection 비용만 275배 곱하지 않는다.
전체 완료 목표에는 가장 긴 선행 작업 경로와 E03/E09 구현 시간도 포함한다.

### 6.4. 전체 실험 완료와 V7 반영

결과 수집은 다음 여섯 항목으로 완료 여부를 판단한다:

1. 예정 seed와 모든 비교 arm의 같은 total-step endpoint.
2. 각 Switch의 자기 경로 D/check/전환 기록.
3. seed별 paired reward 차이와 전체 seed 변동. 질문 bootstrap과 seed 변동을 구분.
4. cache/prefix/selection/training/evaluation/저장 비용 receipt; 미계측은 unknown.
5. code commit, plan/input/prefix hash, node, 시작·종료 시각, 실패·재시도 기록.
6. V6 제출 결과와 새 결과를 별도 표로 유지. 모델·프로토콜이 다른 실행을 합산 평균하지 않음.

전체 실험 종료 목표는 **10월 25일**, 결과·비용 검증은 **10월 28일**,
V7 완성 초안은 **11월 1일**, 최종 점검은 **11월 4일 18:00 KST**다.
실제 자원·wall-time 확인 전의 내부 목표이며, 미완료 실험을 완료 처리하지 않는다.
실험이 끝나는 대로 V7 표·본문·부록에 반영하고, 리뷰 공개 전 PDF·TeX·변경 요약을
모두 준비한다. 상세 단계는 [리뷰 준비 일정](REVIEW_SCHEDULE_2027_KO.md)을 따른다.

## 7. 과거 실험·진단 명령 위치

현재 온라인 실험과 과거 진단은 별개다. 아래는 필요 시 과거 결과를 확인하거나
해당 frozen protocol을 보충할 때 찾을 위치이며, E01–E10 완료 수에 합치지 않는다.

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

E03/E09/E10은 기존 실행을 다시 호출해서 만들 수 없다. 전용 명령과 출력 위치는
[4절](#4-e03e07-e09-e10--추가-arm-120개)에 있다. 이 문서 작성으로 작업을 자동 시작하거나
주기적 실행을 등록하지 않았다.

**2026-09-30 점검:** 관련 회귀 테스트 31개, 문서 shell 블록 19개 문법 검사,
배정표 40개 명령의 인자 전달 검사를 통과했다. CPU 분석 명령은 임시 group 경로의
합성 기록으로 MATH/MBPP 출력을 확인했다. 실제 GPU 학습·원격 작업 상태 검증은 포함하지 않는다.

**2026-09-30 구현 추가 (`997cab9`):** E03 `replicate<k>-<arm>`, E09 `direction_*`, E10 `sr_hold`를
같은 launcher에 등록했다. 전체 SR-GC 회귀 테스트 318개 통과(torch 환경 9 skipped), frozen core
hash 불변, launcher 셸 문법 검사 통과. 배정표에 80개 tuple을 추가했다. GPU 실행은 없다.
