# Limitation 후속 실험: 구현 목록과 실행 기록

구현·명령 점검: 2026-10-01, `master` (추가 runner 매-update 저장, attention 계승, 종료 전달 수정). 아래 날짜별 실행 기록은 보존한다.
대응 원고: V7 `sections/discussion.tex`의 Discussion and Limitations.
실행 코드는 이 저장소에만 유지한다. 논문 쪽 목록은 `v7/EXPERIMENTS.md`다.

실행할 때는 [통합 명령 모음](REBUTTAL_COMMANDS_KO.md),
[추가 arm 120개 배정표](REBUTTAL_EXTRA_TASKS.tsv),
[공식 리뷰 일정과 내부 준비 계획](REVIEW_SCHEDULE_2027_KO.md)을 사용한다.
아래는 실험 목적·조건·구현 기록을 자세히 설명한다.
기존 실험의 수치·완료 범위와 교정이 필요한 비용은
[실험 결과 기록](EXPERIMENT_RESULTS_LEDGER_KO.md)에 출처와 함께 정리했다.

**2026-09-28 저자 목표 갱신:** 명령 모음의 E01–E09(09-30에 E10 추가) 전체 실험을 완료하고 V7
원고까지 준비한 뒤 11월 5일 리뷰 공개를 맞는다. P0–P3는 순서이지 선택적으로
생략할 목록이 아니다. CPU/GPU·최대 동시 노드 수는
[자원 표](REBUTTAL_COMMANDS_KO.md#61-실험별-gpu와-최대-동시-노드),
구현·실행·원고 내부 마감은 [일정](REVIEW_SCHEDULE_2027_KO.md)에 정리했다.

**상태:** 저자는 아직 실행하지 않은 실험이 많다고 알렸다. 아래는 구현과 실행
명령을 확인한 목록이지 GPU 실험 완료 목록이 아니다. 이번 작업에서 새 7B 학습은
시작하지 않았다. 로컬 `nvidia-smi`는 드라이버 통신 실패를 반환했고, 실제 사용할
H100 노드의 접속 정보와 빈 allocation은 확인되지 않았다. 원격 실행을 완료 또는
진행 중으로 기록하지 않는다. 기존 작업을 중단하거나 기존 결과를 덮어쓰지 않았다.

## 코드 위치와 명령 빠른 찾기

### P0/P1 최대 동시 노드 수

**한 작업은 4-H100 노드 1개를 사용한다.** 아래는 seeds 5-9, 독립 반복 k=1,2,
고정 전환 `switch_fixed200` 한 조건 기준이다. 모든 해당 prefix가 준비되고
아직 끝난 continuation이 없을 때의 작업 수 상한이며, 그 규모의 실측 성능이나
현재 빈 노드 수를 뜻하지 않는다.

| 대상 | MATH만 | MBPP만 | 두 데이터셋 합계 | 실행 조건 |
| --- | --- | --- | --- | --- |
| P0 cache/prefix만 진행 중 | 최대 5노드 | 최대 5노드 | **최대 10노드 / 40 GPU** | seed별 cache -> prefix 순서, 아직 continuation이 준비되지 않은 단계 |
| P0 기본 네 arm | 최대 20노드 | 최대 20노드 | **최대 40노드 / 160 GPU** | seed별 step-25 prefix 준비 후 Random/SR/On/Switch는 서로 독립 |
| P1 독립 반복 E03 | 최대 20노드 | 최대 20노드 | **최대 40노드 / 160 GPU** | 5 seeds x k=1,2 x SR/Switch; 각 seed의 prefix만 필요 |
| P1 고정 전환 E04 | 최대 5노드 | 최대 5노드 | **최대 10노드 / 40 GPU** | 5 seeds x switch_fixed200; 각 seed의 prefix만 필요 |
| P1 합계 | 최대 25노드 | 최대 25노드 | **최대 50노드 / 200 GPU** | 독립 반복과 고정 전환 사이에 선후 의존성 없음 |
| P0 continuation + P1 합계 | 최대 45노드 | 최대 45노드 | **최대 90노드 / 360 GPU** | P0 네 arm 완료를 기다릴 필요 없이 해당 prefix 준비 후 병렬 실행 |

P0는 `sh scripts/run_srgc.sh all run`으로 공유 queue에서 자동 배정한다.
P1은 이 queue에 포함되지 않으므로 **다른 빈 노드에 서로 다른 dataset/seed/arm/k를
직접 배정**한다. 같은 tuple을 여러 노드에서 중복 실행하지 않는다. 노드 하나에서
P0 worker와 P1을 함께 실행하지 않는다. 1노드 순차 실행도 가능하다.
현재 배정 가능 수는 `min(빈 노드 수, prefix가 준비된 미완료·미실행 작업 수)`이며,
이미 실행 중인 작업은 추가 배정 수에 세지 않는다.

### 2026-10-01 추가 실험 시작 오류 수정

`srgc_sr_refresh.py`의 P1 등 추가 arm에서 `PackageNotFoundError: math-verify`가
발생한 원인은 P0에 있던 오프라인 verifier 준비가 빠졌기 때문이다. 이제 group
저장소 경로를 확인한 뒤, NCCL 초기화와 패키지 버전 기록 전에 동일한 hash 검증
번들을 준비한다. pip 설치, 채점 규칙 변경, frozen core 변경은 없다.
실패한 작업은 수정 코드를 받은 뒤 기존 명령으로 재개한다. 실행 중인 정상 작업은
중단하지 않는다. 이 수정은 별도로 보고된 간헐적 NCCL 오류의 원인을 확정한 것이 아니다.

추가 실험의 `.sr_refresh.execution.lock` 등 잠금 오류도 실행 경로를 수정했다.
이전 launcher는 `torchrun`을 직접 실행했고 추가 arm의 잔존 rank는 공통 정리 대상에서
빠져 있었다. 이제 [추가 실험 supervisor](../scripts/srgc_extra_worker.py)가 작업별 launch
잠금, P0와 공유하는 GPU 잠금, NCCL admission을 거쳐 시작하고 종료 시 자식 rank를 정리한다.
살아 있는 기존 launcher의 작업은 보존하며, 소유 launcher가 없는 추가 arm 프로세스만
정리한다. 같은 dataset/seed/arm/k가 이미 실행 중이면 GPU 로딩 전에 `BUSY`와 exit 75로
종료하고 재시도를 반복하지 않는다. **잠금 파일을 삭제하지 않는다.** 실제 실패 원인은
`seed-N/launches/<arm>/<attempt>/task.log`와 `worker.json`에 남으며,
독립 반복은 `seed-N/replicate-<k>/launches/<arm>/` 아래에 남는다.

### 저장소와 진입점

- 코드 저장소: [33modeling/offpolicy-misranking](https://github.com/33modeling/offpolicy-misranking), branch **`master`**.
- 현재 PC의 코드 위치: `/home/kms/dev/offpolicy-misranking`.
- 현재 PC의 논문 목록: `/home/kms/dev/offpolicy-v3/offpolicy-misranking-paper-v2/v7/EXPERIMENTS.md`.
- 아래 명령은 **코드 레포 루트 기준**이다. H100에서는 해당 노드의 코드 checkout으로 이동한다. `OM_WORK`는 저장 경로 설정이며 코드 checkout과 같은 위치라고 가정하지 않는다.
- 실행 중인 checkout에서 `git pull` 또는 branch 전환을 하지 않는다. 빈 노드용 checkout과 기존 실행의 frozen code/plan을 구분한다. 이번 변경은 명령 정리이며 실험을 새로 실행하지 않았다.

| 우선순위 | 실험 | Shell 진입점 | 실제 구현 위치 |
| --- | --- | --- | --- |
| P0 | OLMo MATH/MBPP 네 arm | [run_srgc.sh](../scripts/run_srgc.sh) | [run_srgc_rebuttal.py](../scripts/run_srgc_rebuttal.py), [run_experiment.py](../srgc_rebuttal/run_experiment.py), [srgc.py](../srgc_rebuttal/srgc.py) |
| P0 | 비용·결과·checkpoint | 같은 `run_srgc.sh`의 `results/costs/backup` | [reports.py](../srgc_rebuttal/reports.py), [cost_report.py](../srgc_rebuttal/cost_report.py), [cost_ledger.py](../srgc_rebuttal/cost_ledger.py), [srgc_checkpoint_backup.py](../scripts/srgc_checkpoint_backup.py) |
| P1 | 동일 prefix의 독립 재현 | `sh scripts/run_srgc_sr_refresh.sh math 5 replicate1-sr`, `... replicate1-switch` (k=1,2) | [srgc_replicate.py](../scripts/srgc_replicate.py)의 `ReplicateMixin`; 같은 launcher, 출력은 `seed-N/replicate-<k>/` (`997cab9`). 기존 `run` 재호출이나 `switch_repeat`는 대체가 아님 |
| P1 | fixed-step-200 전환 대조 | `sh scripts/run_srgc_sr_refresh.sh math 5 switch_fixed200` | `srgc_switch_fixed.py`; checkpoint 200에서 전환하여 update 201부터 SR. 2026-09-29 경계 수정, 새 run 필요 |
| P2 | 후보 40개 SR 갱신 | [run_srgc_sr_refresh.sh](../scripts/run_srgc_sr_refresh.sh) | [srgc_sr_refresh.py](../scripts/srgc_sr_refresh.py)의 `SRRefreshEngine`, `scope=candidates` |
| P3 | 같은 유지 간격의 cached-SR 대조 | `sh scripts/run_srgc_sr_refresh.sh math 5 sr_hold` | `srgc_sr_refresh.py`의 `SRRefreshEngine`, `scope=cached` (`997cab9`) |
| P2 | 반복 전환 | 같은 `run_srgc_sr_refresh.sh` | [srgc_switch_repeat.py](../scripts/srgc_switch_repeat.py)의 `SwitchRepeatEngine`; 위 runner가 호출 |
| P3 | 전체 pool SR 갱신 | 같은 `run_srgc_sr_refresh.sh` | `srgc_sr_refresh.py`의 `SRRefreshEngine`, `scope=pool` |
| P3 | Qwen3.5-9B 온라인 네 arm | [run_srgc_qwen35.sh](../scripts/run_srgc_qwen35.sh) | [run_srgc_qwen35.py](../scripts/run_srgc_qwen35.py), [srgc_qwen35.py](../scripts/srgc_qwen35.py), [srgc_qwen35_rank.py](../scripts/srgc_qwen35_rank.py) |
| P3 | 초기 gradient 방향 matched ablation | `sh scripts/run_srgc_sr_refresh.sh math 5 direction_removed` (`direction_magnitude`, `direction_replaced`) | [srgc_direction_ablation.py](../scripts/srgc_direction_ablation.py)의 `DirectionAblationEngine` (`997cab9`) |

### OLMo: 시작·조회·비용

`run`은 빈 4-H100 노드에서만 실행한다. MATH와 MBPP 명령은 서로 다른 작업 배정
예시이며, 같은 노드에서 두 GPU worker를 동시에 실행하지 않는다.
`status/results/costs`는 새 학습을 시작하지 않는다.

```sh
# 같은 코드의 활성 실험 재개; 실험 코드 identity 변경 시 별도 cohort 생성/합류
sh scripts/run_srgc.sh math run
sh scripts/run_srgc.sh mbpp run

# 상태, 결과, 비용
sh scripts/run_srgc.sh math status
sh scripts/run_srgc.sh math results
sh scripts/run_srgc.sh math costs
sh scripts/run_srgc.sh mbpp status
sh scripts/run_srgc.sh mbpp results
sh scripts/run_srgc.sh mbpp costs

# 저장된 checkpoint 백업 1회
sh scripts/run_srgc.sh math backup
sh scripts/run_srgc.sh mbpp backup
```

MATH plan은 [pair_seeds.json](../srgc_rebuttal/experiments/pair_seeds.json) 또는
[additional_seeds.json](../srgc_rebuttal/experiments/additional_seeds.json),
MBPP는 [mbpp_pair_seeds.json](../srgc_rebuttal/experiments/mbpp_pair_seeds.json) 또는
[mbpp_seeds.json](../srgc_rebuttal/experiments/mbpp_seeds.json)이다.
[default_plan](../scripts/srgc_pair_inputs.py)과 [route_plan](../scripts/srgc_shared_storage.py)이
활성 cohort를 고른다. 기존 실행을 위해 plan 파일을 수동으로 바꾸지 않는다.
구체적인 결과 root는 `status` 출력에서 확인한다.

### OLMo 추가 arm: seed별 실행

해당 seed의 prefix 완료 후 별도 빈 노드에서 실행한다. 아래는 seed 5의 명령이며
`5`를 `6`, `7`, `8`, `9`로 바꾼다. 한 줄은 한 arm이고 seed 전체 자동 순회가 아니다.
같은 노드에서는 앞 작업 완료 후 다음 작업을 실행한다.

```sh
# P1: total step 200까지 On-policy, update 201부터 SR
sh scripts/run_srgc_sr_refresh.sh math 5 switch_fixed200
sh scripts/run_srgc_sr_refresh.sh mbpp 5 switch_fixed200

# P2: 후보 40개 SR 갱신
sh scripts/run_srgc_sr_refresh.sh math 5 candidates
sh scripts/run_srgc_sr_refresh.sh mbpp 5 candidates

# P2: 반복 전환
sh scripts/run_srgc_sr_refresh.sh math 5 switch_repeat
sh scripts/run_srgc_sr_refresh.sh mbpp 5 switch_repeat

# P3: 전체 pool SR 갱신
sh scripts/run_srgc_sr_refresh.sh math 5 pool
sh scripts/run_srgc_sr_refresh.sh mbpp 5 pool

# P1 (E03): 같은 prefix의 독립 반복. k는 1, 2. 같은 k의 sr/switch가 한 쌍(공통 sampling stream)
sh scripts/run_srgc_sr_refresh.sh math 5 replicate1-sr
sh scripts/run_srgc_sr_refresh.sh math 5 replicate1-switch
sh scripts/run_srgc_sr_refresh.sh mbpp 5 replicate1-sr
sh scripts/run_srgc_sr_refresh.sh mbpp 5 replicate1-switch

# P3 (E09): 방향 정보만 제거/대체한 On-policy 대조 3조건
sh scripts/run_srgc_sr_refresh.sh math 5 direction_removed
sh scripts/run_srgc_sr_refresh.sh math 5 direction_magnitude
sh scripts/run_srgc_sr_refresh.sh math 5 direction_replaced

# P3 (E10): 갱신 없이 유지 간격만 On-policy와 맞춘 cached-SR 대조
sh scripts/run_srgc_sr_refresh.sh math 5 sr_hold

# 추가 arm 결과와 selection 비용 요약 (독립 반복은 seed·k별 paired 차이로 함께 출력)
sh scripts/run_srgc_sr_refresh.sh math results
sh scripts/run_srgc_sr_refresh.sh mbpp results
```

추가 arm에는 `status`/`costs` 명령이 없다. 상태는 콘솔과
`seed-N/<arm>-progress.json`, 상세 비용은 `seed-N/cost-receipts/<arm>/`를 본다.
`<arm>`은 `switch_fixed200`, `sr_refresh`, `switch_repeat`, `sr_refresh-pool`, `sr_hold`,
`direction_removed`, `direction_magnitude`, `direction_replaced`이다. 독립 반복은
`seed-N/replicate-<k>/` 아래에 `replicate.json`(replicate id, sampling seed, prefix hash)과
원래 arm 이름의 `<arm>-{latest.pt,progress.json,endpoint.json}`, `cost-receipts/<arm>/`를 둔다.
결과 요약을 전체 비용 합계로 해석하지 않는다. 자세한 산출물과 재시작 조건은 아래 3-4절에 있다.

### Qwen: 별도 환경과 전용 queue

`56be4d1`에서 추가된 전용 온라인 실험이다. 과거 `run_qwen35_9b.sh` selection
매트릭스와 구분한다. [상세 준비 안내](QWEN35_SRGC_KO.md)와
[환경 명세](../configs/srgc_qwen35/requirements.txt)를 따르고, OLMo 환경을
업그레이드하지 않는다. 아래 `QWEN_PYTHON` 예시는 group-volume `OM_WORK`가
설정되어 있고 별도 Qwen 환경이 이미 준비되어 있을 때 사용한다.

```sh
export QWEN_PYTHON="$OM_WORK/.venv-qwen35/bin/python"

# 모델 준비는 다운로드 가능한 환경에서 1회; doctor는 실제 GPU admission을 대신하지 않음
sh scripts/run_srgc_qwen35.sh all download
sh scripts/run_srgc_qwen35.sh all doctor
sh scripts/run_srgc_qwen35.sh all prepare

# 각 빈 4-H100 노드에서: MATH/MBPP queue의 작업을 자동 배정
sh scripts/run_srgc_qwen35.sh all run

# 결과에는 비용 보고도 포함; 별도 costs 하위 명령은 없음
sh scripts/run_srgc_qwen35.sh all status
sh scripts/run_srgc_qwen35.sh all results
```

한 dataset만 실행/조회하려면 `all`을 `math` 또는 `mbpp`로 바꾼다.
Qwen은 자기 초기 정책으로 cache와 prefix를 새로 만들며 OLMo 결과를 재사용하지
않는다. 감사 후 v2 기본 Qwen root는 `$OM_WORK/srgc-rebuttal/qwen35-9b-v2`이고
`SRGC_QWEN_ROOT`로 지정할 수 있다. `prepare`가 생성하는 plan은
`<Qwen root>/experiments/qwen35-9b-{math,mbpp}.json`, 결과는
`<Qwen root>/runs/{math,mbpp}/seed-N/`, TXT 보고는 `<Qwen root>/reports/`다.
실제 9B GPU admission/실험 결과는 아직 확인되지 않았다.

## 우선순위

2026-09-28 지정. 기준은 **핵심 결과의 재현성, SR-GC 시점 선택의 추가 가치,
비용 비교의 신뢰성**이다. 코드가 이미 있다는 이유만으로 더 중요한 미구현 대조보다
앞세우지 않는다. 아래 순서는 신규 자원 배정과 구현의 우선순위이며, 진행 중인
작업의 중단이나 낮은 순위 실험의 취소를 뜻하지 않는다.

| 순서 | 우선순위 | 실험/작업 | 먼저 하는 이유 | 준비 상태 |
| --- | --- | --- | --- | --- |
| 1 | P0 | 추가 seeds 5-9의 MATH/MBPP 네 arm 완성 및 전체 비용 수집 | 관측 이득의 재현성과 실제 계산 비용을 함께 검증하는 기본 증거 | 구현됨; 기존 진행 유지, 누락 결과 확인 |
| 2 | P1 | 동일 prefix에서 SR/Switch의 독립 학습 반복 | 같은 조건의 실행 변동과 Switch 이득을 직접 구분 | 구현됨 (`997cab9`, `replicate<k>-<arm>`); 반복 수 R=2와 stream 유도 규칙을 사전 고정, GPU 미실행 |
| 3 | P1 | 사전 고정 total-step-200 전환 대조 | 전환 자체의 효과와 SR-GC timing rule의 추가 가치를 구분 | `switch_fixed200` 구현 및 경계 수정, GPU 결과는 별도 검증 필요 |
| 4 | P2 | 후보 40개의 SR 성공률 갱신 `sr_refresh` | 오래된 캐시를 유지하는 전략과 갱신 전략의 성능·비용 비교 | 구현됨; 순수 갱신 효과 분리용 배치 유지 간격 통제 `sr_hold`도 구현 (`997cab9`, E10) |
| 5 | P2 | 반복 전환 `switch_repeat` | 한 번만 전환하고 점검을 끝내는 선택의 성능·비용 trade-off 확인 | 구현됨; 전환 후 scoring 비용 포함 |
| 6 | P3 | 전체 400개 갱신 `sr_refresh-pool` | 후보 범위를 넓힌 갱신의 추가 이득과 비용 확인 | 구현됨; 4번 다음 확장 |
| 7 | P3 | 다른 backbone에서 동일 온라인 Switch | 모델 의존성과 일반화 검증 | Qwen3.5-9B 전용 온라인 runner 준비; CPU 검증, 9B GPU admission/실험은 대기 |
| 8 | P3 | 초기 gradient 방향의 matched ablation | 초기 이점에 대한 인과적 설명 보강 | 구현됨 (`997cab9`, `direction_removed/magnitude/replaced` 3조건 사전 고정); 핵심 결과 검증 이후 배정 |

### 자원 배정과 완료 기준

- **기존 실행은 유지:** MATH/MBPP의 정상 작업을 끄거나 처음부터 다시 시작하지 않는다. 새로 배정할 자원이 경쟁하면 주 결과인 MATH의 누락 paired 결과를 먼저 완성하고 MBPP를 완성한다. 이는 MBPP의 기존 진행 중단이나 계획 seed 제외를 뜻하지 않는다.
- **비용은 1번부터 동시 수집:** 모든 실험에서 cache 생성/재사용, prefix, selection, training, 평가·저장 비용과 불완전 계측을 함께 기록한다. 비용만 뒤로 미루거나 unknown을 0으로 채우지 않는다.
- **구현은 GPU 작업과 병행:** 2번의 replicate runner와 회귀 테스트는 09-30에 준비했고, 3번과 함께 prefix가 준비되면 배정한다. 새로 할당 가능한 GPU는 준비된 상위 순위 작업에 먼저 배정한다. 남는 별도 노드는 P2에 쓸 수 있지만 상위 작업을 밀어내지는 않는다.
- **구현된 것의 실행 순서:** OLMo 기본 네 arm 및 비용, `replicate<k>-sr`/`replicate<k>-switch`, `switch_fixed200`, `sr_refresh`, `switch_repeat`, `sr_refresh-pool`, `sr_hold`, Qwen 온라인 네 arm, `direction_*` 순서다. P2/P3의 추가 continuation을 P1 대조보다 먼저 완료해야 하는 것은 아니다.
- **seed·비교 조건은 결과와 무관하게 고정:** 계획된 seeds 5-9를 유지하고 모든 결과를 수집한다. 좋은 seed만 골라 다음 실험을 하거나 유리한 결과가 나온 시점에 반복을 종료하지 않는다. P1의 반복 수는 R=2(`replicate1`, `replicate2`), sampling stream은 `sampling_seed(base seed, k)`로 코드에 고정했다. 결과를 본 뒤 반복을 추가하면 그 사실을 표에 남긴다.
- **P0 완료:** 예정된 네 arm/seed의 같은 total step 결과, paired 차이, 자기 경로의 전환 이력과 비용 receipt를 검증한다. 일부 arm만 끝난 평균을 최종 결과로 쓰지 않는다.
- **P1 완료:** 2번은 동일 prefix·캐시·평가 조건의 SR/Switch 반복을 짝지어 보고하고, 3번은 같은 조건의 고정 전환과 Switch를 직접 비교한다. 결과가 무차이 또는 불리해도 함께 보고하며, 두 실험의 완료를 효과 입증과 동일시하지 않는다.

6번은 refresh 한 번당 후보 응답 수가 320개에서 3,200개로 늘어난다. 이는
전체 wall-time이 정확히 10배라는 추정이 아니다. 1-5번이 답하는 핵심 질문을
먼저 다루고, 나머지도 리뷰 전 완료 범위에 포함해 실측 자원에 맞춰 배정한다.

## 1. Limitation과 실험 대응

| 항목 | 이미 있는 구현 | 이번 실행 대상 / 남은 일 | 상태 |
| --- | --- | --- | --- |
| 학습 seed 간 변동 | MATH/MBPP seeds 5-9, Random/SR/On-policy/Switch | 이미 돌고 있는 작업은 유지하고 미완료 arm만 이어서 실행 | 구현됨, 원격 완료 확인 필요 |
| 다른 task로의 확장 | 동일 네 arm의 MBPP plan과 입력 | MATH와 분리된 MBPP 결과 수집 | 구현됨, 원격 완료 확인 필요 |
| 자신의 경로에서 주기적 판단 | 현재 `Engine`, 25-update refresh, single-reference D | 각 Switch가 자기 checkpoint에서 판단한 기록 수집 | 구현됨, 과거 A/B replay와 별도 결과 |
| 공통 refresh 절차의 end-to-end 비용 | phase/stage ledger, cache, prefix, continuation, 평가·저장 비용 | 완료/미완료 계측을 구분한 cost export | 구현됨, 실측 결과 미수신 |
| SR cache refresh | `sr_refresh`, `sr_refresh-pool` | 두 scope 모두 MATH/MBPP seeds 5-9에서 기존 SR와 비교 | 구현됨, 이번 점검에서 신규 GPU 실행 안 함 |
| 반복 전환 | `switch_repeat` | 동일 seed의 일회 전환 `switch`와 비교 | 구현됨, 이번 점검에서 신규 GPU 실행 안 함 |
| 고정 schedule 대비 SR-GC timing 가치 | 기존 네 arm만으로는 이를 직접 검증하지 않음 | 사전 고정 total-step-200 대조, 동일 prefix/평가/길이 | `switch_fixed200` 구현됨; 실행 완료나 효과 입증을 뜻하지 않음 |
| 같은 시작 상태의 독립 training replicate | 추가 base seed 실험과 다른 질문 | prefix/캐시를 고정하고 분기 이후 sampling stream만 바꾸는 paired SR/Switch 반복 | `replicate<k>-<arm>` 구현됨 (`997cab9`); GPU 미실행 |
| 초기 gradient 방향의 인과적 효과 | retrospective 진단 및 objective 비교는 있음 | 방향 정보만 제거/대체하고 나머지를 맞추는 ablation | `direction_removed/magnitude/replaced` 구현됨 (`997cab9`); GPU 미실행 |
| SR 갱신 효과와 배치 유지 간격 효과의 분리 | `sr_refresh`는 갱신과 유지 간격이 동시에 바뀜 | 갱신 없이 유지 간격만 맞춘 cached-SR 대조 | `sr_hold` 구현됨 (`997cab9`); GPU 미실행 |
| 다른 backbone에서도 온라인 Switch가 유효한가 | Qwen3.5-9B 전용 온라인 adapter 및 MATH/MBPP 5시드×4arm 준비 | [실행 안내](QWEN35_SRGC_KO.md); Qwen 캐시/prefix 새로 생성, 실제 9B GPU admission 후 실행 | CPU 검증; GPU 결과 미생성. 과거 selection 매트릭스와 별도 실험 |

모든 행을 한꺼번에 실행할 수 있는 단일 runner가 있는 것은 아니다. 특히
`switch_repeat`는 SR과 On-policy 사이를 여러 번 **전환**하는 방법이며, 같은
실험을 다른 난수로 **반복 재현**하는 runner가 아니다. CPU 테스트 성공도
논문 limitation이 실증적으로 해결됐다는 뜻이 아니다.

## 2. 구현된 온라인 실험의 정확한 차이

공통 plan은 OLMo-3-1025-7B, GRPO, LoRA rank 16, seed별 25-update 공유 prefix,
total 275 updates, 4-GPU 실행이다. 각 update의 학습은 4문제 x 새 응답 8개다.
후보 pool 400개와 validation/evaluation 분할은 plan/input hash로 고정한다.

| Arm | 후보/선별 | 전환 이후 | 주 비교 |
| --- | --- | --- | --- |
| `random` | 매 update마다 중복 없는 무작위 40개에서 무작위 4개 | 해당 없음 | 유지 대조군 |
| `sr` | 매 update마다 무작위 40개에서 캐시 SR 점수 상위 4개 | 해당 없음 | 캐시를 갱신하지 않는 유지 대조군 |
| `on_policy` | 25 updates마다 무작위 40개를 현재 gradient로 선별, 상위 4개를 다음 refresh까지 유지 | 계속 gradient scoring | 지속 선별 대조군 |
| `switch` | On-policy와 같이 시작, 40-vs-40 D의 기존 temporal rule | SR로 한 번 전환한 뒤 gradient scoring/check 중단 | 추가 seed의 핵심 비교 |
| `sr_refresh` | 25 updates마다 무작위 40개 x 8응답으로 현재 성공률을 다시 구하고 SR 상위 4개 유지 | 계속 성공률 갱신 | 기존 `sr`, `on_policy`와 비교 |
| `sr_refresh-pool` | 25 updates마다 전체 400개 x 8응답으로 성공률 갱신, 상위 4개 유지 | 계속 전체 pool 갱신 | `sr_refresh`와 범위/비용 비교 |
| `switch_repeat` | 기존 Switch와 같은 전환 규칙으로 시작 | SR 상태에서도 check를 계속하고 양의 D 확인 시 On-policy로 복귀 가능 | 일회 전환 `switch`와 비교 |
| `sr_hold` | 25 updates마다 무작위 40개를 **캐시** SR 점수로 정렬해 상위 4개를 다음 refresh까지 유지 (rollout 없음) | 계속 캐시 사용 | `sr_refresh` − `sr_hold` = 갱신 효과, `sr_hold` − `sr` = 유지 간격 효과 |
| `direction_removed` | On-policy와 같은 refresh·후보 40·scoring·validation gradient; SR 비교 gradient 없음. 선별만 채점된 40개 중 균등 무작위 4개 | 계속 (전환 없음) | `on_policy`와 비교: 방향·크기 정보 모두 제거, 계산 범위 동일 |
| `direction_magnitude` | 같은 절차; 선별을 projected gradient norm 순으로 | 계속 | `on_policy`와 비교: 크기만 남기고 방향 제거 |
| `direction_replaced` | 같은 절차; validation 방향을 refresh마다 뽑은 무작위 단위 벡터로 대체해 cosine 정렬 | 계속 | `on_policy`와 비교: 방향은 쓰되 validation 정보 없음 |
| `replicate<k>-<arm>` | `<arm>`(random/sr/on_policy/switch/switch_fixed<N>)과 동일; 분기 이후 sampling stream(후보 추출·응답 생성·tie-break)만 `sampling_seed(base seed, k)`로 교체 | `<arm>`과 동일 | 같은 k의 arm끼리 paired; 평가는 base seed 규칙 유지 |

샘플 중복 금지는 **한 번의 40개 추출 내부**에 적용된다. 이전 update/refresh에서
쓴 문제가 이후에 다시 나올 수 있다. SR-GC의 SR 비교 40개는 별도의 미사용 문제
우선 preview이며 실제 SR 학습용 무작위 40개와 구분한다.

`sr_refresh`와 기존 `sr`는 캐시 갱신 여부뿐 아니라 선택 배치의 유지 간격도
다르다. 따라서 이 비교를 "갱신 여부만 바꾼 순수 ablation"으로 쓰면 안 된다.
그 효과만 분리하는 동일 유지 간격의 cached-SR 대조가 `sr_hold`(E10)다.

On-policy scoring은 후보 40개와 단일 validation 50개만 계산한다. Switch는
D를 확인하는 시점에만 SR 비교 40개를 추가하며, 후보와 중복된 문제는 한 번만
계산한다. 방향 ablation과 고정 시점 전환 대조에도 SR 비교 gradient는 없다.
수정 전 샘플의 비용 기록은 그대로 보존하며 수정 후 실행과 섞지 않는다.
반면 `sr_refresh`는 후보 성공률만 얻으며 scoring
backward/validation 생성은 하지 않는다. 후보 320응답이라는 이유로 총 scoring
예산이 On-policy와 정확히 같다고 쓰지 않는다. 반복 전환은 SR 상태에서도 이
scoring 비용이 발생하므로 기존 일회 전환의 미미한 산술 비용과 혼동하지 않는다.

## 3. 실행 순서와 명령

아래 명령은 **코드 레포 루트, 해당 실험에 할당된 빈 4 x H100 노드**에서 실행한다.
기존 Pair/MBPP Python 환경을 그대로 쓰며 패키지를 재설치하지 않는다.
노드마다 하나의 GPU worker만 실행한다. 기본 `run_srgc.sh ... run`에는 시작 시
동일 사용자 GPU 프로세스를 정리하는 코드가 있으므로, 다른 학습이 도는 노드에서
추가 실험을 시작하는 용도로 호출하지 않는다.

### A. 기존 네 arm 및 prefix

기존 실행 노드에서는 상태만 확인한다. 아직 worker가 없는 빈 노드에서만 run을
추가한다. dataset별 명령을 구분하면 해당 활성 cohort의 경로를 확인하기 쉽다.

실행·조회 명령은 문서 상단의 "OLMo: 시작·조회·비용"에 모았다.

주 queue는 seed별 prefix 완료 후 네 continuation을 노드에 배정한다. 동일 명령은
실험 코드 identity가 같으면 기존 활성 cohort와 checkpoint를 사용하며 새 replicate를
만들지 않는다. identity가 바뀌면 새 cohort 생성이나 active 경로 변경 없이 중단한다.
원래 코드로 재개하며, 과거 자동 전환 기록(`automatic-restart.json`)이 있으면
`status/results`가 이전 실행을 찾아 별도 표로 보여준다. 실행 간 결과는 합치지 않는다.
Pair seed-3/4 캐시 재사용 plan과 별도 준비 입력 plan을 섞지 않는다. Pair 입력이
사용되면 seeds 5/7/9는 source seed 3, seeds 6/8은 source seed 4의 데이터/캐시를
재사용하므로 다섯 독립 데이터 분할이라고 부르지 않는다.

### B. 고정 전환과 Limitation의 추가 arm

해당 seed의 **검증된 prefix가 완료된 다음** 실행한다. 같은 seed의 기본 네 arm이
모두 끝날 때까지 기다릴 필요는 없지만, 별도의 빈 노드를 사용해야 한다.
아래 한 줄은 한 seed의 한 arm만 실행한다. `5`를 `6`, `7`, `8`, `9`로 바꾸어
각 seed를 실행한다. 같은 노드에서는 앞 명령이 끝난 다음 다음 명령을 실행한다.

실행 명령과 구현 파일은 문서 상단의 "OLMo 추가 arm: seed별 실행"에 모았다.

추가 실행 목록은 2 datasets x 5 seeds x 12 continuation = **120 continuations**다.
E04 고정 전환 10개, E05–E07 30개, E03 독립 반복 40개(k=1,2 × sr/switch), E09 방향 대조 30개(3조건),
E10 `sr_hold` 10개이며, 기본 네 arm이나 prefix는 포함하지 않는다.
각 continuation은 prefix 이후 250 updates이며, 각 dataset의 prefix/기본 arm
진행 상태에 따라 실행 가능한 작업 수가 달라진다. 총 GPU 시간은 실측 후 추정한다.

추가 arm은 현재 기본 네-arm 자동 queue에 등록되어 있지 않다. 따라서
`run_srgc.sh all run` 하나로 위 120개가 실행되지는 않는다. 추가 arm별 lease는
중복 실행을 막지만, 여러 노드가 자동으로 다른 seed/arm을 골라주는 기능은 아니다.
노드별로 다른 `(dataset, seed, arm)`을 배정한다.

### C. 재시작·상태·결과

추가 arm은 동일 명령을 다시 실행하면 같은 arm의 `latest.pt`에서 이어서 진행하고,
이미 완료된 동일 실험은 재학습하지 않는다. 모델·optimizer·선별 상태를 복구한다.
**추가 runner의 저장 간격은 매 update**다. 저장 완료 후 progress JSON도 갱신한다.
중단 시 저장하지 못한 update는 다시 수행될 수 있고, 기존 비용 receipt는 남는다.
선별 주기 25 updates와 최종 평가 조건은 그대로다. 새 arm은 prefix의 attention을
계승하고, 기존 추가 arm은 실제 사용하던 kernel(구 runner는 eager)을 유지한다.
자세한 중단·재개·15노드 배정 예는 [명령 모음](REBUTTAL_COMMANDS_KO.md)을 따른다.

결과·비용·백업 명령은 문서 상단의 빠른 찾기를 따른다.

추가 runner에는 별도 `status` 하위 명령이 없다. 기본 queue의 status/results가
추가 arm을 자동 집계한다고 안내하지 않는다. 학습 콘솔의 `TRAIN ... step=N/275`,
활성 root 아래 `seed-N/<arm>-run.json`, `<arm>-progress.json`을 확인한다.
최종 산출물은 `<arm>-endpoint.json`이며, 상세 비용은 같은 seed 아래
`cost-receipts/<arm>/`, `invocations/<arm>/`에 남는다. 독립 반복은 같은 파일들을
`seed-N/replicate-<k>/` 아래에 두고 `replicate.json`에 replicate id·sampling seed·prefix hash를
남기며, endpoint의 `replicate` 항목이 manifest와 다르면 results가 거부한다. 기존 `backup`은
기본 네 arm의 checkpoint만 복사하므로 추가 arm·replicate checkpoint는 별도로 보존한다.
위 추가 results는 selection 비용 요약이지 전체 cold-start 비용 합계 보고서가 아니다.

## 4. 저장·비용·완료 기준

- 기본 저장 위치는 `${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking/srgc-rebuttal`이며, 실제 cohort는 활성 plan pointer를 따른다. 사용자 볼륨으로 fallback하지 않는다.
- 실행 중인 cohort의 frozen core/input/plan hash를 고치거나 mismatch 검사를 풀지 않는다. 다른 hash면 원 실행 버전과 source를 확인한다.
- 초기 cache 생성 또는 기존 cache의 재사용 출처, prefix, continuation, model/setup, checkpoint 저장·복구, evaluation을 분리한다. 새 실행에서 cache를 재사용했다고 과거 생성 비용이 0이 되는 것은 아니다.
- D 산술은 selection 비용 안의 부분 항목이다. 합계에 두 번 더하지 않는다. shared prefix/cache는 arm별 표시와 실험 전체 합계에서 중복 회계하지 않는다.
- 최종 reward/total step, plan/input/core hash, 공통 prefix hash, per-question reward와 비용 계측 완결 여부가 필요하다. 누락 cost를 0으로 채우지 않는다.
- `results` 파일이 생겼다는 사실만으로 paired 비교 완료를 선언하지 않는다. 실제로 필요한 모든 arm과 seed가 있는지 확인하고 불리한 결과도 유지한다.
- 추가 변형의 script 파일 자체는 frozen core hash 대상이 아니다. 실행 코드 commit과 launcher/script hash도 작업 기록에 남기고, 실행 도중 다른 변형 코드로 교체하지 않는다.

## 5. 이번 코드 점검과 수정

### 2026-10-01 실행기 점검

- 추가 runner가 shell의 attention 설정을 모델에 전달하지 않던 문제 수정.
  새 arm은 prefix의 kernel을 계승하고, 기존 추가 실행의 eager는 재개 시 유지.
- 추가 arm과 독립 반복의 checkpoint/progress 저장을 매 update로 변경.
  fsync와 atomic replace를 적용하고 저장 실패 시 이전 checkpoint/progress 보존.
- shell SIGTERM을 자기 worker에 전달하고 종료를 기다린 뒤 재시도 없이 중단.
  진행 timeout은 exit 124로 기록. `results --json`을 shell에서도 지원.
- 15개 실행 형태의 매-step 저장/재개, 실패 저장, shell 종료 전달 회귀 테스트 추가.
  전체 suite: 338개 중 329개 통과, 환경 조건에 따른 9개 skipped. 실제 H100 학습은 미실행.
- frozen core SHA-256 유지:
  `9435e80003f41e1f65cb9dcc4074f1b06f063880d4823f433d5cd8fb5e2d74a7`.
  학습 규칙·입력·seed·기존 결과 변경 없음.

### 2026-10-01 추가 전체 재검수

- 범위: E03 독립 반복, E04 고정 전환, E05/E07 SR 갱신, E06 반복 전환,
  E09 방향 대조, E10 유지 간격 대조, E08 Qwen 확장과 공통 실행·결과 경로.
- 추가 arm의 완료 판정을 결과 검증과 통일. 다른 prefix/replicate의 endpoint,
  잘못된 평가 문항, 평균과 맞지 않는 보상, NaN/Inf·음수 비용을 완료로 인정하지 않음.
- 결과 하나가 손상돼도 나머지 정상 결과를 출력. 오류는 JSON `errors`와 비정상
  종료 코드 1로 표시하며, 잘못된 값을 0으로 대체하거나 정상 값으로 포함하지 않음.
- 과거 코드의 결과는 기록된 run identity로 읽기 전용 조회. 실행·재개 때의
  current-code 검사는 유지. prefix checksum 검증은 결과 조회 시 seed당 한 번 수행.
- Qwen 진행 timeout을 124로 남기고 다른 준비 작업을 계속 처리.
  사용자 중단 130과 일반 실패 1을 구분하고, 중단을 성공 종료로 보고하지 않음.
- Qwen rank가 OLMo의 `SRGC_ATTENTION` 환경을 물려받지 않고 plan의 eager를 사용.
  `all results`도 한 데이터셋 오류 때문에 다른 데이터셋 출력을 생략하지 않음.
- OLMo frozen core hash는 위 값 그대로. **Qwen adapter hash는 변경**되므로 기존
  Qwen run은 frozen checkout 유지. 수정본으로 새 실험을 할 때만 별도 root에 준비.
- 문서의 120개 tuple을 실제 shell에 전달하되 학습 실행은 대역 처리하여 명령 전달을 검증.
  실제 학습·원격 worker·실험 결과는 이번 재검수에서 시작하거나 변경하지 않음.
- 최종 회귀 테스트: 349개 중 340개 통과, Transformers 4 환경에서 Qwen 9개 건너뜀.
  별도 Transformers 5 환경에서는 해당 테스트를 포함한 Qwen 33개 모두 CPU에서 통과.
  작은 모델의 생성·gradient·학습·재개까지 확인했으며 실제 H100 분산 실행은 미검증.
- 문서 링크·앵커 83개, shell 예제 48개, 수정 Python 파일 9개의 구문 검사 통과.
  두 launcher의 `sh -n`, `git diff --check` 통과.

### 이전 점검 기록 (당시 코드 기준)

1. 추가 launcher가 `additional_seeds.json`/`mbpp_seeds.json`을 고정 선택하던 문제를 수정했다. 기본 launcher와 같은 `default_plan()`을 사용하여 활성 Pair 재사용 cohort를 따른다.
2. shell의 plan 조회 subprocess에서 설정한 환경은 torchrun에 전달되지 않는다. 실제 GPU rank에서 group-volume runtime cache 환경을 설정하도록 했다. prefix가 없거나 변경되었으면 모델 로드 전에 중단한다.
3. 사용자가 명시한 Python 경로가 없을 때 system Python으로 조용히 바꾸지 않고 실패하도록 했다.
4. 추가 results에서 실험 identity/공통 prefix/최종 update 수를 확인하고, 누락 비용은 `unknown`, 불완전 계측은 별도 표시한다.
5. `switch_repeat`를 서로 다른 module 이름으로 import하여 전체 테스트에서 class identity가 달라지던 문제를 canonical import로 수정했다.
6. (2026-09-30, `997cab9`) E03 독립 반복: `scripts/srgc_replicate.py`의 `ReplicateMixin`이 update 동안만 `config.seed`를 `sampling_seed(base seed, k)`로 바꾼 config를 쓰게 하여 후보 추출·scoring/training rollout·tie-break stream만 교체한다. SR 캐시 정렬·validation/evaluation 집합·checkpoint의 config·identity 검사는 base seed를 유지한다. 같은 k의 arm은 stream을 공유하고, checkpoint는 replicate id/sampling seed/protocol이 일치해야 재개되며, 기록된 arm의 checkpoint나 다른 replicate에서 분기하지 못한다.
7. (2026-09-30, `997cab9`) E09 방향 ablation: `scripts/srgc_direction_ablation.py`의 `DirectionAblationEngine`이 On-policy refresh를 선별 단계만 바꿔 재현한다. 후보 40·SR 비교 40·validation gradient·training stream seed가 On-policy와 같음을 테스트로 고정했고, 기록에는 실제 cosine(`ranking_scores`)과 ablation 점수(`ablation_scores`)를 함께 남겨 방향 분해 분석이 그대로 적용된다.
8. (2026-09-30, `997cab9`) E10 `sr_hold`: `SRRefreshEngine`의 `cached` scope. `sr`와 같은 추출·캐시 정렬 규칙을 쓰되 On-policy처럼 25 updates 유지하며 rollout이 없다. launcher/CLI/results/방향 분석에 세 계열을 등록했고 results는 replicate를 seed·k별 paired 차이로 출력한다.

`srgc_rebuttal/*.py`의 frozen 학습 패키지와 실험 JSON은 바꾸지 않았다.
점검 전 core SHA-256은
`1869fe1cf898d4ff3a6d5e9054790836442b5e0b81b485fb04bc27de4ebab20a`다.
수정 후에도 같은 hash임을 확인했다.
그룹 경로 및 변형 engine의 새 회귀 테스트는
[`test_extra_arm_launch.py`](../srgc_rebuttal/tests/test_extra_arm_launch.py),
[`test_sr_refresh.py`](../srgc_rebuttal/tests/test_sr_refresh.py),
[`test_switch_repeat.py`](../srgc_rebuttal/tests/test_switch_repeat.py)에 있다.

전체 테스트의 첫 실행에서는 로컬 `torch` 부재로 cache 복구 및 매-step checkpoint
테스트를 실행하지 못했다. GPU 환경을 바꾸거나 테스트를 통과했다고 처리하지 않는다.
검증 결과는 아래 실행 기록에 남긴다. H100 admission 및 7B 실행 검증은 별도로 필요하다.

## 6. 다른 작성된 실험 코드

전체 과거 실험은 [전체 실험 명세](EXPERIMENTS_COMPLETE_GUIDE_KO.md)에 이미 있다.
아래 코드도 존재하지만 위 온라인 Limitation 실험과 같은 결과로 합치지 않는다.

| 기존 코드 | 용도 | 구분할 점 |
| --- | --- | --- |
| `run_srgc_newseeds.sh` | 과거 seed 3/4의 d0/100/400에서 사전 고정 D와 후속 학습 | 현재 seeds 5-9 온라인 queue가 아님 |
| `run_selector_pair_srgc_repeat.sh` | 보존 checkpoint의 반복 진단 | 독립 training replicate나 `switch_repeat`가 아님 |
| `run_selector_pair.sh` | 42개 Pair continuation과 비용/목표 비교 | 과거 frozen 실행의 후속 처리와 새 periodic-refresh 실행 구분 |
| `run_rloo.sh` | 학습 objective를 바꾼 대조 | direction의 인과적 ablation이 아님 |
| `run_additional_experiments.sh` | Qwen 크기별/다른 domain 매트릭스 | 같은 온라인 Switch를 새 backbone에서 검증한 결과가 아님 |
| `run_reference_axes.sh`, `run_reliability_budget.sh` | reference 예산과 진단 신뢰도 | 진단 횟수를 실제 온라인 비용에 무조건 가산하지 않음 |
| `run_mopps_comparison.sh`, `run_low_order.sh`, `run_method_choice.sh` | 다른 선택·재사용 방법의 후속 비교 | 각 frozen protocol의 조건과 완료 기준 필요 |

이 목록은 모든 역사적 실험을 지금 재시작하라는 의미가 아니다. 현재 요청의
Limitation 대응과 코드/실험의 이름이 비슷하다는 이유로 서로 대체하지 않는다.

## 7. 실행 기록

| 날짜 | 항목 | 실제 수행 | 결과/다음 조건 |
| --- | --- | --- | --- |
| 2026-09-28 | 구현 inventory | Limitation, 현재 plan/engine/launcher/test 확인 | 구현됨/미구현 분리, 신규 과학적 결과 없음 |
| 2026-09-28 | 변형 engine + launcher 회귀 | 로컬 CPU 테스트 | 19개 통과, 실제 GPU 성능 검증 아님 |
| 2026-09-28 | 선별 규칙·캐시 재사용·비용·storage·shell 회귀 | 관련 9개 test module 실행 | 88개 통과, core hash 불변 확인 |
| 2026-09-28 | 전체 suite 첫 점검 | 210 tests 수집·실행 시도 | torch 부재 2 errors, import identity 1 failure 발견 후 수정; 16 skipped |
| 2026-09-28 | 전체 CPU 범위 재점검 | torch 필수인 두 항목을 명시적으로 제외하고 나머지 208 tests 실행 | 192개 통과, 선택 의존성 관련 16개 skipped; 나머지 오류 없음 |
| 2026-09-28 | H100 실행 | 실행하지 않음 | 로컬 드라이버 사용 불가, 실제 대상 노드/빈 allocation 확인 필요 |
| 2026-09-28 | 우선순위 지정 | P0-P3와 신규 자원/구현 순서 기록 | 문서 변경만 수행; 작업 실행·중단·계획 변경 없음 |
| 2026-09-30 | E03/E09/E10 구현 (`997cab9`) | `replicate<k>-<arm>`, `direction_*`, `sr_hold` engine·launcher·results·분석 등록, 회귀 테스트 추가 | 전체 SR-GC suite 318개 통과 (proto-ml; torch 환경 9 skipped), frozen core hash 불변, shell 문법 검사 통과 |
| 2026-09-30 | H100 실행 | 실행하지 않음 | 새 arm의 GPU 결과 없음; 배정표에 80개 tuple 추가 |

두 제외 항목은 `test_build_cache.CacheTests.test_resume_after_last_receipt_exports_without_loading_model_or_regenerating`와
torch를 import하는 `test_step_checkpoints` 모듈이다. 이 범위를 통과했다고
기록하지 않는다. H100의 기존 환경에서는 `python -m unittest discover -s srgc_rebuttal/tests`
전체를 다시 실행해 확인한다. shell 문법, CLI help, 문서의 로컬 링크/스크립트 경로도 확인했다.

새 GPU 작업을 실행할 때 이 표에 `(dataset, seed, arm)`, node/allocation,
시작 시각, code commit, active plan 경로와 hash, 로그 경로, prefix hash를 남긴다.
완료 시 endpoint/cost receipt 경로, 총 updates, reward, 비용 완결 여부와 실패/재시도
내역을 붙인다. 노드에서 학습 프로세스와 로그가 확인되기 전에는 `실행 중`으로 쓰지 않는다.
