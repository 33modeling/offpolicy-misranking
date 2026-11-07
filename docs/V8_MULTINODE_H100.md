# V8 Branch Bank: 여러 H100 노드에서 실행

같은 공유 폴더를 보는 4×H100 노드마다 같은 명령을 실행한다. 노드 하나가
4GPU 작업 하나를 맡고, 끝나면 다음 seed·분기·평가 작업을 가져온다. 노드 3개면
독립 작업을 최대 3개까지 함께 실행한다. 한 학습 작업을 여러 노드로 쪼개지는 않는다.

범위는 기존 seeds 5–9의 `rebuttal_bank`이다. OLMo MATH가 주 실험이고 Qwen
MBPP가 재현 실험이다. 새 seed의 개발·확증 실험이나 E8 양방향 전환은 포함하지 않는다.

## 각 노드에서 실행

기존에 설정한 다음 환경을 그대로 사용한다.

| 변수 | 용도 |
| --- | --- |
| `V8_PYTHON` | 기존 H100 환경의 Python 실행 파일 |
| `V8_CODE_ROOT` | 원래 seed 실험에 사용한 고정 trainer checkout. 변경하지 않는다. |
| `V8_ROOT` | 모든 노드에 같은 절대경로로 연결된 공유 실험 폴더 |
| `EXPORT_CODE_ROOT` | 최신 실행 도구를 받는 별도 offpolicy-misranking checkout |

기존 `run_h100.sh ... rebuttal` 전체 실행기를 사용 중이면 체크포인트를 저장하며
종료한 뒤, 모든 노드에서 아래 실행기로 바꾼다. 두 전체 실행기를 섞어 돌리지 않는다.
기존 결과·학습 기록·체크포인트를 삭제하거나 초기화할 필요는 없다.

현재 paper runner는 `offpolicy-misranking-paper-v2`의
`3d0c08336950d2697dc66cbe3570a230e449f33b`이다. 아래 명령은 그 checkout의
최상위 폴더에서 실행한다. `EXPORT_CODE_ROOT`는 `V8_CODE_ROOT`와 분리한다.

```sh
git -C "$EXPORT_CODE_ROOT" fetch origin master
git -C "$EXPORT_CODE_ROOT" checkout --detach FETCH_HEAD
export V8_RUNNER_ROOT="$PWD/v8/experiments"
sh "$EXPORT_CODE_ROOT/scripts/run_v8_nodes.sh" olmo_math
```

위 명령을 배정받은 노드마다 실행한다. 모든 노드의 실행 도구·paper runner·trainer·
설정·Python/package 버전은 같아야 한다. 먼저 시작한 노드가 준비를 한 번 수행하고,
나중에 시작한 노드는 같은 큐에 합류한다. 큐 생성 후 도구를 업데이트하려면 전체
worker를 종료하고 검증해야 한다. 버전이 다른 worker는 기존 큐에 합류하지 못한다.

GPU 번호가 다른 4GPU 그룹이면 실행 명령에 `--gpus 4,5,6,7`을 붙인다. 같은
노드에서 겹치는 GPU 그룹을 다시 지정하면 모델을 실행하기 전에 거부한다.

Qwen MBPP는 해당 seed 입력이 있는 환경에서 cohort만 바꾼다.

```sh
sh "$EXPORT_CODE_ROOT/scripts/run_v8_nodes.sh" qwen_mbpp
```

동일한 GPU 그룹에서 OLMo와 Qwen worker를 함께 실행하지 않는다. 다른 노드나
서로 겹치지 않는 4GPU 그룹을 배정한다.

## 현황·재개

```sh
sh "$EXPORT_CODE_ROOT/scripts/run_v8_nodes.sh" olmo_math --status
```

출력의 `phase`, `known_task_counts`, `leased_tasks`, `failure`를 확인한다. 아직
발견하지 않은 후속 작업은 전체 개수로 미리 세지 않는다. worker는 실행 중 60초마다
현황을 출력하고, 각 자식 프로세스 출력은 그대로 터미널과 로그에 남긴다.

Ctrl-C/SIGTERM은 자식 실행기에 종료를 전달한다. 체크포인트 저장을 기다린 다음
같은 명령으로 재개한다. 마지막 작업자가 사라졌다는 시간 추정만으로 lease를
빼앗지 않는다. 실제 실패는 새 배정을 멈추고 원래 실패·시도 기록을 보존한다.
원인을 고친 뒤 명시적으로 재시도한다.

```sh
sh "$EXPORT_CODE_ROOT/scripts/run_v8_nodes.sh" olmo_math --retry-failed
```

## 실행 순서와 보존

- 준비·입력 확인 → trajectory와 부모 체크포인트가 준비된 분기 → 모든 학습 완료
  확인 → 비용만으로 예산 고정 → 예산 연장 → 최종 평가 → 수집·분석 순서다.
- 부모 trajectory가 계속 학습해도 이미 저장한 불변 체크포인트의 분기는 다른
  GPU 그룹에서 먼저 시작한다. 실행 중인 작업이 있다는 이유로 단계를 건너뛰지 않는다.
- 기존 runner의 학습·방법·optimizer·난수·seed 설정은 유지한다. 같은 task의
  별칭은 한 번만 실행하며, 다른 정의가 같은 ID를 쓰면 거부한다.
- 학습 배치 reward와 25-update 체크포인트는 기존 runner가 보존한다. 25-update마다
  held-out 평가를 강제하지 않는다. 최종 평가는 별도로 실행하며, 측정하지 않은
  중간 held-out 값을 학습 reward로 대신 채우지 않는다.
- 평가 작업은 seed별 입력 bundle과 checkpoint·평가 protocol의 해시를 사용한다.
- 큐·시도·로그는 `$V8_ROOT/<cohort>/.multi-node/`, 기존 학습 결과는
  `$V8_ROOT/<cohort>/runs/`에 남는다. `EXPORT_CODE_ROOT`가 설정되어 있으면
  마지막에 기존 history exporter도 실행한다.

## 검증 범위

70개 소프트웨어 검사와 실제 v8 CPU ToyBackend의 작업자 3개·28개 작업을
검증했다. 중복 실행 없이 완료했고 seed마다 다른 평가 문항 집합도 구분했다.
원래 trainer Python 30개 파일의 해시는 그대로다.

다중 H100의 처리량·속도 향상과 실제 공유 파일시스템의 노드 간 잠금은 아직
측정하지 않았다. 모든 노드는 POSIX `flock`이 노드 사이에서도 동작하는 같은 공유
파일시스템을 사용해야 한다. 새 노드를 구매·할당하거나 GPU 작업을 자동 제출하는
기능은 없다. 검증 기록은 `research/v8-multinode-20261108/`에 있다.
