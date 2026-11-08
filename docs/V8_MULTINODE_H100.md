# V8: 한 명령으로 OLMo·Qwen 노드 자동 배정

모든 4×H100 노드에서 같은 명령을 실행한다.

```sh
sh "$EXPORT_CODE_ROOT/scripts/run_v8_nodes.sh"
```

별도 모델명이나 노드별 역할을 지정하지 않는다. 한 worker가 OLMo MATH와 Qwen
MBPP 양쪽의 준비된 작업을 확인하고, 한 번에4GPU 작업 하나를 실행한다. 작업이
끝나면 다음 cohort부터 확인한다. 한쪽이 체크포인트를 기다리거나 완료되면 다른
쪽의 준비된 작업을 가져온다. 모델별 고정 노드 할당이나 GPU 분할은 하지 않는다.

## 실행 환경

현재 paper runner는 `offpolicy-misranking-paper-v2`의
`3d0c08336950d2697dc66cbe3570a230e449f33b`이다. 그 checkout의 최상위 폴더에서
실행한다. 필요한 환경은 이미 설치한 H100 환경을 사용한다.

| 변수 | 용도 |
| --- | --- |
| `EXPORT_CODE_ROOT` | 최신 실행 도구를 받는 별도 offpolicy-misranking checkout |
| `V8_CODE_ROOT` | 원래 seed 실험에 사용한 고정 trainer checkout. 변경하지 않는다. |
| `V8_ROOT` | 모든 노드에 같은 절대경로로 연결된 공유 실험 폴더 |
| `PAIR_PYTHON` | 기존 OLMo 환경 Python. 미지정 시 공통 Python 사용 |
| `SWITCH_PYTHON` | 기존 Qwen 환경 Python. 미지정 시 공통 Python 사용 |
| `V8_PYTHON` | 공통 Python 및 controller 실행 Python. 선택 사항 |

controller는 `V8_PYTHON`, `PAIR_PYTHON`, `SWITCH_PYTHON`, `python3` 순서로
사용 가능한 설정을 선택한다. cohort 자식은 `V8_PYTHON_OLMO_MATH` 또는
`V8_PYTHON_QWEN_MBPP`의 명시 설정, 해당 `PAIR_PYTHON`/`SWITCH_PYTHON`, 공통
Python 순서로 선택한다. 각 자식 환경의 실제 Python/package 버전도 별도로 확인한다.
새 환경을 만들거나 package를 자동 설치하지 않는다.

도구 checkout을 갱신한 뒤 같은 단일 실행 명령을 모든 노드에서 사용한다.
`EXPORT_CODE_ROOT`는 `V8_CODE_ROOT`와 분리한다.

```sh
git -C "$EXPORT_CODE_ROOT" fetch origin master
git -C "$EXPORT_CODE_ROOT" checkout --detach FETCH_HEAD
export V8_RUNNER_ROOT="$PWD/v8/experiments"
sh "$EXPORT_CODE_ROOT/scripts/run_v8_nodes.sh"
```

모든 노드의 도구·paper runner·trainer·설정과 cohort별 환경 버전은 같아야 한다.
GPU 번호가 다른4GPU 그룹이면 `--gpus 4,5,6,7`을 붙인다. 겹치는 GPU 그룹은
cohort에 관계없이 모델을 실행하기 전에 거부한다.

기존 전체 실행기를 사용 중이면 체크포인트를 저장하며 종료한 뒤 모두 같은
실행기로 전환한다. 기존 결과·학습 기록·체크포인트를 삭제하거나 초기화하지 않는다.
생성된 큐는 coordinator 버전을 고정한다. 기존 큐의 binding과 다른 버전은 거부한다.
이 변경은 이미 실행 중인 옛 버전의 큐를 자동으로 다시 쓰거나 이전하지 않는다.

## 동시 학습 상한

| 현재 실험 | 이론상 동시 학습 작업 상한 | 모든25-step prefix가 준비된 첫 시점 |
| --- | ---: | ---: |
| OLMo MATH | 55 | 10 |
| Qwen MBPP | 55 | 10 |
| 합계 | 110 | 20 |

각 실험의5seed마다 SR 분기10개와 On trajectory 또는 그 완료 후 열리는 Uniform
분기1개가 동시에 실행 가능한 학습 작업의 상한이다. 부모 On과 후속 Uniform은
동시에 실행되지 않는다. 각 작업은4GPU다. 두 실험의 준비된 학습 작업을 모두
동시에 실행한다는 그래프상 상한은110노드/440H100이다. 이 수치는 구매·할당 권장량,
실제 H100 동시 실행 확인 또는 속도 향상 측정값이 아니다. 체크포인트 생성 시점과
작업 종료 시점에 따라 실제 동시 작업 수는 작아진다.

실행기 자체에는 고정 노드 수 제한이 없다. 평가 단계의 작업 수는 예산이 실제로
선택한 checkpoint와 중복 제거 결과에 따라 달라지므로 위 학습 상한과 구분한다.
노드를 추가하면 준비된 작업이 있을 때 자동 합류한다.

## 체크포인트 재사용과 단계

범위는 기존 seeds5–9의 rebuttal Branch Bank다. OLMo MATH는 주 실험, Qwen
MBPP는 다른 모델·과제의 재현 실험이다. 두 cohort로 모델 효과만 분리하지 않는다.
새 seed의 개발·확증 실험과 E8 양방향 전환은 포함하지 않는다.

두 실험 모두 기존 `prefix-ready.json`과 `prefix.pt`를 찾아25-update 공통 상태에서
분기한다. checkpoint 해시·step·trainer 구현을 확인하고 모델·optimizer 상태를
불러온다. 저장된 process RNG가 있으면 복원한다. 같은 v8 작업은 `latest.pt`에서
재개한다. 다른 방법의 기존275-update 결과를 새 v8 작업 완료로 자동 대체하지 않는다.

각 cohort는 독립적으로 준비·입력 검증 → trajectory와 checkpoint가 준비된 분기 →
모든 학습 완료 확인 → 비용만으로 예산 고정 → 예산 연장 → 최종 평가 → 수집·분석
순서를 지킨다. 다른 cohort의 단계와 예산을 섞지 않는다. 한쪽의 GPU 작업 실패는
기록하고 그 cohort의 새 배정을 멈춘다. 정상 cohort의 남은 작업을 마친 뒤 전체
worker는 실패 종료 코드를 반환한다. 환경·입력 사전검사 오류는 가려서 진행하지 않는다.

기존 trainer와 방법·optimizer·seed·난수 설정은 그대로다. 학습 배치 reward와
25-update checkpoint를 보존하고 주기적 held-out 평가는 강제하지 않는다. 최종 평가는
별도이며, 측정하지 않은 중간 held-out 값을 학습 reward로 대신 채우지 않는다.

평가는 seed별 bundle과 실제 선택한 checkpoint의 run/update/hash를 사용한다.
예산 endpoint가 연장 run이나 전환 전 trajectory에 있으면 그 위치에서 평가한다.
선택한 checkpoint가 없거나 달라졌으면 실패로 기록한다.

## 현황과 재개

```sh
sh "$EXPORT_CODE_ROOT/scripts/run_v8_nodes.sh" --status
```

양쪽의 `phase`, `known_task_counts`, `leased_tasks`, `failure`가 함께 출력된다.
worker는60초마다 현황을 출력한다. 자식 출력은 터미널과 각 cohort 로그에 남긴다.
노드마다 같은 GPU 그룹에서 worker 하나를 실행한다.

Ctrl-C/SIGTERM은 자식 실행기에 전달된다. 체크포인트 저장을 기다린 뒤 같은 명령으로
재개한다. 실제 실패 원인을 고친 뒤에는 같은 명령에 `--retry-failed`를 붙인다.
기존 시도·실패를 지우지 않고, 시간 추정만으로 lease를 빼앗지 않는다.

큐·시도·로그는 `$V8_ROOT/<cohort>/.multi-node/`, 학습 결과는 각 `runs/`에 남는다.
각 cohort의 마지막 수집 뒤 기존 history exporter도 실행한다.

## 검증 범위

기존 단일 cohort 검사와 추가 자동 배정 검사를 함께 실행했다. 실제 v8 CPU
ToyBackend의 작업자3개가 두 cohort를 모두 처리하고, 동일 worker의 GPU 작업이
겹치지 않으며 모든 예산 endpoint를 평가하는지 확인한다. 노드 역할 고정 없이
cohort를 오가고 한 cohort의 대기·작업 실패가 정상 cohort 작업을 막지 않는지도 검사한다.
현재5seed/10SR 분기 설정의55개 준비된 학습 작업은 별도 의존관계 fixture로 검증한다.
원래 trainer Python30파일의 해시는 그대로다.

실제 다중 H100 처리량과 네트워크 파일시스템의 노드 간 잠금은 아직 측정하지 않았다.
같은 공유 파일시스템에서 POSIX flock이 노드 사이에 동작해야 한다. 노드 구매·할당이나
GPU 작업의 자동 제출은 수행하지 않는다.

검증 기록: `research/v8-auto-dispatch-20261108/`. 초기 분리 실행과 예산 endpoint
수정 이력은 `research/v8-multinode-20261108/`, `research/v8-budget-endpoint-20261108/`에 보존한다.
