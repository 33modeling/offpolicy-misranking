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

## 실행 순서를 반영한 병렬 수

**110노드를 실제 실행의 최대 동시 노드 수나 필요한 노드 수로 안내한 답변은
철회한다.**55+55는 부모 체크포인트가 모두 준비됐다는 정적 학습 그래프의 폭이다.
실제 실행에서는 부모가 진행하는 동안 앞선 분기가 끝나므로 그 수를 노드 필요량으로
바로 환산할 수 없다.

모든 seed25 prefix가 준비된 첫 시점의 작업은 OLMo10개·Qwen10개, 합계20개다.
이후 준비된 작업 수는 부모 checkpoint 생성 속도와 기존 분기의 완료 속도에 따라
변한다. 실행기는 준비된 작업만 노드에 배정한다.

기존 cohort 설정에 기록된 평균GPU시간을 사용하고, 모든 seed가 같은 속도로
진행하며 준비된 SR분기를 즉시 시작한다고 가정한 **주 학습 구간만의 근사 추정**은
다음과 같다. 작업 하나는4GPU다.

| 계산 범위 | 추정 동시4GPU 작업 피크 |
| --- | ---: |
| OLMo의 On trajectory와 SR분기 | 40 |
| Qwen의 On trajectory와 SR분기 | 30 |
| 두 실험을 같은 시각에 시작한 위 구간의 합계 피크 | 55 |

각 cohort의 피크 시각이 달라 합계 피크를40+30으로 계산하지 않는다. 이 계산은
model초기화·I/O·checkpoint 비용과 실제 seed별 속도 차이를 제외한다. Uniform분기,
예산 연장, 최종 평가도 포함하지 않는다. 특히 OLMo 평가 시간은 기존 설정에 없다.
따라서 **55 역시 전체 파이프라인 최대치나 확보할 노드 수가 아니다.** 실제 전체
동시 실행 최대치와 노드별 완료 시간은 아직 측정하지 않았다.

근사 계산의 입력·가정·전체 event timeline은
`research/v8-concurrency-correction-20261108/`에 보존한다. 새로운 GPU 측정이나
속도 향상 결과로 사용하지 않는다.

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
현재5seed/10SR 분기 설정의55개 준비된 학습 작업 fixture는 정적 그래프 검사이며 실제 시간에 따른 동시 실행 최대치 검증이 아니다.
원래 trainer Python30파일의 해시는 그대로다.

실제 다중 H100 처리량과 네트워크 파일시스템의 노드 간 잠금은 아직 측정하지 않았다.
같은 공유 파일시스템에서 POSIX flock이 노드 사이에 동작해야 한다. 노드 구매·할당이나
GPU 작업의 자동 제출은 수행하지 않는다.

검증 기록: `research/v8-auto-dispatch-20261108/`. 초기 분리 실행과 예산 endpoint
수정 이력은 `research/v8-multinode-20261108/`, `research/v8-budget-endpoint-20261108/`에 보존한다.
