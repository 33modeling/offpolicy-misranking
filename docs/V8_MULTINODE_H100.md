# V8: 한 명령으로 OLMo·Qwen 노드 자동 배정

모든 4×H100 노드에서 같은 명령을 실행한다.

```sh
sh scripts/run_v8_nodes.sh
```

별도 모델명이나 노드별 역할을 지정하지 않는다. 한 worker가 OLMo MATH와 Qwen
MBPP 양쪽의 준비된 작업을 확인하고, 한 번에4GPU 작업 하나를 실행한다. 각 cohort의
원본 Runtime 초기화와 CPU 준비는 독립적으로 진행한다. Qwen의 준비가 끝나지 않아도
검증·준비가 끝난 OLMo는 시작할 수 있고, 반대 방향도 같다. 공유 잠금 안에서 실제 점유 중인 작업 lease를
세고 적게 배정된 cohort부터 새 작업을 가져온다. 기존 Qwen 작업10개가 실행 중이면
추가 worker는 준비된 OLMo 작업을 우선한다. 한쪽이 체크포인트를 기다리거나 완료되면
다른 쪽의 준비된 작업을 가져온다. 공유 잠금은 배정 판단·작업 claim 동안만 잡으며,
모델 로딩·학습·준비 작업 중에는 잡지 않는다. 모델별 고정 노드 할당이나 GPU 분할은
하지 않는다.

## 기존 환경 자동 사용

기존 코드 저장소에서는 위 명령 한 줄을 실행한다. 오프라인 배포 묶음에서는
최상위의 `sh run.sh` 한 줄을 실행한다. 새로 export하거나 경로 인자를 붙이지 않는다.

새 bootstrap은 기존 `scripts/run_srgc.sh`, `scripts/run_srgc_qwen35.sh`의 규칙을
사용한다. 기존 `OM_WORK`/`GROUP_VOLUME`/`OM_USER`를 우선 사용하며, 미지정 기본은
`/group-volume/minsoo3.kim/offpolicy-misranking`이다. 기존 `.venv-cu126/bin/python`을
재사용하고 `PAIR_PYTHON`/`SWITCH_PYTHON` 등 기존 override도 보존한다.

원래 trainer는 기존 `V8_CODE_ROOT`/`OM_CODE_ROOT`, 저장된 locator의 원본 코드 경로,
현재 저장소·상위 저장소 및 기존 user-volume 위치에서 찾는다. 기존 run의 구현 해시와
prefix 해시는 원래 검증 경로에서 확인한다. runtime은 이제 이 코드 저장소의 `v8/experiments`에 함께 포함된다. 그 사본을
우선 사용하며 별도 원고 저장소나 수동 runtime 경로를 요구하지 않는다. source는 paper commit
`3d0c08336950d2697dc66cbe3570a230e449f33b` 그대로다.

공유 결과 경로는 기존 `V8_ROOT`, 또는 같은 작업 폴더의 `v8/`를 쓴다. exporter는
자기 위치를 사용한다. 원래 경로·checkpoint·venv를 옮기거나 다시 만들지 않는다.
명시된 Python이 깨졌거나 기존 입력 신원이 다르면 다른 실험으로 조용히 바꾸지 않고
실패 이유를 출력한다.

GitHub 연결이 없는 기존 클러스터에는 소스·최신 실행기·고정 v8 runtime을 포함한
오프라인 묶음을 사용한다. 노드에서 git fetch, package 설치, 모델 재학습용 초기화,
수동 경로 입력은 요구하지 않는다. 모든 노드는 같은 묶음 버전을 사용한다.

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

## 저장된 원본 런타임 선택

기존 saved-prefix 런처의 원본 package 검증을 재사용한다. OLMo의 알려진
보존 런타임 두 버전은 전체 보존 package·engine 바이트를 검증한 뒤 자식이
모델을 만들기 전에 실제 원본 모듈을 불러온다. 현재 엔진을 옛 해시로 바꾸어
표시하거나 다른 체크포인트를 새 결과로 대체하지 않는다.

Qwen은 기존 코드가 기록한 `sha256(engine_digest + adapter_digest)`를 그대로
검증한다. 기본 엔진 해시만 비교하지 않는다. 각 seed의 run.json과
prefix-ready.json도 같은 원본 신원이어야 한다. 모르는 원본 해시나 모델 revision
차이는 그대로 거부하고 원인별로 출력한다. prefix가 없으면 코드 경로 대신
missing_prefix로 표시한다.

수정 전 준비 단계에서 실패했고 아직 작업/입력 고정 기록이 하나도 없는 큐만,
동일한 trainer/config 환경에서 새 검증 코드로 준비를 다시 시도한다. 이전
binding·실패 기록은 setup-history에 보존한다. 학습이 시작된 큐는 이 방식으로
바꾸지 않는다.

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
sh scripts/run_v8_nodes.sh --status
```

양쪽의 `phase`, `known_task_counts`, `leased_tasks`, `failure`가 함께 출력된다.
worker는60초마다 현황을 출력한다. 자식 출력은 터미널과 각 cohort 로그에 남긴다.
노드마다 같은 GPU 그룹에서 worker 하나를 실행한다.

`--status`의 `dispatch_workers`에는 각 worker가 실제 기록한 제어 코드 해시와
정책 버전이 나온다. 표 형식 `status`는 `dispatch current <hash>`(현재 checkout과
같음), `dispatch different <hash>`(다름), `dispatch unverified`(기록 없음)를 구분한다.
정책 파일이 디스크에 있다는 것만으로 실행 중인 worker에 적용됐다고 판단하지 않는다.
현재 배정 정책은 `independent-cohort-startup-live-leases-v3`다. 유휴 worker는 배정 전에 모든 작업을
중복 검사하지 않으며, 그 노드의 입력 검증을 통과한 뒤에만 claim할 수 있다.
원본 Runtime 초기화에는120초 제한을 두고, 한 cohort가 초기화에 응답하지 않으면
오류를 표시하고 다른 cohort를 계속한다. 학습 실행 시간에는 이 초기화 제한을 적용하지 않는다.

Ctrl-C/SIGTERM은 자식 실행기에 전달된다. 종료 유예는10초이며 강제 종료되면 마지막으로
저장된 체크포인트에서 같은 명령으로 재개한다. 종료 직전 업데이트의 저장은 보장하지 않는다.
실제 실패 원인을 고친 뒤에는 같은 명령에 `--retry-failed`를 붙인다.
기존 시도·실패를 지우지 않고, 시간 추정만으로 lease를 빼앗지 않는다.

큐·시도·로그는 `$V8_ROOT/<cohort>/.multi-node/`, 학습 결과는 각 `runs/`에 남는다.
각 cohort의 마지막 수집 뒤 기존 history exporter도 실행한다.

## 노드 실행기 점검 결과 (11-08 Claude)

- GPU 점유 잠금은 기존 SRGC worker와 같은 `$OM_WORK/srgc-rebuttal/gpu-node-locks/`의
  GPU UUID 파일(sha256)을 쓴다. 기존 SRGC worker가 같은 GPU를 잡고 있으면 V8 worker는
  시작하지 않고 그 이유를 출력한다. 이전 `/tmp/srgc-v8-gpu-locks`는 다른 사용자가 만든
  폴더나 쓰기 금지 `/tmp`에서 `Permission denied`로 멈췄다.
- 할당된 GPU 중 하나라도 4,000 MiB 넘게 쓰고 있으면 새 GPU 작업을 받지 않고 5분마다
  해당 GPU와 프로세스 번호를 출력한다. nvidia-smi가 4개 GPU를 보여 주지 않는 노드는
  큐를 건드리기 전에 거절한다.
- 한 cohort의 준비 단계나 작업이 실패해도 그 cohort만 멈추고 다른 cohort는 계속한다.
  끝에 실패한 cohort와 이유를 다시 출력하고 종료 코드 1을 반환한다.
- 큐 기록과 코드가 다르면 무엇이 다른지와 할 일을 출력한다. 다른 노드가 큐를 새 코드로
  바꾸면 이전 코드로 돌던 worker는 그 cohort 작업을 더 받지 않는다.
- 다른 묶음으로 시작한 큐는 일반 실행에서 자동으로 바꾸지 않는다. 아직 update가 하나도
  끝나지 않았다면 모든 노드의 worker를 멈춘 뒤 한 노드에서만
  `sh scripts/run_v8_nodes.sh --new-queue`를 한 번 실행한다. 기존 큐는 지우지 않고
  `.multi-node.superseded-<시각>`으로 그대로 보관하고 새 큐를 만든다. 완료된 update가 있거나
  작업 lease가 잡혀 있으면 거부한다. 나머지 노드는 같은 묶음으로 일반 명령을 실행한다.
- cohort별 Python은 기존 실행기와 같은 순서로 고른다. OLMo MATH는 `PAIR_PYTHON`,
  Qwen MBPP는 `QWEN_PYTHON` 다음 `SWITCH_PYTHON`, 없으면 기존 `.venv-cu126`이다.
  OLMo가 MBPP/Qwen 인터프리터를 물려받던 문제를 고쳤다.
- `--status`의 `leased_tasks`에 작업 로그 경로와 `log_written_seconds_ago`(마지막 출력 뒤
  지난 초)가 나온다. audit 단계는 공백이 들어간 경로도 그대로 넘기고, `harvest.sh`는
  worker가 로그를 쓰는 중에도 `tar`의 "file changed" 경고로 실패하지 않는다.

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

## 로딩 표시와 종료

기존 큐를 재개할 때는 `binding.json`의 coordinator/runner 해시와 Python·package 버전을
먼저 읽는다. 현재 코드와 다르면 오프라인 묶음의 `v8/queue-runtimes.zip`에서 원래 코드를
자동 선택한다. 설치나 GitHub 접속은 필요 없으며 큐·학습 결과·control.lock을 바꾸지 않는다.
특정 과거 버전이라고 가정해 배포한 단일 호환 묶음은 이 자동 선택 묶음으로 대체한다.
원본 코드가 묶음에 없거나 기존 Python 환경이 달라졌으면 해당 원인을 출력하고 거부한다.
신원 검사를 끄거나 새 코드에 옛 해시를 붙여서 통과시키지 않는다.
기존 큐에도 현재의 공통 배정기를 적용한다. 각 cohort의 원본 Runtime은 그 큐에 기록된
코드·Python으로 별도 자식 프로세스에서 실행하고, 새 배정기는 두 cohort의 실제 lease를
함께 비교한다. 원래 binding.json을 고치지 않는다. 새 제어 정책과 배정기4파일 해시,
원본 binding은 `$V8_ROOT/.v8-dispatch-policies/`의 별도 불변 기록에 남긴다.
프로세스 간 통신은 상속받은 로컬 socketpair이며 네트워크 listener를 만들지 않는다.
현황 확인 명령은 원본 코드 전환 없이 읽기만 수행한다.

새 제어 정책은 원본 `cluster_child.py`의 해시를 검증한 뒤 별도 `v8_task_entry.py`로
같은 child의 `main()`을 호출한다. 원본 Context가 backend를 만든 뒤 샘플링 인자와
기존 종료 조건을 보존한 always-false 관찰 조건을 추가한다. 5초마다 실제 증가한
디코딩 길이·호출 번호·반환된 decode step을 기록한다. `generation_start`는 진입 기록이고
생성 완료나 optimizer update가 아니다. `generation_finished`도 생성 함수 반환만 뜻한다.
생성 호출 안에서30분간 디코딩 단계가 늘지 않으면 해당 rank가 스택 기록을 시도하고
exit70으로 실패한다. 기존 checkpoint는 남기며, 성공으로 처리하거나 자동 재학습하지 않는다.
이 제한은 전체 학습 시간이나 selection refresh 총시간의 제한이 아니다. 로그 경로가
깨졌거나 콘솔 쓰기가 막혀도 감시 종료가 그 쓰기를 무한 대기하지 않는다.
CPU 작은 무작위 OLMo 모델에서 응답8개·기존 종료조건·출력 바이트·난수 상태가
원래 generate와 같은지 검사했다. 실제 H100 생성 정체의 원인과 해결 여부는 별도 확인이 필요하다.

현재 checkout을 선택한 경우도 실행 전 별도의 불변 사본으로 고정한다. 실행 도중
`git pull`이 되어도 이 worker가 이후 띄우는 child/config/helper는 바뀌지 않는다.
이미 사용한 사본도 로컬 캐시에 남겨 후속 pull 뒤 같은 큐를 다시 열 때 재사용한다.
두 cohort의 Python 경로가 달라도 공통 배정기를 사용한다. 각 원본 Runtime의
NCCL 사전검사와 학습은 해당 cohort의 Python에서 실행한다.
한 cohort의 원본/Python/메타데이터 오류는 그 cohort에만 표시하며 유효한 나머지는 계속한다.
일부 실패를 숨기지 않도록 유효한 cohort가 끝난 뒤 전체 종료 코드는 실패로 남긴다.

`git pull`은 이미 실행 중인 worker의 메모리나 고정 실행 사본을 바꾸지 않는다.
수정 이후 시작한 worker부터 공통 배정기를 사용한다. 실행 중인 Qwen 학습을
멈추거나 큐를 초기화할 필요 없이 유휴 노드의 새 worker가 OLMo를 가져올 수 있다.
기존 worker를 나중에 재실행해도 같은 큐와 저장된 checkpoint에서 재개한다.
25-update snapshot은 보존하고 `latest.pt`만 재개용 최신 상태로 갱신한다.

콘솔 출력을 받는 쪽이 멈춰도 학습·종료가 대기하지 않도록 supervisor는 출력을 로컬
임시 폴더의 `v8-console-<uid>/<launch>.log`에 보존하고 별도 thread로 전달한다.
종료 경로는 콘솔 쓰기나 interpreter 최종 flush를 무한 대기하지 않는다. 기존 tag 없는
worker의 종료 중 새로 생긴 rank도 해당 노드·UID·V8_ROOT 범위에서 다시 찾아 정리한다.
상태 화면의 GPU 유휴 기반 경고는 같은 hostname의 작업에만 적용한다. 다른 노드의
GPU 상태를 이 노드의 수치로 추정하지 않으며 FQDN을 생략하지 않는다.

`Loading weights` 출력만으로 학습 정지를 판단하지 않는다. 현재 상태를 바꾸지 않고
GPU 사용과 이 노드의 현재 attempt 기록·로그를 확인할 때:

```sh
sh scripts/v8_probe.sh
```

현재 노드의 해당 V8 실행을 종료할 때:

```sh
sh scripts/v8_stop.sh
```

원래 checkpoint/queue/lock 파일은 지우지 않는다. 종료 신호 뒤10초 동안 기다리고
남아 있는 해당 실행의 프로세스만 강제 정리한다. 다른 UID·root·launch 또는 재사용된
PID는 신호 대상에서 제외한다. kernel의 중단 불가능한 I/O 대기는 남은 PID로 출력하며
GPU가 해제됐다고 성공 처리하지 않는다.

새 기본 실행기는 별도 supervisor를 사용한다. Ctrl-C/TERM/HUP을 처리하고 두 번째
Ctrl-C는 즉시 정리를 요청한다. 자식 출력 pipe의 EOF를 기다리는 무한 join을 사용하지
않는다.15초마다 rank가 실제 기록한 rollout/gradient_scoring/policy_update 단계와
기록 나이를 출력한다. 이 출력은 완료 update나 checkpoint 저장 확인과 구별한다.

종료 supervisor 자체는 wrapper·진단·종료 경로만 바꾸며 기존 큐의 identity를 바꾸지 않는다.
현재 master에는 별도의 child 실행 환경·NCCL 사전 검사·verifier fallback 수정도 포함된다.
따라서 전체 묶음의 child/coordinator가 이전 묶음과 동일하다고 취급하지 않는다.
이미 작업이 생성된 큐는 첫 업데이트 전이어도 runtime identity가 다르면 자동 변경을 거부한다.
로그가 비었거나 작업 lease가 없어도 다른 노드의 idle worker가 없다는 증거가 아니므로,
기존 큐와 control.lock을 옮기거나 바꾸지 않는다. 현재 실행에는 동일한 수치 코드의
호환 묶음으로 supervisor·종료·진단 경로만 추가할 수 있다.
모델 로딩에서 실제로 멈췄는지에 대한 원격 원인은 로그/노드 관측 전에는 미확인이다.
