# Worker 등록 전 잠금 대기와 자동 재시도

사용자 보고: “새로 잡 할당이 안되고 무한 런칭이야”. 실제 원격 노드의 콘솔·PID·잠금 소유자는 확보되지 않았다. 아래는 로컬에서 재현한 코드 결함과 그 수정 범위다.

Gemma IT child는 worker 등록 전에 패키지 overlay, 모델 snapshot 검증, 실험 session 잠금을 기다릴 수 있었다. 세 경로 모두 blocking flock을 사용했고 coordinator는 같은 child의 종료만 기다렸다. 새 코드를 pull해도 이미 그 안에서 기다리는 Python은 갱신되지 않는다. 기존 준비 plan fast path는 이 세 잠금을 처리하지 않았다.

`startup_locks.py`는 model-chain child의 시작 전 잠금 획득만 nonblocking으로 바꾼다. peer가 소유하면 단계·잠금 정보를 남기고75로 돌아간다. coordinator의 기존 재시도 경로가 같은 모델의 새 child를 실행한다. 잠금이 비었을 때는 원래 함수의 패키지 버전·모델 크기·해시·identity 검증을 그대로 수행한다. 다른 모델로 건너뛰거나 task 완료를 만들지 않는다.

- 실험 session은 기존 shared join을 유지한다. exclusive reset owner가 있으면 chain join만75로 돌아가며, 명시적 reset과 비체인 시작은 기존 보호를 유지한다.
- 패키지·snapshot·다운로드 lease는 실제 획득에 실패했을 때만75로 처리한다. 검증 함수 내부 예외는 원래 원인을 유지한다. worker 등록 후에는 이 재시도 보정을 적용하지 않는다.
- frozen adapter/engine92개 고유 파일을 기존 dff44713과 비교해 모두 같음을 확인했다. 학습 설정·입력·기존 reward·checkpoint를 수정하지 않았다.

## 구버전 child가 이미 잠금에서 대기할 때

같은 실행 checkout에서 다음 명령을 사용한다.

```sh
git pull --ff-only origin master
sh scripts/unstick_gemma_launch.sh --dry-run
sh scripts/unstick_gemma_launch.sh
```

이 도구는 현재 사용자·현재 노드에서 model-chain의 Gemma IT child만 검사한다. 실제 `/proc/locks`의 waiting 기록과 FD의 device/inode를 대조하고, 신호 직전에 PID·process group·coordinator의 시작 세대와 잠금을 재확인한다. session FD가 열려 있다는 사실만으로 종료하지 않는다. torchrun·rank wrapper가 있는 그룹도 제외한다. 확인된 시작 전 대기 child만137로 종료되어 기존 coordinator가 현재 코드를 다시 읽는다. coordinator 전체나 학습 작업을 재시작하는 명령이 아니다. 커널 근거가 없으면 종료하지 않으며 잠금 파일도 삭제하지 않는다.

기존 shell wrapper는 인자를 전달하지 않아 `--help`나 `--dry-run`이 무시될 수 있었다. 인자를 전달하도록 수정하고 Python CLI가 신호 처리 전에 도움말/읽기 전용 모드를 해석하도록 했다.

## 검증

- 수정 전 실제 peer-held file lease에서 패키지·모델2개가 각각5초 제한을 넘겼다(실패2, 대조군 통과4). session도 별도 동일 조건에서5초 이상 대기했다.
- 수정 후 세 경로는75로 반환하고, peer가 해제한 뒤 재실행하면 큐 진입 callback까지 도달했다. 잠금 inode와 실제 소유권이 그대로임도 검사했다. 합성 소형 모델·패키지를 사용한 CPU 검사이며 실제 GPU task 실행을 뜻하지 않는다.
- 핵심 startup/등록/선택적 구버전 복구36개 통과(9.42초). 실제 신호 검사는 테스트가 직접 만든 PID에만 한정했다.
- model chain·node lease·패키지 환경·MBPP 배정·resume 관련307개 통과(82.96초). 별도 reset5개 통과(4.64초). 서로 다른 검사 총348개이며 전체 저장소 전수 검사 결과가 아니다.
- 수정된 production Python은3.8–3.12 구문 검사, shell은 `sh -n`, 패치는 `git diff --check`를 통과했다.

원격40개 노드의 실제 등록·할당 회복이나 속도는 아직 측정하지 않았다. 실제 peer 작업이 계속 잠금을 소유하고 있으면 새 child도 기다리는 상태로 재시도한다. 잠금을 빼앗거나 정상 진행으로 표시하지 않는다. 원격의 현재 정체 원인은 해당 노드의 콘솔/읽기 전용 probe와 대조해야 한다.
