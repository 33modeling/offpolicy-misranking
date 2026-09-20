# 같은 상태에서의 selector 전환 검증

새 진입점: `bash scripts/run_selector_pair.sh`.
기존 `run_selection_switch.sh`, `run_switch_quality.sh`, `run_switch_difficulty.sh`
실험과 결과는 수정하지 않는다. 기존 `fresh_r`, `difficulty`는 아티팩트 키이며,
설명과 결과의 이름은 **on-policy gradient alignment**, **cached success-rate**다.

## 무엇을 검증하는가

같은 모델·옵티마이저·선행 학습 이력에서 두 selector를 비교한다.
고정된 목표 보상 `tau`까지의 비용으로 다음 라벨을 만든다.

```
H = cached의 총 GPU초 - on-policy의 총 GPU초
H_hat > 0 : on-policy 유지
H_hat <= 0: cached success-rate로 전환
```

총비용에는 gradient scoring / 캐시 읽기, 데이터 구성, 입력 검증,
학습 프로세스 시작, 이전 실패·재시도 비용이 포함된다. adaptive에는
현재 상태 진단과 예측 비용도 한 번 더한다. 중간 체크포인트는 평균
step 시간으로 환산하지 않고 해당 실행의 실제 비용 기록과 시각을 연결한다.

GPU를 할당받은 동안의 wall time × GPU 개수이며 FLOPs/순수 커널 시간은 아니다.
학습 cap과 scoring 장부는 분리하지만 **비교하는 총비용에는 scoring을 다시 포함**한다.
동일 총예산의 최종 성능 실험이라고 해석하면 안 된다.

## 실험 설계

- 기존의 인증된 selected-prefix 25 / 50 / 100 updates를 재사용한다.
  새 prefix를 이 스크립트가 자동 학습하지는 않는다.
- 개발 seed 0, 1, 2: 각 상태에서 두 selector를 실행한다. 총 18개 continuation.
- 개발 데이터만으로 고정 ridge(alpha=1, threshold=0)를 학습한다.
  입력은 직전 reward, mixed-reward group 비율, 캐시 성공률 표준편차,
  선행 update 수의 로그다. 향후 평가 보상은 입력으로 쓸 수 없다.
- 테스트 seed 3, 4: 6개 상태의 모든 결정을 먼저 저장한다.
  그 뒤에만 on-policy, cached, adaptive, random을 실행한다. 총 24개 continuation.
- adaptive는 선택한 selector를 **자기 작업 디렉터리에서 다시 scoring·학습·평가**한다.
  결과가 좋은 대조군이나 선택한 대조군의 결과를 복사하지 않는다.
- 모델, optimizer, 원본 캐시, prompt pool, learner 설정, evaluation 설정의
  해시가 일치하지 않으면 paired comparison을 거부한다.
- 단계별 상태는 같은 seed에 종속된다. seed별 요약을 내고, 질문 bootstrap만으로
  학습 seed의 불확실성을 대체하지 않는다.

이 설계는 **세 후보 상태 각각에서 한 번 내리는 전환 결정**을 검증한다.
한 학습 경로에서 반복적으로 판단하는 online stopping policy나 전역 최적 전환
시점을 검증하는 코드는 아니다. 보고서에도 이 범위를 명시한다.
모든 상태에서 cached가 유리하거나 모든 결정이 동일하면, 그것을 학습 단계별
전환 경계를 발견한 결과라고 주장하지 않는다. 개발 라벨의 부호 분포, 테스트
선택 횟수, H 예측 오차와 실제 adaptive 비용 차이를 함께 보고한다.

## 준비와 실행

저장소에서 **이 스크립트 하나만 실행한다.** `init`, `prepare`, `run` 인자나
환경변수를 따로 입력할 필요가 없다. 같은 공유 저장소를 사용하는 **각 빈 4-GPU
노드에서 같은 명령**을 tmux 또는 배치 스케줄러 안에서 실행한다.

```bash
bash scripts/run_selector_pair.sh
```

설정 생성 → 원본 검증·준비 → 개발 곡선 수집 → predictor 학습 → 테스트 결정
고정 → 테스트 학습·평가 → 보고서까지 진행한다. 목표 미도달 등 실험의 검증
조건을 충족하지 못하면 해당 단계에서 중단하며, 결과를 만들어 통과시키지 않는다.

큐는 같은 seed/step 안에서도 **독립된 분기 단위**로 노드를 배정한다.
개발 9개 상태의 두 selector는 최대 **18개 4-GPU 노드**, 테스트 6개 상태의
네 대조군은 최대 **24개 4-GPU 노드**에서 동시에 실행할 수 있다. 이는 배정 가능한
상한이며 클러스터 할당량이나 동일 배속을 보장하지 않는다. 공유 부모 평가 등은
잠금으로 직렬화한다. 개발 완료·predictor 학습·테스트 결정 고정 이후에만 테스트를
시작하므로 개발과 테스트를 합쳐 42개 노드에서 동시에 실행하지는 않는다.
기존 실험 수인 개발 18개, 테스트 24개는 바뀌지 않으며 추가 노드는 중복 실행하지 않는다.

새 실험의 기본값은 목표 보상 **0.35**, 분기당 학습 예산 **87,120 GPU초**,
중간 평가점 9개, eval-k 8이다. 학습 예산은 분기당 24.2 GPU시간이며 scoring과
평가 비용은 별도다. 목표에 실제로 도달한다는 보장은 없다. 수치는 결과를 보기
전에 고정되며, 기존에 고정된 실험의 값을 바꾸지 않는다.

기본 저장소는 `/group-volume/minsoo3.kim/offpolicy-misranking`이다.
그 아래 `runs/selection-switch-v1`의 인증된 prefix와
`runs/olmo3-1025-7b-base-rlzero-grpo-h100-v2` matrix를 사용하고,
새 결과는 `runs/selector-pair-v1`에 저장한다. 원본에는 다섯 seed의
25/50/100 checkpoint와 optimizer가 모두 있어야 한다. 원본 저장소가
마운트되지 않았거나 검증을 통과하지 못하면 GPU 학습 전에 중단한다.

`pair.json`이 없으면 자동 생성하고, 예전 `init`이 남긴 미고정 설정의 빈
target/budget도 자동으로 채운다. 직접 지정한 값은 유지한다. `request.json`이
이미 고정되었거나 준비가 완료되었다면 기본값을 덮어쓰지 않고 그대로 재개한다.
`c78ca17`, `5086e15`, `c054c67`의 실험 manifest/decision도 보존하며, 검토된 실행부 수정만
별도 영수증으로 기록한다. 빈 `{}`나 가짜 학습 결과로 검증을 우회하지 않는다.

기본 GPU 종류는 NVIDIA H100 80GB HBM3다. 준비 완료 후 target, cap, seed,
scoring 방법을 바꿀 수 없다. 실행 중인 checkout은 업데이트하지 않는다.
검토되지 않은 코드 변경이 발견되면 중단한다.

새 worker들은 실행 동안 root의 `.pair.lock`을 공유 잠금(SH)으로 유지한다.
분기 실행 중에는 상태의 `.state.lock`도 SH로 유지하고, 분기별 배타 잠금(EX)으로
중복 배정을 막는다. 상태 입력 준비는 짧은 별도 EX 잠금으로 보호하며 모든 분기가
검증된 뒤 상태 결과를 발행할 때는 상태 EX 잠금을 요구한다. 구버전 상태 큐의
EX 잠금과도 충돌하므로 구버전과 같은 상태를 중복 실행하지 않는다.
설정 준비와 런타임 영수증 발행,
predictor 학습·테스트 결정 고정은 각각의 공유 단계 잠금으로 보호한다.
개발 9개 상태의 결과와 fitting 조건이 검증되어야 predictor를 고정하며,
테스트 6개 상태의 모든 결정이 저장·검증되기 전에는 테스트 학습을 시작하지 않는다.

구버전 단일 controller가 배타 잠금(EX)을 잡고 있으면 새 worker는 GPU 작업을
시작하지 않고 `[WAIT]`를 출력하며 안전한 전환을 기다린다. root·런타임 영수증·
공유 단계 잠금은 15초 간격으로 확인하되 **각 잠금 대기는 최대 180초**다.
준비 충돌로 반환된 exit 75도 셸에서 15초 간격으로 최대 12회 재시도한 뒤
exit 76으로 종료한다. 이 상한은 무한 대기를 막는 운영상 제한이며, 논문의
학습 예산·selector 비용 cap이나 target을 바꾸는 값이 아니다.

상한 전에 잠금이 풀리면 자동으로 계속한다. **exit 76으로 종료된 뒤에는**
원인이 해결되어도 자동 재시작하지 않으므로 같은 실행 명령을 다시 실행해야 한다.
종료만으로 실험이 복구되거나 완료된 것은 아니다. 잠금을 삭제하거나 기존
프로세스를 죽여 우회하지 않으며, 저장된 결과·비용·체크포인트를 초기화하지 않는다.

대기 종료 진단은 정확한 잠금 경로와, 로컬 커널에서 확인할 수 있을 때 해당
잠금 소유 PID를 출력한다. 별도로 진행 메타데이터에서 관측한 노드·단계도
표시하지만, 그 기록이 실제 잠금 소유자를 확정하는 증거는 아니다. 소유 PID가
로컬에서 보이지 않는다고 노드가 죽었다고 판단하거나 잠금을 제거하지 않는다.

`previous single-controller` 대기가 계속되면, 대기 중인 노드의 저장소에서
아래 명령으로 잠금 증거를 **4 KiB 이하 TXT 하나**에 저장한다. Python 표준
라이브러리만 사용하며 frozen 실행 코드를 import하거나 변경하지 않는다.
프로세스 종료·잠금 삭제·GPU 작업·학습 재시작은 하지 않는다.

```bash
bash scripts/check_selector_pair.sh
```

마지막 `[saved]`의 `selector-pair-lock-*.txt`를 전달한다. `ROOT_LOCK`은 그 순간의
공유 잠금 획득 가능 여부, `OWNER confirmed-local`은 로컬 커널의 배타 잠금
소유 PID다. `OBSERVED` 노드는 진행 기록일 뿐 실제 잠금 소유자를 확정하지 않는다.
`CHECKOUT`도 현재 파일의 Git revision이며 이미 실행 중인 Python의 revision이
아니다. 보고서만 만들고 잠금 문제가 복구되었다고 판단하면 안 된다.

구버전에서 처음 전환할 때는 **기존 구버전 pair를 실행 중인 노드에서만**
실행 터미널의 `Ctrl+C`로 정상 종료를 요청하고, 자식 프로세스·비용 영수증
정리가 끝나 명령이 반환된 뒤 checkout을 갱신하고 위 명령을 실행한다.
실행 중인 checkout에 먼저 `git pull`을 하지 않는다. 추가 빈 노드는 새 코드를
받은 뒤 같은 명령으로 참여한다. 별도 MBPP 실행이나 다른 사용자의 프로세스는
중단 대상이 아니다.

### 구버전 root 잠금 소유 노드에서 안전하게 인계

`previous single-controller`는 구버전 controller의 실제 배타 잠금 때문에 새
worker가 참여하지 못한다는 뜻이다. 잠금 파일의 존재 자체가 원인은 아니다.
2026-09-19 전달된 진단에는 `run284168-wts-3`의 최근 학습 진행이 관측되었지만,
이는 **잠금 소유 노드로 확인되었다는 뜻이 아니다**. 대기 노드
`run284441-wts-59`에서는 소유 PID가 보이지 않았다.

`restart_selector_pair.sh` **파일 하나만** 기존 저장소의 `scripts/`에 복사한 뒤,
**기존 Pair를 실행한 노드에서** 다음 하나를 실행한다. Python 도우미가 파일 안에
포함되어 있어 별도 Python 도우미 복사가 필요하지 않다. 실행 중인 checkout에
`git pull`을 먼저 하지 않는다.

```bash
bash scripts/restart_selector_pair.sh
```

주의: 이 인계 도구가 고정한 `6066d11`은 **상태 단위 큐**이며 개발 최대 9개,
테스트 최대 6개 노드다. 위의 분기 단위 18개/24개 큐를 설치하는 명령이 아니다.
분기 큐 전환은 기존 작업이 정리된 뒤 검증된 새 실행 코드로 별도 진행해야 하며,
실행 중인 checkout을 업데이트하거나 확인 목적으로 정상 작업을 재시작하지 않는다.

도구는 로컬 커널에서 실제 배타 잠금 소유 PID를 확인하고, 같은 사용자·Pair root·
checkout의 controller인지 다시 검증한다. 재시작할 코드는 기존 checkout을 그대로
재실행하지 않고, 검토된 분산 실행 버전 `6066d11091c71ef9ec2c43dbc271ff0824c6bc4e`를
`.work/pair-runtimes/`의 별도 경로에 준비한다. 필요한 Git 객체가 없을 때만 정확한
commit을 가져오며 checkout·branch·실행 중인 소스는 바꾸지 않는다. 이미 준비된
경로도 파일 내용을 다시 검증하고, 변조·누락이 있으면 덮어쓰지 않고 중단한다.
이렇게 해야 구버전 checkout에 bash만 복사한 경우에도 구버전 EX controller를
다시 띄우지 않는다. 원래 학습 경로의 소유권과 새 실행 경로의 frozen 호환성을
각각 확인한 다음 종료한다. 학습 중이면 trainer의 기존 검증기로
유효한 로컬 체크포인트를 확인한 뒤 그 controller에만 TERM을 전달한다. 기존
종료 처리가 소유 자식 작업을 정리하고 비용 영수증을 닫으며, controller·자식·
기존 launcher의 종료와 root 잠금 해제를 확인한 후 같은 root와 단계로 재개한다.
이 과정은 **마지막 유효 체크포인트부터의 재개**다. 아직 저장되지 않은 update는
재수행할 수 있으며, 그때 이미 사용한 GPU 비용은 지우거나 환불하지 않는다.

다른 노드여서 소유 PID를 확인할 수 없거나 소유권·체크포인트 검증에 실패하면
TERM 전에 중단하고 실행 중인 작업은 그대로 둔다. TERM 이후 정리 완료나
비용 종료 영수증을 확인하지 못하면 강제 종료·중복 시작을 하지 않고 중단한다.
도구는 lock 파일 삭제, 예산 초기화, 결과·체크포인트 삭제, checkout 업데이트를
하지 않으며 별도 MBPP 작업을 종료하지 않는다. root가 이미 공유 잠금을
허용하면 controller를 종료하지 않고 같은 고정 분산 버전의 GPU·노드 입장 검사를
거쳐 참여한다. Git 다운로드·별도 코드 검증 실패 시 현재 controller는 종료하지 않는다.
유지보수 시 Python 도우미를 수정하면 `python3 scripts/build_selector_pair_handoff.py`로
단일 파일을 재생성하고, 같은 명령의 `--check`로 원본과 일치하는지 검증한다.

`status`는 실행 잠금을 요구하지 않는 읽기 전용 조회다. 실행 중에도 조회할 수
있으며, 준비되지 않은 root를 생성하거나 frozen manifest/런타임 영수증을 바꾸지 않는다.
MBPP status의 출력 함수를 공유하며 한글 집계, 전체 진행표, 번호가 붙은 노드별
작업표와 작업 없는 노드 목록을 같은 형식으로 표시한다. 개발 18개와 테스트 24개를
합쳐 총 42개 분기를 집계하며, 결과 영수증과 곡선 기록이 있어야 DONE이다.
시간 한도 사용률은 완료율과 구분한다. 전체 과학적 검증은 report 단계에서 수행한다.

```bash
bash scripts/run_selector_pair.sh status --watch
```

MBPP처럼 매 갱신마다 조회 프로세스를 새로 실행한다. 기본 갱신 간격은 15초다.
`--watch 5`로 양의 정수 간격을 지정하고, `--all`로 과거 노드와
분기 경로도 볼 수 있다. `--json`은 같은 조회의 구조화된 결과를 출력한다.

단계별 실행도 가능하다.

```bash
bash scripts/run_selector_pair.sh develop  # 개발용 18개 곡선 수집
bash scripts/run_selector_pair.sh report   # 미도달/부적격 목표를 포함한 개발 결과
bash scripts/run_selector_pair.sh fit      # CPU, 개발 데이터만으로 H 예측기 고정
bash scripts/run_selector_pair.sh freeze   # 테스트 6개 결정 고정; 진단 비용 기록
bash scripts/run_selector_pair.sh test     # 별도 adaptive 포함 24개 학습·평가
bash scripts/run_selector_pair.sh status
bash scripts/run_selector_pair.sh report
```

`freeze`도 점유 중인 4-GPU 할당에서 수행한다. 진단은 CPU cache/log 읽기지만
기다리는 GPU 할당 시간을 포함한다. 결과 파일은 다음과 같다.

- `pair.json`: 사전 고정 target, 데이터·코드·실험 설정 해시.
- `development/sN-tN/result.json`: paired 곡선, 관측 H, censoring 상태.
- `model.json`: 개발 데이터만 사용한 predictor와 provenance.
- `test-decisions.json`, `decisions/*/decision.json`: 테스트 시작 전 고정 결정.
- `test/sN-tN/result.json`: 실제 adaptive 곡선, 비용 절감, 오판 비용.
- `report.json`, `curves.csv`: 개발/테스트 곡선과 seed별 요약.
- `branches/*`: 원본 policy, optimizer, 평가 응답, 비용 장부, 완료 해시.
- `{development,test}/sN-tN/queue-branches/*.json`: 검증된 분기 곡선과 상태·프로토콜 연결.
  파일 존재만으로 완료를 인정하지 않고 원본 결과와 다시 비교한다.
- `pair-branch-queue-runtime.json`: 검토된 실행부 전환 영수증. 기존 manifest,
  예산, 결정, 결과, 비용 장부는 덮어쓰지 않는다.
- `queue-workers/*.json`: 노드명, worker PID, 개발/테스트 단계, 담당 상태·분기·arm,
  RUN/WAIT/DONE, 최근 갱신 시각, 확인한 상태·분기 수와 해당 worker의 실패 기록.

`WAIT`는 해당 worker가 상태 작업을 실행하지 않고 배정·공유 단계 완료를
기다리는 상태다. 노드 전체 GPU가 비었다는 인증은 아니다. 필요한 완료 기록이
이미 있는 재실행은 추가 NCCL 사전 검사 없이 결과 검증·보고서 경로를 진행할
수 있지만, 파일 존재만으로 검증을 생략하거나 완료를 인증하지 않는다.

분기 큐에서 다른 노드를 기다릴 때는 잠금 대기와 다른 **무진행 180초** 기준을
사용한다. 검증된 완료 상태·분기 수가 늘거나, 실제로 기다리는 busy 분기에서 최근
`running` heartbeat의 `updated` 값이 새로 증가해야 무진행 시간을 갱신한다.
구버전의 상태 EX 잠금을 기다릴 때만 해당 상태 전체의 진행을 확인한다.
다른 분기의 진행, `WAIT` 기록 갱신, 같은 과거 heartbeat를 다시 읽는 것은
기한을 연장하지 않는다. 정상적인 peer 진행이 계속 확인되면 분기 큐 대기는
180초를 넘을 수 있다. 해당 진행이 180초 동안 없으면 exit 76으로 안전하게
빠져나오며, peer의 프로세스·잠금 파일·결과·비용 기록은 건드리지 않는다.

worker 기록의 `queue_wait_wall_seconds`는 분기 큐에서 측정한 대기시간이다.
분기 곡선의 selector 비용축에 이 대기를 숨겨 합산하지 않으며, 그렇다고 대기한
GPU 할당 비용을 0으로 취급하지도 않는다. 전체 운영 비용을 보고할 때는 이
대기와 준비·공유 단계 대기 등 노드 할당 비용을 별도로 확인해야 한다.

## 목표 보상과 해석상의 제한

`--target-reward`는 `[0, 1]` 단위이며 양수여야 한다. 두 continuation의 최종
보상을 본 뒤 `min(final_G, final_D)`로 정하지 않는다.
시작 보상 이상이 아니라 **시작 보상보다 엄격히 높은 목표**여야 한다.
이미 목표를 넘은 상태는 `target_not_above_parent`, 끝까지 못 넘은 상태는
`right_censored`다. 이 경우 H를 0이나 임의의 최댓값으로 대체하지 않는다.

현재 v1 fitting은 개발 9개 상태 모두의 H가 관측되어야 한다. 일부 상태만
성공했다는 이유로 해당 상태만 골라 fitting하지 않는다. 목표 미도달이면 곡선과
상태는 저장하고 fitting을 중단한다. cap 확대나 target 변경은 새 사전계획·새
root에서 해야 하며, 이전 시도를 숨기면 안 된다. 검열된 라벨용 생존분석은
이 구현에 들어 있지 않다.

기본값은 시작점 + 최대 9개 중간 checkpoint + 끝점 평가다. 원래 learner의
5-update checkpoint 중 완료 학습량의 지정 비율에 가장 가까운 것을 사용한다.
중간점·끝점 모두 같은 응답 수(eval-k)를 쓴다. 첫 **평가된** checkpoint의
도달 비용을 보고하며 선형 보간으로 정확한 switch update를 만들어내지 않는다.
평가 사이의 순간적인 도달이나 목표 이상의 성능 유지까지 보장하지 않는다.

평가 비용은 별도의 reporting 비용이다. 가상의 배포 시 online 평가·중단 비용까지
포함했다고 주장할 수 없다. `curve_evaluation_cost`, `parent_evaluation_cost`,
원본 result의 평가 장부를 보존한다. shared prefix와 기존 캐시의 역사적 비용은
원본 실험에 남아 있으며, 0이라고 가정하지 않는다. 개발 fit 비용도 따로 기록한다.

## 중단 / 재개

GPU 작업 전에 Switch/MBPP와 같은 4-rank NCCL/DDP 사전 검사를 수행한다.
실제로 재검사에 성공한 통신 설정만 자식 작업에 전달한다. 검사 기록과 공유
노드 검사 비용은 `node-preflight/`에 남는다. 검사 실패 시 exit 78로 종료하고
학습을 시작하지 않는다. CPU 테스트 통과가 실제 노드의 CUDA 정상 동작을 보장하지는 않는다.

분기 작업의 일반 실행 오류는 저장 작업을 보존하고, 노드를 재검사한 뒤 한 번만
자동 재시도한다. 두 번째에도 실패하면 재검사에 통과한 노드에서 다른 분기를
진행한다. 계약/비용 검증 오류는 해당 분기를 재시도하지 않는다. 남은 실패는
분기의 `pair-attempt.json`과 해당 `queue-workers/*.json`에 기록한다.
한 worker는 같은 실행 pass에서 실패한 분기를 끝없이 다시 배정하지 않으며,
독립적인 다른 분기를 진행한다. 공유 부모 평가가 다른 worker에서 진행 중이면
학습 실패나 GPU 재입장 사유로 취급하지 않고, 저장된 결과와 평가를 재사용해
남은 곡선 발행만 재개한다. 미완료 개발 결과로 predictor를 학습하거나
테스트 결정을 만들지 않는다. 노드 종료로 분기 잠금이 해제되면 다른 worker가
기존 체크포인트와 비용 장부를 보존한 채 이어서 수행할 수 있다.
종료 신호와 노드 사전 검사 실패는 후속 작업으로 넘어가지 않는다.

학습 예산이 소진되고 유효한 최종 저장/결과 후보가 없는 분기는 검증 비용부터
다시 적립하지 않고 거부한다. 최종 policy/stop 또는 결과 후보가 있으면 기존
검증 절차를 통해 남은 평가·곡선·발행만 재개할 수 있다. 예산 초기화나 증액,
목표 변경, 체크포인트 삭제는 하지 않는다.

`Resource temporarily unavailable`만으로 CUDA OOM이라고 단정하지 않는다.
런처는 Python 시작 전 OpenBLAS/MKL/OpenMP/Rayon/NumExpr의 CPU 스레드 수를
1로 제한하고 토크나이저 병렬화를 끈다. 이 설정은 네 GPU의 rollout·scoring·학습·평가
자식 프로세스에도 적용된다. GPU 수, generation batch, 응답 수, seed는 바꾸지 않는다.
따로 환경변수를 입력할 필요 없이 아래의 같은 명령으로 재개한다.

시작/실패 로그의 `[pair-resources]`에는 프로세스 제한과 읽을 수 있는 cgroup의
`pids.current/max`가 표시된다. 노드 전체의 PID 한도가 이미 소진된 경우에는 이
제한만으로 해결되지 않을 수 있다. `pair lock busy: ...`는 별개의 실행 잠금 충돌이다.
잠금 파일을 삭제하거나 다른 작업을 자동 종료하여 우회하지 않는다.

검토된 이전 버전의 실행은 원본 manifest·이전 업그레이드 영수증을 그대로 보존하고
`startup-resources-runtime.json`에 새 코드 해시와 CPU 제한을 별도로 고정한다.
이 운영 설정은 처리 시간에 영향을 줄 수 있으므로 변경 전후 시간을 동일 환경의
측정처럼 취급하지 않는다. 실패·재시도 비용은 기존 장부에 계속 포함한다.
이번 운영 수정은 `pair-operations-runtime.json`으로 별도 기록하며, 검토된 이전
코드의 manifest와 기존 런타임 영수증을 덮어쓰지 않는다.
중복 실행 조회 수정은 `pair-lock-observation-runtime.json`에 별도로 기록한다.
분산 큐 수정은 `pair-distributed-runtime.json`에 기록하며, 검토된 `1aebf1d`
코드의 manifest와 이전 operations·lock-observation 등 모든 기존 영수증을
그대로 보존한다. 대기 상한·소유자 진단 수정은 `pair-wait-guard-runtime.json`에
별도로 기록하며, 검토된 `5fd2410`의 distributed 영수증까지 이전 8개 영수증을
덮어쓰지 않는다. 기존 단일 실행과 분산 실행의 고정 설정·학습·결과·비용은
그대로 유지한다. 실행 중인 pair의 checkout을 변경하기 위한 자동 pull/restart
기능은 추가하지 않았으며, 조회는 기존 실행을 중단하지 않는다.

같은 root에서 `bash scripts/run_selector_pair.sh`를 다시 실행한다. 완료된 selection, checkpoint, 결과,
동결된 decision을 재사용한다. 중간 checkpoint의 cost receipt는 checkpoint
디렉터리와 함께 atomic하게 저장되고 archive에도 보존된다. 최종 policy가 저장된
직후 끊긴 경우, 검증된 policy와 종료된 train 장부에서 최종 비용을 복구한다.

종료 시간이 확인되지 않는 열린 cost event는 공짜로 처리하지 않는다.
atomic 완료 영수증이 있으면 기존 복구 코드가 재생하고, 없으면 종료 기록을
확인할 때까지 중단한다. 기존 복구 도구의 `--root`에는 전체 pair root가 아니라
문제가 생긴 `branches/on_policy` 등 개별 switch root를 준다. 기존 비용을
waive/reset하여 제거한 실행을 이 연구의 깨끗한 재시도로 간주하면 안 된다.

CPU 검증:

```bash
PAIR_PYTHON=.work/.venv-cu126/bin/python bash scripts/run_selector_pair.sh cpu
```

CPU 테스트는 실제 모델/GPU 수치 검증이 아니다. 정식 GPU run 전 개발 branch의
동작과 메모리·비용을 점검해야 하며, 테스트 통과만으로 논문에 새 결과를 추가하면 안 된다.
분기 큐 테스트는 실제 CPU 프로세스 18개/24개의 동시 배정, 동일 상태 준비의 직렬화,
중복 실행 방지, 결과 저장 직후 프로세스 종료와 재개, 기존 결정·비용·영수증 보존을
확인한다. 이는 클러스터에서 18개/24개 노드를 할당받거나 GPU 실행이 완료되었다는 뜻이 아니다.
