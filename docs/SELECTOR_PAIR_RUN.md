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
환경변수를 따로 입력할 필요가 없다. 4-GPU 할당 노드에서 tmux 또는 배치
스케줄러 안에서 실행한다.

```bash
bash scripts/run_selector_pair.sh
```

설정 생성 → 원본 검증·준비 → 개발 곡선 수집 → predictor 학습 → 테스트 결정
고정 → 테스트 학습·평가 → 보고서까지 진행한다. 목표 미도달 등 실험의 검증
조건을 충족하지 못하면 해당 단계에서 중단하며, 결과를 만들어 통과시키지 않는다.

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

한 root는 한 controller만 실행한다. 여러 노드에 분산하는 큐가 아니다.
다른 controller가 같은 root를 사용 중이면 잠금 오류로 종료한다.

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
