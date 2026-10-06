# 전체 코드 저장소 점검

기준: `master`의 `e9ef3f5129b405c2b0b49c8a1fa49cae9e7ce8f8` 이후 작업 트리.
대상은 Pair, MBPP, Qwen, 기존 SR-GC, 추가 실험 N01-N08, 비용·상태·결과 수집 도구다.
논문 저장소·PDF·웹 게시본과 실제 실험 결과는 수정하지 않는다.

## 확인하고 수정한 문제

| 항목 | 영향 | 수정·회귀 검증 |
| --- | --- | --- |
| N01-N07 큐의 작업 시간 초과 | `run_child`의 `TimeoutError`가 큐 전체를 종료해 다른 작업과 자동 재시도가 중단됨 | 오류·exit 124 기록, 기존 3회 실패 한도 적용, 다른 작업 계속 처리 |
| 구형 비용 집계의 누락 경계 | 중간 progress 또는 산출물이 없으면 여러 단계의 시간을 다음 단계 하나로 귀속시킴 | 인접한 단계 경계가 모두 있을 때만 산출; 공동 gradient 단계는 oracle·validation 산출물 모두 확인 |
| 잘못된 타이머 | 역전된 timestamp, 음수·NaN·무한대·boolean 학습 시간이 비용으로 들어갈 수 있음 | 불명확한 경계 시간은 미확인으로 유지하고 잘못된 학습 타이머는 거부 |
| 신규 실험 상태 오류 격리 | 깨진 manifest·progress·queue JSON 하나로 전체 status가 종료되거나, 다른 dataset/seed의 manifest를 결과에 섞을 수 있음 | 해당 seed 또는 작업만 `invalid`로 표시, 나머지는 계속 조회; results에서도 dataset/seed 확인 |
| 신규 endpoint 평균 보상 검사 | `abs(mean - NaN) > tolerance`가 false여서 NaN 평균 보상을 정상으로 인정 | 유한한 숫자인지 먼저 검사한 뒤 문제별 평균과 대조 |

관련 코드:
[`cost_accounting.py`](../src/cost_accounting.py),
[`cli.py`](../srgc_research/cli.py),
[`storage.py`](../srgc_research/storage.py),
[`report.py`](../srgc_research/report.py).

회귀 테스트:
[`test_repo_cost_audit.py`](../tests/test_repo_cost_audit.py),
[`test_audit.py`](../srgc_research/tests/test_audit.py).
첫 수정 전 독립 재현에서 14개 assertion이 실패했고, 정상 d0 대조 1개는 통과했다.
이후 잘못된 progress·queue JSON 형태에 대한 6개 실패도 재현해 수정했다.
신규 회귀 사례 21개를 추가했다.

이 수정은 구형 비용 **집계 도구**의 잘못된 귀속을 막는 것이다.
산출물 mtime 기반 추정을 새 실측으로 바꾸거나, 이미 작성된 논문의 시간 값을
자동으로 다시 계산해 반영한 것은 아니다. 원본 로그·결과는 그대로 둔다.

## 점검 범위

- 시작 시점 추적 파일 937개. Python 662개 문법·compile, JSON 37개 파싱 통과.
- Shell 115개를 실제 interpreter 기준으로 문법 검사하고 ShellCheck error 검사 통과.
  `_mbpp_experiments.sh`는 Bash에서 source하는 배열 사용 파일이므로 POSIX sh로 판정하지 않는다.
- 실행 코드 전체의 정의되지 않은 이름·할당 전 참조·잘못된 구문·가변 기본 인자 검사 통과.
- MATH·MBPP seed 5-9 입력 10개를 실제 입력 검증기로 확인했다.
  각각 후보 400, validation 100, evaluation 300이며 ID·정규화된 문제 텍스트의 분할 중복이 없다.
  저장소의 cache가 빈 것은 GPU 생성 전 입력이며 오류나 이미 완료된 캐시로 표시하지 않는다.
- 수동 검토는 선별·학습 gradient, 재개 시 모델/optimizer/attention 보존,
  노드·task lease, 캐시 저장 위치, 비용 단계와 포함 관계, 완료 판정·결과 출처에 집중했다.
  모든 파일의 모든 줄을 수동 검토했다는 의미는 아니다.

## 검증 기록

전체 회귀 검사: **5,408 passed, 16 skipped, 330 subtests passed**.
실패·수집 오류는 없고, 소요 시간은 1,290.76초다. 경고 90건은 출력되었다.
건너뛴 항목은 명시적 CUDA 장치가 필요한 15개와 RLOO에 적용되지 않는
nested-curve meter 검사 1개다. 이를 GPU 검증 통과로 합산하지 않는다.

최종 변경 상태를 별도로 검증한 신규 패키지·비용 테스트는 **76개 통과**다.
전체 실행의 수집 이후 추가한 progress·queue JSON 형태 회귀 6개도 여기에 포함된다.
두 실행의 중복 테스트 수를 더해 보고하지 않는다.
전체 실행에는 사용자의 기존 미추적 CFCS 테스트 16개도 포함했지만,
해당 코드·테스트는 이번 커밋에 포함하지 않는다.

로컬 검사 원본은 `/tmp/offpolicy-repo-audit-verified-20261006.xml`과
`/tmp/offpolicy-focused-audit-final-20261006.xml`에 남겼다.

별도 완료한 검사:

- 기존 Transformers 5.14.1 / PEFT 0.20.0 / PyTorch 2.13 CPU 환경에서
  Qwen 모델·확장 16개와 기존 Qwen 실행 경로 93개 통과.
- 선언된 gate 의존성을 별도 임시 경로에서 사용한 gate 검사 53개 통과.
  OLMo 전용 Transformers 4.57 환경의 Qwen import 오류와 sklearn 미설치는 코드 결함으로 세지 않았다.
- 수정 관련 신규 패키지·비용 검사 최종 76개 통과.
- 4개 CPU rank의 실제 작은 OLMo/LoRA 모델로 feature scoring, rollout 재사용,
  GRPO/Adam 진단, checkpoint·endpoint 복구 검사를 통과했다.
  모델 연산은 실제 작은 모델을 사용하고 응답 생성은 결정적 테스트 입력으로 대체했다.
  Gloo 기반 기능 검사이며 H100/NCCL 성능 측정으로 해석하지 않는다.
- Ruff 및 `git diff --check` 통과.

프로젝트 의존성이 준비된 검증 환경에서의 실행 명령:

```sh
PYTHONPATH=src:scripts:tests:. CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q tests srgc_rebuttal/tests srgc_research/tests \
  --import-mode=importlib --disable-warnings --tb=short
```

이번 검사는 기존 CPU 환경과 임시 gate 의존성 경로를 이용했다.
운영 노드의 Python 환경·패키지를 설치하거나 변경하지 않았다.

## 실행 중인 실험 보호

기존 `srgc_rebuttal/*.py`의 해시는 수정 전후 모두 다음과 같다.

```text
12cf5ef830ebfd92fa8a87ea62dc7df734cd9ceab57fbce18fc4b2548385f960
```

보존된 두 구버전 runtime의 manifest·engine 해시도 검증했다.
진행 중인 mechanism, 고정 runtime, 학습 cache, checkpoint, 원본 결과에 쓰기 작업을 하지 않았다.
사용자의 미추적 연구·hotfix 파일은 수정·커밋 대상에서 제외한다.

## 남는 검증 범위

로컬 CPU·작은 실제 모델·임시 저장소·모의 노드 검증이다.
H100 메모리 상한, 실제 CUDA/FLA kernel, 여러 물리 노드의 NCCL 통신,
운영 그룹 볼륨 장애·부하 상황과 전체 학습 성능은 이번 점검으로 확인한 것이 아니다.
실제 GPU 실험 결과를 만들어 넣거나 기존 논문 수치를 수정하지 않았다.
