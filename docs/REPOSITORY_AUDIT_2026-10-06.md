# 전체 코드 저장소 점검

## 2026-10-19 V7 실험 코드 재점검

기준 commit은 `031e2ea5d9578ed1e7cf09e06d1c89588c211e22`다. 현재 V7 실행 코드의
`srgc_research/tests`와 `srgc_rebuttal/tests` 전체를 실행하고, 수동 검토는 모델별
배정·재개, 25-update 선별, SR-GC, 분산 응답 저장과 결과·재시작 비용 추출에 집중했다.
이 기록은 아래 10월 18일 저장소 전체 검사와 범위가 다르다.

| 재현한 문제 | 영향 | 수정·검증 |
| --- | --- | --- |
| rank별 응답 기록 검증 실패 | 한 rank가 토큰 변환 또는 응답 수 검사에서 먼저 종료하면 다른 rank는 원래 원인 대신 Gloo 통신 오류를 받음 | 기존 응답 gather 한 번으로 오류도 전달하고 모든 rank에서 같은 검증 오류로 종료. 4개 실제 CPU 프로세스·두 손상 유형으로 수정 전 실패와 수정 후 전달을 확인 |
| OLMo 이전 실행 결과의 텍스트 누락 | 이전 reward는 JSON에 남지만 새 stdout 표와 저장 txt에서 빠져 완료 결과를 놓칠 수 있음 | `PREVIOUS RUN`과 원본 plan 경로·이전 결과를 별도 표시. 현재 미완료 결과와 합치지 않으며 txt/JSON 동시 보존을 검사 |

분산 저장 실패 시 응답 archive를 게시하지 않고 해당 phase 비용은 미완료로 남긴다.
성공한 저장 경로의 추가 gather·모델 호출·응답 생성은 없다. 기록 확장의 새 버전은
기존 별도 receipt로 연결하며, 기존 protocol·archive·checkpoint를 덮어쓰지 않는다.
RuntimeError 등 치명적 CUDA 예외는 추가 gather 없이 원래 예외 그대로 전파한다.
이 동작은 별도 fault-injection 검사로 확인했으며, 이 수정이 사용자가 보고한 원격
`CUDA unspecified launch failure`의 원인이라는 뜻은 아니다.

On-policy 기본 선별은 25 updates마다 후보 40개에서 상위 4개를 고르고, 각 update는
선택 문제 4개에 새 응답 8개씩을 사용한다. SR-GC의 비교 집합과 학습 배치를 구분하고,
두 연속 음수 및 음수·비음수·음수 창의 합 부호 규칙을 그대로 둔다. 학습 엔진·Qwen/Llama/Gemma
adapter 해시, plan, 입력, 원고, 캐시, optimizer checkpoint와 실험 결과는 변경하지 않는다.

| 검사 | 결과 |
| --- | --- |
| 수정 전 두 실험 패키지 전체 회귀 | 1,961 passed · 324 subtests passed · 1 failed. 실패는 OLMo 이전 실행 텍스트 누락으로 재현·수정 |
| 새 분산 저장 재현 | 수정 전 두 손상 유형 모두 실패: rank 2의 검증 오류가 다른 rank에서 통신 오류로 바뀜 |
| 수정 후 핵심 회귀(실제 로컬 디스크) | 4 passed · 4 subtests passed. 두 분산 손상 유형, CUDA RuntimeError 보존, 이전 실행 stdout/txt/JSON 분리 보존 |
| 수정 후 두 실험 패키지 전체 회귀 | **1,964 passed · 325 subtests passed · 실패·제외 0** (442.10초) |
| 정적·보존 검사 | Python 298개 AST · shell launcher 16개 문법 · 변경 공백 · 문서 상대 링크 14개 통과. 세 모델의 engine/adapter 해시 동일 |

전체 회귀의 마지막 실행은 pytest 임시 fixture를 메모리 파일시스템에 두어 로컬 디스크의
fsync 대기를 줄였다. 저장 함수·fsync·원자적 교체·실제 파일 잠금 검사는 그대로 실행했으며,
별도 핵심 회귀는 실제 로컬 디스크에서도 통과했다. 중복·중단한 검사 실행은 최종 통과 수에
더하지 않았다. Python 3.12·CPU PyTorch 2.13.0·Transformers 5.14.1·PEFT 0.20.0 환경이다.
GPU가 없는 CPU 환경이며 실제 H100/NCCL, 9B/12B GPU 메모리 사용량과 실제 그룹 볼륨의
노드 간 잠금은 검증하지 않았다. 원격 학습 프로세스를 중단하거나 재실행하지 않았다.
67개 경고는 CPU 분산 검사에서 발생한 Python 3.12의 멀티스레드 `fork` 사용 폐기 예정 경고다.



## 2026-10-18 재점검

기준 commit은 `1435fb356ade48b39982b53159d840e51a5fcdf8`다. 추적 중인 전체 소스의
정적 검사와 회귀 검사를 수행하고, 수동 검토는 재개·작업 순서·실행 잠금·중간 보상·
status/results 경계에 집중했다. 모든 파일의 모든 줄을 수동으로 읽었다는 뜻은 아니다.
진행 중인 실험의 학습 조건·동결 runtime·모델 adapter·입력·결과는 보존했다.

| 재현한 문제 | 영향 | 수정 |
| --- | --- | --- |
| `all resume`의 데이터셋 전환 시 재적용 | MATH 실행 중 새로 건 MBPP 중지 요청이 MBPP 진입 때 다시 지워질 수 있음 | 한 coordinator 실행에서 모델마다 resume를 한 번만 적용하고, 다음 데이터셋은 run으로 진입 |
| 잘못된 coordinator worker JSON | 배열/문자열 기록 하나로 status가 종료되거나 잘못된 started/device 정보가 정상 세션 선택에 간섭 | JSON object·시간·모델·장치·순서 필드를 검사하고 잘못된 기록만 조회에서 제외; 원본 파일 보존 |
| 손상된 중간 rollout archive | 잘린 gzip, 잘못된 압축 데이터, 배열 형태의 trace/training/group/history 때문에 results 전체 또는 자동 게시가 예외로 중단 | 읽기 경계에서 검증 오류로 변환하고 해당 arm의 curve 오류로 기록; 정상 endpoint와 다른 arm은 partial export에 보존 |
| 긴 경로의 status 화면 | 80열에서 ROOT 경로가 여러 줄을 차지해 화면 길이 제한을 넘거나, 미준비 화면의 경로가 폭을 초과 | ROOT 표시를 한 줄로 제한하고 긴 경로는 축약; 경로 안의 연속 공백과 JSON의 전체 경로는 보존 |

수정 위치는 [모델 연결 실행](../srgc_research/dispatch/model_chain.py)과
[결과 내보내기](../srgc_research/dispatch/model_results.py),
[기존 실험 현황판](../scripts/selection_switch_status.py)이다. 중지 요청 재현은 세 모델,
archive 손상 재현은 여섯 손상 유형×세 모델을 검사한다. 정상 endpoint를 invalid나 0점으로
바꾸지 않고, 손상 파일도 자동 삭제·수정하지 않는다. 관련 회귀는
[연결 실행 테스트](../srgc_research/tests/test_model_chain.py)와
[rollout 보존·내보내기 테스트](../srgc_research/tests/test_rollout_records.py),
[현황판 테스트](../tests/test_selection_switch_status.py)에 추가했다.

정적 검사: Python **762개** AST 파싱, JSON **39개** 파싱, Shell **118개**의 실제
interpreter 기준 문법 검사와 ShellCheck error 검사 통과. `_mbpp_experiments.sh`는
Bash에서 source하는 파일로 검사했다. 운영 확장 파일은 Ruff 검사를 통과했고,
기존 현황판 파일·handoff 테스트에는 기존 대비 새 Ruff 진단이 없다. 전체 변경의 공백 검사를 통과했다.
기존 소스 전체의 스타일 경고를 동결 코드에 일괄 적용하지 않았다.

동결 learner·세 모델 adapter·기존 resume 모듈·checkpoint/rollout·보상 기록 확장을
포함한 **63개 소스 파일**은 기준 commit과 바이트/해시가 같다. 새 실험을 실행하거나,
원격 노드의 실행 중인 Python에 수정 사항을 적용한 작업은 아니다.

전수 검사 후반의 Gemma/Llama 기본 Python 선택 실패 두 건은 앞선 Pair handoff 테스트가
`PAIR_PYTHON`·GPU 환경 변수를 남긴 데서 발생했다. 각 테스트를 따로 실행하면 두 건 모두
통과하지만 handoff와 같은 프로세스에서 실행하면 **98개 통과·2개 실패**로 재현됐다.
[handoff 테스트](../tests/test_selector_pair_handoff.py)에 테스트별 환경 복원을 추가했다.
실제 handoff가 검증한 owner의 interpreter와 GPU 설정을 재시작에 전달하는 동작은 보존했다.

| 검사 | 확인한 결과 | 범위 |
| --- | --- | --- |
| 저장소 전체 1차 회귀 | **6,183 passed, 16 skipped, 330 subtests passed, 3 failed**, 수집 오류 0 | 전체 검사에서 긴 경로 화면 실패 1건과 테스트 환경 누수 2건을 발견; 아래 최종 검사에서 해결 확인 |
| 수정 후 최종 결합 회귀 | **585 passed, 1 skipped, 실패·오류 0** | handoff 이후 모델 런처를 같은 프로세스에서 검사; 앞선 세 실패와 재개·SIGKILL·status/results·rollout·중간 보상·분석 회귀를 모두 포함 |

전수 검사에서 제외한 16개는 명시적 CUDA 장치가 필요한 15개와 RLOO에 적용되지 않는
nested-curve meter 검사 1개다. 최종 결합의 제외 항목도 같은 RLOO 검사다.
현황판의 신규 12개 회귀는 전수 수집 이후 추가되어 최종 결합 검사에 포함했다.
두 실행의 중복 테스트 수는 합산하지 않는다. 최종 추가 회귀는 총 **43개**다.
검사 원본은 `/tmp/offpolicy-source-audit-final-v3.xml`과
`/tmp/offpolicy-source-audit-final-focused.xml`에 남겼다.

검증 환경은 Python 3.12, PyTorch 2.13 CPU, Transformers 5.14.1, PEFT 0.20.0이며,
gate 의존성은 선언된 별도 CPU 패키지 경로를 사용했다. 실험 노드의 환경을 변경하지 않았다.
로컬 CPU 검사와 실제 H100/NCCL 검증을 구분한다. 중간 학습 배치 보상과 고정 held-out
보상은 기존처럼 별도로 보존·내보낸다. 사용자 미추적 CFCS 테스트·hotfix 파일은 커밋에서 제외한다.

---

아래는 2026-10-06 점검 당시의 보존 기록이다.

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
