# Gemma 중간 평가 중단 시 복구와 status 구분

**실제 원격 실행 문제는 미해결이다.** 사용자는 후속 보고에서 `RUN → EVAL` 반복 중
학습 진행 자체가 안 되며, 이전 재사용 코드는 문제가 없었다고 명시했다.
[미해결 추적 이슈 #1](https://github.com/33modeling/offpolicy-misranking/issues/1)에
확인한 코드 변경, 실행 근거의 부재 및 실제 노드에서 확인해야 할 종결 조건을 기록했다.
아래 CPU 복구 검사를 이 운영 문제의 해결 증거로 사용하지 않는다.

## 추가 분류: status 오류 재현

사용자는 이후 **학습 오류인지 status 오류인지 먼저 구분하라**, **status 자체가
엉망이다**라고 정정했다. 원격 학습 장애로 단정하지 않고 조회 코드만 분리해서 검사했다.
기준 `e82cb60a6ad6789e91b501b1287f5494e2ac7102`와 수정한 status를 같은 합성 실행
파일에 적용해 다음 표시 오류를 재현했다. 조회 전후 원시 실행 파일은 동일했다.

| 원시 기록 | 기존 표시 | 수정 표시 |
| --- | --- | --- |
| training checkpoint 50 started: 50회 완료, update 51 진행 중. 과거 eval 50 파일의 mtime이 같거나 더 최신 | EVAL @50 | RUN 50/275, update 51 진행 중 |
| 유효한 최종 endpoint가 저장됐지만 실행기 lease가 아직 유지됨 | FINAL EVAL @275 | FINALIZING, result saved, 실행기 종료 대기 |
| 다른 root/PT/Qwen 실행기 기록이 함께 존재 | 현재 Gemma의 MODEL QUEUE 하단에 섞임 | 현재 모델·프로토콜·root와 맞는 실행기만 표시 |

첫 오류는 실제 update 증거보다 mtime 비교를 우선한 상태 판정 문제다. 두 번째는
평가 완료와 프로세스 정리 대기를 구분하지 않은 표시 문제다. 최종 결과는 기존
reporter의 plan/input/implementation/prefix/300개 reward 검증을 재사용하고, 조회 중
endpoint 파일이 바뀌면 준비 완료로 표시하지 않는다. lease를 해제하거나 완료 task를
조작하지 않는다. 결과가 없거나 잘못되면 `FINALIZING`/`DONE`을 만들지 않는다.

이 수정은 `model_status.py`, `qwen_status.py`, `status_current.py`의 조회 경로만 바꾼다.
학습·평가·재시도 실행기, frozen adapter/engine, checkpoint 및 실험 조건은 변경하지 않는다.
[재현 검사](../srgc_research/tests/test_status_phase_diagnosis.py)와
[상태 분류 증거](evidence/gemma-status-diagnosis-20261023.json)에 코드 검증을 기록한다.
수정 전 동일한 17개 조건에서 13개가 실패하고, 수정 후 전체 status 회귀 **151개가
65.31초에 통과**했다. 검증 원시 로그는 `/home/kms/reports/gemma-status-diagnosis-2026-10-23/`에 보존했다.
이는 실제 GPU 노드가 정상 학습 중이거나 고장 났다는 판정이 아니다.

**이번 status 수정 적용에는 학습이나 노드 재시작이 필요 없다.** 저장소에서
`git pull --ff-only` 후 기존 `bash scripts/run_srgc_gemma4.sh status`만 다시 실행한다.

사용자 보고: Gemma가 `RUN → EVAL → RUN → EVAL`을 반복하고 학습 진행 및 `DONE` 도달이 막혔다.
기준 코드는 `20d15e58dd861e2b2418996e32a8866653c2d99d`다.

25 update마다 고정 held-out 평가를 하고 다음 학습으로 돌아가는 순서 자체는 기존
중간 reward 기록 요구사항이다. 25, 50, …, 250에서 평가하며 275에서는 원래
runner의 최종 평가와 endpoint 검증을 거친다. 이 간격·평가 문항·8개 응답·학습
objective·모델·seed·총 update 수를 변경하거나 평가를 생략하지 않았다.
이 일정이 존재한다는 사실만으로 사용자가 보고한 정체를 정상 동작으로 분류할 수는 없다.

## 재현한 복구 문제

기존 recorder가 `update()` 안에서 긴 held-out 평가를 끝낸 후에야 바깥 runner가
경계 update의 복구 checkpoint를 저장했다. 실제 CPU AdamW와 작은 선형 모델로
평가 실패를 주입하니 다음 상태를 재현했다. reward 값은 합성 값이며 Gemma 성능이 아니다.

| 작업 | 학습 scalar의 완료 update | 평가 중단 후 복구 checkpoint |
| --- | ---: | ---: |
| prefix | 25 | 24 |
| on_policy continuation | 50 | 49 |

따라서 평가 도중 재시작하면 이미 수행한 마지막 optimizer update까지 다시
실행했다. 이 현상은 코드에서 재현했지만, 사용자의 원격 노드가 실제로 같은
경계에서 실패하는지는 원격 로그를 확보하지 못해 확인하지 못했다. 정상적인
서로 다른 25-step 평가와 동일 step 재시도는 구분해야 한다.

## 수정

- 실제 학습 rollout과 scalar를 저장한 다음 경계의 모델·optimizer 복구 checkpoint를
  먼저 저장하고 중간 평가를 실행한다. scalar/rollout 저장 실패 시 복구 지점을 앞당기지 않는다.
- 복원 시 해당 경계의 미완료 평가를 먼저 마친다. 특히 prefix의 25-update 학습
  루프가 이미 끝났어도 평가를 누락하고 prefix-ready를 발행하지 않는다.
- 275-update 복원은 추가 optimizer update 없이 기존 최종 평가·endpoint 경로로 간다.
- status 셀은 중간 평가를 `EVAL … @50`, 최종 평가를 `FINAL EVAL … @275`로 구분한다.
  상세 줄에는 중간 평가 후 다음 update와 실제 attempt 번호를 표시한다. 유효한
  endpoint가 없으면 평가 prompt를 모두 저장해도 `DONE`으로 표시하지 않는다.

학습 계산은 동결된 adapter/engine에서 유지한다. 추가 경계 checkpoint I/O는 기존
timer에 기록하고 recorder 구현 변경은 별도 recording-extension receipt로 남긴다.
기존 protocol, reward, rollout, checkpoint를 새 실험으로 바꿔 쓰지 않는다.

## 검증 범위

[새 실행 검사](../srgc_research/tests/test_boundary_evaluation_resume.py)는 공개 Gemma IT
recorder 진입점을 사용하되 모델·GPU 시작만 작은 CPU optimizer로 대체한다.
prefix와 네 arm의 평가 실패·복구, 정상 실행과 복구 실행의 모델/optimizer 동일성,
25-update 평가 순서, 275 도달, 학습 scalar 저장 실패를 검사한다. 별도로 원래
`run_experiment.main`을 사용해 prefix와 네 arm의 endpoint 발행, 마지막 endpoint
쓰기 실패 후 275에서 복구, 최종 `run.json: complete` 및 status `DONE`을 검사한다.
CPU 검사에서는 world size와 projection dimension을 테스트용으로 대체한다.
이는 12B Gemma 가중치나 4-GPU 학습의 실측 검증이 아니다.

[status 검사](../srgc_research/tests/test_status_current.py)는 중간/최종 평가 구분,
다음 단계, 재시도 번호, 검증 전 `DONE` 금지와 기존 읽기 전용 조회를 검사한다.
[검증 기록](evidence/gemma-boundary-resume-20261023.json)에 결과와 원시 로그를 기록한다.
테스트 준비 중 발생한 CPU allocation/projection fixture 오류 및 변경 전 UI 기대값의
실패 로그도 삭제하지 않고 보존했다.

실제 원격 Gemma의 현재 seed·arm·update·가중치 변화·성능·완료 여부는 미확인이다.
저장소 push만으로 이미 실행 중인 원격 프로세스가 새 코드를 읽는 것은 아니다.
앞서 안내한 실행기 재시작은 코드 적용 방법이며, 현재 정체를 해결했다는 증거가 아니다.
현재 실제 task의 연속 update/evaluation 저장 수, attempt와 원래 rank 오류를 확보하지 못했다.
실제 정체 task의 진척 및 최종 endpoint/DONE을 확인하기 전에는 이 이슈를 완료로 처리하지 않는다.
