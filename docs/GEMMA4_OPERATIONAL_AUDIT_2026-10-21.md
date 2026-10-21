# Gemma 시작·중지와 모델 연결 실행 점검 — 2026-10-21

기준 코드는 `85c514c07d837df3a0280a27e722c00e2ab995ea`다. 입력 준비, 중지,
재개 및 공개 shell 명령의 분기를 검사했다. 이 문서의 결과는 **코드 동작 검사**이며
새 학습 실험이나 모델 정확도 측정이 아니다.

## 재현한 오류와 수정

| 조건 | 기존 동작 | 수정 후 동작 |
| --- | --- | --- |
| MATH 또는 MBPP 한쪽만 준비하고 `stop --now` | 없는 다른 plan을 읽다가 중지 명령 실패 | 요청한 두 데이터셋의 중지를 기록하고 기존 plan·queue identity를 보존 |
| 첫 입력 준비 도중 즉시 중지 | 준비 자식에 중지 확인 함수가 없어 계속 실행 | 준비 자식의 실행 그룹을 종료하고 다음 단계·GPU admission을 시작하지 않음 |
| plan 생성 전 중지 | 계획이 없어 중지 실패 | `runs` 밖에 준비 중지 기록을 저장하고 다음 시작을 막음; status에도 중지 표시 |
| `all --root '/a path'` 등 action을 생략한 옵션 사용 | Qwen·Llama·Gemma 모두 모델 연결 실행을 우회 | 명시적인 `all run --root '/a path'`와 같은 경로로 전달 |
| 도움말을 요청하는 공개 명령 | 일부 형식이 학습 coordinator로 전달될 수 있음 | 도움말은 모델 연결 실행에 들어가지 않음; Gemma는 모델용 Python이 없어도 표시 |

운영 수정은 [중지 제어](../srgc_research/dispatch/gemma_text_control.py),
[공개 Gemma 진입점](../srgc_research/dispatch/gemma_text_run.py),
[상태 조회](../srgc_research/dispatch/model_status.py), 세 모델 시작 shell에 있다.
[검사 코드](../srgc_research/tests/test_gemma_text_recheck.py)는 다음을 확인한다.

- 실제 공개 shell로 한쪽 plan만 준비된 cohort를 중지한다. 이전 구현 해시가 들어 있는
  queue 파일도 덮어쓰지 않으며, 입력과 원래 JSON 파일의 바이트를 비교한다.
- 실제 CPU 자식 프로세스를 시작한 뒤 공개 shell로 `stop --now`를 요청한다. 해당 PID가
  종료됐고 새 plan·run이 생성되지 않았는지 확인한다. 모델 로딩·학습은 실행하지 않는다.
- 완만한 중지는 현재 준비를 마친 뒤 다음 단계를 막는다. 미리 기록한 즉시 중지를
  완만한 중지가 취소하지 않으며, 손상된 중지 JSON도 다시 기록할 수 있다.
- plan이 없을 때 중지해도 일반 `resume`으로 준비를 다시 시작할 수 있다. 호환되지
  않는 plan에 대한 resume은 중지 기록을 지우기 전에 거부한다.
- MATH 연결 실행 중 MBPP만 중지해도 현재 MATH 단계는 막지 않는다.
- 공백을 포함한 root, 옵션만 지정한 기본 실행, 세 모델의 전달 인수와 도움말을 검사한다.

## 검증 기록

최초 재현 검사는 수정 전 6건 모두 실패했다. Qwen·Llama로 확대한 옵션 검사에서도
수정 전 6건이 실패하고, 먼저 수정된 Gemma 3건은 통과했다. 최종 검사 결과는
[검증 요약](evidence/gemma4-operational-audit-2026-10-21.json)에 기록한다.

로컬 원시 기록은 `/home/kms/reports/gemma-operational-audit-2026-10-21/` 아래의
`reproduced-failures.log`, `chain-reproduced-failures.log`, `final-publication-tests.log`,
`scientific-preservation.json`이다. 테스트에 쓰인 보상·checkpoint fixture는 합성 데이터이며
논문 실측으로 집계하지 않는다. 실제 CUDA 생성 결과는 앞선
[PT 응답 형식 진단](evidence/gemma4-pt-completion-diagnosis-2026-10-21.json)에 따로 남아 있다.

## 실험 보존과 적용 범위

고정 과학 코드 71개를 기준 커밋과 바이트 단위로 비교했다. 변경 파일은 0개다.

- text-v3 adapter: `2c0dfb6a80114fd87a98af0610a8740d3f2353535586888fdd519422c257c157`
- 공통 engine: `12cf5ef830ebfd92fa8a87ea62dc7df734cd9ceab57fbce18fc4b2548385f960`
- 보존한 Gemma v1 adapter: `ba0a57d8ffa8802bb52c04b307ccad38a50f232132f0a58345176e3a33a552b1`

현재 text-v3의 가중치, 입력 형식, sampling, verifier, seed, 선택 규칙, 보상 기록,
중간 checkpoint 및 GPU 시간 계산을 유지한다. 새 실험 버전이나 별도 실행 모드는 만들지 않는다.
공개 시작 명령은 `sh scripts/run_srgc_gemma4.sh all`이다.

운영 진입점 수정은 업데이트 후 새로 실행한 Python 프로세스에 적용된다. 실행 중인
학습 프로세스는 이 점검 때문에 재시작하지 않았다. 원격 H100의 전체 preflight·init·학습은
이번에 실행하거나 측정하지 않았으므로 정확도 회복이나 학습 완료를 주장하지 않는다.
