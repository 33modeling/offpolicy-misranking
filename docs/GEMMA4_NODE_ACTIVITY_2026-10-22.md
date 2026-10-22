# Local node / active 집계 누락 수정

사용자 보고: `Local node`가 실행 중인데도 active에 잡히지 않거나 노드가 누락된다.
원격 노드 로그는 이 세션에 없으므로 특정 원격 실행의 실패 원인이나 학습 진척은
단정하지 않는다. 아래 결과는 실제 상태 코드와 준비 입력을 사용한 CPU 회귀 검사다.

## 확인한 원인과 변경

기존 `NODES`는 데이터셋의 `.queue/workers`만 집계했다. 모델 체인 실행기는
별도의 `MODEL QUEUE`에 표시되면서도 active 합계에서는 빠졌다. 워커 기록이
아직 없거나 오래됐을 때, 유효한 작업 lease로 확인되는 RUN 작업도 노드에서
누락되거나 stale로만 집계됐다. 추가로 체인의 대기 간격은 최대 60초인데
stale 기준은 30초여서 정상 대기 중인 실행기가 stale로 바뀔 수 있었다.

- 노드 집계는 데이터셋 워커, RUN 작업 lease, 모델 체인 heartbeat를 함께 읽는다.
  같은 호스트는 한 번만 세고, 현재 작업의 진행 상황을 우선 표시한다.
- RUN 작업 lease가 유지되면 워커 기록이 없어도 해당 호스트를 active로 표시한다.
  heartbeat 누락·지연은 별도 사유로 남긴다.
- 살아 있는 실행기는 `LAUNCH`, `WAIT`, `HANDOFF`로 구분한다. 실행기만 있는
  경우 RUN 작업 수와 optimizer update를 만들어 내지 않는다.
- `Local node`에 현재 상태를 직접 표시한다. JSON의 `nodes`와 터미널 출력은
  같은 집계 함수를 사용한다.
- 새 체인 기록에 실제 모델 프로토콜과 실험 root를 저장해 다른 root 및 PT/IT
  기록이 섞이지 않게 한다. 이전 기록은 root 미기록 사실을 명시한 실행기로만 표시한다.
- 체인은 긴 대기 중에도 5초마다 기존 상태의 heartbeat를 갱신한다.

## 검증과 증거

기준 코드 [095c027](https://github.com/33modeling/offpolicy-misranking/commit/095c02765dfe15b242920459610686e4cb779679)의
상태 모듈과 수정 코드를 같은 준비 입력·lease에 적용했다.

| 재현 조건 | 기존 집계 | 수정 집계 | 수정 상태 |
| --- | --- | --- | --- |
| RUN 작업 lease 유지, 워커 기록 없음 | 0 노드 | 1 노드 | RUN |
| 체인 heartbeat 유지, 데이터셋 워커 등록 전 | 0 노드 | 1 노드 | WAIT, RUN 작업 0 |

자동 검사 **192건이 39.77초에 통과**했다. 신규 검사는 Gemma IT·Qwen·Llama의
등록 전 실행기, 워커 heartbeat 누락·지연, 중복 제거, 다른 root·프로토콜 제외,
stale·종료 기록, 이전 체인 기록, 공개 Gemma shell과 데이터셋 단독 조회,
60초 대기 중 heartbeat 유지를 포함한다. 기존 상태·히스토리·모델 체인·MATH
캐시 인계 검사도 함께 통과했다.

- [신규 회귀 검사](../srgc_research/tests/test_node_activity.py)
- [노드 집계](../srgc_research/dispatch/qwen_status.py)
- [공통 상태 CLI](../srgc_research/dispatch/model_status.py)
- [체인 heartbeat와 실험 범위 기록](../srgc_research/dispatch/model_chain.py)
- [재현 요약과 frozen digest](evidence/gemma-node-status-20261022.json)

원시 로그와 비교 출력은 `/home/kms/reports/gemma-node-status-20261022/`에 보존한다.
최초 테스트 실행 중 발견한 잘못된 테스트 CLI 인자와, 첫 수정에서 발견한 이전
worker ID 연결 회귀의 실패 로그도 덮어쓰지 않고 보존했다. 최종 통과 기록은
`regression-final.log`다. 테스트의 작업 receipt는 합성 데이터이며 실제 학습
reward나 원격 GPU 실행 결과가 아니다. GPU를 사용하거나 실험을 재시작하지 않았다.

수정 전후 학습 adapter/engine digest는 일치했다. 캐시·체크포인트·원시 reward와
기존 실험의 frozen runtime을 변경하지 않았다.
