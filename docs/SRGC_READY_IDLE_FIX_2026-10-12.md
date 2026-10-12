# Qwen·Llama·Gemma READY/IDLE 배정 수정

기준 코드: `63cbfcf554f97a863b9bcf3a34fa4c79870fb829`, `master`.
대상: `run_srgc_qwen35.sh`, `run_srgc_llama31.sh`, `run_srgc_gemma4.sh`의 현재 실행 entry.

## 원인과 동작

기존 `ResumeFirst.claim()`은 공유 resume backlog가 비워질 때까지 backlog 밖의
모든 작업을 거부했다. 다른 노드가 재개 작업을 실행하는 동안에도 독립적인
READY 작업을 배정하지 않아 빈 노드가 계속 IDLE로 남았다.
재시도 대기, 다른 데이터셋의 stop 기록도 같은 대기를 만들었다.

현재 entry는 `srgc_research/dispatch/ready_dispatch.py`를 사용한다.

- 지금 실행할 수 있는 재개 작업을 먼저 배정한다.
- 재개 작업이 이미 다른 노드에 배정됐거나 재시도 대기·한도에 걸려 있으면
  요청한 데이터셋의 독립적인 READY 작업을 배정한다.
- cache→prefix→continuation 의존성과 task/execution lease를 계속 검사한다.
- 배정을 건너뛴 작업의 lease는 바로 해제한다. 다음 작업이 학습하는 동안
  재시도 대기 작업의 lease까지 점유하지 않는다.
- 실패 한도에 도달한 작업은 더 실행하지 않는다. 남은 독립 작업을 처리한 뒤
  실패 작업 이름을 출력하고 오류로 종료한다. 미완료 작업을 완료로 바꾸지 않는다.
- `math`/`mbpp`는 다른 데이터셋의 기존 backlog만 복구한다. 그쪽의 새 작업은
  시작하지 않는다. `all`은 두 데이터셋의 READY 작업을 모두 처리한다.
- 중지된 다른 데이터셋의 기록은 보존하며 요청한 데이터셋을 막지 않는다.

## 기존 실행 보존

`qwen_resume.py`, `model_resume.py`, `llama31/`, `gemma4/`, 학습 engine은
수정하지 않았다. 고정된 파일의 내용으로 기준 commit의 digest를 재계산해
현재 값과 일치하는지 확인했다.

| 대상 | 변경 전과 동일한 SHA-256 |
| --- | --- |
| Qwen adapter | `20e90f784b19afa6602b839ff2f7934eb207f03868a4052896edc9355b7ffd8b` |
| Llama adapter | `1a7ae6ecf3b7fd24643797c97ce56750d32a9e49678125d8ddd2eabb5b61b53a` |
| Gemma adapter | `ba0a57d8ffa8802bb52c04b307ccad38a50f232132f0a58345176e3a33a552b1` |
| 공통 engine | `12cf5ef830ebfd92fa8a87ea62dc7df734cd9ceab57fbce18fc4b2548385f960` |

Plan, 입력, 결과, optimizer checkpoint, 실패 이력과 retry budget revision은
변경하지 않는다. 적용 방법은 [현재 실행 안내](CURRENT_EXPERIMENTS.md)에 있다.
이미 실행 중인 Python worker에는 새 배정기가 자동으로 로드되지 않는다.

## 검증

최종 수정본에서 아래 직접 관련 검사는 **144개 모두 통과**했다.

```sh
python3 -m pytest -q \
  srgc_research/tests/test_resume_drain.py \
  srgc_research/tests/test_qwen_resume.py \
  srgc_research/tests/test_model_resume.py \
  srgc_research/tests/test_qwen_status.py \
  srgc_research/tests/test_llama_run.py \
  srgc_research/tests/test_gemma_run.py
```

세 모델 각각에 대해 같은 데이터셋/다른 데이터셋의 실행 중 작업, 실제 두
프로세스의 서로 다른 READY 작업 배정, prefix 의존성, checkpoint 보존,
재시도 한도·대기, stop, 선택하지 않은 task lease의 즉시 해제를 검사했다.
수정 전에는 READY/IDLE 관련 12개 재현 검사가 실패했다.
변경 Python 파일의 Ruff 검사와 `git diff --check`도 통과했다.

추가로 실행한 research 전체 검사에서는 **488개 통과, 9개 실패**를 확인했다.
이후 최종 lease 보강은 위 144개 검사로 다시 검증했다. 전체 검사 실패 9개는
별도 detached checkout의 수정 전 기준 코드에서도 같은 환경으로 점검했다.

| 실패 종류 | 전체 검사 | 수정 전 코드에서 해당 검사만 재실행 |
| --- | ---: | --- |
| Llama tokenizer metadata/테스트 fixture 호환성 | 6개 실패 | 6개 모두 실패 재현 |
| Gemma FP32 gradient 일치 허용 오차 | 1개 실패 | 실패 재현 |
| Information mixed precision 오류 재현 기대 | 1개 실패 | 실패 재현 |
| Information 동시 retry budget 검사 | 1개 실패 | 통과, 간헐성 원인 미확정 |

위 검사 실패를 수정했다고 주장하지 않는다. tokenizer/gradient 학습 코드를
바꾸면 고정 실험 identity가 달라지므로 이번 배정 수정에 포함하지 않았다.

로컬 NVIDIA driver에 연결할 수 없고 `/group-volume`도 마운트되어 있지 않다.
실제 H100 노드에서 새 worker가 작업을 배정받고 학습을 진행하는지는 미확인이다.
