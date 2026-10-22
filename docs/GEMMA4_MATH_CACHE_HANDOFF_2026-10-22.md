# Gemma IT: MATH 캐시 후 바로 MATH 학습

사용자 수정 지시: MATH를 MBPP 캐시 완료까지 기다리게 하지 않는다.

기존 실행기는 `all`의 MATH 단계에서 캐시 전용 모드를 켜면서 데이터셋 범위를
해제했다. 그 결과 MBPP의 캐시·lease·preflight까지 MATH 시작 조건에 들어갔다.
새 실행 순서는 MATH 캐시 → MATH prefix·학습 → MBPP 캐시 → MBPP 학습이다.

- `gemma_it_run.py`: 현재 데이터셋의 캐시만 준비하고 바로 해당 학습으로 연결한다.
  모델 체인이 없는 직접 실행과 reset도 같은 순서를 따른다.
- `resume_drain.py`, `ready_dispatch.py`: 내부 `SRGC_CACHE_DATASET` 범위를
  배정·복구 backlog·완료 판정에 적용한다. MBPP 캐시가 없거나 점유되어도 MATH를 막지 않는다.
- preflight·GPU admission도 현재 데이터셋만 검사한다.
- 공개 명령은 그대로 `bash scripts/run_srgc_gemma4.sh all`이다.

학습 adapter/engine, 모델·tokenizer, 입력·sampling, reward·checkpoint와 측정 설정은
바꾸지 않았다. 기존 캐시·체크포인트를 지우거나 실험 root를 초기화하지 않는다.
이미 실행 중인 원격 Python 프로세스에 코드를 주입하지 않는다.

새 회귀 검사는 실제 준비 입력·queue·lease를 사용한다. 수정 전 6건 중 5건 실패,
수정 후 6건 모두 통과했다. GPU 실행은 대체한 CPU 배정 검사이며 실제 optimizer
업데이트나 원격 MATH 학습 시작을 측정한 결과로 보고하지 않는다.
원시 검사 기록: `/home/kms/reports/gemma-cache-handoff-20261022/`.

관련 Gemma IT·캐시·resume drain·모델 체인·상태 회귀 검사는 총 286건이
58.01초에 통과했다. 기존 multiprocessing fork 경고 12건이 있었고 실패는 없었다.
실행 시 `CUDA_VISIBLE_DEVICES=''`로 GPU를 사용하지 않았다.

수정 전후 frozen runtime digest도 일치했다.

- adapter: `adb962509386e02d8ed4265d1b7f3e82f9ea6b2ff124feb8d49e706eb6c33d82`
- engine: `12cf5ef830ebfd92fa8a87ea62dc7df734cd9ceab57fbce18fc4b2548385f960`

현재 세션에는 원격 `/group-volume`이나 4-GPU 노드 접속이 없다. 로컬에는 Gemma
학습 프로세스가 관찰되지 않았으며, 실제 원격 캐시 완료 수와 학습 진척은 미측정이다.
기존 로컬의 별도 SFT 실행은 중단하지 않았다.
