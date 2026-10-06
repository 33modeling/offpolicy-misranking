# Qwen3.5-9B 온라인 전환 확장

2026-10-06 Pair 입력 이름 수정: 기존 Pair MATH 입력의 `math500` 표기를
`math_train` plan과 다르다는 이유로 거부하던 검사를 수정했다. 원래 Pair 출처가
기록된 입력만 허용하며 문제·정답·분할·50개 reference는 바꾸지 않는다.
`ded3d11`에서 이 오류로 멈춘 MATH 준비본은 **실행 기록·캐시가 전혀 없을 때만**
원본 plan을 보관하고 현재 adapter로 연결한다. 입력 파일은 수정하지 않는다.
실행이 시작된 실험이나 알 수 없는 코드 버전은 자동 복구하지 않는다.

2026-10-06 시작 오류 수정: `No package metadata was found for math-verify`는
Qwen의 패키지 검사보다 오프라인 verifier 연결이 늦어서 발생했다. 이제 기존
OLMo와 같은 검증된 `math-verify==0.9.0` 묶음을 `$OM_WORK/runtime-deps`에
먼저 준비하고 현재 프로세스와 자식 프로세스에 연결한다. 인터넷·pip 설치나
공유 venv 변경 없이 기존 시작 명령을 그대로 사용한다.

2026-10-06 재점검: 수동 다운로드에도 공유 잠금과 기존 모델 보호를 적용했다.
완료 직후 중단된 worker는 재실행 시 GPU 학습 없이 결과 요약을 복구한다.
여러 queue의 환경 검사를 모두 통과한 뒤 환경 정보를 기록하며, 재개에 사용하는
rollout·attention·진단 기록 코드도 adapter hash에 포함한다.
이번 수정은 adapter hash를 바꾸므로 **새 실험에서 사용한다. 기존 Qwen 실험은
당시 checkout과 root를 유지하며, 기존 plan/hash/잠금을 삭제하거나 바꾸지 않는다.**

2026-10-04 코드 수정: 사용자 중단은 재시도 한도에서 제외하고, 새 MBPP 채점은
`parent-checked-values-v3`로 구분한다. 기존 root를 새 코드로 자동 재개하거나
이전 보상 캐시를 v3로 재명명하지 않는다. 기존 작업은 당시 checkout을 유지한다.
[수정·검증 기록](V7_EXPERIMENT_FIXES_2026-10-04.md)과
[현재 실행 안내](REBUTTAL_COMMANDS_KO.md#5-실행-환경과-보존-원칙) 참고.

2026-09-28 준비. 대상은 **`Qwen/Qwen3.5-9B` post-trained**이며
`Qwen3.5-9B-Base`와 다른 모델이다. 고정 revision은
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
기존 Qwen checkpoint별 selection 매트릭스와 별도로, 현재 OLMo 온라인
네 arm 실험을 Qwen에서 수행한다. 모델과 초기 학습 상태가 함께 달라지므로
순수 architecture ablation으로 해석하지 않는다.

같은 날 메모리·여러 노드·저장 경로를 추가 감사해 실행 프로토콜을
**`qwen35-9b-v2`**로 분리했다. 이전 v1 plan/cache/checkpoint를 덮어쓰거나
새 코드로 이어서 실행하지 않는다. 기존 실행이 있다면 그 코드 snapshot을 유지한다.

## 실험 구성

| 항목 | 설정 |
| --- | --- |
| 데이터 | MATH, MBPP 각각 원래 OLMo plan과 같은 문제·정답·분할 |
| 시드 / 비교군 | 5–9 × Random, SR, On-policy, Switch: 데이터셋당 20개 continuation |
| 길이 | Qwen에서 새로 만든 25-update 공통 prefix → total step 275 |
| On-policy | 25 updates마다 후보 40문제 × 8응답, gradient 점수 상위 4문제를 다음 refresh까지 사용 |
| 학습 | 4문제 × 새 응답 8개, GRPO, LoRA rank 16 / alpha 32 |
| SR / Random | 매 update 무작위 후보 40문제에서 SR 상위 4개 / 무작위 4개 |
| SR-GC | 기존 40-vs-40, validation 50개와 temporal rule 유지; 자기 Switch 경로에서 측정 |
| 초기 SR 캐시 | Qwen 초기 정책에서 400문제 × 8응답을 새로 생성; OLMo reward/prefix 재사용 금지 |
| 평가 | 같은 300문제 × 8응답, total step 275 결과와 실제 phase별 비용 |
| 생성 | temperature 1, top-p 1, top-k 0, 최대 2,048 새 토큰; thinking off |
| 메모리 | 생성 micro-batch 2, log-prob micro-batch 1, vocabulary projection 64토큰, non-reentrant activation checkpointing |
| 모델별 차이 | Qwen tokenizer chat wrapper; attention q/v 및 DeltaNet 4개 projection에 LoRA |

On-policy/Switch의 refresh 비용에는 후보 두 집합의 합집합 및 validation
계산이 포함된다. 40×8은 On-policy 후보 자체의 응답 수이며 전체 SR-GC
비교 비용을 320개 응답으로 축소하여 기록하지 않는다.

총 40개 continuation 외에 데이터셋별 5개 cache 작업과 5개 prefix 작업이 있다.
한 노드는 **H100 80GB 4장**으로 한 작업을 수행한다. 여러 노드에서 같은
명령을 실행하면 공유 queue가 서로 다른 작업을 배정한다. 1개 노드로도
순차 실행 가능하며, 노드가 늘어나면 독립 시드/arm을 병렬 처리한다.

## 시작 명령

빈 4-H100 노드마다 원하는 데이터셋의 명령을 한 번 실행한다. **5개 노드면 같은
명령을 5개 노드에 각각 실행**한다. 선택한 데이터셋의 seeds 5–9와 네 arm을
공유 queue에서 자동 배정한다. 한 데이터셋은 초기 cache/prefix 최대 5노드,
네 arm이 준비된 뒤 최대 20노드다. `all`은 양쪽 합계 초기 최대 10노드,
이후 최대 40노드이며, 선행 작업 진행 상태에 따라 일부 노드는 대기할 수 있다.

```sh
# MATH
sh scripts/run_srgc_qwen35.sh math

# MBPP
sh scripts/run_srgc_qwen35.sh mbpp

# 두 데이터셋을 같은 queue worker에서 처리
sh scripts/run_srgc_qwen35.sh all
```

`math`는 MATH 20개, `mbpp`는 MBPP 20개, `all`은 양쪽 40개 continuation을 처리한다.
데이터셋만 입력하며 시드는 자동 배정한다. 인자 없이 실행하면 `all`과 같다.
패키지 검사, 없는 모델 다운로드, 없는 입력 준비, GPU admission, 학습 순서로
진행한다. 다운로드·입력 준비만 공유 잠금으로 직렬 처리하고 학습은 병렬 배정한다.
이미 준비한 plan은 활성 OLMo cohort가 바뀌어도 다시 만들지 않는다. 기존 결과와
checkpoint를 보존하며 코드·입력 불일치나 기존 모델 손상은 자동 초기화 없이 중단한다.
새 root나 다른 모델 실험을 자동으로 추가하지 않는다.
기존 `runs/<dataset>/`에 기록이 있는데 plan만 없으면 새 plan을 만들지 않고 중단한다.
이때는 해당 실행의 원래 plan을 복구해야 한다. 다운로드·입력 준비 로그는
`<Qwen root>/startup-logs/<실행 ID>.log`에 남긴다. 준비 도중 Ctrl-C/SIGTERM으로
중단하면 자식 프로세스 종료를 확인한 뒤 준비 잠금을 해제하며 학습을 시작하지 않는다.

### 환경·저장 경로

실행 코드는 `master`. OLMo와 같은 Python 환경을 기본으로 사용한다.
`math/all`은 `PAIR_PYTHON`, `mbpp`는 `SWITCH_PYTHON`을 따르며,
미지정 시 `${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python`을 사용한다.
`QWEN_PYTHON`은 기존 실행의 환경을 명시적으로 유지할 때만 쓰는 override다.
별도 `.venv-qwen35` 생성이나 환경변수 지정은 필요 없다. 패키지 목록은
[`configs/srgc_qwen35/requirements.txt`](../configs/srgc_qwen35/requirements.txt).
CUDA PyTorch는 노드의 검증된 빌드를 유지한다. FLA 0.5.2가 필요하며
추론뿐 아니라 dense scoring/LoRA 역전파가 실제 GPU admission을 통과해야 한다.
Transformers 5.14.1 / PEFT 0.20.0 / FLA 0.5.2를 검사하고, 최초 worker의
PyTorch·CUDA·cuDNN·Python·나머지 패키지 버전을 각 queue에 기록한다.
다른 환경의 노드나 환경이 바뀐 재시작은 작업을 받기 전에 거부한다.
공유 환경이 이 버전을 충족하는지는 시작 명령에서 검사한다. launcher는 패키지를
자동 설치하거나 업그레이드하지 않는다. 공유 환경에서 학습이 실행 중이면
패키지를 바꾸지 않는다. 기존 Qwen 결과의 환경 일치 검사도 유지한다.

`MODELS_DIR` 기본값은 `$GROUP_VOLUME/models`; 모델 경로를 따로 지정하려면
`SRGC_QWEN_MODEL_PATH`를 사용한다. 기존 다운로드라면 정확한 Hub revision의
`.om_snapshot.json` 검증이 필요하다. 단순히 폴더 이름이 같은 모델은 허용하지 않는다.
`prepare`는 활성 OLMo source plan의 질문 분할을 읽되 Qwen 캐시를 비워 둔다.
기존 `all download/doctor/prepare/run` 인터페이스는 수동 점검용으로 유지하지만
각각 실행할 필요는 없다. 특정 cohort를 비교해야 할 때만 최초 준비 전에
`math prepare --source-plan <path>` 또는 `mbpp prepare --source-plan <path>`를 사용한다.
이미 준비된 plan에는 이 옵션을 다시 적용하지 않는다.

기본 결과 위치는 `$OM_WORK/srgc-rebuttal/qwen35-9b-v2`이며
`SRGC_QWEN_ROOT`로 별도 group-volume 하위 경로를 지정할 수 있다.
**모든 노드에서 같은 코드, 환경, 모델 snapshot, Qwen root를 사용한다.**
준비·다운로드·결과 export 단계부터 group volume과 경로를 검사한다.
마운트 경로가 없으면 홈 저장으로 대체하지 않는다. group 밖으로 향하는
root, model 경로, plan 출력 경로, 기존 심볼릭 링크를 거부한다.
`OM_WORK`가 홈 경로라면 기본 group work 경로로 정규화한다.
Hugging Face·Torch·Triton·CUDA 캐시와 임시 파일도
`$OM_WORK/qwen-runtime-cache` 아래에 둔다. 각 rank가 과거 receipt 전체를
반복 스캔하지 않고 해당 plan의 실제 입력·출력 경로를 검사한다.

### 조회·중단·재개

아래는 관리 명령이며 추가 실험이나 필수 실행 순서가 아니다.

```sh
# 마지막 Qwen GPU 검사 실패 원인만 조회. 모델 로드·학습·파일 변경 없음.
sh scripts/run_srgc_qwen35.sh all error

# CPU에서도 조회 가능
sh scripts/run_srgc_qwen35.sh all status
sh scripts/run_srgc_qwen35.sh all results

# 현재 작업을 마치고 중단 / 이후 중단 표시 해제
sh scripts/run_srgc_qwen35.sh all stop
sh scripts/run_srgc_qwen35.sh all resume
sh scripts/run_srgc_qwen35.sh
```

`qwen-smoke.log`의 `admission failed`는 실제 Qwen 생성·역전파 검사 실패다.
`error` 명령은 선택한 데이터셋의 마지막 검사 로그에서 원래 rank 예외를 출력한다.
시작 명령도 같은 원인을 예외 메시지 아래에 붙인다. 이 출력 보강은 adapter/engine
hash를 바꾸지 않으므로 검사에서 멈춘 기존 plan·queue를 다시 만들 필요가 없다.
로그를 확인하지 않고 GPU 검사를 생략하거나 OOM·CUDA 오류를 성공으로 처리하지 않는다.

### 가중치 로딩 후 NCCL CUDA 802

작은 NCCL 검사가 통과해도 실제 Qwen 검사에서
`ncclUnhandledCudaError`와 `Cuda failure 802 'system not yet initialized'`가
발생할 수 있다. 일반적인 `unhandled CUDA error`만으로 원인을 판단하지 않는다.

현재 시작 명령은 **해당 Qwen 검사 로그에 NCCL 오류와 CUDA 802가 함께 기록된
경우에만** 기존 NCCL 복구 순서인 `NCCL_NVLS_ENABLE=0`, `NCCL_CUMEM_ENABLE=0`,
`NCCL_P2P_DISABLE=1`을 한 단계씩 추가한다. 이미 명시된 설정은 바꾸지 않는다.
각 단계마다 작은 NCCL 검사와 실제 Qwen 생성·역전파 검사를 모두 다시 실행한다.
최대 세 번의 추가 검사 후에도 실패하면 중단하며, 학습 작업은 배정하지 않는다.
OOM·다른 CUDA 오류·사용자 중단에는 이 복구를 적용하지 않는다.

검사에서는 NCCL INFO 로그를 기본으로 사용하되 사용자가 지정한 로그 설정은
유지한다. 성공한 통신 설정만 학습에 전달하고, 실패한 검사 receipt의 비용도 합산한다.
각 시도의 원본 로그·receipt를 보존하며 최상위 `qwen-admission.json`에 합산 비용과
마지막 검사 경로를 기록한다. 중단으로 receipt가 남지 않은 시도는
`cost_accounting_complete=false`로 표시하며, 이때 비용 합계는 하한이다.
이 운영 경로 수정은 adapter/engine hash나 입력을
바꾸지 않는다. 다만 통신 방식에 따라 실행시간과 부동소수점 합산 순서는 달라질
수 있으므로 각 worker의 실제 설정을 결과와 함께 보존한다.

이 복구는 시스템 장애 수리를 보장하지 않는다. 모든 단계에서 802가 계속되면
호스트 드라이버·CUDA 라이브러리·NVSwitch/Fabric 상태를 관리자가 점검해야 한다.
[NVIDIA CUDA 초기화 안내](https://docs.nvidia.com/nim/large-language-models/2.0.13/troubleshooting/cuda-driver.html).

매 worker 시작 시 기존 4-rank NCCL 검사 뒤에 **실제 9B 모델 생성·scoring
역전파·GRPO update**를 검사한다. 이 검사의 보상은 backward 확인용 합성
보상이며 실험 cache/결과에 기록하지 않는다. 실패하면 cache 학습을 시작하지
않고 `.queue/admission/.../qwen-smoke.log`를 확인하도록 중단한다.
짧은 예제만 검사하지 않고 준비된 두 데이터셋에서 가장 긴 prompt와
2,048-token 합성 응답으로 역전파를 검사한다. rank별 peak allocated/reserved
메모리를 로그에 남기고, 성공·실패한 admission의 GPU 시간을 별도 JSON에 보존한다.
이미 완료된 queue를 다시 조회·실행할 때 모델 admission을 반복하지 않는다.

학습은 매 update 저장하고 공통 prefix의 모델/optimizer를 네 arm이 공유한다.
완료 cache/arm은 재실행하지 않으며 코드·plan·입력이 다르면 resume을 거부한다.
OLMo와 동일한 물리 GPU UUID lock을 사용한다. 기존 worker나 lock 파일을
삭제하는 실행 옵션은 없다. 실패 원인 수정 후에만 `run --retry-failed`를 사용한다.

2026-10-01 재점검: 진행 timeout은 exit 124로 남기고 다음 준비 작업을 처리한다.
`--retry-failed`일 때만 60초 간격으로 재시도하며 누적 상한은 기본 3회다.
사용자 중단은 130, 일반 실패는 1로 구분하고 사용자 중단을 성공 종료로 처리하지 않는다.
각 rank는 `SRGC_ATTENTION` 환경보다 plan의 고정 attention(`eager`)을 우선한다.

위 2026-10-01 수정은 Qwen adapter hash를 바꿨으므로 그 전에 시작한 Qwen 실험의 checkout을
업데이트하지 않는다. 기존 run은 원래 frozen 코드로 재개하고, 수정본으로 새로
실험하려면 사용하지 않은 root에 `prepare` 후 모든 노드에서 같은 `--root`로 실행한다.
기존 plan의 hash를 편집하거나 과거 결과를 새 버전으로 재분류하지 않는다.
새 checkout에서도 `status/results --root`는 과거 plan/입력/실행 identity를 검증해
조회할 수 있지만, `run/prepare/stop/resume`의 버전 검사는 그대로 유지한다.
2026-10-02 단일 시작 명령 추가는 adapter/engine hash를 변경하지 않았다.

`all` worker도 MATH/MBPP별 worker receipt와 로그 경로를 정확히 기록한다.
전용 root가 다른 Qwen worker끼리도 공통 GPU UUID 잠금을 사용한다.
2026-10-02 재점검에서 공통 보호 코드와 Qwen worker가 같은 GPU 잠금을 중복
획득해 자기 자신에게 `BUSY`가 나는 문제를 수정했다. 같은 프로세스·스레드의
중첩 호출만 기존 잠금을 재사용한다. 다른 프로세스·스레드의 동일 GPU 획득은 차단한다.
Qwen worker와 공통 보호 코드를 함께 실행해 admission 호출까지 도달하는 회귀 테스트를 추가했다.
이 수정은 학습 엔진과 Qwen adapter hash를 변경하지 않는다.
정리 대상은 SRGC 실행으로 한정하고 무관한 `torchrun`을 종료하지 않는다.
캐시 완료를 task 잠금 획득 후 다시 확인해 다른 노드와의 완료 경합을 처리한다.
비용 receipt가 없는 완료 cache는 export 복구부터 수행하고 prefix를 시작하지 않는다.

문제당 응답은 여전히 **8개**다. 생성만 2개씩 나누며 OOM이면 해당 호출의
난수 상태를 복원하고 1개씩 다시 생성한다. 응답을 버리거나 길이를 줄이지 않는다.
최소 배치에서도 실패하거나 다른 CUDA 오류가 나면 실패를 그대로 보고한다.
8개 동시 생성과 난수 소비 순서가 달라질 수 있어 v1과 결과를 합치지 않는다.
optimizer checkpoint는 GPU에서 deep-copy하지 않고 CPU로 복사하며,
5/25-step 경계 저장도 flush/fsync 후 원자적으로 교체한다.

## 검증 범위와 결과 보고

2026-10-02 재검증: PyTorch 2.13.0+cpu / Transformers 5.14.1 / PEFT 0.20.0 환경에서
`srgc_rebuttal/tests` 전체 **428개 통과, 실패·건너뜀 없음**. 공통 보호 코드와 Qwen
worker의 통합 시작, 중첩 GPU 잠금, 다른 프로세스·스레드의 중복 점유 차단을 포함한다.
실제 H100 다중 노드 검증을 대신하지 않는다.

CPU의 작은 실제 Qwen hybrid 모델에서 8응답 생성, dense gradient, 양쪽 층의
LoRA update, optimizer 상태 복원 및 동일 결과 재현을 검사했다. 공식 snapshot의
`model.language_model.*` 이름을 text-only decoder에 정확히 매핑하며, 누락된
text weight가 있으면 실행을 거부하는 검사도 포함한다.

회귀 검사 명령은 `python -m unittest discover -s srgc_rebuttal/tests -v`다.
환경은 Python 3.12, PyTorch 2.13 CPU, Transformers 5.14.1, PEFT 0.20.0.
감사 결과와 실행한 검사는 [추가 감사 기록](QWEN35_SRGC_AUDIT_KO.md)에 정리한다.
고정 revision의 실제 tokenizer로 MATH/MBPP 10개 입력 bundle을 생성하고,
40개 continuation plan 및 양쪽 status/results export를 검증했다.
기존 OLMo engine SHA-256은
`1869fe1cf898d4ff3a6d5e9054790836442b5e0b81b485fb04bc27de4ebab20a`로 유지했다.

이 준비 환경에는 사용할 수 있는 H100이 없어 **실제 9B GPU 실행·메모리·속도는
아직 검증하지 않았다**. 실행 시 위 admission으로 확인한다. 아직 reward 결과는 없다.
다섯 시드를 모두 수집하고 각 데이터셋 내 Qwen의 paired 차이를 보고한다.
OLMo와 Qwen의 reward/시간을 하나의 평균으로 합치지 않는다. source plan/input
hash와 chat/thinking/LoRA 차이를 결과와 함께 보관한다.

공식 근거:
[Qwen3.5-9B 모델 카드](https://huggingface.co/Qwen/Qwen3.5-9B),
[Transformers Qwen3.5 text-only 모델 문서](https://huggingface.co/docs/transformers/model_doc/qwen3_5).
