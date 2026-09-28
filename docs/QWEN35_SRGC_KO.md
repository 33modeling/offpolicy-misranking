# Qwen3.5-9B 온라인 전환 확장

2026-09-28 준비. 대상은 **`Qwen/Qwen3.5-9B` post-trained**이며
`Qwen3.5-9B-Base`와 다른 모델이다. 고정 revision은
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
기존 Qwen checkpoint별 selection 매트릭스와 별도로, 현재 OLMo 온라인
네 arm 실험을 Qwen에서 수행한다. 모델과 초기 학습 상태가 함께 달라지므로
순수 architecture ablation으로 해석하지 않는다.

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
| 모델별 차이 | Qwen tokenizer chat wrapper; attention q/v 및 DeltaNet 4개 projection에 LoRA |

On-policy/Switch의 refresh 비용에는 후보 두 집합의 합집합 및 validation
계산이 포함된다. 40×8은 On-policy 후보 자체의 응답 수이며 전체 SR-GC
비교 비용을 320개 응답으로 축소하여 기록하지 않는다.

총 40개 continuation 외에 데이터셋별 5개 cache 작업과 5개 prefix 작업이 있다.
한 노드는 **H100 80GB 4장**으로 한 작업을 수행한다. 여러 노드에서 같은
명령을 실행하면 공유 queue가 서로 다른 작업을 배정한다. 1개 노드로도
순차 실행 가능하며, 노드가 늘어나면 독립 시드/arm을 병렬 처리한다.

## 준비와 실행

실행 코드는 `master`. 기존 OLMo Python 환경을 업그레이드하지 말고
별도 CUDA 환경을 `QWEN_PYTHON`으로 지정한다. 패키지 목록은
[`configs/srgc_qwen35/requirements.txt`](../configs/srgc_qwen35/requirements.txt).
CUDA PyTorch는 노드의 검증된 빌드를 유지한다. FLA 0.5.2가 필요하며
추론뿐 아니라 dense scoring/LoRA 역전파가 실제 GPU admission을 통과해야 한다.

```sh
export QWEN_PYTHON="$OM_WORK/.venv-qwen35/bin/python"
sh scripts/run_srgc_qwen35.sh all download
sh scripts/run_srgc_qwen35.sh all doctor
sh scripts/run_srgc_qwen35.sh all prepare
```

`MODELS_DIR` 기본값은 `$GROUP_VOLUME/models`; 모델 경로를 따로 지정하려면
`SRGC_QWEN_MODEL_PATH`를 사용한다. 기존 다운로드라면 정확한 Hub revision의
`.om_snapshot.json` 검증이 필요하다. 단순히 폴더 이름이 같은 모델은 허용하지 않는다.
`prepare`는 활성 OLMo source plan의 질문 분할을 읽되 Qwen 캐시를 비워 둔다.
비교할 cohort를 명시할 수도 있다:

```sh
sh scripts/run_srgc_qwen35.sh math prepare --source-plan /absolute/path/to/math-plan.json
sh scripts/run_srgc_qwen35.sh mbpp prepare --source-plan /absolute/path/to/mbpp-plan.json
```

모델 가중치 다운로드 전 CPU에서 입력을 준비하려면 `prepare`에
`--allow-tokenizer-download`를 붙인다. 이 옵션은 고정 revision의 tokenizer만 받는다.

기본 결과 위치는 `$OM_WORK/srgc-rebuttal/qwen35-9b`이며
`SRGC_QWEN_ROOT`로 별도 group-volume 하위 경로를 지정할 수 있다.
**모든 노드에서 같은 코드, 환경, 모델 snapshot, Qwen root를 사용한다.**

```sh
# 각 빈 4-H100 노드에서 실행. all은 MATH/MBPP 두 queue를 모두 처리한다.
sh scripts/run_srgc_qwen35.sh all run

# CPU에서도 조회 가능
sh scripts/run_srgc_qwen35.sh all status
sh scripts/run_srgc_qwen35.sh all results

# 현재 작업을 마치고 중단 / 이후 중단 표시 해제
sh scripts/run_srgc_qwen35.sh all stop
sh scripts/run_srgc_qwen35.sh all resume
sh scripts/run_srgc_qwen35.sh all run
```

매 worker 시작 시 기존 4-rank NCCL 검사 뒤에 **실제 9B 모델 생성·scoring
역전파·GRPO update**를 검사한다. 이 검사의 보상은 backward 확인용 합성
보상이며 실험 cache/결과에 기록하지 않는다. 실패하면 cache 학습을 시작하지
않고 `.queue/admission/.../qwen-smoke.log`를 확인하도록 중단한다.

학습은 매 update 저장하고 공통 prefix의 모델/optimizer를 네 arm이 공유한다.
완료 cache/arm은 재실행하지 않으며 코드·plan·입력이 다르면 resume을 거부한다.
OLMo와 동일한 물리 GPU UUID lock을 사용한다. 기존 worker나 lock 파일을
삭제하는 실행 옵션은 없다. 실패 원인 수정 후에만 `run --retry-failed`를 사용한다.

## 검증 범위와 결과 보고

CPU의 작은 실제 Qwen hybrid 모델에서 8응답 생성, dense gradient, 양쪽 층의
LoRA update, optimizer 상태 복원 및 동일 결과 재현을 검사했다. 공식 snapshot의
`model.language_model.*` 이름을 text-only decoder에 정확히 매핑하며, 누락된
text weight가 있으면 실행을 거부하는 검사도 포함한다.

회귀 검사: `python -m unittest discover -s srgc_rebuttal/tests -v` 전체
229개 통과(Python 3.12, PyTorch 2.13 CPU, Transformers 5.14.1, PEFT 0.20.0).
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
