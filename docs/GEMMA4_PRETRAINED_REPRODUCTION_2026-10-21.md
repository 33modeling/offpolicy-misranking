# Gemma 4 12B 학습 전 생성 오류 재현

후속 수정과 새 실행 경로는 [Gemma text-v2 복구 안내](GEMMA4_TEXT_RECOVERY_2026-10-21.md)에 기록했다.
아래 내용은 수정 전 오류를 재현한 기록이며 기존 실험 원본은 보존한다.

실제 `google/gemma-4-12B` PT 가중치에서 이미지 종료 토큰을 반복하는 오류를 재현했다. 공식 생성 설정의 특수 토큰 억제가 실험 로더에서 제거된 것이 이 재현의 직접적인 설정 차이다. 억제를 복원하면 반복은 사라지지만 정답률 회복은 아직 확인되지 않았다. 원격에서 보고된 전체 init 1%대의 원인을 모두 설명했다고 결론 내리지 않는다.

## 실제 모델과 환경

- 모델: `google/gemma-4-12B`, revision `023679ed352de9bb66cc873c9009ce3482585c08`.
- 공식 가중치: 23,919,549,408 bytes, SHA-256 `fe054ae05ff7f44318fd8ae90d58992531455c7ed31356704088f0f2d8c8009a`.
- 공식 파일을 RAM 파일시스템으로 다운로드하며 SHA-256을 계산해 대조했다. 디스크 읽기 지연 때문에 디스크 사본은 검증 근거로 사용하지 않았다.
- 실제 모델 파라미터 11,959,730,176개, 가중치 tensor 677개. 공식 full-model 로더에서 누락·불일치 없이 로드했다.
- RTX 3050 6GB, CUDA 13.0, Torch 2.13.0, Transformers 5.19.0. 실제 BF16 가중치를 CPU에 두고 Accelerate로 GPU에 순차 올려 연산했다. 양자화와 학습은 하지 않았다.
- 실험 로더의 CPU 경로가 FP32를 선택하므로, 진단에서 staging dtype만 BF16으로 지정했다. 실제 실험의 CUDA BF16 dtype과 맞추기 위한 조치다. 원래 로더 소스는 변경하지 않았다.

## 가중치 매핑과 토크나이저 대조

공식 `Gemma4UnifiedForConditionalGeneration`과 실험의 `Gemma4UnifiedForCausalLM` text loader를 별도 프로세스에서 비교했다.

| 입력 | 프롬프트 토큰 | 마지막 토큰의 전체 vocabulary logits |
|---|---|---|
| `2 + 3 =` | 동일 | 비트 단위 동일, 최대 차이 0 |
| RL-Zero 형식의 `What is 2 + 3?` | 동일 | 비트 단위 동일, 최대 차이 0 |
| MATH seed 5의 두 번째 candidate | 동일 | 비트 단위 동일, 최대 차이 0 |
| MBPP seed 5의 두 번째 candidate | 동일 | 비트 단위 동일, 최대 차이 0 |

두 짧은 greedy 생성도 토큰 단위로 동일했다. 간단한 RL-Zero 수학 문제에서는 `Answer: 5`와 EOS가 나왔고 기존 Gemma rank의 수학 채점기로 1점을 확인했다. 이 검사에서는 랜덤 초기화나 BOS 중복, text weight 매핑 오류가 재현되지 않았다. 원격 실행의 모델 파일까지 확인한 것은 아니다.

## 실제 생성 실패

MATH seed 5 candidate `math_train-6b07de8f60714bbd`의 원래 프롬프트를 그대로 사용했다. 정답은 6이다. 두 비교 모두 greedy, `max_new_tokens=64`, BOS=2/EOS=1/PAD=0이며 특수 토큰 억제만 다르다.

| 생성 설정 | 실제 출력 토큰 | 보이는 응답 | 기존 채점 |
|---|---|---|---|
| 현재 실험처럼 `suppress_tokens=None` | 줄바꿈 1개 + `<image|>` 63개 | 줄바꿈뿐 | 0 |
| 공식 억제 `[258883,258882]` 복원 | 줄바꿈, `Answer`, `:`, EOS | `Answer:` | 0 |

공식 설정의 `<audio|>`/`<image|>` 억제가 `GenerationConfig(...)` 재생성 과정에서 제거된다. `<image|>` 반복은 `skip_special_tokens=True` decode에서 사라지므로 저장 응답에는 빈 문자열처럼 나타난다. 이 사례는 실제 생성 오류이며 전체 400문제 × 8응답의 성공률 측정은 아니다. 원래 실험의 temperature=1 샘플링과 이 greedy 재현도 구분해야 한다.

복원 후에도 이 문제는 정답 없이 끝났다. 동일 문제를 질문·답 완성 형식과 두 개의 일반 수학 예시를 붙인 형식으로도 검사했으나 정답이 아닌 4를 생성했다. 입력 형식만 바꾸면 해결된다거나 PT의 정상 init이 몇 %라고 주장할 근거가 없다. 현재 체크포인트는 PT이며 Instruct 체크포인트로 교체한 결과도 아니다.

## 기존 실험 처리

현재 Gemma 결과를 정상 검증 완료로 취급하지 않는다. 원격 초기 응답에서 빈 응답 비율과 실제 보상 분포를 확인하고, 수정된 text 생성 정책으로 같은 cohort의 init을 별도 측정해야 한다. 생성 분포를 바꾸는 경우 학습의 log-probability 계산도 같은 출력 공간을 사용해야 한다. 억제만 바꾼 캐시를 기존 prefix/checkpoint와 섞어 같은 실험으로 보고하면 안 된다.

기존 Gemma/공통 engine의 고정 Python 파일 44개, 원래 입력·cache·보상·checkpoint·원고는 변경하지 않았다. 공개 변경은 별도 진단 코드, 재채점 진단과 이 보고서다. 수정된 학습 runner가 준비됐다고 보고하는 문서가 아니다.

## 근거와 재현

- [컴팩트 실측 근거](evidence/gemma4-initial-policy-reproduction-2026-10-21.json): 실제 토큰 ID, 비교 결과, 모델 해시, 채점 결과. 약 4.4 KB.
- 전체 응답/전체 logits는 로컬 `/home/kms/reports/gemma4-12b-initial-live-2026-10-21/`에 보존했다.
- `srgc_research.dispatch.gemma_pretrained_probe`는 별도 출력 디렉터리에만 기록한다. snapshot에는 공식 revision으로 다운로드하며 검증한 `download-provenance.json`이 필요하다. 모델을 다운로드하거나 학습하지 않는다.

```sh
python -m srgc_research.dispatch.gemma_pretrained_probe \
  --snapshot /path/to/verified/gemma-4-12B \
  --output-dir /path/to/new/diagnostic \
  --mode adapter --gpu-offload --tokens 64 \
  --case math_train-6b07de8f60714bbd --generate-real
# 다른 새 출력 경로와 --official-suppression을 지정해 억제 복원 비교
```

MATH 재채점은 cache 단계에도 설치되는 `scripts.srgc_verifier_fallback.tolerant_math_reward`를 사용했다. 진단 회귀 검사 10개가 통과했다. 기존 수치와 기록을 덮어쓰지 않았다.
