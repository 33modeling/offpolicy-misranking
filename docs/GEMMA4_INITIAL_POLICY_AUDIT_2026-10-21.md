# Gemma 4 학습 전 성공률 코드 점검 — 2026-10-21

## 확인 범위와 결론

이후 실제 Gemma 4 12B 가중치의 GPU 검사에서 이미지 종료 토큰 반복을 재현했다. 아래는 그 전의 토크나이저 검사 기록이다. 실제 가중치 검사와 확인된 오류는 [실물 재현 보고서](GEMMA4_PRETRAINED_REPRODUCTION_2026-10-21.md)를 먼저 참조한다.

사용자가 보고한 학습 전 1%대 성공률의 원인은 아직 확정하지 않았다. 이 PC에는 해당 원격 Gemma 실행의 초기 응답 receipt가 없다. 낮은 값을 PT 모델의 정상 성능이나 구현 오류로 단정하지 않는다. 다음 검사는 공식 파일과 로컬 소스의 정합성을 확인한 것이며, 실제 12B 가중치/H100 생성 검증을 대신하지 않는다.

- 고정 모델: `google/gemma-4-12B`, revision `023679ed352de9bb66cc873c9009ce3482585c08`.
- 공식 `tokenizer.json` 32,170,070 bytes, SHA-256 `12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6`. config·generation_config·tokenizer_config도 기존 고정 해시와 대조했다. 가중치는 다운로드하지 않았다.
- 별도 검사 환경에 고정 `transformers==5.19.0`, `peft==0.21.0`, `tokenizers==0.23.3`을 설치했다. 기존 venv와 GPU 작업은 변경하지 않았다. CPU Torch는 `2.13.0`이다.
- 로컬 OLMo 원본 10개 입력 bundle의 8,000개 prompt record, 서로 다른 1,600개 prompt에서 실제 공식 토크나이저를 사용했다. wrapper와 원본 Rust tokenizer의 토큰 열이 같고 BOS는 한 번만 붙었다. prompt에 EOS/PAD가 끼는 사례가 없었다.
- 서로 다른 prompt·정답 텍스트 2,863개의 decode 결과도 HF wrapper와 원본 Rust tokenizer가 같았다. 숫자 정답과 Python 코드의 decode, BOS/EOS/PAD 제거도 확인했다.
- BOS=2, EOS=1, PAD=0, vocabulary=262,144. 이 입력에서 최장 prompt는 3,786 tokens다. wrapper가 prompt를 자르는 설정은 없다.
- 실제 라이브러리의 작은 Gemma unified checkpoint로 full/text loader logits, 누락 가중치 거부, FP32/BF16 log-probability/gradient, 생성·OOM 재시도·재개 및 운영 경로 검사를 수행했다. 74개 검사가 통과했다. 이는 임의 초기화된 작은 모델 검사이며 실제 12B 정답률의 검증이 아니다.
- 성공률 집계는 400문제 × 8응답에 대한 `100 * sum(rewards) / 3200`이다. 32개 정답이면 1%, 48개면 1.5%로 표시된다. 퍼센트 이중 환산은 재현되지 않았다.

## 확인한 생성 설정 차이

공식 고정 generation_config는 temperature=1.0, top_k=64, top_p=0.95이고 audio/image 종료 토큰 258883·258882를 억제한다. 현재 실험 adapter는 기존 GRPO 비교 조건에 맞춰 top_k=0, top_p=1.0으로 바꾸고 suppress_tokens를 상속하지 않는다. Gemma PT에는 chat template 대신 BOS 한 번과 원래 과제 지시문을 쓴다.

이 차이는 확인된 설정 차이다. 낮은 성공률의 원인으로 확정한 것이 아니다. 원래 실험의 sampling·prompt·보상 기준을 임의로 바꾸거나 더 잘 나온 값을 기존 결과로 대체하지 않았다. 공식 기본값과 비교 생성이 필요하다면 별도 진단 실험과 별도 결과로 남겨야 한다.

공식 원본:
- https://huggingface.co/google/gemma-4-12B/blob/023679ed352de9bb66cc873c9009ce3482585c08/tokenizer_config.json
- https://huggingface.co/google/gemma-4-12B/blob/023679ed352de9bb66cc873c9009ce3482585c08/generation_config.json
- https://huggingface.co/google/gemma-4-12B/blob/023679ed352de9bb66cc873c9009ce3482585c08/config.json

## 원격의 저장 초기 응답 검사

기존 실행 노드의 같은 checkout에서 아래를 실행한다. GPU 생성·학습·재시작 없이 saved bundle과 `<bundle>.cache/*.json`의 초기 응답을 읽는다. 실제 로컬 tokenizer/model directory를 발견하면 metadata 파일의 고정 해시와 BOS/EOS/PAD, 원래 generation_config도 출력한다. tokenizer/model을 다운로드하거나 weights를 로드하지 않는다.

```sh
git pull --ff-only origin master
sh scripts/run_srgc_gemma4.sh all audit
```

MATH/MBPP·seed별 bundle의 초기 성공률과 남아 있는 raw receipt의 성공률을 따로 표시한다. raw receipt의 보존 범위, 빈 출력, Answer 표기, 다른 문제로 이어 쓰는 표기, Python fence·문법·함수 존재 여부와 원시 응답 예시를 터미널에 출력한다. 이런 표기 자체를 오답이나 원인으로 간주하지 않는다. bundle 집계와 raw receipt reward가 다르면 오류로 표시한다. raw receipt가 없으면 분석 불가로 남기며 0으로 채우지 않는다.

```sh
# 현재 CPU 채점기로 각 seed의 저장 응답 3개만 재판정. 새 생성이 아니다.
sh scripts/run_srgc_gemma4.sh math audit --rescore 3
# 필요할 때만 JSON도 별도로 저장한다. 터미널 표시는 유지한다.
sh scripts/run_srgc_gemma4.sh all audit --output /path/outside-experiment/gemma-initial-audit.json
```

MATH 재채점은 Gemma `rank.py`가 cache 단계에서도 설치하는 `tolerant_math_reward`를 사용한다. 초기 문서의 “fallback은 training에만 적용된다”는 설명은 잘못이어서 바로잡았다. 현재 환경에서 재채점 값이 달라지면 recorded reward를 수정하지 않고 차이/오류를 보고한다. 라이브러리 환경 차이도 따로 확인해야 한다. 초기 응답 텍스트는 이미 skip_special_tokens로 decode됐으므로 EOS·오디오/이미지 특수 토큰의 실제 생성 여부와 토큰 제한 도달 여부는 텍스트만으로 복원할 수 없다.

MBPP 원본 프롬프트는 언어명 `python`을 여는 fence의 다음 줄에 쓰는 형식도 지시한다. 그 형태를 문자 그대로 따른 정상 함수는 기존 verifier에서 `python` 식을 실행하다 실패할 수 있다. 단순 정답 함수로 기존 점수 0, 언어명 줄만 제거한 진단 점수 1을 재현했다. 실제 Gemma 초기 응답에 이 형태가 얼마나 있는지는 아직 확인하지 못했다. audit은 `literal_python_language_line` 개수를 세고, 명시적인 `--rescore`에서 이 형태의 별도 진단 점수를 표시한다. 원래 보상이나 verifier는 변경하지 않는다.

학습 engine과 `dispatch/gemma4/` 안의 44개 고정 Python 파일, 원래 plan·입력·cache·checkpoint·결과는 변경하지 않았다. 새 audit 코드는 고정 adapter의 digest 대상 밖에 두었다. 원고와 교정 사이트는 변경하지 않았다.

## 회귀 검증

고정 라이브러리 환경에서 Gemma 모델·운영·status·모델 연결 회귀 164개가 통과했다. 이후 추가한 초기 cache 진단의 읽기 전용·부분 cache·재채점 오류 검사 9개도 통과했다(앞선 검사와 일부 중복). 감사 명령의 shell 실행은 Torch/Transformers/PEFT import를 강제로 실패시키는 대역에서도 성공했다. 원래 파일의 바이트와 기존 결과는 보존됐다.
