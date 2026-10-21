# Llama·Qwen 입력·생성·초기 채점 점검 — 2026-10-21

이번 로컬 검사에서는 Gemma PT의 입력 이어 쓰기와 같은 원인을 재현하지 못했다.
다만 두 모델이 공유하는 MBPP 채점기에서 코드 블록 형식에 따른 오채점은 재현했다.
원격 실험의 실제 영향률과 새 정확도 수치는 아직 측정하지 않았다.

## 실제 토크나이저 검사

Transformers **5.14.1**에서 seed 5의 MATH·MBPP 각 800개 입력을 모델별로 검사했다.
800개는 후보 400개·검증 pool 100개·최종 평가 300개다. 모델별 **1,600/1,600개**에서
실험 입력의 토큰열이 native `apply_chat_template(tokenize=True)`와 일치했다.
이는 문제를 풀어 본 정확도 검사가 아니다.

| 모델 | 확인한 입력 처리 | 종료 처리 |
| --- | --- | --- |
| Qwen3.5-9B | thinking off, assistant 시작과 닫힌 think 구간 일치; BOS 추가 없음 | tokenizer EOS 248046, PAD 248044 |
| Llama-3.1-8B-Instruct | 채팅 프롬프트의 BOS 128000이 정확히 한 번; 추가 BOS 억제 | 로더가 원래 EOS 목록 128001·128008·128009를 보존; PAD가 없으면 EOS로 보완 |

Qwen은 실험 revision의 공개 토크나이저 파일과 고정 해시를 대조했다. Llama 원본 저장소는
인증 없는 접근에 401을 반환했다. 공개 NousResearch 사본에서 **원본 고정 해시와 같은
tokenizer.json·config·generation config·special token map**을 구했고,
다르게 직렬화된 tokenizer config는 실제 production의 `validate_tokenizer_metadata`로 검사했다.
앞서 시도한 Unsloth 사본은 토크나이저 해시가 달라 최종 대조에 사용하지 않았다.
원격 노드에 있는 실제 8B·9B 가중치의 동일성이나 전체 추론 성공을 주장하지 않는다.

## MBPP 코드 블록 오채점

공통 프롬프트는 코드 블록의 첫 별도 줄에 `python`을 쓰는 형식을 요청한다.
기존 추출기는 그 줄을 코드에 남겨 실행하므로 정상 함수도 실패할 수 있다.
동일한 합성 함수와 두 assert를 실제 자식 채점기로 검사했다.

| 응답 형식 | 기존 채점 | 형식만 해석한 별도 진단 |
| --- | --- | --- |
| 시작 fence와 같은 줄에 `python` | 1 | 1 |
| 시작 fence 다음 별도 줄에 `python` | 0 | 1 |

Qwen·Llama는 둘 다 `srgc_rebuttal.verifiers:code_reward`를 쓴다.
기존 실험의 원시 응답이 이 형식으로 생성됐는지 확인해야 실제 영향을 알 수 있다.
공통 채점기·기존 보상·진행 중인 학습 조건을 바꾸거나 진단 점수로 원고 수치를 대체하지 않는다.

## 저장 응답의 읽기 전용 점검

[점검 모듈](../srgc_research/dispatch/instruct_initial_audit.py)은 터미널 표와 선택적인 새 JSON을
제공한다. 원시 응답의 모델·문제 hash·sampling seed를 확인하고, 빈 응답·코드 형식과
제한된 재채점을 기록한다. 형식 진단 점수는 기존 보상과 다른 필드다. 원시 응답이 없으면
미확인으로 표시하며 보상만 보고 원인을 추정하지 않는다.

실험 노드에서 필요할 때 아래처럼 실행한다. 모델 추론이나 학습을 시작하지 않는다.

```sh
python -m srgc_research.dispatch.instruct_initial_audit qwen35 all --rescore 16
python -m srgc_research.dispatch.instruct_initial_audit llama31 all --rescore 16
```

기존 모델·캐시·토크나이저·생성·역전파·재개 검사 **180개**, subtest 16개가 통과했다.
새 점검기와 기존 Gemma 점검기 호환 검사 **26개**도 통과했다. 새 점검기는 합성 fixture로
오채점·누락·다른 모델·변조된 원시 기록을 구분하고 원본이 그대로인지 검사한다.
학습 코드와 해시에 포함된 파일 89개는 점검 전과 바이트가 같다.

[검증 요약](evidence/llama-qwen-init-audit-2026-10-21.json)에 자산 출처·해시·측정 범위를 기록했다.
원시 로그와 native 대조 script는 `/home/kms/reports/llama-qwen-token-audit-2026-10-21/`에 있다.
