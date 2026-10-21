# Gemma PT 응답 종료 규칙과 text-v4 재시작 — 2026-10-22

사용자 보고: Gemma-4-12B PT의 MATH 초기 캐시 보상이 약 14%다. 같은 파이프라인의 Qwen(80%대), Llama-3(30%대), OLMo-3(60%대)는 공개 수치와 대체로 맞는다. Gemma만 비정상이다.

## 원인

PT 체크포인트는 요청한 마지막 줄(`Answer: ...`)을 쓴 뒤에도 EOS를 내지 않고 문서를 이어 쓴다. 새 질문을 스스로 만들고 그 답까지 쓴다. 로컬 실제 12B 응답에서 ` Jerry is 4 years old.\nQuestion: The sum of the ages of`처럼 다음 질문으로 넘어가는 것을 확인했다(`/home/kms/reports/gemma4-12b-initial-live-2026-10-21/gemma4-live-prompt-styles/report.json`). MBPP 응답도 코드 블록 뒤에 다른 질문과 코드를 이어 썼다(`docs/evidence/gemma4-pt-completion-diagnosis-2026-10-21.json`).

채점기는 응답의 마지막 답을 읽는다. 같은 `math_verify` 설정으로 확인했다.

- `"...Answer: 6\n\nQuestion: Tom has 3 apples and buys 4. How many?\nAnswer: 7"` → 7을 추출해 오답(정답 6).
- MBPP verifier도 마지막 코드 블록을 실행한다.

따라서 맞는 풀이도 스스로 지어낸 후속 문제의 답으로 채점된다. 다른 세 모델은 답 뒤에 EOS를 내므로 이 문제가 드러나지 않았다.

## 수정

`srgc_research/dispatch/gemma4_text/stopping.py`: 생성 중 각 응답이 요청한 형식을 마치면 그 응답의 생성을 멈춘다. 토큰을 추가하거나 고쳐 쓰지 않는다.

- MATH: `Answer:`로 시작하는 줄이 줄바꿈으로 끝났을 때, 또는 줄 머리에 새 `Question`/`Problem`/`Q`/`Exercise` 머리글이나 지시문 `Solve the following problem`이 나올 때.
- MBPP: 첫 번째 닫힌 코드 블록.
- 캐시, preflight, GRPO rollout이 모두 같은 `model.generate`를 쓰므로 세 곳에 같은 규칙이 적용된다. 멈춘 응답 뒤에 붙는 batch padding은 GRPO 시퀀스에서 잘라 실제로 샘플된 토큰만 학습한다.
- 샘플링 분포(temperature 1, top-p 1, top-k 0, 공식 modality 억제)와 입력, 문제, seed, verifier는 그대로다.

## 재시작

`EXTENSION`을 `gemma4-12b-text-v4`로 올렸다. 기본 결과 위치가 `$OM_WORK/srgc-rebuttal/gemma4-12b-text-v4`로 바뀌어 입력 준비, preflight, 초기 캐시, prefix, 학습이 처음부터 새로 시작된다. text-v3의 입력, 캐시, checkpoint는 지우지 않고 그대로 둔다.

## 검증

- 새 테스트 `srgc_research/tests/test_gemma_answer_stop.py` 4개: 종료 위치, 코드 블록, padding 제거, 실제 Gemma tokenizer에서 행별 종료와 micro-batch 재사용.
- Qwen3-0.6B CPU 실제 `generate`로 한 행만 멈췄을 때 그 행이 padding으로 채워지고 잘라낸 길이가 맞는지 확인했다.
- 기존 Gemma/후속 측정 테스트 포함 186개 통과(이 변경만 적용한 깨끗한 worktree).
- 원격 H100 캐시는 아직 다시 만들지 않았다. 수정 뒤의 초기 정답률은 미측정이다. 이 규칙이 14%를 얼마나 회복하는지는 v4 캐시로 확인한다. 회복이 부족하면 다음 후보는 공식 생성 설정의 top-k 64·top-p 0.95를 지운 전체 어휘 샘플링이다(현재 프로토콜은 다른 모델과 같게 의도적으로 전체 어휘를 쓴다).
