# Qwen·Llama·Gemma IT: 모델 차이와 최근 변경 영향

소스 기준: `dff44713e40c39ffc079a5e302997c7eca058ef1`. 최근 변경 비교: `72a9a33128c5d095c45fc11f8adf6180da732e9e`.

현재 SRGC 실험 저장소의 Qwen3.5-9B·Llama3.1-8B-Instruct·Gemma4-12B-IT 구현 비교. 전체 모델 계열이나 외부 최신 모델 순위 분석이 아니다.

- 최근 감사 기준 72a9a331→dff44713에서 Qwen12·Llama23·Gemma IT44개 adapter manifest와 frozen engine30파일은 바뀌지 않았다. 모델별 adapter digest가 모두 일치한다.
- 공통 배정·재시도·상태·GPU 시간 reader는 변경됐다. Gemma IT 조기등록·준비 잠금 fast path·MBPP spare-worker 배정은 모델 전용이다. 따라서 Qwen/Llama도 전부 바뀌었다거나 전혀 영향이 없다는 설명 모두 부정확하다.
- CPU fixture에서 세 모델의 학습 기본값·문제 split·source-data 해시는 같았다. prompt/EOS/LoRA target/MBPP fence 추출/runtime은 다르다. 실제 원격 plan·생성 응답·시간·성능은 별도 검증 대상이다.

## 비교 항목

### 모델과 가중치 revision

- Qwen: Qwen/Qwen3.5-9B · posttrained · c202236235762e1c871ad0ccb60c8ee5ba337b9a
- Llama: meta-llama/Llama-3.1-8B-Instruct · posttrained · 0e9e39f249a16976918f6564b8830bc894c89659
- Gemma IT: google/gemma-4-12B-it · instruction_tuned · 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

각 모델 내부의 Random/SR/On-policy/Switch 비교가 주 분석이다. 크기·사전학습·instruction tuning이 달라 절대 성능 차이를 배정 코드 효과로 해석할 수 없다. 모델명과 revision은 코드 상수이며 원격 가중치 파일은 이번에 재검증하지 않았다.

근거: [scripts/srgc_qwen35.py:20](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/scripts/srgc_qwen35.py#L20), [srgc_research/dispatch/llama31/adapter.py:19](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/llama31/adapter.py#L19), [srgc_research/dispatch/gemma4_it/adapter.py:29](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4_it/adapter.py#L29)

### 실제 SRGC 기본 plan과 일반 모델 설정의 구분

- Qwen: SRGC source plan을 복사하고 Qwen 모델·revision·memory/prompt 설정을 덧붙인다.
- Llama: 동일 source plan에서 Llama 전용 모델·프롬프트·memory 설정을 덧붙인다.
- Gemma IT: 동일 source plan에서 IT protocol·sampling·verifier 설정을 덧붙인다.

실제 prepare를 CPU fixture로 실행한 세 모델 모두 seeds 5–9, GRPO, 4 ranks, prefix 25, total 275, 네 arm이다. configs/qwen35_9b_grpo.json의 일반 experiment 블록(seeds 0–4 등)을 이 실험의 실제 설정으로 옮기면 틀린다. 원격 저장 plan의 값은 별도 대조가 필요하다.

근거: [scripts/srgc_qwen35.py:179](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/scripts/srgc_qwen35.py#L179), [srgc_research/dispatch/llama31/adapter.py:162](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/llama31/adapter.py#L162), [srgc_research/dispatch/gemma4_it/adapter.py:181](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4_it/adapter.py#L181)

### 같은 문제를 쓰는지

- Qwen: MATH·MBPP 각각 candidate 400 / ranking validation 50 / held-out evaluation 300.
- Llama: 동일한 candidate·ranking validation·evaluation ID와 source-data 해시.
- Gemma IT: 동일한 candidate·ranking validation·evaluation ID와 source-data 해시.

실제 prepare 코드의 합성 tokenizer/identity fixture에서 ID 및 원문 출처 해시 일치를 확인했다. 프롬프트 문자열은 모델별 template을 거치므로 다르다. 실제 원격 입력 파일이 이 fixture와 동일하다는 주장은 하지 않는다.

근거: [srgc_research/tests/test_model_results.py:42](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/tests/test_model_results.py#L42)

### 선택·업데이트·평가 간격

- Qwen: 후보 집합별 scoring 40, training prompts 4, responses 8; selection/check 25 updates.
- Llama: 동일. projection 4096, projection seed 0, first check 25.
- Gemma IT: 동일. held-out 300문제 × 응답 8개, 최종 step 275.

세 모델의 준비된 plan은 같은 선택 프로토콜이다. training-batch reward와 고정 held-out reward를 분리하고 25 update 간격 기록을 유지해야 한다. 추가 런타임 재시도는 새로운 과학적 seed/arm이 아니다.

근거: [srgc_rebuttal/experiments/additional_seeds.json:31](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_rebuttal/experiments/additional_seeds.json#L31), [srgc_rebuttal/experiments/mbpp_seeds.json:23](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_rebuttal/experiments/mbpp_seeds.json#L23)

### Chat template과 thinking 처리

- Qwen: user 한 메시지; add_generation_prompt=True, enable_thinking=False.
- Llama: user 한 메시지; add_generation_prompt=True. enable_thinking 인자를 전달하지 않는다.
- Gemma IT: enable_thinking=False. IT protocol의 completion cue에는 literal thought channel 구분자가 포함된다.

Gemma의 off 설정과 thought 구분자가 함께 있다는 소스 사실만 확인했다. 실제 reasoning 활성화 또는 고장으로 단정할 수 없다. 같은 문제의 raw prompt/token ID/생성 reasoning 길이를 모델별로 수집해야 의미와 비용 차이를 판별할 수 있다.

근거: [scripts/srgc_qwen35.py:167](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/scripts/srgc_qwen35.py#L167), [srgc_research/dispatch/llama31/adapter.py:145](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/llama31/adapter.py#L145), [srgc_research/dispatch/gemma4_it/protocol.py:12](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4_it/protocol.py#L12)

### BOS 중복과 tokenizer 경계

- Qwen: tokenizer chat template로 완성한 prompt를 사용한다.
- Llama: PromptTokenizer가 add_special_tokens=False를 강제해 template에 이미 들어간 BOS 중복을 막는다. BOS 128000, EOS metadata 128009를 검증한다.
- Gemma IT: BOS 2 / PAD 0 / prefix <bos><|turn>user. IT tokenizer/protocol 경계를 검증한다.

글자 수가 같아도 모델별 token 수와 경계는 다르다. 동일 max_new_tokens를 동일 답변 길이로 취급하지 말고 prompt/generated token 수를 함께 보고해야 한다.

근거: [srgc_research/dispatch/llama31/model.py:251](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/llama31/model.py#L251), [srgc_research/dispatch/gemma4_it/protocol.py:16](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4_it/protocol.py#L16)

### 종료·패딩 토큰

- Qwen: tokenizer EOS를 사용하고 PAD가 없으면 EOS로 보완한다.
- Llama: generation_config의 EOS(복수 가능)를 우선 보존하고 없으면 tokenizer EOS를 쓴다. PAD가 없으면 EOS.
- Gemma IT: EOS [1,106,50], PAD 0을 protocol sampling에 고정한다.

조기 종료·최대 길이 도달 빈도·마지막 token을 비교해야 한다. 다른 모델의 EOS/PAD를 그대로 복사하면 rollout 및 logprob 구간이 달라질 수 있다.

근거: [srgc_research/dispatch/llama31/model.py:67](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/llama31/model.py#L67), [srgc_research/dispatch/gemma4_it/protocol.py:22](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4_it/protocol.py#L22)

### 온도·응답 수·길이

- Qwen: 공통 backend: do_sample=True, temperature=1, top_p=1, top_k=0. 응답 8, 최대 2048 tokens.
- Llama: 동일 공통 값. 모델 generation_config와 EOS는 Llama adapter가 보정한다.
- Gemma IT: 동일 기본 샘플링 값에 Gemma 전용 EOS/PAD·억제 토큰을 명시한다.

세 모델의 표면 설정은 같지만 tokenizer·종료 규칙·분포가 달라 출력 길이와 스타일은 실측 대상이다. 실제 GPU rollout은 이번 CPU 분석에서 생성하지 않았다.

근거: [srgc_rebuttal/torch_backend.py:87](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_rebuttal/torch_backend.py#L87), [srgc_research/dispatch/gemma4_it/protocol.py:20](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4_it/protocol.py#L20)

### 멀티모달 토큰 억제와 logprob support

- Qwen: Gemma IT 전용 suppress-token mask를 설치하지 않는다.
- Llama: Gemma IT 전용 suppress-token mask를 설치하지 않는다.
- Gemma IT: 텍스트 실험에서 audio/image token 258883·258882를 억제한다. scoring에서도 native softcap 뒤 mask_logits를 적용한다.

정책 비율을 비교할 때 sampling과 scoring이 같은 support를 써야 한다. Gemma 전용 제한은 Qwen/Llama와 다른 조건이며 그대로 비교 표에 남긴다. 이것만으로 성능 개선/저하를 확정하지 않는다.

근거: [srgc_research/dispatch/gemma4_it/protocol.py:23](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4_it/protocol.py#L23), [srgc_research/dispatch/gemma4_it/memory.py:10](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4_it/memory.py#L10)

### LoRA 적용 위치

- Qwen: full_attention 층은 q_proj/v_proj, hybrid 층은 in_proj_qkv/z/b/a를 실제 layer_types에 맞춰 적용한다.
- Llama: 각 transformer 층의 q_proj/v_proj만 적용한다.
- Gemma IT: value projection이 있으면 q/v, value가 공유되는 층은 q/k. 선언 TARGETS=q/v/k를 모든 층에 일괄 적용하는 방식이 아니다.

동일 r=16이어도 trainable parameter 수와 조정되는 계산 경로가 다르다. 모델 비교에는 실제 trainable parameter 수·층별 target 목록을 함께 남겨야 한다. 원격 모델 구조의 실측 층 수는 이번에 확인하지 않았다.

근거: [scripts/srgc_qwen35.py:139](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/scripts/srgc_qwen35.py#L139), [srgc_research/dispatch/llama31/adapter.py:109](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/llama31/adapter.py#L109), [srgc_research/dispatch/gemma4_it/adapter.py:123](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4_it/adapter.py#L123)

### 공통 optimizer와 LoRA 수치

- Qwen: LoRA r16/alpha32/dropout0/bias none; AdamW lr1e-5, betas(.9,.999), eps1e-8, weight decay0.
- Llama: 동일 frozen engine 값.
- Gemma IT: 동일 frozen engine 값.

최근 운영 수정은 이 수치를 바꾸지 않았다. 모델별 layer target과 logits 처리 차이는 위 항목에 별도로 남긴다.

근거: [srgc_rebuttal/run_experiment.py:99](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_rebuttal/run_experiment.py#L99), [srgc_rebuttal/torch_backend.py:56](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_rebuttal/torch_backend.py#L56)

### 메모리와 scoring 경로

- Qwen: hybrid attention을 지원하는 Qwen memory backend.
- Llama: Llama용 backend subclass와 prompt tokenizer wrapper.
- Gemma IT: IT native softcap·억제 token mask를 반영한 scoring backend.

현재 prepare 기본값은 generation micro-batch2 / logprob micro-batch1 / logit chunk64, nonreentrant gradient checkpointing, CPU checkpoint snapshot, eager attention으로 같다. 모델 크기·구조별 VRAM/속도는 달라질 수 있으나 원격 측정값이 없다.

근거: [scripts/srgc_qwen35_memory.py:20](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/scripts/srgc_qwen35_memory.py#L20), [srgc_research/dispatch/llama31/memory.py:20](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/llama31/memory.py#L20), [srgc_research/dispatch/gemma4_it/memory.py:15](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4_it/memory.py#L15)

### MATH reward

- Qwen: srgc_rebuttal.run_experiment:math_reward.
- Llama: 동일 verifier 경로.
- Gemma IT: 동일 verifier 경로.

같은 reward 함수라도 답 추출에 영향을 주는 출력 형식·종료·truncation이 다를 수 있다. 문제별 raw response와 판정 근거를 함께 비교해야 한다.

근거: [srgc_rebuttal/run_experiment.py:22](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_rebuttal/run_experiment.py#L22)

### MBPP code fence 추출 차이

- Qwen: srgc_rebuttal.verifiers:code_reward.
- Llama: 동일 common code_reward.
- Gemma IT: IT verifier가 py/python/python3·fence 내부 언어 줄을 정규화한 뒤 같은 parent-side code_reward에 위임한다. protocol version=parent-checked-values-v3-fence-language-line-v1.

세 모델 모두 현재 parent-side assertion 검사 경로를 공유한다. Gemma만 안전 검사를 한다는 해석은 틀리다. 동일 raw 응답에서 fence 형식만으로 판정이 달라질 여지가 있으므로 모델 간 절대 MBPP 점수 비교 전 공통 정규화 verifier로 재채점하고 버전을 적어야 한다.

근거: [srgc_rebuttal/verifiers.py:51](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_rebuttal/verifiers.py#L51), [srgc_research/dispatch/gemma4_it/verifier.py:20](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4_it/verifier.py#L20), [srgc_research/dispatch/gemma4_it/protocol.py:9](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4_it/protocol.py#L9)

### 초기 reward cache와 provenance

- Qwen: 각 모델·seed의 초기 policy로 candidate reward cache를 만든다.
- Llama: Qwen·OLMo reward를 재사용하지 않고 Llama cache를 만든다.
- Gemma IT: 독립 IT root/cache와 sampling·code_verifier_version provenance를 사용한다.

현재 prepare를 실행한 fixture의 cached_rewards는 세 모델 모두 비어 있었다. source question ID가 같다는 이유로 다른 모델의 생성 응답·reward cache를 공유하면 selection 비교가 깨진다. 실제 저장 cache는 이번에 열람하지 못했다.

근거: [scripts/srgc_qwen35.py:159](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/scripts/srgc_qwen35.py#L159), [srgc_research/dispatch/gemma4_it/adapter.py:393](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4_it/adapter.py#L393)

### 모델·입력·adapter identity 보존

- Qwen: adapter manifest12개와 frozen engine 해시를 plan·run provenance에 연결한다.
- Llama: adapter manifest23개. 저장 model snapshot identity와 tokenizer metadata를 검증한다.
- Gemma IT: adapter manifest44개(IT+공유 legacy Gemma+shared files). PT/text/IT root를 분리한다.

72a9a331→dff44713 범위에서 세 adapter digest와 frozen engine30파일이 그대로였다. 운영 파일 일부는 adapter manifest 밖에 있으므로 이 결과가 전체 실행 코드 불변을 뜻하지는 않는다. 자세한 파일 목록·해시는 분석 원본 JSON에 남긴다.

근거: [scripts/srgc_qwen35.py:30](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/scripts/srgc_qwen35.py#L30), [srgc_research/dispatch/llama31/adapter.py:37](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/llama31/adapter.py#L37), [srgc_research/dispatch/gemma4_it/adapter.py:48](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4_it/adapter.py#L48)

### 패키지 준비와 격리

- Qwen: transformers5.14.1 / peft0.20.0 / fla-core0.5.2 등 Qwen runtime 요구조건을 검사한다.
- Llama: transformers>=4.57.0 및 기존 Torch/PEFT를 검사한다. Gemma용 pinned overlay를 자동 설치하는 경로는 아니다.
- Gemma IT: shared atomic package overlay: transformers5.19.0 / peft0.21.0 / accelerate1.10.1 / tokenizers0.23.3 등. 기존 CUDA Torch>=2.5를 보존한다.

환경이 같다고 가정할 수 없다. Qwen의 hybrid kernel과 Gemma package overlay는 서로 다른 cold-start 비용을 만든다. 원격 installed versions와 실제 import 경로가 필요하다.

근거: [scripts/srgc_qwen35.py:53](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/scripts/srgc_qwen35.py#L53), [srgc_research/dispatch/llama31/adapter.py:68](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/llama31/adapter.py#L68), [srgc_research/dispatch/gemma4/runtime.py:17](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4/runtime.py#L17)

### 가중치 검증 marker의 공유 범위

- Qwen: verified-models/<hostname>/<pathhash>.json: hostname별 marker.
- Llama: verified-models/<pathhash>.json: hostname 구분 없는 공유 marker.
- Gemma IT: verified-models/<pathhash>.json: hostname 구분 없는 공유 marker.

세 경로 모두 inode/size/mtime/ctime 변화 시 재검증할 수 있다. Llama도 Gemma처럼 marker를 공유한다. 노드별 inode 등 fingerprint가 다르면 공유 marker가 번갈아 무효화되고 재해시 비용이 생길 가능성이 있다. 실제 노드 간 fingerprint가 달랐다는 관측은 없으므로 현재 지연의 확정 원인으로 쓰지 않는다.

근거: [scripts/srgc_qwen35.py:96](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/scripts/srgc_qwen35.py#L96), [srgc_research/dispatch/llama31/model.py:188](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/llama31/model.py#L188), [srgc_research/dispatch/gemma4_it/model.py:152](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4_it/model.py#L152)

### 컴파일·kernel cache

- Qwen: rank runtime이 노드·rank별 private compiler cache를 구성하고 local/world rank4개 조건을 검사한다.
- Llama: Qwen 전용 FLA/hybrid rank cache 보정을 사용하지 않는다.
- Gemma IT: Gemma package overlay 및 rank runtime은 Qwen FLA 경로와 다르다.

새 노드 첫 import/compile과 반복 rollout 시간을 분리해야 한다. 모델마다 cache warm 여부가 다르면 총 wall time 비교에 cold-start 편향이 생긴다.

근거: [scripts/srgc_qwen35_rank_runtime.py:15](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/scripts/srgc_qwen35_rank_runtime.py#L15)

### 완성 plan을 재사용할 때의 준비 잠금

- Qwen: 원래 prepare_missing이 .start-prepare.lock을 먼저 잡고 saved plan·runtime·모델을 검사한다.
- Llama: 원래 prepare_missing이 준비 lock 안에서 snapshot/runtime/저장 plan을 검사한다.
- Gemma IT: d7730f1 운영 wrapper는 완성 plan identity를 준비 write lock 밖에서 검사한다. missing plan은 peer 소유 시 Busy75. frozen adapter는 그대로다.

Gemma의 ready-plan fast path는 Qwen/Llama에 자동 적용되지 않는다. Qwen/Llama가 비슷한 lock 대기를 하는지는 각각의 PID/lock owner/wchan과 준비 로그로 확인해야 한다. 원래 보호 잠금을 무조건 지우면 동시 준비가 충돌한다.

근거: [scripts/srgc_qwen35_start.py:113](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/scripts/srgc_qwen35_start.py#L113), [srgc_research/dispatch/llama31/cli.py:40](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/llama31/cli.py#L40), [srgc_research/dispatch/gemma_it_bootstrap.py:16](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma_it_bootstrap.py#L16)

### launcher와 dataset worker 등록 시점

- Qwen: 현재 startup 조기등록 wrapper 대상이 아니다. generic worker의 등록 hook만으로 Gemma startup record가 생기지 않는다.
- Llama: 동일. 조기 준비단계 heartbeat는 이 변경으로 추가되지 않는다.
- Gemma IT: d7730f1부터 heavy prepare 전에 PREPARING startup record와 단계 heartbeat를 남긴다. 실제 queue preflight가 dataset worker 등록을 마친다.

model launcher active; dataset worker not registered는 준비 전/중이라는 상태 구분이지 prefix 학습 중이라는 뜻이 아니다. 46dda5a/d7730f1은 관측성과 Gemma 준비 대기를 보완했다. 실제 원격 등록 성공은 이번 분석에서 확인하지 않았다.

근거: [srgc_research/dispatch/startup.py:25](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/startup.py#L25), [srgc_research/dispatch/gemma_it_run.py:288](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma_it_run.py#L288)

### cache/train 이전 real-response admission

- Qwen: Qwen 전용 admission/provenance 검사. Gemma IT fixed-response panel과 동일한 경로는 아니다.
- Llama: Llama 전용 startup/model 검증. Gemma IT panel은 사용하지 않는다.
- Gemma IT: fixed real-response baseline panel을 cache/full training 전에 검사한다. 실패 panel은 원인 변화 없는 재시도만으로 통과 처리하지 않는다.

Gemma는 학습 시작 전 추가 GPU 작업이 있다. panel 실행·실패·복구 시간을 prefix/arm 학습 시간과 섞지 말아야 한다. 4c6d01a는 누락·실패 panel 복구 경로를 보완했으며 점수를 바꾼 증거는 아니다.

근거: [scripts/srgc_qwen35_admission.py:18](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/scripts/srgc_qwen35_admission.py#L18), [srgc_research/dispatch/gemma4_it/preflight.py:20](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma4_it/preflight.py#L20), [srgc_research/dispatch/gemma_it_run.py:21](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma_it_run.py#L21)

### MATH가 남아 있을 때 MBPP 배정

- Qwen: ready_dispatch의 기본값: late MBPP arms 조기 허용 없음.
- Llama: 동일. 기본 MATH barrier를 유지한다.
- Gemma IT: d375b93/4c6d01a에서 idle worker가 먼저 MATH claim을 시도한 뒤 spare worker만 준비된 MBPP arm을 받을 수 있게 했다. MBPP 자기 prefix/panel/task lease는 필요하다.

최근 MBPP 조기 배정은 Gemma IT opt-in이다. Qwen/Llama도 동일하게 바뀌었다고 보고하면 틀리다. 노드가40개라는 사실만으로40개 arm이 동시에 runnable인 것은 아니다. task 수·선행조건·소유 lease를 봐야 한다.

근거: [srgc_research/dispatch/ready_dispatch.py:28](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/ready_dispatch.py#L28), [srgc_research/dispatch/gemma_it_run.py:106](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma_it_run.py#L106)

### 요청 task도 admission guard 적용

- Qwen: 공통 resume_drain의 task_allowed 검사를 받는다.
- Llama: 동일.
- Gemma IT: 동일. Gemma 전용 panel/queue gate와 함께 적용된다.

b2b7c46의 requested queue 경로 보완은 세 모델 공통이다. 사용자가 MBPP task를 지정했다고 선행조건을 건너뛰지 않는다. 이는 learner 수치 변경이 아니라 실행 허용 조건 변경이다.

근거: [srgc_research/dispatch/resume_drain.py:105](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/resume_drain.py#L105), [srgc_research/dispatch/resume_drain.py:50](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/resume_drain.py#L50)

### 선택 scoring 중단 후 재개와 collective timeout

- Qwen: selection_resume.MODELS whitelist 대상 아님.
- Llama: selection_resume.MODELS whitelist 대상 아님.
- Gemma IT: Gemma family만 적용: 완료 scoring block을 policy fingerprint별로 재사용하고 selection collective timeout을6시간으로 늘린다.

rank_runtime의 외부 MODELS 목록이 넓어도 내부 selection_resume.MODELS는 gemma4/gemma4_text/gemma4_it뿐이다. 9858dbc/cd0cb20 효과를 Qwen/Llama에 일반화할 수 없다. 동일 operation 재시도와 실제 재계산 비용은 별도 집계한다.

근거: [srgc_research/dispatch/rank_runtime.py:80](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/rank_runtime.py#L80), [srgc_research/dispatch/selection_resume.py:37](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/selection_resume.py#L37), [srgc_research/dispatch/selection_resume.py:35](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/selection_resume.py#L35)

### model-chain lease·barrier·재시도

- Qwen: 노드/할당GPU별 chain lease, 같은 모델의 MATH+MBPP 완료 barrier를 사용한다.
- Llama: 동일. chain 안에서는 manual GPU cleanup을 건너뛴다.
- Gemma IT: 동일 chain barrier. 모델 내부 Gemma MBPP spare-worker 허용은 다음 모델로 넘어가는 조건과 다르다.

child exit75(Busy)/76(Yield)는 재시도,129/137은 중단 복구,130/143은 정지 경로다. 이미 실행 중인 Python 프로세스가 git pull만으로 새 코드로 바뀌지는 않는다. 같은 checkout에서 새로 실행한 child만 새 소스를 읽는다.

근거: [srgc_research/dispatch/model_chain.py:27](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/model_chain.py#L27), [srgc_research/dispatch/llama_run.py:138](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/llama_run.py#L138)

### 잔여 프로세스 정리 범위

- Qwen: 기존 process guard·resume 보호를 사용한다. 이번 Gemma orphan-wrapper 변경 대상은 아니다.
- Llama: direct/manual start는 자기 사용자·할당GPU4개·PID/start time 검증 후 busy 기준에서 cleanup 가능. model-chain child는 이를 생략한다.
- Gemma IT: dff4471에서 rank_runtime wrapper 아래 고아 학습 rank까지 Gemma IT 정리 대상 식별을 보완했다.

프로세스 종료는 실제 실행에 영향을 주는 변경이다. 단순 표시 수정으로 묶지 않는다. 이번 분석에서 정리 명령을 실행하거나 원격 학습을 재시작하지 않았다.

근거: [srgc_research/dispatch/llama_run.py:138](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/llama_run.py#L138), [srgc_research/dispatch/gemma_it_run.py:72](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/gemma_it_run.py#L72)

### 상태 메시지·이전 child 종료·만료 노드

- Qwen: 공통 model_status의 node/child exit/expiry 표시를 받는다.
- Llama: 동일.
- Gemma IT: 동일 + Gemma startup 준비 단계 기록.

7bb0a93/f61bf09은 만료 노드를 현재 대기열처럼 보이는 문제와 이전 child 종료 이유 누락을 줄였다. launcher active·lock wait·prefix·task lease held는 서로 다른 단계다. 상태 문구가 자세해졌다는 사실은 작업이 실제 진행됐다는 증거가 아니다.

근거: [srgc_research/dispatch/model_status.py:33](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/model_status.py#L33), [srgc_research/dispatch/status_current.py:7](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/status_current.py#L7)

### GPU 시간·중복·중단 비용

- Qwen: 공통 time/status/restart-cost reader 변경을 받는다.
- Llama: 동일.
- Gemma IT: 동일. admission panel/selection 재시도 등 추가 단계는 따로 구분한다.

8dfc085 등은 같은 operation ID의 복사본은 중복 제거하되 다른 실제 replay를 보존한다. invocation 전체 시간과 그 안의 phase를 다시 합산하지 않는다. 누락 시간을0으로 간주하지 않으며 실측0과 미측정을 구분한다. 비용 표시 수정이 archived retained-update 과학 결과를 바꾸지는 않는다.

근거: [scripts/srgc_time_status.py:27](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/scripts/srgc_time_status.py#L27), [srgc_research/dispatch/restart_costs.py:13](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/srgc_research/dispatch/restart_costs.py#L13), [scripts/srgc_gpu_accounting.py:23](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/scripts/srgc_gpu_accounting.py#L23)

### checkpoint·raw rollout·중간 reward

- Qwen: 공유 step checkpoint / resumable rollout / direction records를 manifest에 포함한다.
- Llama: 동일 계열 공통 파일과 Llama 전용 resume.
- Gemma IT: 동일 계열 + Gemma selection block 재개.

기존 checkpoint·reward·25-step held-out 기록·실패 로그를 덮어쓰거나 다른 모델 identity로 재라벨링하면 안 된다. 이번 분석은 plan 생성용 임시 CPU fixture만 사용했고 실제 입력·체크포인트·결과를 수정하지 않았다.

근거: [scripts/srgc_step_checkpoints.py:9](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/scripts/srgc_step_checkpoints.py#L9), [scripts/srgc_resumable_rollouts.py:34](https://github.com/33modeling/offpolicy-misranking/blob/dff44713e40c39ffc079a5e302997c7eca058ef1/scripts/srgc_resumable_rollouts.py#L34)

## 최근 변경

- [9858dbc97d94](https://github.com/33modeling/offpolicy-misranking/commit/9858dbc97d944f84080892b7b73375b8a4554731) · Gemma family · 완료된 selection scoring block 재개. Qwen/Llama: whitelist 밖이므로 설치되지 않음. Gemma: 적용.
- [cd0cb20503bc](https://github.com/33modeling/offpolicy-misranking/commit/cd0cb20503bc0d7294b8be3dca138d18c180b262) · Gemma family · selection collective 대기시간·진행상태 보완. Qwen/Llama: selection_resume whitelist 밖. Gemma: 적용.
- [b2b7c46f29ea](https://github.com/33modeling/offpolicy-misranking/commit/b2b7c46f29eab5bebf984ef39f50972e912f73e4) · 공통 + 현황판 · requested queue에도 task_allowed; launch/evidence 이력 보강. Qwen/Llama: 공통 admission·상태 경로 적용. Gemma: 공통 + panel 조건.
- [8dfc08559300](https://github.com/33modeling/offpolicy-misranking/commit/8dfc08559300d83721c70a9dbd342636e69647fc) · 공통 상태·시간 · GPU 비용 중복/결측과 dispatch 단계 구분. Qwen/Llama: 시간·상태 reader 적용. Gemma: 시간·상태 reader 적용.
- [d375b93bd332](https://github.com/33modeling/offpolicy-misranking/commit/d375b93bd3327db3cdb0778b2c8a2bcb6846d2ea) · Gemma IT · idle worker의 준비된 MBPP arm 조기 claim. Qwen/Llama: 기본 barrier 유지. Gemma: opt-in 적용.
- [4c6d01a3df40](https://github.com/33modeling/offpolicy-misranking/commit/4c6d01a3df40f214b773b71f03cae27df0d74526) · Gemma IT · MBPP bootstrap·panel 복구·빈 backlog 처리. Qwen/Llama: 전용 경로 아님. Gemma: 적용.
- [d7730f15a9bc](https://github.com/33modeling/offpolicy-misranking/commit/d7730f15a9bcb1b747924178ac340de6e3bbd150) · Gemma IT · 초기 PREPARING 등록·ready-plan fast path. Qwen/Llama: 전용 wrapper 미적용. Gemma: 적용.
- [46dda5a23a42](https://github.com/33modeling/offpolicy-misranking/commit/46dda5a23a42f1d947af6530f28ff0ea8f6af945) · Gemma IT · startup heartbeat 파일 수 제한. Qwen/Llama: 전용 wrapper 미적용. Gemma: 적용.
- [7bb0a9387e01](https://github.com/33modeling/offpolicy-misranking/commit/7bb0a9387e01f6657af1ab2e3f1828a518a3c18b) · 공통 상태 · 만료 node 표시와 chain 순서 설명. Qwen/Llama: 표시 적용. Gemma: 표시 적용.
- [f61bf099a146](https://github.com/33modeling/offpolicy-misranking/commit/f61bf099a1461640cd1bf89e5742c1a0ef8e818c) · 공통 상태 · node 옆 최근 child 종료 원인 표시. Qwen/Llama: 표시 적용. Gemma: 표시 적용.
- [dff44713e40c](https://github.com/33modeling/offpolicy-misranking/commit/dff44713e40c39ffc079a5e302997c7eca058ef1) · Gemma IT · wrapper 아래 orphan rank 식별/정리 보완. Qwen/Llama: Gemma 전용 정리 대상 아님. Gemma: 적용.

## 확인·가설·미해결

### 코드 확인 · Qwen/Llama에도 적용된 변경

requested task admission, 공통 model-chain 재시도·상태 표시와 GPU 시간 해석 경로. learner/adapter manifest 불변과 운영 경로 변경은 동시에 성립한다.

### 코드 확인 · Gemma에만 적용된 시작·배정 보완

selection block 재개/6시간 collective timeout은 Gemma family, startup 조기등록·ready-plan fast path·spare-worker MBPP arm 및 orphan wrapper 정리는 Gemma IT. Qwen/Llama는 해당 whitelist/opt-in 밖이다.

### 조건부 가설 · 공유 snapshot marker가 반복 재해시를 일으킬 가능성

Llama와 Gemma 모두 hostname 없는 marker에 inode/mtime/ctime fingerprint를 저장한다. 노드별 fingerprint가 실제로 다를 때만 이 가설이 성립한다. 각 노드 marker·stat·lock owner·검증 시작/종료 로그로 확인해야 한다.

### 원격 미확인 · 현재 멈춤·느림의 실제 원인

원격 노드와 /group-volume 런타임 기록에 접근하지 못했다. 현재 active/PENDING 노드 수, prefix 진행, MBPP 등록 성공, GPU 이용률, rollout 속도, 실측 시간은 확인하지 않았다. 로컬 CPU 통과로 정상 학습을 확정하지 않는다.

### 미해결 · 사용자 보고 SyntaxError

정확한 실행 명령·traceback·파일/줄·Python 버전이 아직 없어 재현되지 않았다. 별도 로컬 전수 구문 검사는867 Python·119 shell에서 오류0이었으나 사용자 오류를 해결했다는 뜻은 아니다. 해당 보고는 미해결로 유지한다.

### 추가 측정 · 공정한 속도·보상 비교에 필요한 기록

같은 문제ID·seed·arm·update에서 prompt token, generated token, 종료 원인, reasoning/코드 fence 형식, verifier 버전과 판정, trainable parameter 수, warm/cold start, lock 대기, phase별 wall/device 시간, 실패/재시도 비용, checkpoint·raw rollout provenance를 남겨야 한다. 기존 보존 기록은 삭제하지 않는다.

## 검증

pinned Git source 비교, 실제 adapter prepare 코드의 CPU fixture, default plan/split ID 비교. 합성 tokenizer와 가중치 identity를 사용하므로 실제 model inference 검증이 아니다.

- 분석 원본/실제 prepare/manifest 해시/소스 줄 검증과 기존 evidence/retention: 30 passed.
- Node module 구문 검사 통과. 실제 브라우저1440px·390px에서30개 항목, 전체 펼치기/접기, 검색 후 새로고침, 가로 넘침 없음, 기존 STATUS-034 hash, 선택적 분석 JSON 누락 격리와 page error0을 확인했다.
- 외부 네트워크 요청과 로컬 서버 없이 브라우저 요청을 로컬 파일로 라우팅했다. 실제 서비스 인증·배포 내용 검증이나 원격 GPU 실행 검증이 아니다.
- 합성 model snapshot identity는 실제 가중치 SHA256 증거로 게시하지 않았다. 전체 manifest와 비교 해시는 [model-analysis.json](../research-bug-board/model-analysis.json)에 보존한다.
