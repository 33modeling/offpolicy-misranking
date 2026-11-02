# math500/s0 fresh 신뢰도 및 코드 감사 — 2026-09-08

결론: 보상 계산의 실제 버그를 재현했다. 낮은 fresh 값을 단순히 seed 부족으로 해석하면 안 된다. 다만 실제 응답별 오채점 비율과 순위 변화는 원본 산출물이 없어 아직 확정할 수 없다.

## 근거와 검사 범위

- 결과: transfer/offpolicy-misranking/digest-olmo3-1025-7b-base-rlzero-grpo-h100-v2-math500-s0-20260907T155239Z.txt.
- 함께 확인: status-history_0907_1.log, status-history_0908.log. 상태 로그는 기록 당시 상태이며 실시간 접속 결과가 아니다.
- 생성 커밋: `0e4cd412ce2d09be482ee3dbba4fa1a5e969dff4`.
- 분석 checkout: `cef766b7e6ef94d031c1b6ad60e3e7b86fa6e83a`.
- 생성 커밋을 별도 임시 디렉터리에 추출해 현재 코드와 대조했다.
- 생성/seed/EOS/재개, reward, GRPO update, gradient 및 projection, validation/candidate 분할, estimator ratio, shard 병합/계약, top-k/tie, bootstrap/readout 경로를 검토했다.
- 이 로컬 transfer에는 원본 rollout JSONL, scores JSON, protocol JSON, val_groups.pt, oracle_micro_groups.pt가 없다. GPU 모델/클러스터 재실행은 하지 않았다.

## 확인된 보상 오류

### 1. 현재 checkout에도 남아 있던 수식 일부 추출 오류 — 이번에 로컬 수정

`src/data.py::_math_reward`는 수식 구분자가 제거된 문자열을 `math_verify.parse`에 직접 전달했다. 이 경로는 전체 수식 대신 일부 숫자를 추출하거나 수식을 추출하지 못한다. 저장된 vendor wheel의 math-verify 0.9.0으로 재현했다.

| 모델 답 | 정답 | 수정 전 보상 | 수정 후 보상 |
|---|---|---:|---:|
| `2x` | `2y` | 1 | 0 |
| `2x` | `2` | 1 | 0 |
| `\sqrt{4}` | `2` | 0 | 1 |
| `\frac{1}{\sqrt{2}}` | `\frac{\sqrt{2}}{2}` | 0 | 1 |
| `x+x` | `2x` | 0 | 1 |

수정은 수학 구분자를 복원하고 LatexExtractionConfig만 사용해 자유 텍스트 숫자 추출을 차단한다. 정답과 예측 모두 동일하게 처리한다. `tests/test_math_expression_reward.py`의 10개 사례 중 수정 전 5개 실패, 수정 후 모두 통과했다. 생성 커밋에도 동일한 bare parse 경로가 있으므로 해당 실행에도 적용 가능한 결함이다. 실제 응답에서 얼마나 발현했는지는 미확인이다.

### 2. 생성 커밋의 답 추출/정규화 오류 — HEAD에는 이미 수정되어 있었음

생성 커밋의 `data.py`를 직접 로드해 다음을 재현했다.

| 답 / 정답 | 생성 버전 | 이번 수정 전 HEAD |
|---|---:|---:|
| `\boxed{\frac{1}{2}}` / `\frac{1}{2}` | 0 | 1 |
| `Answer: (3, 1)` / `(3,1)` | 0 | 1 |
| `Answer: 1,2` / `12` | 1 | 0 |
| `#### 8` 후 `#### 9` / `9` | 0 | 1 |

중첩 brace 미지원, 구조적 쉼표 제거, 첫 #### 답 선택 때문이다. 정답을 오답으로 만드는 오류와 오답을 정답으로 만드는 오류가 모두 있다. 이 함수는 rollout 평가와 GRPO 학습이 공유하므로 평가 점수뿐 아니라 학습된 정책도 영향을 받을 수 있다.

## fresh 수치의 정확한 의미

| drift | A/B top-40 overlap | 겹친 문제 수 |
|---|---:|---:|
| 0 | .200 | 8/40 |
| 25 | .200 | 8/40 |
| 100 | .125 | 5/40 |
| 400 | .150 | 6/40 |

이 값은 `noise_floor`라는 이름의 A/B 재현성이다. fresh R의 정답률도 아니고 수학적 상한도 아니다. `fresh r mean=.0053` 역시 보상/accuracy가 아니라 gradient cosine 점수 평균이다. 이전 설명에서 이 구분과 생성 버전의 R 예산을 명확히 하지 않은 부분을 정정한다.

생성 버전은 후보 fresh 32개를 R16/A8/B8로 나눈다. 현재 v3는 R8을 primary로 쓰고 R16을 high-budget sensitivity로 추가한다. 두 버전 모두 validation 100문제를 R50/A25/B25로 나누며 각 문제의 rollout은 8개다. A/B 비교에서는 후보 표본뿐 아니라 validation 방향도 달라진다. 따라서 이 수치는 전체 측정 경로의 재현성이다.

후보 gradient는 4개씩 LOO advantage를 계산한다. 4개가 전부 같은 보상이면 그 micro-group gradient는 정확히 0이다. 이는 구현 오류는 아니지만 희소한 보상과 오채점에 취약하다. behavior K8은 한 그룹으로 계산하므로 후보 fresh micro-group 평균과 유한 표본 분산이 같지 않다. 이를 편향 버그로 단정하거나 그룹 크기를 사후 변경해서는 안 된다.

## 보고서가 실패한 이유와 남은 검증

digest의 regime-map은 `corrected score/oracle protocols are required`에서 중단했다. 생성 커밋은 oracle schema v2, 현재 분석기는 v3를 요구한다. 구버전 결과를 그대로 최신 분석기로 읽으면 반드시 거절되는 차이를 코드로 확인했다. 다만 실제 protocol JSON이 없으므로 이 실행에서 schema 이외에 누락/해시 불일치도 있었는지는 확정할 수 없다.

`scripts/family_readout.sh`는 실패 후 저장된 report.json의 숫자만 출력한다. 따라서 표가 나온 사실이 최신 계약 통과를 뜻하지 않는다. 이 스크립트의 `split-half ceiling` 문구도 부정확하다. digest는 protocol 본문을 포함하지 않고 에러 줄을 220자로 절단해 정확한 원인 확인을 방해한다. 이번 감사에서는 산출물과 보고서 계약을 임의로 우회하지 않았다.

v2→v3 R 예산 재계산은 저장된 micro-group/validation gradient로 가능하지만 A/B는 그대로이므로 낮은 A/B overlap 자체를 고치지 않는다. 보상 재채점은 별도의 작업이다.

## 정상 확인 및 미확정 위험

- 생성 버전의 실제 gradient 경로를 작은 OLMo3 모델에서 독립적인 per-sequence log_softmax/autograd와 비교: 상대 L2 오차 `2.606832083529298e-07`. 응답 경계, 토큰 shift, 합산/평균, padding, projection 연결에 이 검사에서 오류 없음. 실제 7B/H100/bf16 재현을 대신하지 않는다.
- d0에서 모든 ratio가 1, 네 estimator 가중치 동일: 재현 통과. 결과의 d0 KL=0, ESS≈1, 네 precision=.275와 일치한다.
- prefix/suffix/current/full ratio와 detach/clip, 독립 seed 및 EOS 절단, shard coverage/계약, tie 분리, bootstrap 경로는 관련 테스트 통과. CUDA 에러 34줄은 재시도 이력이며 완성 데이터의 오염을 입증하지 않는다.
- GRPO의 1 epoch 설정에서 업데이트 전 ratio=1, clip=0, 평균 advantage에 따른 loss≈0만으로 학습 중단이라고 볼 수 없다. 로그에 nonzero gradient가 있고 drift별 KL/ESS도 변한다.
- 생성 커밋은 LoRA 초기화 전에 명시적 seed가 없고 AdamW weight_decay를 생략했다(기본 .01). HEAD에는 seed/명시적 weight_decay 및 update guard가 추가되어 있다. 재현성/학습 계약 문제이나 학습하지 않은 d0의 낮은 fresh를 설명하는 원인은 아니다.
- projection 4096차원, bf16 gradient 누적, 응답 길이 cap=2048, validation 방향 간 cosine/zero 비율, 원본의 답 중복률과 오채점률은 현재 자료로 효과 크기 측정 불가. projection 차원/seed 민감도와 fp32 대조는 원본 또는 모델 실행이 필요하다.

## 검증 기록

- 생성 커밋 핵심 테스트 49개 통과. 한 테스트는 추출하지 않았던 scripts/run_point.sh 때문에 실패했고 해당 scripts를 추출한 뒤 그 테스트도 통과: 총 50개 통과.
- HEAD 전체 suite 시도: PYTHONPATH 및 math-verify 설치 경로를 보완한 뒤 132개 통과 상태에서 launcher 재시작 통합 테스트가 길어져 중단. 전체 suite 통과로 보고하지 않는다. 이 실행과 그 자식 프로세스만 종료했다.
- 최종 수정 후 reward/gradient/OLMo chunking/GRPO/protocol/contract/재개/compaction/bootstrap/measurement/관련 reward 회귀 테스트 125개 통과, 5.09초.
- 의존성은 네트워크 다운로드 없이 vendor/wheels에서 `/tmp/offpolicy-audit-20260908-deps`에 격리 설치했다.
- 최종 로그: 이 디렉터리의 `focused-tests.log`, `reward-before.log`.

## 복구 순서

1. 원본 run을 보존하고 별도 분석 사본에서 config, 실제 score/oracle protocol, hash, rollout/gradient coverage를 먼저 확인한다.
2. 저장된 token IDs를 같은 tokenizer로 decode하여 구/신 reward를 비교한다. behavior/fresh/validation별 0→1, 1→0 건수와 micro-group 활성 비율을 집계한다. 먼저 보상 버그의 실제 발현량을 확인한다.
3. 보상이 바뀐 경우 기존 합산 projected gradient만으로는 복구할 수 없다. 같은 저장 응답과 동일 정책으로 candidate 및 validation gradient/score를 다시 계산해야 한다. validation이 바뀌면 off-policy score도 다시 계산한다. 생성 자체는 재사용 가능하나 provenance를 새 분석 사본에 기록해야 한다.
4. 후보 표본을 고정하고 validation 방향만 바꾼 진단, validation을 고정하고 후보 분할만 바꾼 진단, 0 gradient 비율, projection 민감도로 노이즈 원인을 분리한다. 진단과 등록된 primary 평가를 구분한다.
5. 올바른 v3 산출물과 10,000 bootstrap으로 fresh gain/retention/margin을 재분석한다. schema 문자열만 v3로 바꾸는 것은 금지한다.
6. 학습 보상도 오염되었다면 재채점만으로 정책 학습을 되돌릴 수 없다. 수정 보상으로 학습한 새 run이 필요하다. 기존 결과는 legacy reward run으로 명시한다.

이번 수정은 로컬 작업 트리에만 있다. 실행 중인 클러스터나 고정된 generation commit에는 적용되지 않았고, 기존 결과/정책 파일도 변경하지 않았다. 다음 실행은 수정 내용을 포함한 커밋을 명시적으로 고정해야 한다.
