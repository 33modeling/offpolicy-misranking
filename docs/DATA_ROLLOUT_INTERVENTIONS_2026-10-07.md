# Switch의 데이터 구성과 rollout을 바꾸는 추가 실험

이번 요청의 우선 질문은 **초반 On-policy의 성능 이득이 왜 생기는가**다.
[초반 근거와 분석 순서](STAGE_MECHANISM_EXPERIMENTS_2026-10-06.md#초반에-왜-이득이-나는지-확인할-분석)에 따라
선택 문제와 실제 rollout의 차이를 먼저 확인하고 아래 개입 중 그 차이를 분리하는 것을 적용한다.

메커니즘 실험과 Qwen 실험은 현재 실행 중이라는 저자 보고를 기준으로 유지한다.
새 후속 실험의 질문은 **어떤 문제를 고르는가, 같은 문제에서 어떤 응답 대비로 배우는가,
같은 생성 예산을 어떻게 배분하는가**다. 아래 N09–N11은 기존 조건을 반복하는 실험이 아니라
각 요인을 조작하는 새로운 대조 설계다. 새 GPU 학습 runner는 아직 구현하지 않았다.
현재 사용할 수 있는 코드는 기존 결과의 계측 분석과 정규화된 원시 rollout 분석이다.

## 기존 결과에서 확인한 출발점

전달받은 MATH와 MBPP seeds 5–9의 네 방법 endpoint에서 계산했다.
아래 비율은 `training.zero_advantage_responses / training.responses`의
**seed별 비율을 평균**한 값이다. `generated_tokens / responses`도 seed별 평균이다.

| 데이터셋 | 방법 | zero advantage 응답 비율 | 응답당 생성 token |
| --- | --- | ---: | ---: |
| MATH | Random | 31.36% | 1124.4 |
| MATH | SR | 13.02% | 891.1 |
| MATH | On-policy | 9.70% | 1048.0 |
| MATH | Switch | 12.12% | 928.3 |
| MBPP | Random | 35.12% | 891.3 |
| MBPP | SR | 5.66% | 745.5 |
| MBPP | On-policy | 14.70% | 846.9 |
| MBPP | Switch | 8.40% | 780.7 |

MBPP에서는 SR이 On-policy보다 zero advantage 응답이 적고, MATH에서는 순서가 반대다.
SR의 응답 길이는 두 데이터셋 모두 On-policy보다 짧다. 이 관측은 **현재 성공률의 구성과
응답의 내용·길이를 분리할 필요성**을 뒷받침한다. SR의 효과가 이 요인 때문에 발생했다는
결론은 아니다. 전체 경로의 모델 상태가 다르고 계측에 실패·재시도가 포함될 수 있다.
특히 MBPP Random seed 5는 endpoint의 250 continuation updates와 달리 8,960개 응답이
계측되어 있어, 32응답씩 250회인 8,000개와 일치하지 않는다. 중복 제거된 학습 분포로 쓰지 않는다.

이 endpoint 파일에는 단계별 선택 문제와 응답 원문이 없다. 전부 정답인 group과 전부 오답인
group도 누적 zero advantage 수로 구분할 수 없다. 원시 응답 없이 pass@k·응답 다양성·오류 유형을
복원하지 않는다. [계산 결과](evidence/2026-10-07-rollout-receipts.json)는 원본 hash, seed별 값,
표본 SD, 누락과 horizon 불일치를 함께 보존한다.

## 다른 논문에서 가져올 분석과 개입

| 원문 | 실제 분석 또는 실험 | 여기서 가져올 부분 |
| --- | --- | --- |
| [DOTS](https://arxiv.org/html/2506.05316v4), Sec. 4 | 현재 정책의 응답 실패율로 문제 난도를 정의하고 선별한다 | 고정 dataset 난도와 현재 모델이 느끼는 난도를 구분 |
| [ARCUS](https://arxiv.org/html/2609.38018v1), App. F | 유효 group 비율, group 크기, pass@k와 생성 비용을 함께 비교 | 문제 수와 문제당 응답 수를 바꾸면서 같은 생성 예산을 유지하는 대조 |
| [Save Your Saturated Data](https://arxiv.org/html/2609.33126v1), Sec. 5–6 | reward variation 복원과 negative trajectory 구성의 효과를 분리 | 응답의 정답 수가 같아도 학습에 쓰는 trajectory 구성이 중요한지 검증 |
| [Lev](https://arxiv.org/html/2609.32484v1), Sec. 4 | 데이터 구성과 선별 기준을 분리하고 도메인별 손익을 확인 | 평균 보상 외에 선별된 문제의 구성과 평가 문제군별 손익을 연결 |

이 문헌들은 아래 Switch 실험의 설계 근거다. 특정 분석에서 좋은 결과가 나왔다는 사실을
Switch에서 같은 효과가 확인됐다는 근거로 사용하지 않는다.

## N09 난도를 맞춘 문제 분포 개입

**질문:** gradient 선별의 효과는 현재 난도가 적절한 문제를 고르는 데서 오는가,
아니면 같은 난도에서도 특정 문제 구성을 고르는 데서 오는가?

같은 모델·optimizer checkpoint에서 공통 후보 40개의 독립 pilot rollout을 생성한다.
pilot의 성공 수와 prompt token 길이 구간으로 후보를 함께 층화한다. On-policy top-4의
각 층별 문제 수를 고정하고, 그 구성에 맞는 세 대조를 만든다.

| 조건 | 그대로 맞추는 것 | 바꾸는 것 |
| --- | --- | --- |
| On-policy top-4 | 출발 상태, 후보, 선택 수 | 실제 선별 집합 |
| Matched Random | pilot 성공 수와 prompt 길이 구간의 공동 분포 | 층 안에서 문제를 무작위로 선택 |
| Matched Spread | 같은 공동 분포 | 문제 표현의 평균 쌍별 거리가 큰 집합 |
| Matched Compact | 같은 공동 분포 | 문제 표현의 평균 쌍별 거리가 작은 집합 |

따라서 같은 성공률 구간을 더 많이 뽑았다는 설명과 문제 범위의 차이를 분리한다.
Spread와 Compact는 같은 수의 문제를 사용하며, 평가 문제를 선택 기준에 쓰지 않는다.
문제 표현은 개입 전 고정한 encoder의 embedding을 사용하고 encoder와 hash를 기록한다.
단어 집합 거리만 사용할 때는 **어휘 분포** 실험으로 명시하며 의미 다양성으로 부르지 않는다.
원자료에 없는 topic·난도 라벨을 새 정답 라벨처럼 취급하지 않는다.

먼저 0·100·400의 기존 checkpoint가 모델·optimizer·입력까지 검증 가능하게 보존됐는지 확인한다.
검증된 상태별로 네 집합을 각각 복원하고 1 update를 적용한 뒤 독립 평가 보상과 고정 평가 loss의
변화를 본다. 짧은 5-update 후속 학습은 새 rollout을 사용하며 최초 matching이 이후에도
자동 유지된다고 가정하지 않는다. checkpoint가 없으면 새 400-update 경로부터 배정하지 않는다.

**필수 출력:** 후보→선택 집합의 현재 성공 수 분포, source에 있는 topic별 구성,
prompt 길이·쌍별 거리·집합 겹침, 누적 문제 재노출 수, 학습 후 평가 변화와 비용.
평가 구성도 원자료의 라벨이 있을 때만 분해한다. 문제를 반복해서 뽑은 횟수와 서로 다른
문제를 접한 수를 따로 보고한다.

**판정:** Matched Random이 top-4의 이득을 회복하면 난도 구성의 설명력이 커진다.
Spread와 Compact 사이에 차이가 남으면 동일 난도에서 문제 구성의 영향이 있다는 증거다.
한 번 측정한 성공률은 noisy estimate이므로 독립 B rollout에서도 matching 오차를 보고한다.
동일 checkpoint에서의 국소 효과가 전체 Switch 경로의 효과를 완전히 설명한다고 쓰지 않는다.

**실패 조건:** 공동 층별 후보가 부족하거나 두 대조의 거리가 실질적으로 같으면
`matching_infeasible` 또는 `no_distribution_separation`으로 남긴다. 결과를 본 뒤
구간을 합치거나 다른 난도로 대체하지 않는다. N07의 한 축별 matched Random에
분포의 방향을 직접 바꾸는 Spread/Compact 개입을 추가하는 설계다.

## N10 정답 수를 맞춘 rollout 다양성 개입

**질문:** 같은 문제에서 같은 수의 정답·오답을 주어도, 서로 다른 풀이 또는 실패 경로가
더 유익한 업데이트를 만드는가?

checkpoint와 문제 네 개를 고정하고 각 문제에서 같은 정책·sampling 설정으로 32개의
응답을 생성한다. 이 공통 pool에서 실제로 채점된 응답 여덟 개씩을 구성한다.
정답·오답 수와 응답 token 길이 구간의 **공동 histogram**을 세 조건에서 정확히 맞춘다.

| 조건 | 응답 선택 |
| --- | --- |
| Matched Random | 같은 reward×length 구간 안에서 무작위 선택 |
| Low Diversity | 구간별 수를 유지하며 응답 경로가 유사한 group 구성 |
| High Diversity | 같은 구간별 수를 유지하며 서로 다른 응답 경로의 group 구성 |

예를 들어 세 조건 모두 정답 4개·오답 4개이며 각 길이 구간의 개수도 같게 만든다.
같은 response를 인위적으로 복제하거나 정답·오답 라벨을 바꾸지 않는다.
부족한 경우에는 실제 구성 가능한 동일 quota를 미리 정한 규칙으로 택하고 기록한다.
all-correct/all-wrong group은 pure diversity 개입의 학습 대비가 없으므로 별도로 집계하고,
mixed group의 효과에 합치지 않는다. 원하는 응답이 나올 때까지 무제한 생성하지 않는다.

거리 지표는 응답 token n-gram과, 검증 가능한 경우에만 풀이 구조를 사용한다.
MBPP 코드의 AST/실행 실패 유형과 MATH의 풀이 단계 표본 분석은 별도 보조 측정이다.
어휘·AST 차이를 의미적으로 서로 다른 풀이와 동일시하지 않는다.
MATH 풀이 유형을 사람이 표시한다면 방법 이름을 가리고 정한 codebook으로 일부를
중복 판정한다. 자동 tagger의 출력을 확정된 유형 정답으로 취급하지 않는다.

세 조건은 같은 checkpoint·optimizer에서 각각 **한 번의 GRPO update**를 수행한다.
generation temperature를 바꾸지 않으므로 temperature와 다양성이 동시에 바뀌는 혼동을 피한다.
현재 backend가 on-policy log-probability를 사용하는 범위를 지키며 이 pool로 여러 updates를
반복해 fresh on-policy 학습처럼 보고하지 않는다.

**필수 출력:** 실제 reward·length matching 오차, group당 고유 응답 수와 거리,
정답/오답별 길이·반복률·종료 이유, raw gradient와 실제 update norm,
독립 평가 변화. token별 entropy나 clipping 비율은 원시 log-probability가 있을 때만 계산한다.

**판정:** reward 분포가 같아도 독립 평가 변화가 다르면 응답 경로 구성의 기여를 지지한다.
gradient norm만 바뀌고 평가가 그대로면 학습 효용 증가로 결론내리지 않는다.
diversity가 높을수록 좋다는 방향도 미리 확정하지 않는다.
대조군 구성이 만드는 off-policy 성격은 진단 범위와 함께 보고한다.

**비용:** 문제 네 개의 32응답 pool은 총 128개다. 조건별 학습에 32개를 썼다는 것과
128개를 생성한 진단 비용을 분리한다. 실패 응답·선별·채점 비용을 제외하지 않는다.

## N11 같은 rollout 예산의 문제 수와 응답 수 배분

**질문:** 후반에 많은 문제를 얕게 보는 것이 유리한가, 적은 문제에서 응답을 충분히
뽑아 reward 대비를 만드는 것이 유리한가? 전환 효과가 기존 4×8 배분에 의존하는가?

| 배분 | 문제 수 | 문제당 응답 | 학습에 적용하는 총 응답 |
| --- | ---: | ---: | ---: |
| 좁게 깊게 | 2 | 16 | 32 |
| 현재 조건 | 4 | 8 | 32 |
| 넓게 얕게 | 8 | 4 | 32 |

모델·optimizer·candidate pool·학습 objective·평가 sampling을 고정한다.
현재 난도와 prompt 길이의 공동 분포 비율이 같은 nested 문제 집합을 구성한다.
2/4/8개에서 정확한 비율 matching이 불가능하면 잔여 차이를 명시한 별도 탐색 결과로
남기고 matched 비교로 쓰지 않는다. 선별용 응답 수는 8로 고정해 scorer 신뢰도까지 바꾸지 않는다.

우선 검증된 같은 checkpoint의 최대 여덟 문제에서 16응답씩 얻고,
각 배분에 맞는 부분집합을 사용해 한 update의 효과를 비교한다.
이 acquisition은 128응답이고 각 대조의 학습 노출은 32응답이다.
여덟 응답을 전부 같은 prompt의 group으로 묶는 규칙은 그대로 지키며 다른 문제의
응답끼리 섞어서 GRPO group을 만들지 않는다. 이후 온라인 짧은 학습이 필요하면
각 조건이 매 update 직접 32개의 fresh 응답을 생성하게 한다.

**필수 출력:** all-correct/all-wrong/mixed group 비율, 실제 학습에 참여한 응답 수,
고유 문제 노출량, 생성·학습 token, update 수, 평가 변화와 GPU 시간.
32응답을 맞춘 것은 token 수나 GPU 비용을 맞춘 것과 다르므로 둘 다 보고한다.
변수는 문제당 응답 수이며, GRPO 표준화와 prompt/response loss 평균 방식은 고정한다.

**판정:** 같은 32응답에서 배분에 따라 후반 효과가 달라지면 문제 범위와 group 신호의
trade-off를 보여 준다. 이를 먼저 On-policy와 SR의 단계별 상태에서 확인하고,
차이가 분명할 때만 별도의 전체 Switch 경로 확장을 고려한다.
N05는 문제 수를 바꾸며 총 응답 수도 늘어나는 실험이고, 여기서는 총 응답 수를 고정한다.

## 실행 순서와 rebuttal에 넣을 근거

| 순서 | 작업 | 답할 리뷰 쟁점 |
| --- | --- | --- |
| 현재 병행 | 메커니즘과 Qwen 결과 수신·검증 | 선별 정보의 국소 기여와 backbone 의존성 |
| 추가 1 | 원시 rollout과 선택 ID의 분포 분석 | 모델·단계별로 실제 학습 데이터가 어떻게 달라지는가 |
| 추가 2 | N09의 matching feasibility와 한 update 대조 | 난도만으로 설명되는 효과와 문제 구성의 효과 |
| 추가 3 | N10의 공통 pool 대조 | reward 분산 외에 어떤 응답 대비가 유익한가 |
| 추가 4 | N11의 32응답 배분 대조 | 작은 batch와 rollout 배분에 의존하는 결론인가 |

N09–N11은 checkpoint에서 요인을 조작하는 새 연구 설계다. 반복 seed를 늘리는 작업,
fixed200 재도입, 기존 메커니즘의 shuffle/fresh-SR 재실행으로 대체하지 않는다.
현재 N01–N08은 기존 설계로 남기고, 이번 요청의 추가 설계는 데이터·응답 구성부터 검토한다.
새 checkpoint를 확보하려고 진행 중인 job의 코드·해시·sampling을 바꾸지 않는다.

## 사용할 수 있는 CPU 분석 명령

현재 구현은 [`rollout_distribution.py`](../srgc_research/rollout_distribution.py)다.
endpoint receipts와 원시 rollout을 서로 다른 명령으로 처리하고, 입력·기존 출력을 덮어쓰지 않는다.

```sh
python3 -m srgc_research.rollout_distribution endpoints \
  /home/kms/results.json /home/kms/results_mbpp.json \
  --responses-per-update 32 --output /tmp/srgc-rollout-receipts.json

python3 -m srgc_research.rollout_distribution traces \
  /path/to/normalized-traces.json --output /tmp/srgc-rollout-traces-analysis.json
```

두 번째 입력은 `schema: srgc-rollout-trace-v1`, `dataset`, `model`, `seed`, `groups`를 가진다.
각 group에는 `arm`, `phase`(training/selection/diagnostic), `update`, `prompt_id`, `responses`가
필요하며, 같은 group의 독립 재측정은 서로 다른 정수 `draw`를 기록한다.
각 응답에는 실제 검증된 binary `reward`가 필요하다. 길이·어휘 다양성을 계산할 때는
prompt를 제외한 `token_ids`가 필요하다. 종료 원인을 계산하려면 `finish_reason`도 보존한다.
token이 없으면 해당 측정은 미확인으로 남긴다. 서로 다른 group size의 성공 histogram도
`성공 수/응답 수`로 구분한다. 이 명령은 원격 cache를 자동 수집하거나 GPU를 사용하지 않는다.

원시 export에서는 실패 시도와 최종 학습 시도를 구분하고 동일 group을 중복 넣지 않는다.
모든 분석 그림에는 단계·dataset·backbone·seed와 사용된 표본 수를 표시한다.
현재 누적 계측 그림은 raw response 분포나 단계별 메커니즘 그림으로 쓰지 않는다.

## 구현 검증

CPU 분석 테스트 17개와 Ruff 검사를 통과했다. 손상된 계측값, 잘못된 seed/arm,
중복 group, 누락 token, 비이진 reward, 반복 문제의 coverage,
입력·기존 출력 보존을 검사했다. 받은 두 export의 원본 hash는 유지됐고,
메커니즘·Qwen·기존 학습 파일 여섯 개의 수정 전후 hash도 일치한다.
이 검사는 N09–N11의 GPU 학습 검증이 아니다.
