# Switch 신규 실험: 목적·우선순위·실행 계획

2026-10-07의 구체적 분석 목표는 **초반 On-policy의 학습 성능 이득이 생기는 이유**다.
[기존 근거와 원인 분석 순서](STAGE_MECHANISM_EXPERIMENTS_2026-10-06.md#초반에-왜-이득이-나는지-확인할-분석)에 따라
선택 문제·실제 학습 보상을 먼저 분석하고 N07/N02를 필요한 부분에 적용한다.
아래 전체 Switch 방법 비교의 우선순위를 초반 원인 분석의 순서로 대신 쓰지 않는다.

2026-10-07 추가 요청: [데이터 구성과 rollout 개입 N09–N11](DATA_ROLLOUT_INTERVENTIONS_2026-10-07.md).
현재 실행 중이라는 저자 보고는 메커니즘과 Qwen이다. 새 후속 설계는 난도를 맞춘 문제 분포,
reward·길이를 맞춘 응답 다양성, 고정 32응답의 배분을 조작한다.
N09–N11의 GPU 학습 runner는 아직 없으며 아래 N01–N08 구현·명령과 구분한다.

2026-10-06. [최근 논문 12편 조사](RECENT_PAPER_EXPERIMENT_SURVEY_2026-10-06.md)를 바탕으로 한 실험 구현·실행 안내.
**현재 상태: N01-N08 구현 완료. CPU 통합·분산 검증과 H100/NCCL 실측은 구분한다.**
N01-N08은 신규 실험 식별자이며
기존 E01-E16과 진행 중인 `mechanism`을 대체하지 않는다.

핵심 질문은 세 가지다. 더 저렴한 gradient 선별과 비교해도 전환이 유용한가,
선별 점수가 실제 학습 효과로 이어지는 관계가 단계별로 어떻게 달라지는가,
결과가 특정 reference·후보·batch 규모에 의존하는가를 확인한다.
성능과 비용을 함께 평가하며 SR-GC는 전환을 돕는 보조 지표로 유지한다.

## 지금 할 것과 기다릴 것

| 구분 | 실험 | 먼저 하는 이유·시작 조건 |
| --- | --- | --- |
| 신규 학습 우선 | **N01: 저비용 gradient 선별 대조** | 비싼 기존 scorer 때문에만 Switch가 유리한지 직접 확인 |
| 기존 결과로 병행 | **N08: 문제별 개선·퇴행 분석** | 받은 평가 JSON으로 평균 뒤의 차이를 확인; 새 학습 불필요 |
| 직접 한계 대응 | **N04: 실제 reference 교체**, **N05: 후보·batch 규모** | reference 구성과 작은 학습 규모에 대한 의존성을 각각 검증 |
| 메커니즘 결과 수신 후 | **N02: 실제 gradient·optimizer 진단**, **N03: 점수 구간별 학습** | 진행 중인 E14-E16을 먼저 해석한 뒤, 기존 결과로 답하지 못하는 부분만 추가 |
| 관측 차이 확인 후 | **N07: 구성 matched 대조** | 성공률·재노출 등의 차이가 실제로 있는 축을 통제 |
| 다음 외부 비교군 | **N06: ARCUS 계열 대조** | gradient 없는 동적 curriculum과 비교; 독립 adapted 구현이며 공식 코드 재현으로 부르지 않음 |

위 순서는 신규 작업 배정 기준이다. 기존 실험을 중단·재배정하지 않는다.
`fixed200`은 채택 제외를 유지하고, `replicate`는 보조 결과로만 둔다.

## 실행 조건 한눈에 보기

공통 기준은 MATH·MBPP, 기존 OLMo/GRPO 환경, seeds 5-9, 응답 8개다.
전체 경로 비교는 **t0에서 시작**하고, On-policy 재선별 간격은 25로 둔다.
기본 후보 40개는 매 선별 시점 400개 pool에서 중복 없이 뽑는다. 선택한 학습 문제는 4개다.
SR-GC는 **40 대 40, 실행당 reference 하나**를 유지한다.

| ID | 비교·개입 | 고정할 조건 | 핵심 출력 |
| --- | --- | --- | --- |
| N01 | 기존 On-policy / 출력층 feature 선별 / Switch | scorer 이외 모델·학습·응답·평가 조건 | reward 곡선, 선별·학습 시간, 목표·GPU 예산별 성능 |
| N02 | On-policy / SR / Random의 실제 학습 gradient·Adam update | 동일 checkpoint·optimizer, 학습 batch 크기 | raw·centered gradient, 실제·zero-gradient 차감 update, 평가 변화 |
| N03 | 점수 4구간별 4문제 / top-4 / Random-4 | 동일 상태에서 분기, 분기 내 선택 문제 유지 | 학습 전 및 1·5·25 updates 후 reward |
| N04 | reference 3집합, 각 집합으로 자체 Switch 경로 | reference 크기, 후보, 학습·평가 조건 | reference별 SR-GC·순위 변동, 전환 시점·reward·비용 |
| N05 | 후보 40/80/160에서 4개 선택; 별도로 후보 40에서 4/8/16개 학습 | 한 번에 한 축만 변경 | reward, prompt·응답·token 노출, GPU 비용 |
| N06 | ARCUS-adapted / Switch | 모델·pool·GRPO·평가·응답 cap | 생성량, 유효 group 비율, 성능·비용 |
| N07 | gradient top-4 / Random / 성공률 또는 재노출 분포 matched Random | 동일 상태·batch 크기, 한 번에 한 구성 축 | matched 전후 학습 이득·구성·비용 |
| N08 | 동일 seed·평가 문제의 방법별 결과 | 평가 ID·설정·실험 출처 | 문제별 증가·감소, 원시 응답이 있을 때만 pass@k |

N02·N03·N04 고정 상태 진단·N07의 비교 시점은 0/100/400이다.
**이 진단의 학습 경로와 본 실험의 275-update 경로는 다른 작업**이며, 비용도 합치지 않는다.
기존 메커니즘 checkpoint를 검증 없이 가져오지 않는다. 별도 `anchors` 작업은
t0부터 400 updates까지 진행해 비교 상태를 만든다. 따라서 기존 결과 수신 전에 이 작업을
새로 배정하지 않으며, 재사용 가능한 checkpoint가 있는지 먼저 확인한다.

N04는 100문제 validation pool에서 기존 reference와 같은 크기의 서로 다른 세 집합을 만든다.
현재 50문제 reference라면 세 집합을 모두 서로 겹치지 않게 만들 수 없으므로 실제 겹침을 기록한다.
후보·최종 evaluation 문제를 reference에 섞지 않으며, 없는 topic 라벨을 만들어 맞추지 않는다.

## 코드와 검증 상태

| 항목 | 위치 또는 상태 |
| --- | --- |
| 저장소·브랜치 | 코드 저장소 `offpolicy-misranking`, `master`; 논문 저장소에는 추가하지 않음 |
| 신규 코드 경로 | [`srgc_research/`](../srgc_research/) 및 [`scripts/run_srgc_research.sh`](../scripts/run_srgc_research.sh) |
| 기존 코드 보호 | 기존 학습·queue·E14-E16 실행 코드 수정 없음; 기존 Qwen 테스트의 MBPP fixture에 누락된 dataset만 보충 |
| 실행·상태·결과 수집 | 신규 전용 wrapper의 `run/status/results/json`; 기존 dispatcher와 별개 |
| 자동 재개 검증 | 여섯 trajectory의 모델·optimizer·원시 평가 기록 일치, 진단 단계별 복원, endpoint 재발행 확인 |
| 비용·결과 검증 | 비용 합계·미완료 timer, 원본 JSON 수집, 평가 ID·cache hash 오류 거부, 미완료 seed 표시 |
| 분산 검증 | 4-rank CPU/Gloo에서 scorer·rollout 재사용·GRPO/Adam 진단·checkpoint 복구 확인 |
| 최종 회귀 테스트 | 신규 45개 포함 754개 통과, 325 subtests 통과, 실패 0개; 별도 Qwen 환경 필요 9개 제외 |
| 정적·명령 검증 | Ruff, Python compile, `sh -n` 통과; MATH/MBPP `status/results` 경로 출력·빈 결과 처리 확인 |
| GPU 검증·실험 | H100/NCCL 실행 검증 및 신규 실험 실행하지 않음 |
| 기존 결과 N08 | MATH·MBPP 원본 JSON 각각 방법 쌍 30개 분석 완료; 새 학습 없음 |

검증에는 실제 작은 OLMo/LoRA 모델과 기존 GRPO·AdamW 경로를 사용한다.
단위 테스트의 생성 응답은 결정적으로 주입하며, H100의 메모리·속도·NCCL까지 검증했다는 뜻은 아니다.
재개 시 평가 응답 배열이 이후 학습 기록에 의해 변하던 참조 공유 오류와
binary float 응답의 pass@k 조합 계산 오류를 수정했다.
기존 결과·진행 중인 runtime을 다시 작성하거나 해시를 해제하지 않는다.

구현 위치: 조건·의존성 `design.py`, scorer/optimizer 계측 `backend.py`,
학습·진단 `study.py`, ARCUS 대조 `arcus.py`, 저장·복구 `storage.py`/`worker.py`,
노드 배정 `cli.py`/`launch.py`/`rank.py`, 결과·N08 `report.py`.

## 명령 형식

저장소 루트에서 `sh`로 실행하며, 별도 Python 옵션이나 `resume` 명령을 요구하지 않는다.
기존 Python 환경을 그대로 사용하고 새 패키지를 설치하지 않는다.

| 실험 | 실행 명령 형식 |
| --- | --- |
| N01 | `sh scripts/run_srgc_research.sh math n01 run` |
| N02 | `sh scripts/run_srgc_research.sh math n02 run` |
| N03 | `sh scripts/run_srgc_research.sh math n03 run` |
| N04 | `sh scripts/run_srgc_research.sh math n04 run` |
| N05 | `sh scripts/run_srgc_research.sh math n05 run` |
| N06 | `sh scripts/run_srgc_research.sh math n06 run` |
| N07 | `sh scripts/run_srgc_research.sh math n07 run` |
| N08 | `sh scripts/run_srgc_research.sh math n08 results /path/results.json` |

MATH는 `math`, MBPP는 `mbpp`를 쓴다. N01-N07의 `all`은 두 데이터셋을 배정한다.
작업별 기존 `PAIR_PYTHON`/`SWITCH_PYTHON`을 선택하므로 서로 다른 환경도 유지한다.
N08은 데이터셋별 JSON을 각각 분석한다.

상태·결과 명령은 해당 ID를 유지하고 마지막 인자만 바꾼다.

```sh
sh scripts/run_srgc_research.sh math n01 status
sh scripts/run_srgc_research.sh math n01 results
sh scripts/run_srgc_research.sh mbpp n01 results
```

빈 4-H100 노드에서 선택한 scope 하나만 실행하며, 여러 노드에서는 같은 명령으로
seed·조건을 중복 없이 배정한다. 이미 점유된 노드는 기존 작업을 중단하지 않고 `BUSY`로 종료한다.
실험 전체를 무조건 동시에 실행하는 기본값은 두지 않는다.
N05는 조건 15개/seed라 비용이 크다. 다른 scope와 같은 크기의 작업으로 안내하지 않는다.
노드 권장 수·예상 시간은 파일럿 계측 전 확정하지 않는다.

노드가 중단되면 **같은 명령을 다시 실행**한다. 매 학습 update와 진단 단계 종료 시
모델·optimizer·진행 상태를 저장하며, 그 사이 완료된 rollout 응답도 그룹 볼륨에서 복구한다.
자동 재개는 미완료 계산을 이어가는 것이지 실행 중인 노드를 다른 노드로 강제 이동하는 기능이 아니다.
완료 기준을 통과한 작업은 다시 돌리지 않는다. 반복 오류는 3회 뒤 `failed`로 표시한다.

N01의 On-policy는 N05 기본 조건과, N01의 Switch는 N06 대조와 동일 작업을 재사용한다.
진단용 `anchors`와 최초 SR cache도 scope 간 공유한다. 기존 v2/prefix 실험 결과와는 합치지 않는다.

## 결과와 비용 기록

신규 작업의 기본 저장 루트는 `$SRGC_STORAGE_ROOT/literature-v1`이다.
`SRGC_STORAGE_ROOT` 미지정 시 기존 그룹 볼륨의 `$OM_WORK/srgc-rebuttal` 아래를 사용한다.
user volume으로 대체하지 않으며 기존 결과 디렉터리에 덮어쓰지 않는다.

| 산출물 | 신규 루트 아래 위치 |
| --- | --- |
| 실험 출처·고정 입력 | `<dataset>/seed-<N>/manifest.json`, `inputs.json` |
| 최초 SR 캐시 | `<dataset>/seed-<N>/sr-cache.json` |
| 작업 결과·전체 원시 기록 | `<dataset>/seed-<N>/<condition>/endpoint.json` |
| 중단 복구 지점 | `<dataset>/seed-<N>/<condition>/state-latest.pt` |
| 세부 비용·실행 로그 | `<dataset>/seed-<N>/<condition>/cost-receipts/`, `task.log` |
| 전체 프로세스 점유 시간 | `<dataset>/seed-<N>/<condition>/invocations/`; 내부 phase와 중복 합산하지 않음 |
| 수집 JSON | `exports/<dataset>-<scope>-results.json` |
| N08 분석 | 입력 JSON 옆 `<입력파일명>-n08.json` |

`results`는 저장 경로를 출력하며 수집 JSON 안에 원본 결과를 포함한다.
미완료·오류·비용 누락을 완료 결과와 구분하고, 아직 시작하지 않은 seed도 누락하지 않는다.
평균에는 완료된 seed 목록과 전체 완료 여부를 함께 기록한다.
N02의 원시 gradient·update tensor는 checkpoint의 `state.vectors`에 저장하며,
JSON에는 norm·alignment·평가값과 해당 위치를 기록한다.
N08 기존 결과 분석 파일은 `/home/kms/results-n08.json`, `/home/kms/results_mbpp-n08.json`이다.

비용은 **최초 SR 캐시 / 선별용 후보·reference 생성 / gradient·feature 계산 /
정렬 / 학습 / 평가 / 저장 I/O**로 나눈다. SR 캐시 생성 비용과 점수 정렬 비용을 섞지 않는다.
최초 캐시는 seed·데이터셋별 별도 신규 측정 후 재사용하며, 기존 실행의 cache를 삭제·재생성하지 않는다.
Cold-start 비교에는 최초 캐시를 한 번 넣고, 재사용 비교에서는 이미 준비된 비용을 분리해 보여 준다.
연구용 반복 reference·centering·optimizer 진단 비용을 실제 Switch의 온라인 비용에 넣지 않는다.

목표·GPU 예산 비교가 필요하면 **최초 실행 전** `SRGC_RESEARCH_MATH_TARGET` 또는
`SRGC_RESEARCH_MBPP_TARGET`에 reward 목표(0-1), `SRGC_RESEARCH_GPU_BUDGETS`에
쉼표로 구분한 GPU-second 예산을 정한다. 이 값은 manifest에 고정한다.
미지정 시 학습·비용 곡선만 수집하며, 결과를 본 뒤 목표를 만들어 도달 시간을 채우지 않는다.
도달 시간은 실제 평가 시점 중 처음 관측된 값이며 그 사이를 보간하지 않는다.

## N01. Gradient 선별을 싸게 해도 전환할 이유가 남는가?

**참고:** [LESSER Sec. 4-5, App. A.4/F](https://arxiv.org/html/2610.03702v1).
논문의 출력층 feature 대체를 참고하되, 논문에 보고된 시간 배수를 우리 로그에 곱하지 않는다.

**비교군:** 기존 full-backprop 기반 On-policy continue, 출력층 gradient 기반 continue,
기존 Switch. 첫 단계에서는 Switch의 규칙이나 SR-GC 정의까지 동시에 바꾸지 않는다.
현재 baseline의 실제 학습 가능 parameter와 projection 범위를 명시하고,
논문의 dense full-gradient 구현과 동일하다고 부르지 않는다.

**고정:** 같은 시작 모델·optimizer, 후보/query 구성, scoring 응답 수, 선택 문제 수,
재선별 주기, 학습 objective·평가. Scoring representation만 교체하며 GRPO 학습 parameter는 유지한다.
같은 checkpoint의 feature 대조에서는 같은 scoring rollout을 쓴다.
학습 경로가 갈라진 뒤에는 각자의 현재 policy로 fresh rollout을 생성한다.
구현은 t0 전체 경로 세 개와 같은 응답을 재사용하는 `n01-features` 진단 한 개다.
출력층 feature는 현재 projection 차원에 맞춘 64×64 Rademacher 이중 투영(4,096차원)이며,
원 논문의 8,192차원 설정과 구분한다. Feature 진단 비용은 trajectory의 배포 비용에 더하지 않는다.

**측정:** 학습곡선, 최종 held-out reward, 사전에 정한 목표까지의 GPU 시간,
같은 누적 GPU 예산에서의 성능. 후보/query 생성, scorer forward/backward,
projection·ranking, 학습, SR 일회성 cache를 각각 기록한다.

**판정:** 싼 scorer가 실제로 빨라졌는데도 Switch가 성능·비용의 유리한 조합을 제공하면
전환의 가치가 비싼 backward 구현 하나에만 의존하지 않는다는 근거다.
계속 선별하는 쪽이 더 좋다면 그 조건을 그대로 보고한다. 비용만 이기고 성능이 낮으면 trade-off다.

**범위·규모:** 이번 구현은 t0에서 시작하는 전체 경로 비교다.
데이터셋·seed별 trajectory 3개와 같은-rollout 진단 1개다.
동일한 신규 baseline만 공유하며, 기존 prefix 결과를 전체 경로 결과로 합치지 않는다.

## N02. 점수의 방향과 실제 업데이트 방향은 같은가?

**참고:** [LESSER App. F.2](https://arxiv.org/html/2610.03702v1),
[MOPD 분석 Sec. 3-4, App. F](https://arxiv.org/html/2610.02179v1).

**질문:** 선택된 문제들의 cosine 평균이 높아도 실제 batch gradient는 다를 수 있다.
또한 서로 다른 gradient가 공통 Adam 이력 때문에 비슷한 update로 보일 수 있다.

**설계:** 보존된 초반·중반·후반 모델/optimizer 상태에서 On-policy, SR, Random이 고른
동일 크기 batch를 비교한다. 각 batch에 대해 다음 네 측정을 분리한다.

1. 실제 학습 loss로 계산한 batch gradient와 그 norm.
2. 독립 random batch의 평균 성분을 제거한 centered batch gradient.
3. 저장된 optimizer 상태를 적용한 실제 parameter update.
4. 같은 상태에서 현재 gradient를 0으로 둔 update를 뺀 증분 update.

증분은 `update(gradient, optimizer_state) - update(zero_gradient, optimizer_state)`다.
원고의 H 진단과 혼동하지 않도록 optimizer 상태를 H로 표기하지 않는다.
AdamW moments, step counter, weight decay, clipping, loss reduction과 정밀도를 맞춘다.
`grad=None`으로 update를 생략하는 것은 zero-gradient 대조가 아니다.

**측정:** 방법 간 raw/centered alignment, update norm, 증분 update 방향,
한 번 업데이트한 뒤 독립 평가의 loss·reward 변화. Loss 진단은 frozen probe 응답에서,
reward 평가는 업데이트된 policy의 새 응답에서 측정하고 둘을 구분한다.
진단 branch는 실행 후 원래 상태로 복구한다. 본 학습 optimizer는 바꾸지 않는다.

**판정:** 어느 단계에서 선별 점수, 실제 batch 방향, optimizer 처리, 실제 성능의 연결이
약해지는지 확인한다. 방향 유사성만으로 인과나 장기 성능을 결론내리지 않는다.

**기존과 차이:** 현재 mechanism은 선택 문제별 B cosine/dot/norm 평균을 저장한다.
합산 gradient vector나 optimizer update 진단은 아니다.
Checkpoint와 원시 vector가 없으면 기존 JSON만으로 복원할 수 없고 별도 계측이 필요하다.
Centering 평균은 비교 batch와 독립적으로 추정하며, 불안정하거나 norm이 0인 측정은 따로 표시한다.

## N03. 높은 선별 점수가 실제 학습 이득을 예측하는가?

**참고:** [LESSER의 similarity-bin 학습, Sec. 4/App. C](https://arxiv.org/html/2610.03702v1),
[Quality-Utility의 동일 문제 통제, Sec. 4-6](https://arxiv.org/html/2606.16152v1).

**설계:** 동일 checkpoint의 후보 40개에 현재 policy의 gradient 점수를 매긴다.
정렬한 후보를 10개씩 네 구간으로 나누고 각 구간에서 4개를 무작위 선택한다.
별도로 기존 top-4, Random-4를 비교한다. 각 분기는 모델·optimizer를 같은 상태로 복원한다.
이 수치는 신규 설계의 시작안이며 기존 runner 설정을 바꾸지 않는다.

점수를 구간으로 나눌 때 동점은 사전 고정 순서로 처리하되, 실제 score 범위와 동점 비율을 보고한다.
전부 같은 점수인데 서로 다른 정보 구간이 생긴 것처럼 해석하지 않는다.
선정된 4개를 유지하며 매 update 문제당 8개 응답을 새로 생성한다.
학습 전과 1·5·25 updates 후를 비교해 즉시 효과와 여러 번 학습한 효과를 구분한다.

**측정:** 구간별 독립 평가 reward 증가, selection score와 개선량의 순서 관계,
유효 reward group 비율, gradient/update norm, 학습·평가 시간.
같은 상태의 여러 후보 추출을 사용해 한 batch의 우연을 구분하되 독립 training seed로 세지 않는다.

**판정:** 높은 구간의 이득이 초기에는 크고 후기에는 줄어드는지, 국소 loss 개선은 유지되지만
held-out reward로 이어지지 않는지 구분한다. 단조 관계가 유지되면 그 결과도 그대로 기록한다.

**기존과 차이:** E16은 selector가 고른 집합끼리 비교한다.
이 제안은 **점수 순위의 여러 구간을 실제로 학습**해 점수의 예측력을 검사한다.
기존 top-4/Random 결과는 시작 상태·후보·난수·평가가 같을 때만 재사용한다.
기본 분기 6개를 세 시점에서 25 updates씩 실행하면 후보 추출 1회당 450 branch updates다.
구현 기본값은 시점마다 서로 다른 후보 추출 3회로, seed·데이터셋당 **1,350 branch updates**다.
`draw-0/1/2`는 같은 학습 상태에서의 후보 반복이지 새 training seed가 아니다.
Carrier·scoring·평가 비용은 별도이므로 가벼운 무학습 분석으로 안내하지 않는다.

## N04. Reference 문제를 바꾸어도 판단과 성능이 유지되는가?

**참고:** [FAC의 지표 오차·전이 검증, App. L](https://arxiv.org/html/2602.10388v4),
[TESSY의 교차 학습, Sec. 4.2](https://arxiv.org/html/2604.14164v2).
Reference 교체 자체는 이 논문들의 실험이 아니라 Switch의 한계에 맞춘 신규 제안이다.

**설계 1, 고정 상태 진단:** 같은 크기의 reference 집합 3개를 미리 정한다.
후보 pool과 최종 evaluation에서 분리하고, 가능하면 주제·난도 구성을 맞춘다.
Reference끼리 불가피하게 겹치면 겹친 비율을 보고한다. 후보와 candidate rollout은 고정하고
reference 구성만 바꿔 ranking, SR-GC 값·부호, top-k를 비교한다.
별도로 같은 reference에서 rollout만 재추출한 변동을 측정해 두 불확실성을 구분한다.

**설계 2, 실행 검증:** 각 reference를 사용하는 Switch를 각자의 학습 경로에서 실행한다.
**한 실행에는 reference 하나만 사용한다.** 세 reference의 투표나 A/B 동시 선택 규칙을 만들지 않는다.
같은 경로의 신호만 재계산해서 다른 경로의 최종 성능을 추정하지 않는다.

**측정:** 부호 일치율, 순위 상관, top-k 겹침, 실제 전환 시점, 최종 reward·비용의 변화.
Reference 변경에 의한 변동과 rollout 재추출에 의한 변동을 나란히 보고한다.

**판정:** Reference가 달라도 성능·비용 결론이 유지되는지 확인한다.
신호 부호가 자주 바뀌어도 최종 성능은 안정적일 수 있으므로 둘을 별도 판정한다.
이 실험은 기존 A/B 진단이나 무작위 reference 방향 대조를 반복하는 것이 아니다.

## N05. 후보 수와 학습 batch 규모를 분리해서 늘리기

**참고:** [ARCUS App. F](https://arxiv.org/html/2609.38018v1),
[Lev Sec. 4.3](https://arxiv.org/html/2609.32484v1),
[LESSER의 선택 budget 비교](https://arxiv.org/html/2610.03702v1).

**질문:** 현재처럼 40개에서 4개를 골라 학습하는 조건에서만 전략 간 차이가 나타나는가?

| 별도 대조 | 시작 설계안 | 고정할 항목 | 해석 |
| --- | --- | --- | --- |
| 후보 범위 | 후보 40/80/160, 선택 4 | 전체 pool 400, 학습 문제 수·응답 수·주기 | 더 넓게 선별하는 비용이 학습 개선으로 돌아오는가? |
| 학습 batch | 후보 40, 선택 4/8/16 | 후보 범위, 문제당 응답 8 | 작은 학습 batch에서만 관측된 현상인가? |

두 축의 전체 조합을 처음부터 돌리지 않는다. 최소 규모 점검 후 필요한 축만 확장한다.
후보 순서는 미리 고정한 공통 무작위 순열을 이용해 작은 후보군이 큰 후보군의 부분집합이 되게 하고,
같은 후보군 안에서는 중복 문제를 뽑지 않는다. 학습량이 달라지는 대조는 누적 prompt 노출·응답·
token 기준과 실제 GPU 예산 기준을 모두 보고한다. 같은 update 수를 같은 계산량이라고 부르지 않는다.

**규칙 보존:** SR-GC의 40 대 40 점검은 그대로 둔다.
후보 규모 확장의 1차 비교는 selector 자체에 집중한다. Switch까지 확장할 때는
선별 후보군과 별도의 40 대 40 진단 집합을 명시하고 비용도 별도로 계측한다.

**측정·판정:** 규모에 따른 방법 간 reward 차이·변동성과 비용 곡선을 본다.
큰 batch에서도 차이가 유지되는지, 또는 평균화로 사라지는지 확인한다.
규모 하나에서 유리한 결과만 골라 기존 결론을 강화하지 않는다.

## N06. 동적으로 성공률을 추정하는 강한 비교군

**참고:** [ARCUS의 belief·pacing·sampling 대조](https://arxiv.org/html/2609.38018v1),
[DataMaster의 부분 모듈 결합](https://arxiv.org/html/2608.10579v1).

**질문:** Gradient 계산을 중단하는 Switch가, 학습 이력만으로 다음 문제를 고르는
동적 curriculum과 비교해 어떤 성능·비용 차이를 만드는가?

**설계:** ARCUS의 성공률 상태 추정, drift, target pacing, post-rollout filtering을
논문 식 (2)-(5)·Appendix D를 참고해 독립 구현한 **ARCUS-adapted**와 기존 Switch를 비교한다.
공식 코드 재현 실험으로 부르지 않는다. 이미 있는 matched fresh-SR 결과는
단순 갱신과 동적 추정의 차이를 읽는 보조 대조로 쓴다. 초기 policy부터 비교하는 것을 주 설계로 둔다.
기존 prefix에서 시작하는 결과를 사용하면 조건부 비교라고 따로 표시한다.

**통제:** 같은 모델·학습 pool·held-out 평가·보상·응답 cap을 사용한다.
ARCUS의 후보 공급이나 그룹 filtering을 기존 40-to-4 규칙으로 바꾸면 원래 ARCUS가 아닌
adapted 조건임을 명시한다. Fresh-SR을 ARCUS라고 이름만 바꾸지 않는다.
동일 update 수 외에 실제 생성 token·GPU 예산을 맞춘 결과도 제시한다.
이번 adapted 설정은 batch 4, 응답 8, 후보 여유율 25%로 매 round 5문제의 응답을 생성하고,
mixed-reward group 중 최대 4개를 학습한다. **생성한 응답을 그대로 학습에 재사용**한다.
유효 group이 없으면 optimizer update를 생략하고 sampling rounds와 실제 update 수를 따로 기록한다.
40개에서 4개를 고르는 gradient selector와 생성량이 같다고 설명하지 않는다.

**측정·판정:** 목표별 도달 시간, 학습곡선, 유효 group 비율, 생성·추정·학습 비용.
더 나은 방법이 조건별로 달라지면 그 경계를 결과로 삼는다.
N01보다 구현 변경이 크므로 독립 구현의 결과는 별도 대조로 보고하고, H100 파일럿 비용을 먼저 확인한다.

## N07. 난도·다양성·재노출의 영향을 분리하기

**참고:** [Lev의 혼합 구성 통제](https://arxiv.org/html/2609.32484v1),
[FAC의 무작위 feature 대조](https://arxiv.org/html/2602.10388v4),
[Saturated Data의 신호 크기와 내용 대조](https://arxiv.org/html/2609.33126v1).

**1단계, 관측:** 기존 선택 이력에서 고유 문제 수, 재선택 빈도, cached/current 성공률,
실제 학습의 mixed-reward group 비율, 문제 유형과 길이를 비교한다.
MATH의 알려진 topic·difficulty를 활용하고 MBPP에 없는 라벨을 임의로 있다고 가정하지 않는다.
캐시 성공률은 현재 난이도의 정답이 아니다.

**2단계, 개입:** 관측 차이가 있는 축 하나만 고정한다. 예를 들어 현재 성공률 구간별 선택 수를
맞춘 뒤 gradient 상위 선택과 구간 내부 무작위 선택을 비교한다. 다른 대조에서는 재노출 횟수나
주제 구성을 맞춘다. 모든 축을 한꺼번에 맞춰 비교 가능한 후보가 사라지는 설계는 피한다.
같은 checkpoint·optimizer에서 같은 크기의 batch로 짧게 학습한다.

**측정·판정:** 조건을 맞추기 전후의 reward 증가 차이, 실제 group 신호, 선택 분포와 비용.
난도를 맞추어도 gradient 이득이 남으면 난도 외 정보의 기여를 지지한다.
차이가 줄어들면 그 축이 유력한 설명이지만 전체 효과의 유일한 원인으로 단정하지 않는다.
Matching에 성공하지 못한 비율과 제외된 후보도 보고한다.

**기존과 차이:** 현재 shuffle은 점수-문제 연결 전체를 끊는다.
이 실험은 난도나 노출량 같은 한 축을 맞춘 상태에서도 추가 방향 정보가 유효한지 본다.
Current 성공률을 맞추기 위한 probe 비용은 진단 비용이며 배포용 selector 비용과 분리한다.

## N08. 평균 뒤의 문제별 개선과 해결 범위

**참고:** [Sampling SFT의 pass@k·능력 보존 분석](https://arxiv.org/html/2610.02140v1),
[Quality-Utility의 지표와 실제 효용 구분](https://arxiv.org/html/2606.16152v1).

**질문:** 비슷한 평균 reward가 같은 문제를 잘 푼다는 뜻인가?
Switch가 이미 풀던 문제의 성공 확률만 올리는지, 풀 수 있는 문제 범위도 바꾸는지 확인한다.

**설계:** 동일 평가 문제에서 방법별 정답률 변화를 짝지어 비교하고,
개선된 문제·퇴행한 문제·변화가 작은 문제의 분포를 보여 준다.
문제당 원시 binary 응답 결과와 sampling 조건이 보존돼 있으면 같은 응답 예산의
평균 정답률과 pass@k를 별도로 계산한다. 집계값만으로 지원하지 않는 k의 결과를 만들지 않는다.
필요한 원시 기록이 없으면 새 학습 대신 보존 checkpoint의 평가만 추가한다.

**측정·판정:** 문제별 성공률 변화, 주제별 손익, 관측 가능한 k 범위의 해결 비율.
기존 평가와 분리된 OOD 평가를 확장할 경우 모든 방법을 같은 조건으로 평가한다.
유한 횟수의 실패는 능력이 없다는 증명이 아니며, 작은 차이의 문제를 확정적인 획득·망각으로 분류하지 않는다.
반복 응답의 표본 오차와 training-seed 변동을 구분한다.

**역할:** 새 학습을 많이 늘리지 않고 성능 차이를 설명하는 보조 분석이다.
Switch의 학습 단계별 원인을 이 분석만으로 입증하지는 않는다.

## 공통 비용·평가 규칙

- 비교의 기본값은 기존 모델·GRPO·seed 5-9·MATH/MBPP다. 새 seed 반복을 먼저 늘리지 않는다.
  Pilot seed·상태는 실행 전에 정하고 유리한 결과만 골라 본 실험에 포함하지 않는다.
- 학습 상태를 분기할 때 모델뿐 아니라 optimizer 상태·결정적 rollout seed·후보/동점 순서를 보존한다.
  기존 constant-learning-rate GRPO를 유지하며 별도 scheduler를 추가하지 않는다.
  학습용 rollout은 각 분기의 현재 policy에서 새로 생성한다. 진단용 고정 응답과 구분한다.
- 목표 성능, 평가 간격, 비교 예산은 결과를 보기 전에 정한다. 최종 test로 threshold를 조정하지 않는다.
  목표 미도달과 예산 미완료는 그대로 표시한다. 기록 없는 시점의 성능을 임의 보간하지 않는다.
- SR cache 최초 생성·채점·저장, 후보/query scoring rollout, gradient/feature 계산,
  ranking, 실제 학습, 평가, checkpoint I/O의 계측 범위를 명시한다.
- Cold-start 총비용과 이미 준비된 cache 재사용 비용을 나란히 보고한다.
  공유 준비 비용을 실행 자원 집계에서는 한 번만 세고, 단독 배포 비용에는 필요한 준비를 포함한다.
  포함 관계인 phase와 세부 stage를 중복 합산하지 않는다.
- 배포용 실행은 단일 reference를 쓴다. 별도 B·centering·optimizer probe·reference 반복은
  연구 진단 비용으로 구분하며, 진단 시간이 무료라는 뜻은 아니다.
- GPU 수와 점유 시간, wall-clock과 GPU-hours를 구분한다. H100 실측 전 예상 완료 시간을 단정하지 않는다.
  누락된 timer는 0이 아니며 평가 완료와 비용 계측 완료도 별개다.
- 데이터셋 안에서 같은 seed의 방법 간 차이를 먼저 계산한다. 같은 seed 번호라고 MATH와 MBPP를
  대응 표본으로 취급하지 않는다. 문제/응답 반복을 독립 training seed로 세지 않는다.

## 기존 실험과 변경 범위

- [진행 중인 mechanism E14-E16](STAGE_MECHANISM_EXPERIMENTS_2026-10-06.md)은 유지한다.
  Shuffle, fresh-SR, 단계별 실제 학습 효과를 새 아이디어로 다시 실행하지 않는다.
- Qwen 모델 확장, matched cache refresh, 재전환 등은 [기존 실행 안내](LIMITATION_EXPERIMENTS_KO.md)에 있다.
  N01-N08과 같은 이름이나 완료 상태를 붙이지 않는다.
- Fixed200 채택 제외와 동일-prefix replicate의 보조 위치는 유지한다. 원본 결과는 삭제하지 않는다.
- 신규 코드는 별도 실험 식별자·결과 경로에 구현했다. 기존 환경·queue·cache·checkpoint와
  V6/V7 원고·PDF·웹 게시본은 변경하지 않는다.
- CPU 검증과 실제 GPU 실험 완료를 구분한다. 원고에 신규 성능·시간 값을 넣은 상태가 아니다.
