# 저장된 rollout·보상 곡선 분석과 이후 실행의 기록

2026-10-16. 분석 코드는 CPU에서 저장 파일만 읽는다. 학습·생성·평가를 시작하거나
checkpoint를 역직렬화하지 않는다. 완료된 Qwen 실험에서 누락된 중간 평가를
최종 보상으로 복원하지 않는다. 학습 배치 보상과 고정 held-out 평가를 구분한다.

## 분석 실행

저장한 공통 results 파일을 입력한다. 모델별 런처의 `results` 출력은 Qwen,
Gemma, Llama 모두 지원한다. 입력 파일과 출력 디렉터리는 별도 경로다.
기존 출력 디렉터리는 덮어쓰지 않는다.

```bash
python3 -m srgc_research.stage_rollout_analysis \
  /path/qwen35-results.json \
  --output-dir /path/new-qwen-analysis
```

여러 모델 결과, MATH/MBPP legacy `source_results` export, 실제
`stage_mechanism` endpoint 또는 영구 저장된 `srgc-rollout-trace-v1` JSON/JSON.gz도
입력할 수 있다. 같은 데이터를 담은 results와 원시 trace를 함께 넣으면 중복으로
거부한다. 모델·seed·plan·입력·구현이 다른 실행은 cohort를 나눠 집계한다.
원래 모델 정보가 없을 때만 `--model`로 명시하며, 기록된 정보와 충돌하면 거부한다.
legacy 반복 실행과 fixed schedule은 주 실험 네 arm에 합치지 않는다.
이전 분석의 요약 JSON은 원시 endpoint를 대신하는 입력이 아니다.

```bash
python3 -m srgc_research.stage_rollout_analysis \
  /path/results.json /path/results_mbpp.json \
  --output-dir /path/new-olmo-analysis
```

`--plots`는 `requirements-switch-plots.txt`의 Matplotlib이 추가로 필요하다.
측정된 점만 PNG/PDF로 표시하며, 최종점만 있으면 최종점만 그린다.
학습 보상과 독립 평가 보상은 별도 패널이다. 원시 token ID는 JSON.gz를 직접
분석할 때 사용한다. compact results에는 응답별 보상과 길이만 보존되므로
그 입력만으로 어휘 다양성을 계산하지 않는다.

| 출력 | 실제 기록에서 계산하는 내용 |
|---|---|
| `coverage.csv` | 학습/평가 기록 step, 25-step 평가 누락, 고유 문제 수, Switch 경계 |
| `training_rewards.csv` | 각 update에 실제 사용한 학습 배치의 평균 보상 |
| `held_out_rewards.csv` | 실제 측정한 고정 held-out 평가 보상 |
| `groups.csv` | 문제별 8응답의 정답 수, all-wrong/mixed/all-correct, 길이, 캐시와 현재 성공률 차이 |
| `windows25.csv` | 25-update 구간별 정답 0~8개 히스토그램, mixed 비율, 길이와 cache gap |
| `switch_checks.csv` | 저장된 checkpoint별 SR-GC contrast `d` |
| `selections.csv` | 저장된 history의 학습 문제 ID·selector·후보 ID |
| `analysis.json`, `README.md` | 전체 결과, 입력 SHA-256, 누락과 해석 제한 |

문제 노출 횟수와 서로 다른 문제 수를 구분한다. prompt가 재등장해도 서로 다른
update의 실제 응답은 별도 group이다. 재시도·겹치는 export는 합산하지 않는다.
공통 prefix가 여러 arm export에 동일하게 들어간 경우에만 중복을 제거한다.
문제당 응답 수가 다르면 히스토그램 분모를 분리한다. token ID가 없으면 길이·
어휘 다양성은 미측정이다. 어휘 차이를 의미적 다양성이라고 부르지 않는다.

`switched_at=τ`는 τ 업데이트 완료 상태에서 판단한 경계이며 첫 SR 학습은
τ+1이다. 각 group/보상 점에 전환까지의 상대 step을 붙인다. 다른 arm도 같은
cohort의 Switch 경계에 맞춰 비교하되, 그 arm이 실제로 전환했다는 뜻은 아니다.
메커니즘 실험의 `anchor_step`은 분기 시작 상태이고 학습 update는 분기 내부
번호다. 원래 주 학습의 global step으로 합치지 않는다. 이 분석은 관찰 분석이며
난도나 gradient alignment가 성능 개선의 원인임을 입증하지 않는다.

## 기존 실행의 세부 GPU 시간 내보내기

`results`는 저장된 비용 원장을 읽어서 arm별 `exclusive_stages`와 `counts`도
JSON에 보존한다. 생성·forward·backward·gradient projection·통신·대기 시간을
각각 GPU-seconds 그대로 내보내며, 이미 기록된 응답 수와 생성 토큰 수도 유지한다.
계획에 기록된 생성/gradient microbatch·logit chunk·checkpointing·attention 설정과
endpoint에 실제 저장된 `checkpoint_policy`도 보존한다. 설정이 기록되지 않았으면
기본값으로 채우지 않는다.

```bash
bash scripts/run_srgc_qwen35.sh all results
bash scripts/run_srgc_llama31.sh all results
bash scripts/run_srgc_gemma4.sh all results
```

기존 공통 결과 파일 `$OM_WORK/results/{qwen35,llama31,gemma4}-results.json`을
다시 내보내면 된다. 이 작업은 CPU에서 원장을 읽고 결과 파일만 갱신하며 학습·생성·
평가를 실행하거나 cache·checkpoint·진행 중인 task를 변경하지 않는다. 이미 완료된
실험도 원장이 남아 있으면 재학습 없이 상세 비용을 복구할 수 있다.

`costs[].arms.<arm>.exclusive_stages`는 측정된 항목이며 `counts`는 그 항목의
기록된 작업량이다. 준비·평가·저장·통신·대기 등을 나눠 동일한 **비용 항목 범위**를
비교할 수 있지만, 생성 배치 차이에 따른 반사실적 실행 시간을 이 값만으로
계산할 수 있다는 뜻은 아니다. `PARTIAL`/null은 그대로 유지한다. 기록이 없으면
0이나 추정값으로 대체하지 않으며, 전체 GPU 시간과 단계별 GPU 시간을 중복 합산하지 않는다.

### status에서 바로 확인

Qwen·Llama·Gemma의 `status`도 같은 원장을 직접 읽는다. `results`를 먼저 만들거나
학습·평가를 다시 실행할 필요가 없다. 기존 GPU TIMES 표 아래에 다음을 표시한다.

- `RECORDED PLAN SETTINGS`: 계획에 실제 저장된 GPU 수, 생성/gradient microbatch,
  logit chunk, checkpointing, attention, 생성 길이 제한, 문제당 응답 수.
- `GPU STAGES`: seed·arm·phase/section별 생성, forward, backward, projection, 통신,
  잔여 시간과 그 밖의 기록된 단계의 GPU-hours. 선별의 candidate/reference,
  학습, 평가를 서로 합치지 않는다. `Comm`에는 gradient reduction을 포함하고
  `Residual`은 `unattributed_and_wait`다. 잔여 시간을 측정된 GPU idle로 해석하지 않는다.
- `RECORDED WORK`: 같은 범위에서 기록된 prompt·응답·생성 token·forward/backward
  응답 수. prompt 수는 중복 노출을 포함하며 고유 데이터 개수가 아니다.

`status --json`의 각 task `timing`은 원래 `exclusive_stages`(GPU-seconds,
rank-wall seconds, calls)와 `counts`를 보존하고 `exclusive_stage_gpu_hours`를
함께 제공한다. dataset의 `recorded_execution_settings`는 기록된 계획 설정이다.
설정이 없는 항목은 현재 코드의 기본값으로 채우지 않는다.
부분 측정은 `PARTIAL`/`>=`, 미측정 단계는 `-`로 표시한다. 세부 단계 기록이 없는
과거 원장은 합계만 표시하며, 일부 세부 기록이 있다고 모든 phase의 세부 시간이
측정됐다고 간주하지 않는다. 상세 표를 기존 합계에 다시 더하지 않는다.
이 조회는 원장·입력·checkpoint·학습 상태를 쓰거나 기존 runtime identity를 바꾸지 않는다.

## 이후 실행에서 보존하는 파일

공개 런처 `qwen_run`, `gemma_run`, `llama_run`의 training child는 공통
recording engine을 사용한다. Qwen의 기존 rank별 컴파일 캐시 분리도 유지한다.
cache/smoke 단계와 동결된 learner·모델 adapter·rank 코드는 변경하지 않는다.
각 seed 폴더 아래에 저장한다.

| 경로 | 언제·어떻게 기록하는가 |
|---|---|
| `reward-curves/<arm>/protocol.json` | seed·plan·입력·구현·기록 확장 hash와 고정 평가 ID를 결합 |
| `reward-curves/<arm>/training.json` | 매 업데이트의 실제 `sample_reward`와 25-update 평균 |
| `reward-curves/<arm>/rollouts/step-N.json.gz` | 성공한 업데이트마다 실제 생성한 학습·선별·reference 응답의 binary reward와 response-only token ID |
| `reward-curves/prefix/step-25.json`, `.pt` | 공통 prefix의 고정 held-out 평가 및 실제 가중치 |
| `reward-curves/<arm>/step-N.json`, `.pt` | 50, 75, …, 250 업데이트의 고정 held-out 평가 및 가중치 |
| `reward-curves/<arm>/step-275.pt` | 최종 가중치; 275 평가는 기존 최종 endpoint를 사용 |
| `<arm>-latest.pt` | optimizer와 selector 상태를 포함한 학습 재시작 checkpoint |

학습 rollout은 매 업데이트 4문제×8응답이다. On-policy 선별이 갱신되는 경계에서는
40개 후보와 선별용 reference 50개에 각각 8응답을 생성한 기록도 보존한다.
선별한 4문제 ID는 다음 25-update 갱신까지 사용하지만 학습 응답은 매 update
새로 생성한다. Switch의 SR-GC check에서는 SR 비교 40문제를 추가하되 후보와
겹친 문제는 한 번만 계산한다. 선별용 reference 50개는 독립 평가 300개와 다르다.
SR·Random에는 발생하지 않은
gradient 선별 응답을 새로 만들지 않는다. 초기 SR cache 성공률은 입력 bundle의
400문제별 8개 binary rewards에서 계산한다. 현재 성공률은 그 시점에 실제 생성한
응답에서 별도로 계산한다. 다른 phase의 응답을 학습 응답으로 취급하지 않는다.

보존용 JSON.gz는 재시작 캐시와 별도이며 완료 후 삭제하지 않는다. 응답·보상·
학습 배치 기록을 저장한 다음 재시작 checkpoint를 전진시킨다. 저장 실패로
checkpoint만 앞서 나가는 것을 막는다. 동일한 완료 기록은 재사용하고 다른
응답이 같은 step에 들어오면 덮어쓰지 않고 오류를 낸다. 중단된 update 자체는
기존 checkpoint에서 다시 실행한다. 평가-only `.pt`는 optimizer를 제거했으므로
학습 재시작에는 `<arm>-latest.pt`를 사용한다.

중간 평가는 같은 300개 held-out ID에 문제당 8응답을 사용한다. 고정된 별도
`reporting-evaluation` seed stream을 쓰며 Python/NumPy/Torch RNG와 model mode를
보존한다. 추가 평가·평가 checkpoint 저장, rollout 보존은 별도 cost receipt에
기록한다. generation/backward를 추가 호출해 원시 학습 응답을 재생성하지 않는다.
실제 GPU에서의 시간·메모리 영향은 별도로 측정해야 한다.

측정 확장 hash가 다른 기존 디렉터리에 자동으로 새 기록을 섞지 않는다.
진행 중이거나 보관된 실험은 당시 runtime을 유지한다. 새 실행에 이 코드를
사용해도 이미 지나간 step의 평가나 응답이 자동으로 생기지는 않는다.
과거 가중치가 남아 있다면 이후 재평가는 가능하지만, 원래 실행 중 측정한 값과
구분해서 기록해야 한다.

## 이번 검증 범위

CPU 시험에서 실제 반환된 응답 캡처, 완료된 재시작 캐시 삭제 후 영구 기록 유지,
중간 재시작, 기록 실패 시 checkpoint 전진 방지, 오염된 reward/token/protocol 거부,
세 모델의 compact export, RNG를 사용하는 학습 상태 보존을 검사한다.
4개 CPU rank의 Gloo 통신으로 각 rank의 학습 group이 빠짐없이 하나의 영구
기록에 모이는지도 검사한다.
이는 H100에서 전체 실험을 새로 수행했다는 뜻이 아니다.

검증 결과: research 회귀 629개 통과 후 최종 수정 범위 101개와 추가 중단 복구
검사 1개를 통과했다. 수정 파일 Ruff·공백 검사를 통과했고 기존 동결 대상
339개 파일은 수정 전과 내용이 일치한다. Qwen/Gemma/Llama의 adapter hash도 유지했다.

받아둔 **OLMo** MATH/MBPP legacy 결과 두 파일에 분석기를 실행한 결과:
주 실험 최종 held-out 40개, SR-GC check 29개, 학습 보상 0개, 원시 group 0개다.
Qwen/Gemma/Llama의 새 결과로 보고하지 않는다. 입력 SHA-256:

```text
MATH  7718fdcf0f22475d8b9dd931036bc4cf476515f89cfe90569dc2f414f1fa422c
MBPP  2c4bc214b5af506ea3e46228da6614b9b54ee6430625b955e405738db56b1d01
```

그 파일들에서는 25~250 held-out 보상과 난도 분포를 복원하지 않았다.
현재 Qwen에서 누락된 곡선이 존재한다고 가정해 분석할 수 있다는 이전 설명은
잘못이었다. 결과 export에 원시 기록이 있는지와 실제 측정됐는지를 먼저 확인한다.
