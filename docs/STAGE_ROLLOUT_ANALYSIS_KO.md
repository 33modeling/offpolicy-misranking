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

## 노드 강제 종료 후 이어 실행 (2026-10-17)

Qwen·Llama·Gemma 공개 런처는 기존 plan과 seed 폴더를 그대로 사용한다.
새 노드를 할당받은 후 **같은 실행 명령**을 다시 실행하면, 기존 체크포인트가
있는 미완료 작업을 우선 가져와 마지막으로 완전히 저장한 update 다음부터
이어 실행한다. 실행 중인 노드 자체를 다시 할당하는 기능은 아니다.

- `<arm>-latest.pt`와 `prefix-latest.pt`에는 모델·optimizer·선별된 문제,
  이미 학습 배치에 사용한 문제 ID·sampling cycle·SR-GC 시간 창과 Switch 경계가
  함께 저장된다. 저장이 끝나기 전 중단된 update만 다시 수행한다.
- 임시 `.tmp` 저장 도중 강제 종료돼도 기존 `.pt`를 먼저 덮어쓰지 않는다.
  재시작 시 완성된 `.pt`를 읽고 다음 정상 저장에서 남은 임시 파일을 교체한다.
- SIGKILL(-9/137), SIGTERM, 인터럽트 및 torchrun이 보고한 root rank의
  SIGKILL은 `interrupted`로 남긴다. 다른 종료가 ERROR로 기록되더라도
  누적 횟수로 재개를 차단하지 않는다.
  현재 시도의 로그에 명시적인 OOM이 있으면 실제 실패로 남긴다. 과거 시도의
  로그를 현재 오류 판정에 섞지 않는다. 입력/증거 검증 오류도 실패로 유지한다.
- 부모가 죽었어도 자식이 실행 잠금을 보유하면 다른 worker가 같은 작업을
  중복 실행하지 않는다. 이미 완료한 작업과 저장된 평가 응답은 재사용한다.
- status의 `RESUME`는 복구 가능한 중단 작업이다. 60초 넘게 heartbeat가
  없으면 `STALE`로 경고하고, **15분 넘었으며 실행·GPU 잠금의 소유 증거가 없는
  worker 표시**는 일반 목록에서 숨긴다. 최근 stale 및 잠금이 살아 있는 작업은
  유지한다. 오래된 종료 세션에도 같은 보존 기간을 적용한다. 재할당된 GPU의
  새 세션 때문에 이전의 종료 세션을 계속 살아 있다고 판단하지 않는다.
- status는 원본 worker receipt·시도 이력·로그·체크포인트·보상 기록을
  삭제하거나 수정하지 않는다. JSON의 `worker_history`와 화면의 `HISTORY`에
  숨긴 수를 표시한다. 세 모델에 같은 기준을 적용한다.

CPU에서 실제 프로세스를 checkpoint 임시 저장 도중 SIGKILL로 종료해
prefix·Random·SR·On-policy·Switch의 복구를 검사했다. 마지막 `.pt` 보존,
optimizer 복원, 중복 없는 update history와 선별/Switch 상태가 중단 없는
실행과 일치한다. 세 모델의 반복 SIGKILL·torchrun rank 종료·OOM 재시도 및
읽기 전용 status 검사도 포함한다. GPU 재학습이나 이미 실행 중인 프로세스의
코드를 교체한 검증은 아니다. frozen learner/adapter/recording hash는 유지한다.

관련 회귀 237개 통과 후 공유 worker의 두 dataset heartbeat 차이를 추가
보완하고 status 검사 51개를 통과했다. 수정 Python 파일의 Ruff 검사와
공백 검사를 통과했다. 세 모델 adapter·공통 engine·보상 기록 확장 hash는
수정 전 commit과 직접 비교해 일치함을 확인했다.

### 기존 ERROR 작업부터 복구 — 2026-10-18 수정

`ERROR x4`의 4는 누적 실행 시도 번호다. 서버의 GPU 강제 종료도 오류로
기록될 수 있으므로 **3회 등 누적 실패 횟수로 재개를 막는 조건을 제거했다.**
현재 Qwen·Llama·Gemma 공개 dispatcher는 횟수에 관계없이 미완료 작업을
체크포인트부터 계속 재개한다. `--max-attempts`는 이전 CLI와의 호환을 위해
남아 있지만 현재 운영 실행기는 이 값을 재개 제한으로 사용하지 않는다.
과거 복구 예산 파일도 보존하되 실행 조건으로 읽거나 초기화하지 않는다.

상위 연결 실행기가 살아 있고 자식 worker만 SIGKILL/137로 종료됐다면
연결 실행기를 종료하지 않고 저장한 작업을 다시 실행한다. `nvidia-smi`
조회 실패도 GPU가 다시 사용 가능해질 때까지 재확인한다. Ctrl-C·명시적
중지·입력/모델 identity 오류는 GPU 강제 종료와 구분한다.

현재 데이터셋에서 **선택된 모든 모델의 중단 작업 복구 → 새 작업** 순서다.
복구 대상은 작은 task receipt와 checkpoint 파일의 존재/수정 시각으로 먼저
찾고, 실제 실행기는 기존 입력·모델·checkpoint identity를 검증한다.
가중치를 역직렬화하거나 CUDA를 초기화해서 우선순위를 조사하지 않는다.
현재 실행 중인 다른 worker가 소유한 작업은 중단 작업으로 취급하지 않는다.
한 모델에서 복구를 끝내면 다른 모델의 멈춘 작업을 먼저 확인하며 새 seed는
전체 복구 대상 확인 후 시작한다. 실패 작업의 체크포인트가 있으면
그 상태에서 이어 실행한다. dependency가 아직 없으면 필요한 선행 작업부터
복구하며, 다른 worker가 실행 잠금을 보유한 작업은 중복 실행하지 않는다.
반복 오류로 즉시 재실행이 폭주하지 않도록 기존 재시도 간격(기본 60초)은
유지한다. 대기 중에도 다른 미완료 작업은 복구할 수 있지만 새 작업이 그
작업을 추월하지 않는다. 횟수를 초과했다는 이유로 ERROR에 고정되거나 모델을
종료하지 않는다. 과거 ERROR 표시는 실제 작업을 잡으면 RUN으로 바뀐다.
실패 이력·로그·보상·checkpoint는 삭제하지 않는다. status 조회 자체가
학습을 시작하거나 예전 실패 기록을 바꾸지는 않는다.

### MATH 전체 완료 후 MBPP

공개 연결 실행은 선택된 모델 전부의 MATH를 완료한 다음 MBPP를 시작한다.
기본 Qwen 시작 순서는 Qwen MATH → Llama MATH → Gemma MATH → Qwen MBPP →
Llama MBPP → Gemma MBPP다. 다른 시작 모델이나 명시적 모델 순서는 각 단계
안에서 유지한다. MATH가 다른 노드에서 실행 중이거나 재시도 대기 중이면
다른 모델의 MATH를 확인한 뒤 기다리며 MBPP로 우회하지 않는다.
MATH 오류가 재시도 중이거나 일시 중지됐을 때도 MBPP는 시작하지 않는다.

단계 전환 시 기존 plan·task 집합·실패 이력·체크포인트를 유지한다.
MATH의 최종 평가와 endpoint 기록이 끝나고 실행 잠금이 해제돼야 넘어간다.
`math` 명령은 MBPP의 오래된 오류를 대신 실행하지 않는다. `mbpp` 명령은
기존에 준비한 MATH sibling부터 완료하며, plan이 없으면 명시적인 오류로
중단한다. 기존에 실행 중인 프로세스에 코드를 주입하거나 원격 작업을
중단한 것은 아니다. 자세한 명령과 상태 표시는 [모델 연결 실행](SRGC_MODEL_CHAIN.md)을 따른다.

## 기존 보상 기록 확장 검증

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
