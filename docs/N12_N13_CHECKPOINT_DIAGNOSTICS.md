# N12·N13 구현과 기존 checkpoint 재사용 — 2026-10-18

N12의 저장 rollout 표준화·고정 20문제 패널과 N13의 40대40/선택 4대4 비교를
`srgc_research/followup/`에 구현했다. CPU 계산·실제 소형 LoRA gradient·중단 후 재개를
검증했다. H100에서 새 측정을 실행한 상태는 아니다. 기존 learner, 세 모델 adapter,
진행 중인 queue, 원본 입력·checkpoint·결과와 frozen runtime은 변경하지 않는다.

## 2026-10-18 재디버깅

- **종료 직전 중단:** `costs.json`을 저장한 뒤 endpoint 전에 종료하면, 다음 attempt의 비용이
  늘어 기존 immutable 비용 파일과 충돌하던 버그를 재현·수정했다. 미완 작업의 파생 비용
  snapshot만 실행 잠금 안에서 갱신하고 완료 endpoint에는 최종 SHA를 기록한다.
  성공한 응답·A/B vector는 재사용한다. NCCL 정리 오류도 원래 실패 원인을 덮어쓰지 않는다.
- **여러 rank의 시작 결정:** 원본 읽기·완료 확인·사본 보존을 rank 0이 실행 잠금 안에서
  수행하고 torchrun TCPStore로 모든 rank에 같은 결정/오류를 전달한다. 각 rank는 같은
  고정 사본과 코드 hash를 검사한 뒤 NCCL을 초기화한다. 완료 확인에는 CUDA가 필요 없다.
- **같은 weights의 시점 누락:** t125와 t250 또는 여러 arm이 같은 정책을 공유하면 측정은
  한 번 재사용하고 `contexts/*.json`에 각각의 실제 stage·arm·checkpoint SHA를 남긴다.
  `n12-panels`는 이 목록을 자동으로 읽고 시점별 행을 유지하며 응답·비용을 중복 계상하지 않는다.
  attention kernel이 다른 측정은 별도 identity·집계로 유지한다.
- **잘못된 완료 표시:** 필수 source/phase 파일, 파일 SHA, 실제 vector/panel 결과와 endpoint
  요약, 비용 coverage가 맞아야 완료로 처리한다. 일부 checksum 항목만 남긴 endpoint나
  원시 기록과 다른 D4 요약은 GPU를 로드하기 전에 거부한다.
- **집계와 출처:** 학습 group의 재추출 draw·중복 문제를 별도 적용 문제로 세지 않는다.
  progress/endpoint에 중복된 같은 SR-GC check는 한 번만 세고 서로 다른 D는 거부한다.
  보상이 같아도 prefix lineage가 다른 평가 곡선을 연결하지 않는다. JSON hash는 실제 읽은
  payload에 묶는다. 잘못된 attention/config/scoring 이름과 생성용 prompt 누락도 미리 검사한다.
- **Gemma 의존성 순서:** 기존 Gemma의 optional media 비활성화를 PEFT import보다 먼저
  적용한다. 불필요한 media 라이브러리의 ABI 오류가 먼저 발생할 수 있던 순서를 수정했다.

재디버깅 최종 검증: **175 tests 통과**, Ruff·diff 검사 통과. 실제 소형 LoRA gradient,
완료 직전 중단과 정리 실패, 원시 결과/완료 요약 불일치, 동일 weights의 다중 context,
4-worker CPU TCPStore의 완료·rank 0 오류·다른 rank 오류 전달을 검사했다.
기존 learner·모델 adapter·queue·원시 기록 소스는 `5ead62e`와 동일하다.
이 검증은 실제 H100/NCCL 실험 완료를 의미하지 않는다.

이 수정은 별도 진단 코드에 한정한다. 신규 measurement hash가 달라지므로 이전 진단
artifact를 새 측정으로 덮어쓰지 않는다. 기존 학습/기록 runtime과 원본 실험은 그대로다.

## 기존 checkpoint를 쓸 수 있나?

**저장된 가중치가 맞으면 두 실험 모두 학습을 처음부터 다시 돌릴 필요가 없다.**
N12 고정 패널은 추론만, N13은 추론·gradient scoring만 한다. Adam update는 하지 않는다.
따라서 optimizer가 없는 평가용 archive도 사용할 수 있다. 기존 일반 backend loader가
optimizer를 요구하므로 새 진단 전용 `load_weights()`를 사용한다. 이 backend에서는
`train()` 호출을 막는다. optimizer 없는 파일을 학습 재개용으로 취급하지 않는다.

| 저장 자료 | N12 | N13 | 확인할 점 |
| --- | --- | --- | --- |
| 기존 `reward-curves/<arm>/rollouts/step-*.json.gz` 또는 compact `rollout_groups` + 전체 seed input/cache | 새 생성 없이 25-update 구성 표준화 | 보상·token만으로 D4 복원 불가 | 실제 학습 phase, 적용 ID, 전체 pool 초기 cache, 출처 hash |
| `prefix.pt`, `<arm>-latest.pt` + 같은 seed의 `run.json` | 고정 패널 응답 생성 가능 | 같은 상태·후보에서 새 scoring 가능 | 파일 안의 실제 step. `latest`는 이전 t125/t250을 보존한다는 뜻이 아님 |
| `reward-curves/<arm>/step-{25,125,250}.pt` + 같은 폴더 `protocol.json` | 가능 | 가능 | 실제 파일명은 `step-125.pt` 등. optimizer가 제거된 `evaluation_only` archive |
| `checkpoint-backups/.../<sha>/checkpoint.pt` + `receipt.json` | 가능 | 가능 | checksum·입력·plan·seed 검증. 기존 backup은 원본당 최신 **2세대**만 유지하므로 모든 중간 시점이 남는 것은 아님 |
| StageStudy의 저장된 `anchor` | Engine 계약과 출처가 맞으면 가능 | 같은 조건에서 가능 | branch 종료 weights와 anchor를 혼동하지 않음 |
| 실제 문제별 projected gradient + ID/reference/정책·측정 출처 | 해당 없음 | CPU 재계산 가능 | 새 vector schema에 원래 C40/S40/reference50를 보존. scalar D/cosine으로 vector를 만들어 넣지 않음 |
| 과거 Pair의 `checkpoint-000xxx` 디렉터리 | 이 loader에서 지원하지 않음 | 이 loader에서 지원하지 않음 | sampling/저장 계약이 다름. 현재 Engine `.pt`로 자동 간주하지 않음 |

평가용 archive는 기존 기록 확장이 적용된 이후 저장된 시점에만 존재한다. 코드가 파일을
저장하도록 되어 있다는 사실과 실제 서버에 파일이 남아 있다는 사실을 구분한다.

### 이번에 직접 확인한 범위

- 저장 코드에서 평가용 archive가 `trainable`, projection, selector/history를 남기고
  `optimizer`만 제거하는 것을 확인했다. 원본 SHA가 유지되는 실제 `.pt` fixture로
  평가 archive·일반 Engine·checksum backup의 읽기와 optimizer 미변경을 검사했다.
- `/home/kms/Downloads`, `/home/kms/research`, 코드 저장소와 작업 디렉터리를 조회했으나
  해당 실험의 실제 `.pt`는 발견되지 않았다. 현재 환경에는 `/group-volume` 마운트와
  이 실험 서버 SSH 접속 설정이 없다. **서버의 t25/t125/t250 잔존 여부는 아직 미확인이다.**
- 로컬 수신 OLMo MATH/MBPP 결과 JSON의 실제 Switch check를 coverage 명령으로
  읽었다. 기존 check **29개**의 D는 있으나 이 export의 D4는 모두 미측정이다.
  해당 compact export에 projected vector가 없다는 확인이며, 서버 원시
  checkpoint/gradient 파일 전체가 없다는 결론이 아니다.

수신 파일 SHA256은 MATH `7718fdcf0f22475d8b9dd931036bc4cf476515f89cfe90569dc2f414f1fa422c`,
MBPP `2c4bc214b5af506ea3e46228da6614b9b54ee6430625b955e405738db56b1d01`이다.
로컬 coverage 결과는
`/home/kms/reports/srgc-n12-n13-2026-10-18/received-check-coverage/report.json`에 보존했다.

## 먼저 서버에서 파일 점검

원래 실험과 같은 Python 환경에서 실행한다. `audit`는 CPU에서 `.pt`를 읽으며
기본 모델이나 GPU를 로드하지 않는다. 아래 변수는 실제 원본 경로로 지정한다.
seed마다 해당 seed의 입력을 사용한다. pool 크기는 manifest의 실제 값을 읽으므로
과거 MBPP 512개를 400개로 바꾸지 않는다.

```bash
diag_plan=/group-volume/<user>/<run>/plans/math.json
diag_bundle=/group-volume/<user>/<run>/inputs/math-seed-5.json
diag_seed_root=/group-volume/<user>/<run>/runs/math/seed-5
diag_output=/group-volume/<user>/srgc-n12-n13

python -m srgc_research.followup audit "$diag_seed_root" \
  --plan "$diag_plan" --bundle "$diag_bundle" --seed 5 \
  --output-dir "$diag_output/audit-math-seed-5"
```

추가 backup 경로가 있으면 `audit` 뒤에 경로를 더 적는다. 출력은 `report.json`,
`rows.csv`, `requested_stage_coverage.csv`다. 파일명 대신 실제 저장 step을 확인한다.
명시적인 prefix 파일/프로토콜로 확인한 t25만 공통 prefix로 표시한다. 같은 weights는
`policy_sha256`로 식별한다. input/plan/seed, 전체 후보·reference ID, cached-SR 순위,
sampling, projection, attention과 tensor dtype/shape를 검증한다. 구조 검증 통과는
실제 대형 모델·GPU 환경 로딩 성공을 의미하지 않으며 별도 필드로 구분한다.

## N12: 기존 rollout 분석

```bash
python -m srgc_research.followup n12-report /path/model-results.json \
  --bundle "$diag_bundle" --output-dir "$diag_output/n12-math-seed-5"
```

또는 compact export 대신 원래 `reward-curves/<arm>/rollouts/step-*.json.gz`를 입력한다.
같은 group이 든 compact export와 원시 trace를 함께 넣지 않는다. raw trace에 모델/과제
표기가 없을 때만 `--model`, `--dataset`으로 명시한다. 입력 SHA가 다른 seed/cohort는
제외 목록으로 내보내며 합치지 않는다.

계산 규칙:

1. 실제 학습 group을 1–25, 26–50, …의 창으로 집계한다. continuation은 t26부터다.
   학습에 반영된 group의 반복 노출 수와 고유 문제 수를 따로 센다. 적용 ID 기록이
   있으면 group이 해당 batch에 속하는지도 검사한다.
2. 초기 cache 8응답의 정답 수를 0/8, 1–3/8, 4/8, 5–7/8, 8/8로 고정한다.
   이는 실제 난도 label이 아니다. seed 전체 후보 pool의 고정 비중으로 mixed 비율,
   성공률, 응답 길이와 0–8 정답 histogram을 표준화한다.
3. 양의 pool 비중인 stratum의 기록이 없으면 표준화 전체 값은 `null`이다.
   일부 strata만 재정규화하지 않는다. 길이 기록 누락 역시 `null`이다.
   실제 `subject`/`topic`/`level`이 있으면 해당 label 안의 같은 집계를 추가한다.
4. 창 전체가 누락됐더라도 원래 horizon이 알려졌으면 빈 창을 남긴다. 한 창의
   25 updates×4 groups가 모두 기록됐는지 `complete_training_window`로 표시한다.
5. `windows_25.csv`, `training_rewards.csv`, `held_out_rewards.csv`, `report.json`을
   분리한다. 실제 평가 기록만 내보내며 학습 reward로 held-out curve를 채우지 않는다.

## N12: 같은 20문제의 정책 변화

```bash
python -m srgc_research.followup n12-panel --bundle "$diag_bundle" --seed 5 \
  --output-dir "$diag_output/panel-definition-math-seed-5"

torchrun --standalone --nproc_per_node=4 -m srgc_research.followup collect \
  --experiment n12 --checkpoint "$diag_seed_root/prefix.pt" \
  --plan "$diag_plan" --bundle "$diag_bundle" --seed 5 --output-root "$diag_output/measurements"
```

별도 GPU 할당에서 원래 모델의 Python·package overlay·오프라인 base model 환경을
사용한다. OLMo/Qwen3.5/Gemma4/Llama3.1의 기존 모델 loader·LoRA 대상·scoring backend를
원래 plan에 따라 선택하며 모델 확장은 plan의 adapter/engine pin도 검사한다.
pin이 다르면 진행하지 않는다. 새 runtime을 기존 frozen 디렉터리에 덮어쓰지 않는다.

20문제는 전체 candidate pool에서 균등·중복 없이 **한 번만** 뽑는다. panel은
`input_sha256 + seed`에 고정되며 모든 arm과 t25/125/250에서 ID를 재사용한다.
각 정책에서는 문제당 **새 응답 8개**를 생성한다. 학습/선별/held-out 평가에 사용하지 않는다.
공통 t25는 위 prefix 명령 한 번으로 측정한다. 동일 정책 weights와 진단 설정이면
다른 checkpoint 파일명으로 다시 호출해도 완료 artifact를 검증한 뒤 기존 측정을 반환한다.
각 호출의 stage·arm은 `contexts/*.json`에 남겨 동일 weights인 두 시점도 그래프/CSV에서
누락하지 않는다. 응답 수는 acquisition당 160이며 context 수를 곱하지 않는다.

나머지 arm의 t125/t250 `.pt`로 같은 `collect` 명령의 `--checkpoint`만 바꾼다.
`panel-measurement.json.gz`의 응답별 reward·full/response token·초기 stratum을 보존한다.
생성 seed와 정책 hash는 파일에 기록한다. 여러 정책 결과는 다음처럼 모은다.

```bash
python -m srgc_research.followup n12-panels \
  /path/t25/panel-measurement.json.gz /path/t125/panel-measurement.json.gz \
  --bundle "$diag_bundle" --seed 5 --output-dir "$diag_output/n12-panel-comparison"
```

같은 문제의 t25 대비 성공률·길이 변화를 `rows.csv`로 내보낸다. t25 미측정이면
차이는 `null`이다. 동일 정책의 동일 결과는 중복 제거하고 충돌하는 재측정은 거부한다.
응답 수는 `160 × 실제로 다른 측정 정책 수`이며 seed/응답 반복을 독립 seed로 세지 않는다.

## N13: 40대40과 선택 4대4

```bash
torchrun --standalone --nproc_per_node=4 -m srgc_research.followup collect \
  --experiment n13 --checkpoint "$diag_seed_root/reward-curves/on_policy/step-125.pt" \
  --plan "$diag_plan" --bundle "$diag_bundle" --seed 5 --output-root "$diag_output/measurements"

python -m srgc_research.followup n13-report /path/measurement/vectors.json.gz \
  --output-dir "$diag_output/n13-comparison"
```

같은 명령의 `--checkpoint`를 t25/250, 다른 arm의 저장 상태로 바꿔 사용한다.

checkpoint의 원래 후보 추출 stream과 **학습 배치에 사용된 ID 이력**으로 C40과
진단 S40을 preview한다. preview는 원래 상태를 변경하지 않는다. SR의 학습 4개는
S40의 앞 4개나 전체 pool 상위 4개가 아니라 **같은 C40 안**에서 원래 cached-SR
순위가 높은 4개다. 이 새 진단은 전환 후 Switch/SR 상태에서도 실행할 수 있지만,
과거 그 상태에서 실제 SR-GC check가 실행됐다는 뜻으로 표시하지 않는다.

후보/SR 중복 문제는 한 번만 scoring하고 원래 reference50 하나를 사용한다.
A에서 cosine 상위 4개 ON4를 원래 tie stream으로 선택하여 SR4와 비교한다.

```text
D40 = dot(v, mean(g[C40]) - mean(g[S40]))
D4  = dot(v, mean(g[ON4]) - mean(g[SR4]))
```

독립 B도 기본으로 측정하여 **A에서 선택한 같은 ON4/SR4**를 B의 gradient로 비교한다.
B에서 재선별하지 않는다. `--a-only`는 A만 측정하는 명시적 옵션이다.
후보/SR은 원래 candidate response group size, reference는 8응답 group을 사용한다.
문제별 projected vector, reference vector, response token/reward, projection seed·
parameter names, 각 scoring seed를 `score-A/B.json.gz`와 `vectors.json.gz`에 남긴다.

과거 저장 vector를 새 schema로 전달하면 CPU에서 재계산할 수 있다. 새 `collect`는
**저장 weights에서 새 rollout을 생성한 재측정**으로 기록하며 원래 응답을 재사용한
측정으로 부르지 않는다. 기존 compact results를 `n13-report`에 주면 각 check의 coverage를
내보내고 vector 없는 D4는 `null`로 둔다. score/D/reward/token으로 복원하지 않는다.

출력은 D40/D4 부호 일치·불일치, ON4/SR4 ID overlap, A/B 차이, 실측 유효 수다.
같은 정책·시점의 중복 acquisition은 합산하지 않는다. D4는 projected gradient 진단이며
실제 AdamW의 1-update 효용은 `null`이다. 기존 Switch 규칙은 변경하지 않는다.

`n13-report --reward-records /path/original-results.json`으로 같은 input/plan/원래
implementation·seed·arm의 실제 held-out 기록을 연결할 수 있다. 현재 stage와 전후
25-update 기록만 사용하며 다음 25 updates의 보상 변화는 경로상의 연관으로 표시한다.
평가가 없거나 출처가 다르면 차이는 `null`이고 학습 batch reward는 연결하지 않는다.

## 중단·비용·검증

성공한 응답을 별도 diagnostic root에 저장한다. 노드 종료 후 같은 명령으로 재개하면
완료 응답을 재사용하고 남은 generation/scoring을 한다. 완료 A/B phase와 endpoint를
재사용하며 원래 checkpoint·model/optimizer·selector 이력에 쓰지 않는다.
완료 endpoint의 artifact SHA가 다르면 덮어쓰지 않고 오류를 낸다.

첫 측정에 사용한 checkpoint·input·plan의 정확한 사본을 `source-checkpoint.pt`,
`source-inputs.json`, `source-plan.json`에 보존하고 SHA를 endpoint에 기록한다.
원래 `latest`가 나중에 갱신·삭제되면 `--checkpoint`를 이 보존 사본으로 바꿔 같은
`--output-root`에서 재개할 수 있다. 원본 plan/input은 같은 파일을 사용한다.
copy 도중 원본이 preflight와 달라졌거나 rank별 정책이 다르면 혼합 측정을 막는다.

원본 사본은 CPU preflight에서 먼저 보존한다. `cost-receipts/`, `costs.json`은 GPU 초기화
이후의 사본 확인·startup·N12 inference/N13 A·B scoring phase 비용이다. CPU preflight를
포함한 전체 할당 시간의 실측으로 부르지 않는다.
기존 학습 GPU-hours에 섞지 않는다. 중단 attempt의 미완 receipt가 있으면
`cost_measurement_complete=false`로 남겨 비용이 완전하다고 보고하지 않는다.
추론/scoring만 하는 실험으로 새 training curve를 만들지 않는다.
`diagnostic_phases`는 N12 inference와 N13 A/B 비용을 나눈 동일 receipt의 재분류이며
기존 phase 합계에 다시 더하지 않는다. CPU preflight 전후의 전체 job 경과 시간으로
phase 비용을 바꿔 보고하지 않는다.

검증 파일은 `srgc_research/tests/test_followup.py`다. 알려진 구성 차이, 누락 stratum,
400/512 pool, 40/4 역부호, SR4 후보 제약, B에서 A 선택 고정, checkpoint 출처·checksum,
optimizer 없는 weights 읽기, 실제 소형 LoRA dense gradient, RNG/Adam/weights 불변,
prompt 도중 강제 종료와 응답 재사용, 출력 덮어쓰기 거부를 검사한다.

2026-10-18 최초 구현 검증: 신규 진단과 기존 stage 분석·rollout/중간 보상 기록·Information
관련 **155 tests 통과**. Ruff와 diff 공백 검사도 통과했다. `6230307` 기준 learner,
scripts, 기존 모델/기록 dispatch와 backend·study·information·storage 소스는 변경하지 않았다.
실제 H100·NCCL 및 원격 노드 실행은 이 검증에 포함하지 않는다.

## 2026-10-19 점검 수정

문서의 실험 정의와 코드를 다시 대조했다. D40/D4 식, C40·S40 preview(원래 `candidate-draw`
stream과 학습 사용 ID 이력), C40 안 cached-SR 순위의 SR4, 원래 `online-ties` stream의 ON4,
reference 50개 평균 `v`, B에서 A 선택 고정, 고정 20문제 panel, 전체 pool 비중 표준화는
문서와 일치했다. 아래 세 가지를 실패 테스트로 먼저 재현한 뒤 수정했다.

- **continuation arm이 게시한 공통 prefix archive 거부:** `reward_curves.capture_boundary`는
  prefix 작업이 t25 archive를 남기지 않았거나 측정이 끊긴 경우, 처음 t25에 도달한 continuation
  arm이 `reward-curves/prefix/step-25.pt`를 게시하게 한다. 이때 저장 Engine의 arm은 그
  fork(sr/random/switch)인데 감사·collect는 `on_policy`만 허용해 같은 prefix weights를
  "arm differs"로 거부했다. prefix protocol의 t25는 저장 arm과 무관하게 공통 prefix로 받는다.
  t125/t250은 여전히 protocol arm 일치를 요구한다. `n12-panels`는 공통 prefix context를 arm과
  무관하게 하나로 묶어 t25 행이 게시 arm 수만큼 중복되지 않게 했다.
- **prefix backup의 공통 prefix 표시 누락:** `checkpoint-backups/.../checkpoint.pt`는 파일명이
  바뀌므로 `prefix.pt` backup도 공통 prefix로 표시되지 않았다. 감사의 t25 coverage가
  on_policy만 present로 나오고 panel 행 arm도 prefix가 아니었다. receipt에 기록된 원래
  `source` 파일명으로 판정한다. `<arm>-latest.pt` backup은 계속 prefix로 보지 않는다.
- **N12 생성 seed 미기록:** 위 안내는 생성 seed를 파일에 기록한다고 했으나
  `panel-measurement.json.gz`에 없었다. `generation_seed`와 rank 분할 수
  `generation_world_size`를 남긴다. 응답별 draw는 `stream_seed(generation_seed, rank, prompt)`다.

진단 코드 파일이 바뀌어 measurement hash가 달라진다. 10-18 코드로 이미 만든 진단 artifact는
덮어쓰지 않고 별도 identity로 남는다. `n12-panels`의 t25 기준은 같은 measurement hash 안에서만
연결되므로, 새 코드로 t125/t250을 측정하면 t25도 새 코드로 다시 측정한다.

검증: `test_followup.py` 73개(신규 4개 함수, 6 case)와 `test_stage_rollout_analysis.py` 9개,
**82 tests 통과**, Ruff·diff 검사 통과. 로컬 CPU 검증이며 원격 H100/NCCL 실측이 아니다.
`src/`, `srgc_rebuttal` engine, 세 모델 adapter, checkpoint·보상 기록 모듈은 변경하지 않았다.

확인하지 못한 것(코드 변경 없음):

- 서버의 실제 t25/t125/t250 파일 존재와 모델별(OLMo/Qwen3.5/Gemma4/Llama3.1) 실제 load·생성·scoring.
- 시작 조정 TCPStore 대기 한도는 60초다. rank 0의 원본 읽기·사본 보존이 이보다 길면 다른 rank가
  시간 초과로 종료한다. 사본 쓰기는 원자적이므로 같은 명령의 재시도는 안전하다.
- 원래 worker가 설정하던 MATH 검증기 overlay(`verifier_environment`)와 확장 모델의
  `setup_storage` cache 경로를 진단 진입점은 설정하지 않는다. 원래 실험과 같은 환경에서 실행한다.
