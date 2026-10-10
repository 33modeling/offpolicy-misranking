# SRGC 선택 정보 분석 코드

같은 checkpoint의 On-policy와 SR이 어떤 문제와 응답 신호를 잡고,
그 신호가 실제 GRPO/AdamW update에서 어떻게 반영되는지 기록한다.
기존 학습을 이어서 400 updates를 돌리는 명령이 아니다.

## 측정 단위

`stage`는 checkpoint까지 완료된 optimizer update 수다. 초기·중기·말기를
비교하려면 해당 시점의 **weights와 optimizer state가 함께 저장된 checkpoint**를
각각 제공해야 한다. endpoint JSON이나 gradient norm으로 과거 weights를 복원하지 않는다.

1. 기존 Engine과 같은 seed·stage 규칙으로 후보 40개를 뽑는다.
2. 현재 모델에서 독립 scoring 응답 A/B를 생성한다. 각 문제에 8개 응답을 저장한다.
   A로 On-policy를 선택하고 B로 선택된 문제의 신호를 확인한다.
   SR은 입력 bundle의 기존 cached success rate를 사용한다. 동점은 원래 selector 규칙을 따른다.
3. 선택 reference와 겹치지 않는 validation 문제 8개를 probe로 고정한다.
   최종 evaluation 300개는 사용하지 않는다.
4. 각 selector가 고른 4개 문제에서 새 학습 응답을 생성한다.
   **동일한 weights와 AdamW state에서 각각 한 번만 실제 GRPO update**를 수행한다.
   측정 후 weights와 optimizer를 복원한다. 두 selector의 학습 응답은 서로 다른 seed로 생성한다.

원래 trajectory의 On-policy는 선택된 문제를 25 updates 유지하고,
SR은 매 update 다시 후보를 뽑는다. 이 측정은 같은 상태의 즉시 update를 비교한다.
장기 성능 차이나 원래 trajectory의 노출 빈도를 재현한 결과로 해석하지 않는다.

## 저장 내용

| 파일 | 내용 |
| --- | --- |
| `manifest.json`, `inputs.json`, `plan.json` | dataset·seed·stage, 입력·plan·checkpoint·측정 코드 hash, 정확한 입력 복사본 |
| `source-checkpoint.pt` | 지정한 checkpoint 복사본. 원본은 변경하지 않는다 |
| `measurement-code/runtimes/<sha>/` | 이 측정에 사용한 별도 코드 snapshot |
| `score-A/B.json`, `.pt` | 후보·reference ID, 원시 응답 token·text·reward, projected dense gradient와 cosine·dot·norm |
| `probe.json`, `.pt` | 독립 probe 응답, 고정된 old log probability, 실제 LoRA GRPO loss gradient |
| `on_policy/sr.json`, `.pt` | 선택 문제, 실제 학습 응답, 문제별 GRPO gradient, clip 전후 batch gradient, weights·AdamW state의 전후 snapshot |
| `endpoint.json` | 완료된 측정과 각 phase JSON/tensor의 hash |
| `measurement-settings.json`, `packages.json` | attention, checkpoint arm과 실제 anchor arm, 원래 코드 hash, 실행 package 버전 |
| `cost-receipts/`, `invocations/`, `costs.json` | generation·backward·진단·저장 시간, 실패 시 미측정 시간의 누락 여부 |

응답마다 update 전후 token log probability를 저장한다. 문제별 gradient는
실제 GRPO loss의 gradient다. 문제별 gradient의 평균과 실제 batch gradient가
수치적으로 일치하는지 검사하며, BF16/FP16 누적 반올림 오차는 dtype별 norm bound로 확인한다.
검사 오차와 허용 범위도 기록한다.

실제 parameter 이동량은 저장된 `theta_after - theta_before`다.
같은 AdamW 이력에서 zero gradient로 한 번 step한 이동량도 기록하고,
실제 이동량에서 이를 뺀 값을 `batch_incremental_update`로 저장한다.
이 차이는 batch 전체의 추가 효과이며, 개별 문제의 AdamW 기여도를 선형 분해한 값은 아니다.
정답만 있거나 오답만 있는 batch의 reward gradient가 0이어도 AdamW 이력 때문에
weights가 움직일 수 있다.

선택용 projected dense LOO gradient와 실제 학습용 LoRA GRPO gradient는
parameter 범위와 목적함수가 다르다. 둘을 같은 벡터처럼 비교하지 않는다.
Probe 정렬과 loss 변화는 고정된 독립 응답에 대한 진단이며, 새로운 평가 응답의 성공률은 아니다.

## 실행

초기 MATH 분석은 아래 한 줄로 실행한다.

```sh
sh scripts/run_srgc_information.sh math
```

기본값은 seed 7, t0, SDPA다. 기존 공유 저장소에서 plan과 input을 읽고,
`<OM_WORK>/selection-information/math/seed-7/t0`에 측정 JSON·tensor를 저장한다.
MBPP는 `math` 대신 `mbpp`, 두 dataset을 차례로 수집하려면 `all`을 사용한다.
인자를 생략하면 `math`다. 중단 후 같은 명령으로 재개한다.
**기본 명령은 HTML을 만들지 않는다.** 보고서는 별도로 요청하고 `report`를 실행할 때만 만든다.

코드는 `master`에서 실행한다. GPU 수집은 기존 환경의 4×H100 admission과
공유 GPU lock을 사용한다. 입력은 cache를 포함한 원래 `srgc-inputs-v1` bundle이어야 한다.
일반 Engine checkpoint 또는 StageStudy의 저장된 `anchor`를 읽는다.
StageStudy의 현재 branch weights를 anchor로 오인하지 않는다.
Checkpoint의 `input_sha256`, `plan_sha256`, seed, 실제 step과 configuration을 확인한다.
별도 sidecar만 있는 `literature-v1/anchors/step-*.pt`는 이 명령의 입력 형식에 포함하지 않는다.

초기 상태는 checkpoint를 생략할 수 있다. 기존 실험과 attention을 맞추려면 명시한다.

```sh
sh scripts/run_srgc_information.sh math collect \
  --plan /group-volume/<user>/offpolicy-misranking/srgc-rebuttal/experiments/pair_seeds.json \
  --inputs /group-volume/<user>/offpolicy-misranking/srgc-rebuttal/inputs/pair-seed-7.json \
  --seed 7 --stage 0 --attention sdpa \
  --output /group-volume/<user>/offpolicy-misranking/selection-information/math/seed-7/t0
```

중간·후기 상태는 해당 시점의 checkpoint를 지정한다.
입력 hash는 JSON의 실제 byte hash다. 포맷만 바뀐 입력도 같은 입력으로 취급하지 않는다.

```sh
sh scripts/run_srgc_information.sh math collect \
  --plan /group-volume/<user>/offpolicy-misranking/srgc-rebuttal/experiments/pair_seeds.json \
  --inputs /group-volume/<user>/offpolicy-misranking/srgc-rebuttal/inputs/pair-seed-7.json \
  --seed 7 --stage 100 \
  --checkpoint /group-volume/<user>/snapshots/on-policy-step-100.pt \
  --output /group-volume/<user>/offpolicy-misranking/selection-information/math/seed-7/t100
```

MBPP는 `math` 대신 `mbpp`와 해당 plan/input/checkpoint를 사용한다.
`PAIR_PYTHON`과 `SWITCH_PYTHON`은 기존 환경의 interpreter를 지정한다.
완료 전 중단되면 **같은 명령과 output**으로 재개한다. 완료된 phase는 hash를 확인하고 재사용한다.
코드·입력·checkpoint·설정이 바뀌면 새 output이 필요하다.
손상된 기록은 덮어쓰거나 정상 완료로 처리하지 않는다.
중단되어 알 수 없는 GPU 시간이 있으면 cost total은 `null`로 남긴다.
측정 완료와 전체 비용 측정 완료는 별도 상태다.

```sh
sh scripts/run_srgc_information.sh all status --output /group-volume/<user>/offpolicy-misranking/selection-information/math/seed-7/t100
```

## 보고서

보고서는 GPU·Torch 없이 생성할 수 있다. Python과 기존 NumPy dependency를 사용한다.
각 단계의 수집 폴더를 반복 지정하면 시점별 문제·성공률·신호·업데이트를 함께 볼 수 있다.
원본 수집 폴더와 report output은 분리한다.

```sh
sh scripts/run_srgc_information.sh all report \
  --measurement /group-volume/<user>/offpolicy-misranking/selection-information/math/seed-7/t0 \
  --measurement /group-volume/<user>/offpolicy-misranking/selection-information/math/seed-7/t100 \
  --measurement /group-volume/<user>/offpolicy-misranking/selection-information/math/seed-7/t400 \
  --output /group-volume/<user>/offpolicy-misranking/selection-information/reports/seed-7
```

`report.html`, `information.json`과 5개 CSV를 생성한다.
HTML에는 선택 문제 원문과 학습 응답, 정답·오답별 log probability 변화가 들어간다.
CSV/JSON 링크는 같은 report 폴더의 파일을 가리킨다.
CSV는 시점별 정보 요약, 선택 문제, 공통 후보, batch update, parameter별 변화다.
독립 B의 mixed 비율과 cosine을 같은 후보 40개의 평균과 비교한다.
A/B 상관, gradient와 cached/current SR ranking의 상관, 선택 집합의 중복도 함께 기록한다.
문제 분야·난도 label이 원본 입력에 없으면 추정해서 붙이지 않는다.

기존 mechanism JSON만 있는 경우에도 선택 문제·성공률·학습 reward 분포를 정리할 수 있다.
동일 파일·동일 endpoint는 한 번만 읽고, 서로 충돌하는 endpoint는 거부한다.
원문을 보려면 정확한 hash의 원래 input bundle을 제공한다.

```sh
sh scripts/run_srgc_information.sh all report \
  --legacy /path/to/pair-results.json --legacy /path/to/mbpp-results.json \
  --inputs /path/to/original-pair-seed-7.json \
  --output /path/to/information-report
```

기존 JSON에 없는 응답 원문·weights·실제 update는 **미기록**이다.
25-update 평균 gradient norm을 첫 update의 gradient나 실제 AdamW 이동량으로 바꾸지 않는다.
미완료 seed는 pending으로 남긴다. A/B 차이나 단일 update만으로 초기 On-policy 우위의
원인이 확정됐다고 쓰지 않는다.

## 검증

```sh
python -m pytest -q srgc_research/tests/test_information.py
python -m pytest -q srgc_research/tests/test_information_wrapper.py
python -m pytest -q srgc_research/tests srgc_rebuttal/tests
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --max_restarts=0 -m srgc_research.tests.information_distributed_smoke
```

작은 실제 OLMo/LoRA 모델에서 직접 GRPO/AdamW step과 측정값을 비교한다.
실제 generation, BF16 weights, clip, zero-reward/momentum, probe finite difference,
실패 후 weights·optimizer 복원, 각 phase 중단/재개, hash 손상, 원본 덮어쓰기 방지,
legacy 중복·누락·충돌과 HTML escaping을 검사한다.
4-rank CPU/Gloo 테스트는 단일 process와 실제 update를 비교하고
rank-zero 파일 저장 실패가 모든 rank에 전달되는지 확인한다.
CPU 검증과 H100/NCCL 실험은 구분한다. 이 변경의 검증 중에는 새 GPU 학습을 실행하지 않는다.

2026-10-10 검증: 추가 테스트 45개 통과. 전체 SRGC 회귀 테스트는 819개 통과,
9개 skip, 325개 subtest 통과. 4-rank CPU/Gloo smoke, Ruff, Python compile,
shell syntax 검사도 통과했다. 검증 환경은 Python 3.12, Torch 2.14.0 CPU,
Transformers 4.57.6, PEFT 0.21.0이다. 현재 장비에서 NVIDIA driver를 사용할 수 없어
H100/NCCL과 7B 모델의 실제 GPU 메모리·실행 시간은 확인하지 않았다.

간단 실행 명령의 추가 검증은 12개 테스트가 통과했다. 인자 생략·dataset별 Python 선택,
두 dataset의 순차 실행·실패 중단·입력 누락·공백이 있는 경로·명시 인자 전달을 검사했다.
기본 명령에서 `report`가 호출되지 않는 것도 확인했다.
