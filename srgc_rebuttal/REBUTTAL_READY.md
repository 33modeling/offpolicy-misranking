# 추가 시드 실험 준비 — 2026-09-27

코드와 실제 문제 입력은 준비했다. GPU 캐시 생성과 7B 학습은 아직 실행하지 않았다.
준비 완료 여부와 입력 해시는 [prepared_inputs.json](experiments/prepared_inputs.json)에 기록한다.
제출본 v6와 기존 실험 보상·비용 수치는 변경하지 않는다.

## 고정한 새 실험

| 항목 | 설정 |
| --- | --- |
| 추가 시드 | 5, 6, 7, 8, 9 |
| 대조군 | Random, SR, On-policy, Switch |
| 공통 시작 | 시드별 동일 모델·optimizer의 On-policy 25-update prefix |
| 종점 | 모든 군 total update 275 |
| On-policy 갱신 | 25 updates마다 무작위 후보 40개를 뽑아 채점하고 상위 문제 4개 선택 |
| 갱신 사이 학습 | 선택한 4개를 유지하며 매 update 4 × 8개의 새 학습 응답 생성 |
| Random | 매 update 전체 400개에서 중복 없이 무작위 40개를 뽑고, 그 안에서 무작위 4개 학습 |
| SR | 매 update 전체 400개에서 중복 없이 무작위 40개를 뽑고, 그 안에서 SR 점수 상위 4개 학습 |
| D | On-policy의 무작위 후보 40개와 다음 미사용 SR 점수 상위 40개의 gradient 평균 차이를 validation 방향과 내적. 학습 배치 4개와 구분 |
| 전환 | Switch 자신의 D만 사용; 기존 두 음수/3-check 평균 규칙 유지 |
| SR 전환 이후 | 전환 update부터 위 SR 방식으로 학습; gradient scoring과 D 검사 중단 |
| 평가 | 같은 300문항 × 응답 8개; 시드별 결과와 대응 차이·시드 간 표준편차 보고 |

On-policy prefix는 t=0에서 선별한다. 이후 선별은 t=25,50,...,250에서 수행하며
각 경계의 모델로 계산한 결과를 다음 update부터 사용한다. 상위 4개와 선별 시점을
checkpoint에 저장하므로 구간 중간에 재시작해도 후보를 추가로 뽑지 않는다.
GPU 시간은 실제 선별 때 측정한 비용만 더하며, update 수를 다시 곱하지 않는다.

2026-09-28 저자 요청으로 Random/SR의 고정 부분집합 반복과 SR-GC 비교 대상을
수정했다. SR-GC는 저자 재확인에 따라 원래의 40 대 40 비교를 유지한다.
후보 40개와 다음 SR 40개를 함께 채점하므로 중복을 제외하면 최대 80개다.
비교만 한 문제는 사용 처리하지 않는다. 실제 학습한 문제는 prefix를 포함해 기록하며,
최신 요청에 따라 Random/SR 학습은 매번 400개 전체에서 40개를 새로 뽑는다.
중복 금지는 한 번에 뽑는 40개 안에서 적용하며, 다음 추출에서는 이전 문제가 나올 수 있다.
이 규칙은 MATH와 MBPP 모두 동일하다. 기존 SR-GC 비교용 미사용 문제 기록은 유지한다.
후보 40개와 최종 4개를 로그에 저장하며, 이전 방식의 checkpoint/결과와 혼합하지 않는다.
간단 실행기는 최초에만 `candidate40-v2` 작업을 만들고, 이미 활성 작업이 있으면
저장된 진행에서 자동으로 이어간다. 명령은 항상 `sh scripts/run_srgc.sh math` 또는
`sh scripts/run_srgc.sh mbpp`다. 정상 실행 중인 worker는 중지하거나 재시작할 필요 없다.
같은 데이터셋의 두 노드는 같은 작업에 합류하며, 실행 중인 작업을 중복 실행하거나
기존 파일을 초기화하지 않는다.

## 준비된 실제 입력

[MATH train](https://huggingface.co/datasets/EleutherAI/hendrycks_math)의 revision
21a5633873b6a120296cce3e2df9d5550074f4a3을 고정했다.
원본 7,500문항 중 정답 상자가 비어 있는 2문항과 정규화 후 중복 1문항을
분할 전에 제거했다. 빈 정답을 추정하거나 모델의 성공 여부로 문항을 고르지 않았다.
제외 ID와 이유는 manifest 및 각 입력의 provenance에 있다.

분할 시드 0으로 후보 400개, validation 100개, 평가 300개를 서로 겹치지 않게 정했다.
다섯 학습 시드가 같은 문제 분할을 사용한다. 온라인 validation은 그 100개 중
고정된 50개를 사용한다. 이 50개는 새 실험의 명시적 설정이며, 제출 실험의 누락된
설정을 복원했다고 주장하지 않는다. 800개 gold answer의 verifier 자기 일치도 확인했다.
OLMo RL-Zero prompt, math-verify 0.9.0, base-model revision은 입력·계획에 기록했다.

입력 파일:

- [seed 5](inputs/seed-5.json)
- [seed 6](inputs/seed-6.json)
- [seed 7](inputs/seed-7.json)
- [seed 8](inputs/seed-8.json)
- [seed 9](inputs/seed-9.json)

초기 정책의 실제 응답 8개씩을 생성해야 cached_rewards가 채워진다. 현재는 비어 있으며,
임의 보상으로 채우지 않았다. 학습 실행기는 캐시가 없는 입력을 거부한다.

## 실행 순서

저장소 루트에서 CPU 입력 검사를 실행한다.

    python -m srgc_rebuttal.plan --check-inputs --allow-pending-cache

CUDA용 PyTorch와 requirements.txt의 패키지가 설치된 각 4-GPU 노드에서 같은
worker 명령을 실행한다. 공유 저장소에서 2개 이상 노드가 작업을 자동 분담한다.

    python scripts/run_srgc_rebuttal.py worker --dataset math

각 문제의 응답과 보상은 즉시 원자적으로 저장한다. 재시작은 저장된 결과를 재사용하고
남은 문제도 같은 문제별 난수로 생성한다. 모델·입력·생성 설정이 달라진 재시작은 거부한다.
캐시가 완성된 시드부터 prefix를 실행하고, 이어 네 비교군을 분배한다.
큐는 캐시 5개, prefix 5개, continuation 20개를 의존 순서에 맞춰 분배한다.
이미 완료된 작업은 재실행하지 않는다. MBPP는 CPU 호스트에서 한 번 입력을 준비한 뒤
각 노드에서 옵션만 바꿔 실행한다. MATH와 MBPP의 결과 경로와 큐는 분리된다.

    python scripts/run_srgc_rebuttal.py prepare --dataset mbpp
    python scripts/run_srgc_rebuttal.py worker --dataset mbpp
사용 가능한 노드만 사용하며 자세한 명령은 [CLUSTER.md](CLUSTER.md)에 있다.

    python scripts/run_srgc_rebuttal.py status --dataset math
    python scripts/run_srgc_rebuttal.py costs --dataset math
    python scripts/run_srgc_rebuttal.py summary --dataset math

## 리뷰에 남길 데이터

각 시드·대조군의 종점 보상, 문항별 보상, 공통 prefix 해시, D 수열,
선별 후보·상위 4개 ID·점수, 전환 시점, 선별·학습·평가 시간과 실행 환경을 저장한다.
불리한 시드나 전환하지 않은 시드도 포함한다. 제출본 시드 3·4와는 먼저 분리해서 보고한다.

선별·학습 비용은 단계별 receipt로 남긴다. 재시도한 완료 연산도 누락 없이 합산한다.
시작 receipt만 있는 중단 연산은 비용 미측정으로 표시하고, 해당 군의 완전한 비용
총합을 보고하지 않는다. 모델 초기화와 checkpoint I/O는 별도 단계로 계측하고,
전체 실행 시간에는 프로세스 내부 작업 대기와 계측 부대비용도 포함한다.
SR-GC 산술 시간은 선별 시간의 일부이므로 다시 더하지 않는다.

현재 작업 컴퓨터에서는 CUDA 드라이버가 작동하지 않는다. 실제 캐시·추가 시드 결과를
생성하려면 사용 가능한 GPU 노드 접속 또는 스케줄러 할당이 필요하다.
