# 실험 실행 안내

코드 저장소 `master` 기준. [실험 목록·우선순위](LIMITATION_EXPERIMENTS_KO.md) · [결과 기록](EXPERIMENT_RESULTS_LEDGER_KO.md).
명령은 코드 저장소 루트에서 실행한다. 실행 중인 checkout에는 pull하지 않는다.

## 1. 기존 P1 결과 수집

```sh
sh scripts/run_srgc_sr_refresh.sh math results
sh scripts/run_srgc_sr_refresh.sh mbpp results
```

별도 옵션 없이 검증된 결과 JSON을 모은다. 출력 마지막의 **COLLECTED JSON**과
**COLLECTED FILES**가 실제 저장 위치다. `all results` 대신 위 두 명령을 사용한다.

| 파일 | 위치 |
| --- | --- |
| 최신 통합 JSON | `<run root>/results/results.json` |
| 개별 원본 JSON 사본 | `<run root>/results/exports/<수집 시각-ID>/raw/seed-N/` |
| 해당 수집본의 통합 JSON | `<run root>/results/exports/<수집 시각-ID>/results.json` |

통합 JSON의 `source_results`에는 문항별 보상·비용·출처를 포함한 원본 내용이 들어 있다.
fixed200와 replicate-1/2를 포함해 검증된 추가 실험 결과를 수집하며 원본과 이전 수집본은 보존한다.
오류 파일은 제외하고 `errors`에 기록한다. 누락은 0으로 채우지 않는다.
`--json`도 파일을 저장하며 stdout에는 JSON, stderr에는 저장 경로를 출력한다.

**P1 상태 확인**

```sh
sh scripts/run_srgc_sr_refresh.sh all status switch_fixed200
sh scripts/run_srgc_sr_refresh.sh all status replicate
```

미완료분을 이어갈 때는 빈 노드마다 아래에서 **한 명령만** 실행한다.

```sh
sh scripts/run_srgc_sr_refresh.sh all switch_fixed200
sh scripts/run_srgc_sr_refresh.sh all replicate
```

## 2. 신규 실험 실행

MATH를 먼저 배정한다. MBPP는 `math`를 `mbpp`로, 양쪽 자동 배정은 `all`로 바꾼다.
**빈 4-H100 노드마다 선택한 명령을 한 번 실행**하면 seed와 조건이 자동 배정된다.

| 순서 | 실험 | 실행 명령 |
| --- | --- | --- |
| 1 | 초반 On-policy·후반 SR 원인 진단 | `sh scripts/run_srgc_mechanism.sh math` |
| 2 | 선별 기준·배치 유지·보상 갱신 대조 | `sh scripts/run_srgc_support.sh math` |
| 3 | 고정 전환 시점 비교 | `sh scripts/run_srgc_switch_validation.sh math timing` |
| 4 | Qwen 모델 일반화 | `sh scripts/run_srgc_qwen35.sh math` |

고정 전환 명령의 `timing`을 생략하면 규칙 대조까지 포함한 70개 queue가 실행된다.
`switch_single`·`switch_consecutive`와 추가 RLOO의 신규 배정은 보류한다.
기존 queue와 진행 중인 작업은 유지한다.

## 3. 상태와 결과 조회

| 실험 | 상태 | 결과 |
| --- | --- | --- |
| 원인 진단 | `sh scripts/run_srgc_mechanism.sh all status` | `sh scripts/run_srgc_mechanism.sh all results` |
| support | `sh scripts/run_srgc_support.sh all status` | `sh scripts/run_srgc_support.sh all results` |
| 고정 전환 | `sh scripts/run_srgc_sr_refresh.sh all status timing` | `sh scripts/run_srgc_switch_validation.sh all results` |
| Qwen | `sh scripts/run_srgc_qwen35.sh all status` | `sh scripts/run_srgc_qwen35.sh all results` |
| 완료한 P0 | `sh scripts/run_srgc.sh all status` | `sh scripts/run_srgc.sh all results` |

조회는 GPU 학습을 시작하지 않는다. 원인 진단·support·고정 전환 results는 화면 출력이며,
1절처럼 JSON을 자동 수집하지 않는다. Qwen은 보고서를 저장하고 `Saved:` 경로를 표시한다.
고정 전환 결과 보고서는 규칙 대조도 함께 표시하지만, 표의 상태 명령은 `timing`만 조회한다.
원인 진단 JSON 출력은 `sh scripts/run_srgc_mechanism.sh all json`,
P0 비용 조회는 `sh scripts/run_srgc.sh all costs`다.

`reported_running`은 heartbeat 기록이며 원격 프로세스 생존 확인이 아니다.
`endpoint_unverified`는 파일은 있지만 검증 완료로 세지 않은 상태다.
완료 수와 남은 작업은 실제 status/results로 확인한다.

## 4. 노드 배정

| 실험 | 데이터셋 하나 | MATH + MBPP |
| --- | ---: | ---: |
| 원인 진단 | 최대 5노드 | 최대 10노드 |
| P1 fixed200 | 최대 5노드 | 최대 10노드 |
| P1 독립 반복 | 최대 20노드 | 최대 40노드 |
| support | 최대 15노드 | 최대 30노드 |
| 고정 전환 5시점 | 최대 25노드 | 최대 50노드 |
| Qwen cache/prefix 준비 | 최대 5노드 | 최대 10노드 |
| Qwen 네 arm 학습 | 최대 20노드 | 최대 40노드 |

한 작업은 4 H100을 사용한다. 표는 모든 작업이 준비되고 미완료일 때의 독립 배정 상한이며,
권장 노드 수나 현재 남은 수가 아니다. fixed200은 고정 전환 grid에서도 재사용한다.
원인 진단은 한 seed 안의 시점·branch를 순차 실행한다. seed당 물리적 update 850회는
공통 경로 400회와 진단 branch 450회의 합이며, 단일 정책의 850-step 학습이 아니다.

## 5. 실행 환경과 보존 원칙

- 기존 그룹 볼륨의 환경과 `OM_WORK` 설정을 사용한다. 사용자 볼륨으로 캐시를 옮기지 않는다.
- 한 노드에 worker 하나만 실행한다. 정상 실행 중인 checkout·환경·모델을 바꾸지 않는다.
- 중단 후에는 같은 명령으로 자동 재개한다. 잠금이나 해시 검증을 임의로 해제하지 않는다.
- 학습 update별 checkpoint와 prompt별 rollout cache를 보존한다.
- 기존 prefix는 당시 runtime·attention·채점기를 유지한다. MBPP v2와 신규 v3 결과를 섞지 않는다.
- 완료 P0·기존 cache/prefix·원고 TeX/PDF·웹 게시본은 변경하지 않는다.

세부 환경은 [클러스터 안내](../srgc_rebuttal/CLUSTER.md),
[Qwen 안내](QWEN35_SRGC_KO.md), [기존 실행 호환성](V7_EXPERIMENT_FIXES_2026-10-04.md)을 따른다.

## 6. 관련 문서

- [전체 실험 목록](LIMITATION_EXPERIMENTS_KO.md): 목적·조건·완료 범위.
- [원인 진단 상세](STAGE_MECHANISM_EXPERIMENTS_2026-10-06.md): E14-E16의 개입·평가·비용.
- [전환 시점 상세](SWITCH_ADDITIONAL_EXPERIMENTS_2026-10-06.md): E11-E13의 조건·분석.
- [정리 전 실행 기록](https://github.com/33modeling/offpolicy-misranking/blob/340bf7e7c7cad3f8a0a9bc229489ff9d4aa430a5/docs/REBUTTAL_COMMANDS_KO.md): 과거 120개 배정표·운영 기록.
