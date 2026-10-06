# 실험 실행 안내

코드 저장소 `master` 기준. [실험 목록·우선순위](LIMITATION_EXPERIMENTS_KO.md) · [결과 기록](EXPERIMENT_RESULTS_LEDGER_KO.md).
명령은 코드 저장소 루트에서 실행한다. 실행 중인 checkout에는 pull하지 않는다.

## 통합 명령

형식은 `sh scripts/run_srgc_experiments.sh 데이터셋 실험 [run|status|results]`다.
실험을 생략하거나 전체 학습을 암묵적으로 시작하지 않는다. 기존 개별 sh 명령도 유지한다.

```sh
sh scripts/run_srgc_experiments.sh list
sh scripts/run_srgc_experiments.sh all mechanism
sh scripts/run_srgc_experiments.sh all mechanism status
sh scripts/run_srgc_experiments.sh all mechanism results
```

실험 이름은 `mechanism`, `qwen`, `timing`, `rules`, `support`, `pool`, `switch_repeat`,
`direction`, `fixed200`, `replicate`, `candidates`, `sr_hold`, `p0`다.
`all replicate results`는 MATH·MBPP를 각각 수집한다. 한쪽 오류가 다른 쪽 수집을 막지 않는다.
실행 환경·모델·seed·캐시 경로·재개는 기존 검증된 runner를 따른다.

## 1. 기존 추가 실험 결과 수집

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

**독립 반복·fixed200 상태 확인**

```sh
sh scripts/run_srgc_sr_refresh.sh all status switch_fixed200
sh scripts/run_srgc_sr_refresh.sh all status replicate
```

결과 수집을 이유로 replicate를 새로 배정하지 않는다. 동일-prefix 반복은 보조 실험이다.
신규 배정 기준은 [V6 Limitations 대응표](LIMITATION_EXPERIMENTS_KO.md#우선순위)다.
과거 P1은 운영용 묶음이며 V6의 실험명이나 우선순위가 아니다.

## 2. 신규 실험 실행

MATH를 먼저 배정한다. MBPP는 `math`를 `mbpp`로, 양쪽 자동 배정은 `all`로 바꾼다.
**빈 4-H100 노드마다 선택한 명령을 한 번 실행**하면 seed와 조건이 자동 배정된다.

| 순서 | 실험 | 실행 명령 |
| --- | --- | --- |
| 1 | 한 backbone 한계: Qwen 재현 | `sh scripts/run_srgc_qwen35.sh math` |
| 2 | 공통 prefix의 전환 시점 | `sh scripts/run_srgc_switch_validation.sh math timing` |
| 2 | temporal confirmation 규칙 | `sh scripts/run_srgc_switch_validation.sh math rules` |
| 3 | SR 캐시 갱신: 유지·동점 일치 대조 | `sh scripts/run_srgc_support.sh math` |
| 3 | SR 전체 pool 갱신 | `sh scripts/run_srgc_sr_refresh.sh math pool` |
| 4 | 재전환과 추가 선별 비용 | `sh scripts/run_srgc_sr_refresh.sh math switch_repeat` |
| 5 | reference 방향 정보의 부분 대조 | `sh scripts/run_srgc_sr_refresh.sh math direction` |
| 보조 | 같은 prefix 이후 학습 반복 | `sh scripts/run_srgc_sr_refresh.sh math replicate` |
| 별도 요청·병행 | 초반·후반 메커니즘 | `sh scripts/run_srgc_mechanism.sh math` |

고정 전환 명령의 `timing`을 생략하면 규칙 대조까지 포함한 70개 queue가 실행된다.
규칙 대조는 V6의 confirmation 한계에 대응한다. 점검 간격 최적화는 포함하지 않는다.
추가 RLOO는 배정하지 않는다. 기존 queue와 진행 중인 작업은 유지한다.

메커니즘은 요청된 실험이며 V6 대응 실험 이후로 미루지 않는다. MATH 최대 5노드,
MBPP 최대 5노드, 양쪽 최대 10노드다. 양쪽을 자동 배정하려면 빈 노드마다
`sh scripts/run_srgc_mechanism.sh all`을 한 번 실행한다.

## 3. 상태와 결과 조회

| 실험 | 상태 | 결과 |
| --- | --- | --- |
| 원인 진단 | `sh scripts/run_srgc_mechanism.sh all status` | `sh scripts/run_srgc_mechanism.sh all results` |
| support | `sh scripts/run_srgc_support.sh all status` | `sh scripts/run_srgc_support.sh all results` |
| 고정 전환 | `sh scripts/run_srgc_sr_refresh.sh all status timing` | `sh scripts/run_srgc_switch_validation.sh all results` |
| 확인 규칙 | `sh scripts/run_srgc_sr_refresh.sh all status rules` | `sh scripts/run_srgc_switch_validation.sh all results` |
| 전체 pool 갱신 | `sh scripts/run_srgc_sr_refresh.sh all status pool` | 1절의 데이터셋별 `results` |
| 재전환 | `sh scripts/run_srgc_sr_refresh.sh all status switch_repeat` | 1절의 데이터셋별 `results` |
| 방향 정보 대조 | `sh scripts/run_srgc_sr_refresh.sh all status direction` | 1절의 데이터셋별 `results` |
| Qwen | `sh scripts/run_srgc_qwen35.sh all status` | `sh scripts/run_srgc_qwen35.sh all results` |
| 완료한 P0 | `sh scripts/run_srgc.sh all status` | `sh scripts/run_srgc.sh all results` |

조회는 GPU 학습을 시작하지 않는다. 원인 진단·support·전환 비교 results/json은 원본을 포함한
통합 JSON과 개별 사본을 자동 저장한다. `status`는 읽기 전용이다.

| 결과 | 통합 JSON 위치 |
| --- | --- |
| 메커니즘 | `<run root>/results/mechanism/results.json` |
| support | `<run root>/results/support/results.json` |
| 고정 전환·규칙 | `<run root>/results/switch_validation/results.json` |

각 폴더의 `exports/<수집 시각-ID>/raw/`에 검증된 원본 사본을 보존한다.
`COLLECTED JSON / COLLECTED FILES`가 실제 경로다. 기존 1절의 통합 파일과 덮어쓰지 않는다.
Qwen은 기존 보고서를 저장하고 `Saved:` 경로를 표시한다.
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
| fixed200 | 최대 5노드 | 최대 10노드 |
| 보조 독립 반복 | 최대 20노드 | 최대 40노드 |
| support | 최대 15노드 | 최대 30노드 |
| 고정 전환 5시점 | 최대 25노드 | 최대 50노드 |
| 확인 규칙 2조건 | 최대 10노드 | 최대 20노드 |
| 전체 pool 갱신 | 최대 5노드 | 최대 10노드 |
| 재전환 | 최대 5노드 | 최대 10노드 |
| 방향 정보 3조건 | 최대 15노드 | 최대 30노드 |
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

## 7. 실행 코드 점검

- 통합 실행부를 새로 작성했다. 기존 학습 엔진·동결 runtime·실험 조건과 메커니즘 코드 해시는 유지한다.
- support는 같은 runtime·attention끼리만 paired 집계한다. 미확인 조건은 원값만 표시한다.
- 메커니즘 상관·overlap·선택 집합 진단값을 원본 점수·보상에서 재계산해 검증한다.
- 손상된 rollout의 개수·토큰·보상을 검사하고 재생성한다. 캐시 저장은 flush/fsync 후 교체한다.
- 잘못된 status 항목·queue receipt가 다른 정상 작업을 가리지 않게 수정했다. 손상 기록의 retry 횟수를 초기화하지 않는다.
- 결과는 검증한 같은 JSON 객체를 수집한다. 저장 중 실패하면 기존 최신 수집본을 유지한다.
- CPU 4-rank 메커니즘 학습·분기·재개와 2-rank 저장 실패 전파를 확인했다. 실제 H100/NCCL 실측은 별도다.

2026-10-06 검증: 전체 `srgc_rebuttal/tests` **668 passed / 9 skipped / 305 subtests passed**.
9개 skip은 로컬 Transformers 4.57.6에 Qwen3.5 모델 클래스가 없어서 발생한 기존 모델 테스트다.
H100 실행 환경을 변경하거나 패키지를 새로 설치하지 않았다. CPU 분산 점검은 GPU 실측이 아니다.
