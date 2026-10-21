# Gemma 생성·채점 수정 — 2026-10-21

Gemma의 일반 시작·상태·결과 명령은 이제 별도 `gemma4-12b-text-v2` 경로를 사용한다.
공식 특수 토큰 억제, 일치하는 학습 log probability와 MBPP 코드블록 추출을 수정했다.
기존 v1 캐시·checkpoint는 보존하고 새 소량 검사 → 캐시 → prefix → 학습으로 진행한다.
[수정·검증·실행 안내](GEMMA4_TEXT_RECOVERY_2026-10-21.md)를 따른다.
실행 중인 예전 coordinator는 새 코드를 다시 import하지 않으므로 정상 중지 후 새 명령으로
시작해야 한다. 이미 실행 중인 다른 모델 작업을 업데이트만을 위해 재시작하지 않는다.
로컬 실제 12B 짧은 생성 검사는 마쳤으며 원격 전체 init·학습이 완료됐다는 뜻은 아니다.

같은 날 후속 재점검에서 text-v2 시작 함수의 연결 순서를 수정했다. 모델 연결 실행에서
MATH 완료 후 MBPP를 기다리며 대기할 수 있던 경로, 현재 데이터셋의 사전 점검,
별도 preflight 중지, 모델용 Python이 없을 때의 상태·중지 명령을 보강했다.
생성·채점·학습 프로토콜과 이미 준비한 text-v2 입력·checkpoint는 유지한다.
상세 재현과 보존 범위는 위 안내의 “시작·중지 경로 재점검”을 따른다.

# Qwen·Llama·Gemma 연결 실행 — 2026-10-19

현재 공개 학습 시작 명령은 **모델 단위**로 연결한다(2026-10-19 사용자 결정).
한 모델의 MATH를 ERROR·중단 작업·남은 학습·최종 평가까지 끝내고, 이어서 같은 모델의
MBPP도 모두 끝낸 뒤에야 다음 모델을 실행한다. Llama 시작은 `Llama MATH → Llama MBPP
→ Gemma MATH → Gemma MBPP → Qwen MATH → Qwen MBPP`, Qwen 시작은 `Qwen MATH → Qwen MBPP
→ Llama MATH → Llama MBPP → Gemma MATH → Gemma MBPP`다. 모델 안에서는 MATH 완료 전
MBPP를 시작하지 않는다. 실패·GPU/peer 대기·복구 전용 worker의 성공만으로는 다음
모델로 넘어가지 않으며, ERROR 우선 복구는 현재 모델·현재 데이터셋 안에서 적용한다.
GPU 강제 종료로 ERROR가 누적돼도 재개를 막지 않도록 실패 횟수 한도를 제거했다.
자세한 동작·보존 범위·기존 IDLE worker 적용 방법은
[모델 연결 실행](SRGC_MODEL_CHAIN.md)을 따른다. 아래 2026-10-12 기록의
“모델마다 별도 노드에서 해당 명령 하나”는 당시 동작이며 현재 기본값은 연결 실행이다.

2026-10-18 **`no math plan in requested queues` 수정**: 실제 plan의 `math_train`과
CLI의 `math`를 배정 판단에서만 매핑한다. 같은 오류가 있던 복구 backlog 필터와
MBPP 선행 금지 조건까지 함께 고쳤고 `04ce845`를 우선 push했다. 추가 점검에서
캐시 인계 파일이 없는 ERROR 복구 누락, endpoint 저장 뒤 실행 lease가 남았을 때의
조기 완료도 수정했다. 실제 adapter가 준비한 세 모델 plan으로 검사하며 원본 plan,
입력, 학습·보상 기록 runtime hash는 유지한다. [상세 수정·검사](SRGC_MODEL_CHAIN.md).

2026-10-18 **ERROR 우선 재개 재수정**: 내부 배정기의 실패 한도 제거와 별개로,
바깥 coordinator가 오류 코드 1·2로 종료된 모델을 재시도하지 않는 버그를 재현·수정했다.
기존 미완료 작업이 남으면 복구 전용 실행을 유지한다. 당시 모델 간 ERROR 우선 순회는
2026-10-19에 현재 모델 완료 후 전환으로 대체했다. 오류 작업의 cache/prefix 복구 우선순위와 일시적인 자식 실행 실패도
함께 수정했다. 공개 Llama 시작 script의 CPU 통합 검사에서 Llama·Gemma 오류 재시도 후
MATH·MBPP 순서 및 checkpoint 보존을 검증했다. [상세 동작·적용](SRGC_MODEL_CHAIN.md)을 따른다.
원격 학습 노드에서 실제 실행됐다는 뜻은 아니다.

2026-10-18 [전체 코드 재점검](REPOSITORY_AUDIT_2026-10-06.md): `all resume`는 모델마다
한 번만 적용하여 MATH 실행 중 새로 건 MBPP 중지 요청을 보존한다. 잘못된 coordinator
기록 하나로 status 전체가 종료되지 않도록 검사한다. 손상된 중간 rollout은 해당 arm의
curve 오류로 표시하고 정상 endpoint·다른 arm은 results에 유지한다. 기존 실험 현황판의
긴 ROOT 경로도 한 줄 안에서 표시하며 JSON의 전체 경로와 경로 안 공백을 유지한다. 학습·기록 runtime과
현재 실험 조건은 보존하며 원격 worker에 수정이 적용됐다는 뜻은 아니다.

# Current Experiment Allocations

2026-10-18 **N12·N13 코드 추가**: [기존 checkpoint 진단·실행 안내](N12_N13_CHECKPOINT_DIAGNOSTICS.md).
저장 rollout의 초기 cache 구성 표준화와 고정 20문제 패널, SR-GC 40대40/같은 후보의 선택
4대4 및 독립 B 비교를 별도 모듈에 구현했다. optimizer 없는 평가 archive도 추론/scoring에
재사용하며 학습 update는 하지 않는다. 실제 서버의 t25/t125/t250 파일 잔존과 H100 실측은
미확인이다. 기존 queue·계산·저장 runtime은 보존한다.

같은 날 N12·N13 재디버깅으로 완료 직전 종료 후 비용 파일 충돌, 같은 weights 재사용 시
stage 누락, 학습 draw 및 역사 check 중복 집계를 수정했다. 여러 rank의 CPU preflight 결정과
오류 전달, 원본 사본 고정, 완료 evidence 검증도 강화했다. 4-worker CPU 시작 검증과 실제
H100/NCCL 측정은 구분한다. 상세 내역은 위 checkpoint 진단 안내를 따른다.

## 현재 실험 — 2026-10-10

사용자가 진행 상태를 확인한 실험은 아래 네 종류다.
원격 프로세스나 완료 여부를 직접 확인한 기록은 아니다. 노드 수는 미확인이다.

| 실험 | 모델·측정 | 실행 명령 |
| --- | --- | --- |
| information | 사용자 보고: 전체 측정 완료. 같은 checkpoint에서 On-policy·SR의 선택 문제, 성공률, 실제 GRPO update 비교 | `sh scripts/run_srgc_information.sh all results` |
| Qwen | Qwen3.5-9B SRGC | `bash scripts/run_srgc_qwen35.sh all` |
| Gemma | Gemma 4 12B PT SRGC | `bash scripts/run_srgc_gemma4.sh all` |
| Llama | Llama 3.1 8B Instruct SRGC | `bash scripts/run_srgc_llama31.sh all` |

기존 결과·입력·optimizer checkpoint를 보존한다. 진행 중인 실험의 학습 코드나
설정을 바꾸지 않는다. 재개할 수 있는 기존 작업을 먼저 배정한다. 그 작업이 다른
노드에서 실행 중이거나 재시도 대기 중이면 빈 노드가 독립적인 READY 작업을 수행한다.
`BACKUP` 로그는 백업 상태이며 학습 완료나 GPU 사용 여부를 뜻하지 않는다.
Information 전체 완료는 `sh scripts/run_srgc_information.sh all status`의
검증된 측정 `10/10`을 기준으로 확인한다.
사용자가 information 전체 완료를 보고했다. 결과 추출은 `all results` 한 줄이며,
MATH·MBPP와 모든 seed·저장된 stage를 `<OM_WORK>/results/information-results.json`
하나에 담는다. 파일 내부의 `coverage`, `pending`, `errors`로 실제 수집 상태를 확인한다.

2026-10-10 다운로드된 information JSON을 확인했다. OLMo-3-7B의 MATH·MBPP,
seed 5–9, t0 측정 10/10이 완료되어 있고 `pending`과 `errors`는 비어 있다.
중기·후기 측정은 포함되어 있지 않다. GPU 비용 ledger는 10개 모두 미완료다.
실제 선택 문제·응답·업데이트와 seed별 비교는
[Information 결과 분석](INFORMATION_RESULTS_2026-10-10.md)에 기록했다.

네 실험의 결과 추출은 각 실행 script에 `all results`를 붙여 실행한다.
MATH·MBPP와 seed별 결과를 `<OM_WORK>/results/`의 실험별 JSON 하나에 합친다.

| 실험 | 결과 파일 |
| --- | --- |
| information | `information-results.json` |
| Qwen | `qwen35-results.json` |
| Gemma | `gemma4-results.json` |
| Llama | `llama31-results.json` |

기본 형식은 compact다. Information은 선택·성공률·실제 update·응답별 scalar 변화,
세 모델은 seed별 최종 reward·문제별 평가 reward·전환 기록·측정된 비용을 담는다.
Token·log probability 배열, 학습 응답 원문, worker heartbeat와 rank progress는 넣지 않는다.
파일 경로와 크기를 출력하며, 같은 실험의 반복 추출은 같은 JSON을 원자적으로 갱신한다.
두 노드가 동시에 추출하면 snapshot 수집부터 쓰기까지 직렬화한다.
미완료 결과와 미측정 비용은 완료값으로 바꾸지 않는다.
현재 GPU 작업을 재시작할 필요 없이 갱신된 script의 `all results`로 추출할 수 있다.

2026-10-12부터 Qwen·Llama·Gemma의 공개 `status`와 `results`는 같은 snapshot
loader를 사용한다. 상태 표, JSON 항목, seed별 초기 캐시 성공률, 최종 reward,
비용 누락과 오류 판정 기준이 같다. 초기 캐시 성공률은 400개 후보의 각 8개
응답으로 구하며, 최종 held-out 평가 reward와 별도로 표시한다.
한 seed의 입력/캐시가 잘못됐으면 그 seed를 INVALID로 표시하고 나머지 작업은
계속 보여준다. 조회는 학습을 시작하거나 기존 실험 기록을 수정하지 않는다.

```sh
sh scripts/run_srgc_qwen35.sh all status
sh scripts/run_srgc_llama31.sh all status
sh scripts/run_srgc_gemma4.sh all status
```

`status`만 쓰면 `all status`이며 `--json`을 붙이면 같은 snapshot을 JSON으로
출력한다. `math status`/`mbpp status`로 범위를 제한할 수 있다. `results`도
동일한 축약형·데이터셋 선택을 지원한다.

275는 마지막 학습 update이며, 이후 300문항 × 8응답의 최종 평가가 남는다.
2026-10-12 공통 status는 이 구간을 `EVAL`로 구분하고, 해당 평가 cache의 저장
문항 수를 확인할 수 있으면 `EVAL 120/300 saved`처럼 표시한다. 저장 파일 수만으로
완료를 판정하지 않는다. 실행 순서와 현재 노드에서 확인할 항목은
[Qwen 275 이후 최종 평가](QWEN35_FINAL_EVALUATION_2026-10-12.md)에 정리했다.

## READY 작업이 있는데 IDLE로 멈추는 배정 오류 — 2026-10-12

Qwen 3.5, Llama, Gemma의 공통 배정기가 공유 resume backlog를 모든 새 작업의
완료 장벽으로 사용했다. 재개 작업을 다른 노드가 실행하고 있어도 READY 작업이
배정되지 않았다. 재시도 대기 중이거나 중지된 다른 데이터셋의 backlog도 같은
대기를 만들었다. 세 모델의 실제 queue/lease를 사용하는 로컬 테스트에서 재현했다.

현재 실행 entry는 별도 `ready_dispatch.py`에서 재개 작업을 우선 배정하고,
당장 재개할 작업이 없으면 요청한 데이터셋의 독립적인 READY 작업을 배정한다.
`math`/`mbpp` 명령은 다른 데이터셋의 기존 미완료 작업만 복구하며 그쪽의 새 작업을
자동으로 시작하지 않는다. `all`은 두 데이터셋의 READY 작업을 모두 배정한다.
실패 한도에 도달한 작업은 반복하지 않고, 나머지 독립 작업을 처리한 뒤 실패한
작업 이름을 출력하며 종료한다. 요청하지 않은 데이터셋의 stop 기록은 유지한다.

고정된 engine·모델 adapter·기존 resume 코드의 파일과 해시는 유지한다.
실험 plan, 입력, 결과, optimizer checkpoint, 과거 실패 기록과 retry budget을
초기화하지 않는다. 실행 중인 Python worker는 배정기를 다시 import하지 않으므로
업데이트된 코드로 **IDLE worker를 종료하고 기존 시작 명령으로 다시 실행**해야 한다.
정상 학습 중인 노드는 해당 작업을 마친 뒤 재시작한다.

```sh
git pull --ff-only origin master
# 해당 모델의 IDLE worker를 Ctrl-C로 종료한 뒤 같은 노드에서 실행
sh scripts/run_srgc_qwen35.sh all
sh scripts/run_srgc_llama31.sh all
sh scripts/run_srgc_gemma4.sh all
```

위 세 시작 명령은 각 모델에 할당된 별도 노드에서 해당 명령 하나만 실행한다.
원격 노드의 실제 재시작과 GPU 진행은 로컬 queue 테스트와 별도로 확인해야 한다.
원인, 보존한 코드 해시와 검증 범위는 [수정 기록](SRGC_READY_IDLE_FIX_2026-10-12.md)에 있다.

같은 날 추가 점검에서 worker 시작 시 다른 데이터셋의 기존 backlog를 확인하기
전에 완료로 종료하는 경로도 수정했다. 두 데이터셋의 model snapshot/runtime을
먼저 검증하고 실행하며, Qwen도 작업 사이 GPU 메모리 해제를 최대 30초 기다린다.
SIGINT/SIGTERM 중단은 실패 재시도에서 제외하고 shell에도 중단 코드로 전달한다.
Information queue는 collector 예외를 failed 기록에 남기고, 종료 신호를 무시하는
자식은 최대 30초 뒤 해당 process group을 정리한다. 기존 retry revision은 유지한다.

## 이전 실행 기록 — 2026-09-25

Recorded: 2026-09-25 04:21 KST (2026-09-24 19:21 UTC).
Source: the user's explicit report in the current conversation, not a remote
process-health check. This is a handoff snapshot, not live status.

| Experiment | User-reported active count |
| --- | ---: |
| run pair | 2 |
| SR-GC switch | 8 |
| RLOO | 1 |
| Total | 11 |

Use this allocation when discussing additional nodes, timing, or restarts.
Do not conflate the two Pair workers with the eight Switch workers. Do not
restart healthy jobs merely to update status/results scripts. Preserve all
checkpoints, including optimizer state. Confirm newer logs or operator reports
before treating a job as failed, completed, or available for reassignment.

Switch inspection commands (no GPU work; existing jobs need not restart):

```bash
bash scripts/run_selector_pair_switch_rewards.sh status
bash scripts/run_selector_pair_switch_rewards.sh results
```

Exports: `~/selector-pair-switch-status.txt` and
`~/selector-pair-switch-results.txt`. Phase wall time and GPU-hours are separate;
their sums across workers are not the concurrent job's elapsed completion time.

## Pair Two-Final Resume (2026-09-25)

The remaining branches identified by the user are On-policy `s1/t50`
(`selection_reduced`) and Random `s4/t100` (`random_full`). The copied
`~/hash.txt` diagnosis pins missing checkpoints 355 and 155 and reports a
recovery-runner hash mismatch; newer saved final policies were inventoried at
457 and 256. These are copied observations, not live GPU status.

`bash scripts/run_selector_pair.sh` now automatically resumes their final and
curve evaluations when the other 40 sealed results are present. It validates
the current final policy, optimizer, source inputs and lineage, without
rebinding the obsolete recovery plan or training again. A busy source/output
lease is respected. Failed evaluations resume only missing shards.

The original results, checkpoint files and cost ledgers are unchanged. New
outputs use `runs/selector-pair-final-eval-v1/seed-{1,4}` by default
(`PAIR_FINAL_EVAL_ROOT` overrides it). The existing `status` command shows
running and completed evaluations, explicitly excluding these over-budget
results from matched-budget paired comparisons. Completion of GPU evaluation
must be confirmed on the allocated node; local fixture tests are not evidence
that a remote job restarted. No web publication is part of this change.

Local verification: 384 tests passed, 8 CUDA-dependent tests skipped. Covered
normal/pinned launcher dispatch, historical runtime handoff, source preservation,
resume-only-missing-shards, busy leases, status, budget recovery and shutdown.

## Export All 42 Endpoints

`bash scripts/run_selector_pair.sh results` and the existing
`bash scripts/run_selector_pair_results.sh` now read the original endpoints and
the two sealed saved-final evaluations together. They run on CPU without
acquiring `primary.lock`, starting training, or rerunning evaluation. The
default TXT is `~/selector-pair-results.txt`.

The exporter checks each supplemental result's seal, plan, source identity,
current policy manifest and question coverage, and includes both final reward
and curve points in `branch_measurements`. A valid canonical result always
takes precedence; the same branch is never counted twice. Only 40 valid
original endpoints plus two valid saved-final evaluations produce `42/42`.
Missing or damaged measurements remain missing, with their errors retained.

`branch_results_complete` and `branch_completion` describe endpoint coverage;
the existing `complete` field still describes strict paired validation, not
endpoint coverage. Supplemental results remain explicitly ineligible for
matched-budget paired comparison. Their presence no longer waits on the
original strict paired report; `--validate-pairs` requests that separate check.
`--final-eval-root` or `PAIR_FINAL_EVAL_ROOT` selects a nondefault output root.
Old `budget-recovery` results are preserved separately and are not substituted
for the newer final policies. No experiment files or web publications change.
