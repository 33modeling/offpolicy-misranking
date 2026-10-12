# Current Experiment Allocations

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
