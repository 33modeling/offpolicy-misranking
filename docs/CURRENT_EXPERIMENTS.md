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
설정을 바꾸지 않는다. 중단된 기존 작업을 먼저 재개하고 새 작업은 그 뒤에 시작한다.
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
