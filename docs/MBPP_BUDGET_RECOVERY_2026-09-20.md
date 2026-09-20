# MBPP Exhausted-Allocation Recovery

## Evidence And Failure

The user supplied `~/mbpp_why.txt`, captured at 2026-09-20 02:00:15 UTC
from checkout `8fe216a`. The current quality condition reported five historical
failures, but the bounded attachment includes only the latest two:

| Branch | Cap (GPU-s) | Recorded Used (GPU-s) | Excess (GPU-s, Approx.) |
| --- | ---: | ---: | ---: |
| s4/t50/selection_reduced | 28376.929754570127 | 28383.562 | 6.632 |
| s4/t25/random_full | 28380 | 28389.436 | 9.436 |

The first branch's log reached update 140 and then recorded an NCCL/CUDA
unspecified launch failure. This is evidence of training progress, not proof
that checkpoint 140 survives or is valid. The replacement node passed admission.
The attachment does not establish a four-node scheduling ceiling, current
cluster completion, or the contents of the actual checkpoint directories.
Older cost-inclusive root failures are separate from this quality condition.

The implementation gap was that the exhausted-allocation guard rejected
continuations before considering valid intermediate checkpoints. Normal final
policy publication could be recovered, but interrupted checkpoint evaluation
had no equivalent path. A second, distinct constraint remains: recorded costs
above the frozen cap cannot pass canonical fixed-budget result validation.
Treating an exhausted branch as DONE, resetting costs, increasing caps, or
restarting from the parent would hide these failures rather than repair them.

## Repair

The operational queue wrapper now intercepts exhausted MBPP branches under the
original branch task lease. It preserves the original trainer, selector,
protocol files, policies, result validators and all prior cost records.

`scripts/mbpp_budget_recovery.py`:

- Validates the frozen inputs, parent lineage, configuration, subset, complete
  checkpoint hashes and per-update statistics using the existing validators.
- Selects the latest valid full checkpoint if final publication is absent,
  pins that choice before evaluation, and never restarts training or selection.
- Evaluates the saved continuation and available registered curve points with
  the original questions, sample counts and sampling seeds. Reuses matching
  sealed canonical evaluation shards and resumes only missing recovery shards.
- Records new work only on a separate reporting ledger under
  `<branch>/budget-recovery/`. The original ledger is not waived or reset.
- Binds the recovery plan, evaluation shards and result; refuses changed inputs,
  unknown costs, damaged saved work and loss of already published shards.
- Publishes `budget-recovery/result.json` and its seal with
  `evaluation_complete=true`, `canonical_complete=false`, the original costs,
  actual excess, additional evaluation costs and per-question rewards.

These measurements are posthoc recovery evidence, NOT a replacement canonical
`result.json`, a gate-fitting label, an official MBPP test-split benchmark, or
proof of equal-budget completion. Canonical final-policy recovery at exactly
the cap keeps its existing behavior. MATH and Selector Pair do not enter this
MBPP recovery hook. No scientific `src/` file changes are required.

Missing/corrupt saved policies stay in review with no GPU work. When only
quarantined branches remain and all other scoped work is complete, the original
queue's exit 80 releases the node without claiming successful experiment
completion. A missing development result can still block dependent gate work;
this repair does not manufacture that result or train a replacement gate.

The dashboard shows saved recovery evaluations as WAIT with a separate remark,
never increments canonical DONE, and retains the original 48-branch denominator.
`why` now includes checkpoint presence/count/latest-name and recovery-result
presence within its existing 16 KiB cap. Presence is explicitly not validation.
Interrupted recovery reporting events use the original branch task lease during
stale-cost reconciliation, just like the existing curve sub-ledger.

## Verification And Deployment Boundary

CPU tests exercise final/intermediate recovery, corrupted newest-checkpoint
fallback, frozen checkpoint selection, partial shard resume, missing result-seal
repair, input tampering, unknown reporting costs, original artifact preservation,
curve sampling, reuse of existing evaluations, cost recovery ownership and
dashboard non-DONE behavior. Rollout loading/generation is simulated; policy,
lineage, checkpoint/statistics, reward-coverage and artifact validators run.

Validation on 2026-09-20: the final focused recovery/publication/fit run passed
42 tests. Broader status/cost checks passed 142 tests, publication/queue/failure
checks passed 134 tests, and multi-node/quarantine checks passed 84 tests with
two deselected cases of the previously reproduced baseline Pair receipt test.
These suites overlap; their counts must not be summed as unique tests.

The GPU filesystem is not mounted on the development machine. No remote
checkpoint, GPU evaluation, node reload or completed branch has been observed
directly here. Pushing the reviewed operational changes to master makes them
available to existing MBPP controllers at their normal between-pass update;
it does not certify when any controller has adopted them. No remote process is
stopped, no node is allocated and the user's nine-node Selector Pair run is not
restarted by this work.
