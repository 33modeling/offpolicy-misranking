# Follow-up audit: September 6, 2026

## Scope and baseline

- Repository: `offpolicy-misranking`, branch `master`.
- Baseline: `e26ed6d`, not the older `audit/p1-integrity` checkout.
- Reviewed today's commit history (KST), with focused inspection of reward
  parsing, generation/OOM handling, optimizer safeguards, matrix completion,
  launcher paths, and status diagnostics. This is not a claim that every line
  or every cluster execution path has been verified.
- Today's changes cover Qwen 3.5/3.8 profiles, local snapshot discovery and
  identity verification, launch/preflight logging, OOM retries, GRPO/EOS and
  reward fixes, completion contracts, and OLMo/Qwen health displays.

## Reproduced and fixed

1. **Structural commas corrupted rewards.** The thousands-separator regex
   stripped `(1,234)` into `(1234)`, collapsing distinct mathematical answers.
   Only complete grouped scalar numbers now lose commas. Coordinates, sets,
   intervals and malformed numeric grouping remain intact.
2. **An earlier hash-delimited answer won.** Multiple `####` answers returned
   the first attempt, not the final correction. The final match now wins,
   consistent with the existing final `Answer:` handling.
3. **Status created a phantom worker.** The new `logs/status-history.log`
   was included by `recent_workers`, so merely checking status could report
   a running worker. Status logs are now excluded from worker discovery.
4. **Deferred d0 hid completed and active points.** `current_point` stopped
   scanning at the first unfinished drift. The actual launcher may defer d0
   evaluation until after training. All completion markers are now counted;
   the most recently active unfinished point is displayed. Recency is a
   diagnostic heuristic, not independent proof of useful computation.
5. **Malformed idle telemetry bypassed schema validation.** Fresh records
   claiming `idle-suspected` or `terminating-idle` were trusted before their
   schema was checked. Invalid records now report UNKNOWN, not an asserted
   stall or recovery.
6. **Zero failures became two zeros.** `grep -c ... || echo 0` emitted `0\n0`
   when there were no matches, causing a shell integer-comparison error.
7. **Qwen reported completion without a complete matrix.** A successful
   prepare/check/launcher exit was called DONE even with missing points;
   empty DONE files were also counted. Completion now requires 40 nonempty
   marker files for this registered Qwen status profile; otherwise rc=0
   produces a warning. This display still does not deep-validate artifacts.
8. **History output masked failures.** An `echo` replaced PIPESTATUS before
   it was inspected. Child-status and history-writer failures are now preserved.

## Verification

- Added `tests/test_status_reward_audit.py`: reward/parser examples, synthetic
  OLMo status fixtures, and execution of the actual Qwen shell script with
  isolated logs and a stub process probe.
- Initial regression run before the fixes: 11 failures, 3 passes.
- Expanded targeted suite: 40 passed, including the existing status and
  September 6 reward/EOS regression modules.
- Ruff checks on changed Python modules and the new test module passed.
- `bash -n scripts/status_qwen35.sh` and `git diff --check` passed.
- Full-suite outcome is recorded below when the run finishes.

## Operational limits

No cluster job was launched, restarted, killed, or modified. No checkpoints,
rollouts, or result files were deleted or rewritten. Real multi-GPU GRPO,
NFS lock visibility, and CUDA/FLA behavior were not validated on the cluster.

Reward changes do not repair already recorded rewards or trained policies.
Existing responses containing structured commas or multiple `####` answers
need an impact audit before affected results are reused. Re-scoring an
evaluation alone cannot undo a training update made with a wrong reward.
