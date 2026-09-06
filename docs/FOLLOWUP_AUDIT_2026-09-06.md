# Follow-up audit: September 6, 2026

## Scope and baseline

- Repository: `offpolicy-misranking`, branch `master`.
- Initial baseline: `e26ed6d` (49 commits since midnight KST), not the older
  `audit/p1-integrity` checkout. Concurrent status commits through `aa6a493`
  were also inspected as they appeared.
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
9. **Recovery cause overwrote point state.** Concurrent commit `3233a47`
   reused `kind` for a recovery error, replacing the active-point classification.
   The row lost its drift and running marker. A separate `recovery_kind`
   retains the point state; a regression exercises the actual compact output.
10. **An error-search regex could not compile.** Concurrent commit `4b8556f`
    introduced an unmatched opening parenthesis in both Qwen grep patterns.
    The shell test caught the resulting stderr. Both searches now share a
    valid base error pattern while recovery/backoff matches remain supported.

## Verification

- Added `tests/test_status_reward_audit.py`: reward/parser examples, synthetic
  OLMo status fixtures, and execution of the actual Qwen shell script with
  isolated logs and a stub process probe.
- Initial regression run before the fixes: 11 failures, 3 passes.
- Expanded targeted suite includes the existing status, launcher safety and
  September 6 reward/EOS regression modules.
- Replaced the obsolete no-Git-keywords launcher test with execution of all
  16 launch/prepare/check/doctor combinations. Added 8 isolated checks that
  status uses only fast-forward merges and does not merge on dirty/offline
  checkouts or reset a diverged branch. The intentional status auto-update
  behavior was not removed to satisfy the older string-based assertion.
- Ruff checks on changed Python modules and the new test module passed.
- `bash -n scripts/status_qwen35.sh` and `git diff --check` passed.
- Initial full run: 241 passed, 1 obsolete launcher assertion failed.
- A subsequent shared-checkout run caught the concurrently introduced grep
  error (267 passed, 1 failed). Final verification uses a fixed copy to avoid
  source changes during collection/execution.
- Fixed-copy full suite: **269 passed in 132.07 seconds**. The copy was taken
  from `aa6a493` plus the working fixes in this audit.
- Final shared-checkout targeted suite: **74 passed in 3.36 seconds**, including
  two additional end-to-end grep checks for `[abort]` and legacy GPU failure
  markers after consolidating/escaping the error pattern.
- Interpreter: `.work/.venv-cu126/bin/python` (PyTorch 2.7.1+cu126). The system
  Python lacked torch/math-verify and was not used for the full validation.

## Operational limits

No cluster job was launched, restarted, killed, or modified. No checkpoints,
rollouts, or result files were deleted or rewritten. Real multi-GPU GRPO,
NFS lock visibility, and CUDA/FLA behavior were not validated on the cluster.

Reward changes do not repair already recorded rewards or trained policies.
Existing responses containing structured commas or multiple `####` answers
need an impact audit before affected results are reused. Re-scoring an
evaluation alone cannot undo a training update made with a wrong reward.
