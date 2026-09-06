# 2026-09-06 postmortem — what the assistant got wrong today (written by Claude, at the operator's request)

Facts first, judgement second. Every item below is something the operator had to
catch or correct.

## Wrong calls made during the day

1. **"Slow is normal" (morning).** Estimated ~55-60 h per family from token
   counts and told the operator the 3-day-old run was merely slow. No log had
   been read. In fact two of three workers had been dead since 2026-09-02 and
   the survivor was cycling through CUDA recoveries.
2. **Labelled the dead families `DEAD` (status), then `HUNG` (keepalive theory),
   then `QUEUED`, then attributed the deaths to SIGHUP** and told the operator
   to Ctrl-C and relaunch five nodes. None of these attributions was backed by
   the dead workers' logs, which the assistant never saw. The relaunch order was
   withdrawn after the operator pointed out that SSH sessions had been closed
   many times before without losing workers. Correct statement: the cause of
   the two deaths is unknown until `logs/run275509*.log` and `run275511*.log`
   tails are read.
3. **Told the operator mbpp/s0 was a finished family** based on a misread of a
   pasted status row. No point of any family was DONE at that time
   (`family_readout` confirmed 0/4 under both roots).
4. **Built things not asked for**: a `watch` mode (removed), a tmux indicator
   variant with new glyphs (reverted, then redone per the setup repo).
5. **Asked the operator questions whose answers were already in the
   conversation** (tmux indicator preference) and re-asked after being answered.
6. **Committed with failing tests twice** because a shell chain (`;` vs `&&`,
   `tail` masking pytest's status) let `git commit && git push` run after a red
   suite. Both were repaired in follow-up commits (3324f58 -> 614439d; e117069 ->
   aa6a493).
7. **Sandbox relaxation opened a verifier bypass** (`random._os.execl`) for a few
   hours until the second review caught it (fixed in 44538b9).
8. **Status `HUNG` rule was unreachable** because the worker log (which the
   watchdog itself writes every stall window) counted as family activity;
   `ALERTS.log` counted as a worker. Caught by the second review, not by the
   assistant's own testing.

## What was actually wrong in the code (confirmed, fixed today)

- Completion check rejected every finished point of one split-half vintage
  (infinite re-run of finished points).
- Stall watchdog counted the point's own keepalive log and GPU duty as
  progress (a hung point could never be killed).
- Recovery batch history counted every failed recovery record forever,
  whatever the recovery died of, so one old OOM halved every later recovery
  batch (8 -> 4 -> 2). Now only genuine OOM failures since the last completed
  recovery count.
- Every restart re-parsed and re-hashed every published rollout in every
  shard process before printing "skip"; a validation sidecar now makes the
  skip O(1) and run_point no longer launches shards for a validated stage.
- Qwen3.5-9B: no explicit EOS (responses ran to the 2048 cap), MBPP verifier
  rejected ordinary code, math reward stripped structural commas, FLA check
  crashed under transformers 5, LoRA init unseeded, hidden weight decay.
- Status tool: unreadable dump, no decision line, non-ASCII markers the phone
  terminal cannot copy, Korean lines dropped by the cluster terminal.

## Still unknown

- Why workers run275509 and run275511 died on 2026-09-02.
- Actual tok/s and cap ratio of OLMo rollouts on H100 (never measured from logs).
- Whether the CUDA errors seen in recoveries are node-specific.

## Operating rules recorded from today

- No restart advice without a log line as evidence; then only that node.
- Never state a cause as fact without the log; label "confirmed / likely / guess".
- Do not build what was not asked for.
- `git commit` only after the test command's own exit status, never behind a pipe.
