# Experiment Operations Audit: 2026-09-21

## Reproduced Defects

- Default MBPP status observed the original quality root while the repair
  launcher worked in a separate root. A verified repair is now the primary
  48-branch view; the original remains visible as history outside that total.
  Repair schema, root identities, source manifest digest and identical copied
  manifest are required. Explicit suite selection and scheduling are unchanged.
- Pair curve evaluation could start a duplicate shard worker after its point
  supervisor exited but a shard still held its lease. The operational adapter
  probes unfinished shard leases after acquiring the point lease and defers a
  busy point through the existing pending path. Independent points and branches
  remain runnable. No lock file is removed and no peer process is stopped.
- One unreadable MBPP cost ledger could abort an entire results export. Ledger
  reads are isolated; readable measured endpoints remain available, while the
  affected cost total stays unknown. A missing or cyclic curve ledger cannot
  certify a closed total.
- An unreadable path in Pair's legacy progress scan could abort status. The
  operational viewer retains its independent known-meter scan, preserving
  readable peers and their RUN state.
- A damaged RLOO arm could remove other independently validated arms from a
  partial export. Operational reporting now isolates arm-local validation
  failures, rechecks the shared contract, and retains certified siblings.
  Invalid arms have no inferred reward; comparisons require both complete
  validated arms. Direct strict point validation is unchanged.

## Verification Scope

CPU tests use real file locks, real subprocesses and the Bash entrypoints.
Curve worker execution is replaced by a CPU fixture, not a paid GPU workload.
Checks include supervisor loss with a surviving shard, partial result reuse,
unchanged lock inodes, byte-preserved policies and costs, duplicate node names,
saved DONE with continuing work, invalid UTF-8, directories in place of files,
cyclic and dangling symlinks, permission failures, and partial TXT exports.

Pair root/runtime/barrier waits were tested with changing peer heartbeats:
their total deadline remains 180 seconds. Completed branches with a held
state-publication lease also stop waiting at the deadline. A separate simulated
four-hour healthy curve keeps its progress and is allowed to finish.

Integrated CPU/process regression: 2,133 passed, 10 skipped (including eight
opt-in CUDA cases). The final guard CLI and wait regressions separately passed
46 tests after the last exit-code cases were added. Bash syntax and generated
handoff-source synchronization checks passed. No remote H100 execution is
claimed. Local integrated report: `/tmp/experiment-operations-20260921.xml`.

These checks do not establish the cause of every remote WAIT. The supplied
older diagnostic showed an exclusive root-lock owner. Current remote lock
ownership still requires a fresh diagnostic; a local test is not proof that a
server allocation resumed. Never unlink a lock or infer owner death from a
hostname or timestamp alone.

The frozen scientific sources, Pair launcher, selectors, trainers, targets,
budgets, checkpoints and measured results are not changed by these repairs.
The isolated restart runtime is pinned to
`95213b3f2317978d2fabbe43f42d688bb55536de`, including the operational curve guard.
The self-contained restart bundle is regenerated from its maintained sources.
Final deployment/guard/handoff regression: 180 passed. This includes staging
the actual pinned Git commit in a temporary checkout and checking that its
guard, launch hook, ownership checks and frozen science match the reviewed
files, with no live checkout update or rewrite on reuse.

## Bash Entry Points

Read-only status after updating the checkout:

```bash
git pull --ff-only
bash scripts/run_mbpp_experiments.sh status
bash scripts/run_selector_pair.sh status
bash scripts/run_rloo.sh status
```

Each results command writes one bounded TXT, including usable partial results:

```bash
bash scripts/run_mbpp_repair.sh results
bash scripts/run_paper_results.sh results pair
bash scripts/run_rloo.sh results
```

Outputs are `~/mbpp-repair-results.txt`, `~/selector-pair-results.txt`, and
`~/rloo-results.txt`. Original MBPP results remain separately available through
`bash scripts/run_mbpp_experiments.sh results`; they are not pooled with repair
attempts as if their cost histories were identical.

For a current blocked Pair allocation, collect one read-only diagnostic TXT:

```bash
bash scripts/check_selector_pair.sh
```

The command prints its output path. It does not restart or stop a worker.
