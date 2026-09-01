# Multi-Cluster Admission Lock Incident

Date: 2026-09-01 KST

## Impact

Three independent four-H100 clusters launched the same canonical command against
one `GROUP_VOLUME`. Only one entered the experiment queue; the other two exited
before claiming work. This wasted scheduled cluster time and delayed the primary
matrix.

## Root cause

`run_rlvr.sh` used `hostname` to name a process-lifetime admission lock and put
that lock below `$OM_WORK/locks` on the shared volume. The three cloned cluster
environments reported the same hostname. They were therefore incorrectly
treated as duplicate processes on one physical node.

The seed/dataset family queue itself was not the defect. Its nonblocking shared
locks correctly assign one complete checkpoint chain to one worker and release
claims after process death. The invalid shared launcher lock prevented two
workers from ever reaching that queue.

## Correction

Revision `fb48565dda10d35aa58452b0b37696ae122558aa` makes the lock boundary
explicit:

- `primary.lock` and `additional-suite.lock` live in node-local
  `/tmp/offpolicy-misranking-$UID` by default.
- A configured `OM_LOCAL_LOCK_DIR` below `GROUP_VOLUME` is rejected.
- Shared storage retains only seed/dataset family locks, collection locks,
  immutable contracts, artifacts, logs, and the final harvest lock.
- Each launcher receives a random worker suffix, preventing same-hostname log
  and smoke-marker collisions.
- Additional experiments wait on the same node-local primary lock before using
  that node's GPUs.

## Current `295dfea` run recovery

Do not pull or edit the checkout of the worker that is still running. To join
the two stopped clusters to the same `295dfea` experiment without changing the
recorded source revision, run the following from their existing clean checkouts,
using a different label on each cluster:

```bash
# stopped cluster 2
bash -c 'hostname(){ printf "rlvr-cluster-2\n"; }; export -f hostname; exec bash scripts/run_rlvr.sh'

# stopped cluster 3
bash -c 'hostname(){ printf "rlvr-cluster-3\n"; }; export -f hostname; exec bash scripts/run_rlvr.sh'
```

This confines the hostname override to the launcher process, creates distinct
legacy admission-lock names, and leaves the checkout and Git provenance clean.
Both workers then enter the existing shared family queue. Future launches should
pull `fb48565` or later and use the ordinary command with no override.

Do not pull the checkout of a process that is still running. Revision
`a223ee31bd0542cbe6c35af66cfb3549d0a8c40e` covers the later stopped-process
case: after a worker exits, a newer supervisor may be pulled and launched with
the ordinary command. The matrix-level `generation.git` marker is created
without requiring a pre-existing snapshot or manifest. If partial artifacts
record an older commit, the supervisor automatically restores that commit in a
node-local detached worktree and resumes only the missing or invalid points.
Mixed provenance fails before a family claim.

## Verification

- Full suite: 86 passed.
- Three launchers with one cloned hostname all entered both matrix phases.
- Three shared-queue workers all claimed at least one family; every cell ran
  once, aside from one intentionally killed point that was retried exactly once.
- The launcher and queue pair passed five consecutive repetitions.
- A duplicate launcher sharing one local lock was rejected.
- A node-lock directory below the shared volume was rejected.
- A three-worker empty matrix selected one atomic generation commit; an
  interrupted matrix resumed after a simulated pull using the recorded commit.
- Malformed markers, mixed run-config commits, and an explicitly wrong resume
  checkout were rejected before new work was claimed.
- Python compilation, fatal Ruff rules, shell syntax, JSON parsing, dependency
  integrity, and whitespace checks passed.

Actual H100 scheduling remains cluster-runtime evidence to capture from the
three console logs; this audit host has no access to that cluster volume.
