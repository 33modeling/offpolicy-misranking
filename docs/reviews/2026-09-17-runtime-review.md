# Runtime Change Review, 2026-09-17

Scope: `12ffa8c..d52ed90`, including the edits initially observed as uncommitted
and subsequently committed in `d52ed90`. No production code or experiment
artifacts were changed by this review. The gate formulas did not change in this
diff. Findings below are reproduced locally, not evidence of a particular
published cluster result being contaminated.

## Findings

### P1: Waived scoring work remains reusable for free

`scripts/waive_stalled_attempts.py:109`, `:152`, `:190`;
`src/selection_switch_gpu.py:1361`.

The expanded candidate policy now admits any recorded failure, including
`fresh-r-candidate` failures before training. No checkpoint is classified as
infrastructure evidence. The full failed event is removed from `cost.jsonl`,
but `discard_outputs()` does not move `fresh-r/`. If three scoring shards finish
and the fourth fails, all four GPUs' elapsed cost is waived while the three
completed shards' files and completion markers survive. `select_once()` skips
those shards on retry. This violates the scoring-inclusive comparison for
`budget` roots such as v1 and long; `matched` reporting-ledger scoring is not
this case.

Reproduced: 400 GPU-s becomes zero, with only shard 3 scheduled on retry.
Discard artifacts belonging to a waived scoring phase, or retain the cost of
work that is reused. Training checkpoint cleanup alone is insufficient.

### P2: One stale event becomes multiple fault strikes

`scripts/_stall_watchdog.py:115`.

Each scan increments the host's strikes even when `event_id` is unchanged and
`workers_of()` returns no workers. A surviving `running` progress record after
its owner disappears is enough. Two scans of the same event produce strike 2,
which the new launcher admission policy blocks indefinitely. There need not
have been a second failed attempt. Deduplicate fault strikes by event identity;
also do not renew the cooldown timestamp for an already handled event.

Reproduced with the same event and no live workers at either scan.

### P2: First-strike cooldown can terminate the node before expiry

`scripts/run_selection_switch.sh:299`, `scripts/run_mopps_comparison.sh:292`;
`scripts/run_experiments.sh:425`.

A fresh first-strike record returns the same exit 78 as a terminal admission
failure. The outer launcher exits after two such passes. If a fresh record is
already present at a pass, the default initial 300-second hold becomes 600
seconds, so two blocked passes can terminate the launcher well before the
1800-second TTL. This also matters when a fault in the switch pass immediately
blocks a subsequent MoPPS pass. Not every timing of a first fault triggers it,
but the advertised automatic cooldown recovery is not guaranteed.

Reproduced using the actual inner TTL block and outer launcher, with only the
hold interval shortened. Distinguish cooldown from terminal admission failure
or wait for the first-strike expiry before applying the terminal failure count.

### P2: Signal evidence does not match the actual meter's receipts

`scripts/waive_stalled_attempts.py:102`; `src/selection_gate_gpu.py:218`, `:272`.

The new killed-attempt classification requires a negative ledger `exit_code`.
The real meter records its own `rc`, initialized to 1 on every exceptional exit,
not the subprocess return code. Thus a worker killed by SIGTERM is recorded as
exit 1. With an existing checkpoint and no watchdog/log/recovery evidence,
the waiver refuses the killed attempt. The new negative-exit unit fixture does
not exercise the production receipt writer.

Reproduced with a CPU subprocess that terminates itself with SIGTERM: worker
code -15, ledger code 1, no waivable event. Preserve structured signal evidence
in receipts and consume it consistently, respecting runtime compatibility.

### P2: Helpers release nodes with unfinished sibling work

`scripts/run_experiments.sh:421` (new sibling loop starts at `:396`).

The new sibling loop does not extend the outer completion condition. Once the
own switch/MoPPS roots are complete, a sibling can return 1 with failed work,
yet the node immediately exits 0 as fully done. It never reaches the next
recovery/waiver/retry pass for that sibling. This loses helpers precisely when
the original roots finish. Include eligible sibling roots in completion when
helping is enabled.

Reproduced with the actual sibling/completion shell block and mock roots.

### P2: Peer pulls do not refresh already loaded launchers

`scripts/run_experiments.sh:367`.

The new updater compares checkout HEAD immediately before and after its own
pull. On a shared checkout, another node may already have advanced HEAD while
this launcher was training or holding. Both queries then return the new HEAD,
so the old launcher functions are never reloaded. Only the node that sees its
own pull advance HEAD is guaranteed to restart. Compare against the revision
loaded at launcher startup, not merely the current checkout before a pull.

Reproduced with the actual update block observing B both times after a peer
has advanced the shared checkout from the running launcher's revision A.

## Verification

- Existing tests: `test_waive_stalled_attempts.py`, `test_run_experiments.py`,
  `test_stall_watchdog.py`: **23 passed** in 76.81 seconds.
- Review probes: **6 passed** in 2.82 seconds, asserting the problematic current
  behavior described above, not asserting that it has been fixed.
- Temporary reproduction script:
  `/tmp/offpolicy-runtime-review-20260917/test_runtime_regressions.py`.
- GPU process discovery was stubbed for launcher tests. Real training jobs,
  cluster roots, and git remotes were not started, stopped, or modified.
- Cluster end-to-end validation was not performed.

Reproduce probes from the repository:

```sh
env PYTHONPATH=src:scripts:/tmp/offpolicy-v4-review-20260915-deps .work/.venv-cu126/bin/python -m pytest -q -s /tmp/offpolicy-runtime-review-20260917/test_runtime_regressions.py
```
