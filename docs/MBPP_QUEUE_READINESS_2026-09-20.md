# MBPP queue readiness correction

## Observed and reproduced scope

The user reports that MBPP stops growing beyond four active nodes. No global
four-node cap was found in the launcher or branch scheduler. Four GPUs are
required per admitted node; that is not a cluster-wide node limit.

The local `mbpp-why` attachments are dated September 18 and 19. They contain
historical allocation and admission failures, not sufficient current evidence
to identify why the fifth node is idle now. This correction is not a claim of
remote recovery or a cluster scheduler quota increase.

A concrete readiness defect was reproduced: after all 18 development final
results exist, the original worker attempts a gate fit even while required
convergence curves are missing. Full validation then fails, and that worker
records the fit as attempted and does not retry it after completing curves.
This can leave the six held-out gated branches unavailable for the remainder
of the pass. Repeated passes can also repeat expensive validation while peers
are still producing curves. Missing prerequisites are not failed labels.

## Change

The standard MBPP node queue now defers only MBPP convergence fits until each
development arm has its final result and curve in the point registered by its
frozen `suite.json`. Other protocols and existing models still reach the
original fit. Once ready, the original code validates all artifacts and fits
the gate; file presence does not certify correctness.

This change is confined to `scripts/queue_selection_switch_gpu.py`, which the
standard controller selects with `SWITCH_QUEUE_PASS=1`. Standalone `fit` and
direct worker commands are unchanged. Scientific source hashes, selection,
training, budgets, ledgers, checkpoints, completed results, branch leases and
the 48-branch design remain unchanged. The Selector Pair launcher/runtime is
unchanged. Callback overrides are restored even when the worker raises.

## CPU verification

- The pre-change readiness tests reproduced premature validation, premature
  fit-lock contention, and failure to release gated work in the same pass.
- Missing curves now skip the fit without writes or fit-lock acquisition.
- Ready inputs still run original validation; invalid frozen models are not
  bypassed. Unregistered point directories cannot supply or block readiness.
- A completed development result is reused, pending curves finish, all 30
  held-out branches become eligible at the proper barriers, and a repeated
  pass does not retrain completed work.
- A subprocess test starts four active CPU workers, joins eleven more, and
  observes 15 overlapping branch claims using the real exclusive task leases.
  All 48 branches finish without duplicate claims. Existing development
  results retain their exact bytes and mtimes; the frozen manifest is unchanged.
  GPU admission, training and evaluation are simulated, not remote GPU tests.

One unrelated existing test failure was independently reproduced in a clean
`master` worktree at `8fe216a`: the migrated Selector Pair quarantine test
expects one new receipt, but existing code also creates
`pair-status-runtime.json`. No Pair code or test is changed to conceal it.

Final focused regression: **151 passed, 2 deselected** across MBPP readiness,
15-worker concurrency, node controllers, GPU admission, pinned runtime
launching, branch quarantine and runtime preservation. The two deselections
are the existing Pair receipt test's two parameterizations; tested separately,
one passes and the migrated case fails identically on unchanged `master`.

## Delivery boundary

The correction is isolated on `fix/mbpp-queue-readiness`, based on `master`,
without the separate Selector Pair branch-queue change. No live controller was
restarted and no experiment artifact was edited. Pushing this fix branch does
not move the auto-pulled `master` branch. Applying it to live MBPP controllers
is a separate operational step; current server evidence is still needed to
confirm the reported four-node symptom has been resolved.
