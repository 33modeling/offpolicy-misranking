# Experiment Queue Audit

Scope: MBPP, RLOO and Selector Pair. CPU regressions and real local process/lease
tests only; no remote GPU run or server deployment is claimed.

## MBPP

The supplied WHY report contains exhausted branch errors followed by controller
`rc=1` and repeated holding. File timestamps are not used to reject that evidence.
Exhaustion is a branch allocation condition, not proof that the cluster quota or
every independent task is exhausted.

The operational worker performed exhausted-policy recovery inline before later
independent work. A regression reproduces recovery starting before any of 41
independently runnable branches. Recovery now runs after the ordinary queue
pass, reacquires the task lease without waiting, and cannot stop other ordinary
assignments through a recovery failure. Peer ownership is respected. Exhausted
training is never restarted or refunded. Review-only dependencies remain
incomplete and use the existing exit-80 release path, not a completion marker.

Gate-dependent arms still require validated development labels. The fix does not
invent missing labels, fit on posthoc recovery results, or silently change the
frozen gate or comparison conditions.

## RLOO

The original entry point stopped the complete run at the first branch exception.
The shell launcher now uses an isolated operational queue outside frozen `src`
contracts. It records branch failures and proceeds to other arms and source
points. Invalid evaluation receipts never become DONE. Completed work is reused.
Lock acquisition failures are distinct from EAGAIN during task execution.

Runtime/I/O failures require successful bounded NCCL re-admission before another
GPU task. Failed admission stops the node with code 78. Failed passes return 1;
peer-owned incomplete passes return 75. No automatic same-arm retry loop is added.
SIGINT/SIGTERM unwind the existing meter cleanup so children, leases and cost
events are finalized. Training/objective/optimizer/input sources are unchanged.

Because RLOO historically hashes every `src` file, the reviewed Pair observation
patch also needs an exact compatibility pin. Preparation preserves the original
RLOO contract and appends `queue-observation-runtime.json`; only the pinned Pair
change and the reviewed RLOO validation change are allowed. Training and input
tampering still fail. Status uses held meter leases to supplement clock-skewed
heartbeats, without relabeling those heartbeats as fresh.

## Selector Pair

The existing distributed queue isolates branch/state failures, preserves frozen
state order, skips peer leases and bounds idle waits. Budget errors are not
runtime retries; GPU errors require re-admission. Regression coverage now
explicitly includes allocation exhaustion in distributed dispatch. The subsequent
curve observation correction is documented in
`SELECTOR_PAIR_CURVE_PROGRESS_2026-09-20.md`; it changes operational code and records
that upgrade without rewriting scientific manifests or historical receipts.

The quarantine migration test had an obsolete expected-file list: the current
released code also creates `pair-status-runtime.json`. Its assertion now checks
that receipt, the curve-progress upgrade, and runtime bindings while still requiring all prior receipts
and saved work to remain byte-identical. No migration validation was relaxed.
