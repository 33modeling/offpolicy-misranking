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

### Pair startup recovery and duplicate hostnames

Pair previously checked occupied GPU memory without first recovering children
left behind when a phase owner died. The operational node admission helper now
performs local recovery before taking the node lease and before the launcher's
memory check. Recovery requires all of: the exact output root, a unique cost
event environment marker, a log path inside that root, a matching progress or
finished-event receipt, and exclusive acquisition of the existing phase cost
lease. Held or missing leases do not authorize cleanup. The lease remains held
through PID/start-time-checked termination and bounded CUDA PID release checks.
No host-name or wall-clock inference, GPU reset, root-wide process sweep, cost
refund, result rewrite, or scientific-code/hash change is involved. Processes
without this evidence remain untouched; failed CUDA release blocks admission.

On shared filesystems Pair/RLOO node admission also used a bare-hostname fallback,
which collided for distinct hosts with identical names. Updated launchers take
an exclusive boot-identity lock and a shared legacy-hostname guard. Distinct
boot identities can share a hostname; changing a display node ID cannot bypass
the same boot's lock. Existing exclusive legacy-hostname owners remain protected.
This is Linux boot identity, not a claim that every container platform exposes
unique host hardware identifiers.

Validation includes real local process recovery versus held-lease preservation,
same-event/different-root isolation, invalid or missing ownership evidence,
CUDA-release failure, duplicate-hostname ownership, and legacy-lock coexistence.
Remote GPU release and the user's current server launch are not yet verified.
MBPP startup failures and all alternate status views remain separate open checks.
