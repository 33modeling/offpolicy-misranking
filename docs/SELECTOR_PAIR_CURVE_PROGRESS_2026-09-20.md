# Pair curve observation and queue waits

The legacy progress traversal stops after 2 seconds or 2,048 entries. Growing
cost-event and selector trees could hide a running branch's `curve/progress.json`
from both status and idle peer-wait detection. Known meter paths are now read
directly before that bounded diagnostic traversal.

Peer waits observe changes to event/elapsed/timestamp metadata using the waiting
node's monotonic clock. A static old or future timestamp cannot keep an idle node
waiting indefinitely. State locks are never bypassed or removed.

The Pair dashboard checks existing curve meter leases when wall-clock timestamps
are outside the freshness window. It distinguishes ownership from a fresh
heartbeat; finished cost receipts take precedence. No GPU work starts in status.

The exact previous Pair runtime is pinned for upgrade. Historical receipts,
scientific manifests, checkpoints, evaluations, costs and caps remain unchanged;
`pair-curve-progress-runtime.json` records the operational upgrade.

Verification: 403 Pair/compatibility CPU tests passed, including live curves with
clock skew, exhausted traversal budgets, finite dead-peer waits and preservation
of the released runtime's artifacts. This is not remote GPU execution verification.
