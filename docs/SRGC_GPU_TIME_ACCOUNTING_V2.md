# SRGC GPU time accounting v2

The current status/results path reads GPU costs through
[`scripts/srgc_gpu_accounting.py`](../scripts/srgc_gpu_accounting.py).
It rebuilds derived totals from existing receipts; it does not change the learner,
recorded experiment identity, checkpoints, rewards or original measurement files.
The accounting schema and reader SHA-256 distinguish this calculation from archived
summaries.

## Units and scope

A receipt records allocated GPU-seconds, not GPU utilization or CUDA kernel active
time. Where both fields exist, its GPU-seconds must equal synchronized wall seconds
multiplied by the recorded allocated GPU count. Conversion to GPU-hours divides by
3,600 once. Four allocated GPUs for ten seconds are 40 GPU-seconds, or
0.011111 GPU-hours. Rank allocation and stage partitions are checked when recorded.

| View | Meaning | Addition rule |
| --- | --- | --- |
| Recorded phase costs | Startup, preparation, selection, training, checkpoint and evaluation receipts | Sum distinct operation IDs within their bound scope |
| Inclusive invocation | The allocation interval containing its measured phases and orchestration | An alternative total; never add its contained phases again |
| Shared prefix/cache | Work shared by a seed's continuation arms | Count once in experiment allocation totals |
| Real replay | A new operation ID, including a retry at the same checkpoint | Keep it in recorded resource cost |
| Receipt copy | Another file describing the same operation ID and measurements | Exclude the repeated copy; keep all source files |
| Queue wait | No measured allocation interval in the input evidence | Unknown, not inferred from worker age or receipt mtime |

Finished receipts contribute measured costs. An unclosed start has unknown duration
and makes the scope incomplete. The reader does not fill that interval with zero,
current wall time or a predicted runtime. A measured zero remains distinct from no
measurement. Partial detail displays use per-phase completion evidence; ambiguous
zero defaults in older partial aggregates stay unknown when no completed phase
can be identified. Conflicting copies, invalid allocation conversions, nonfinite values,
nonreconciling stages and wrongly scoped invocation/phase receipts are errors.

A cache's archived aggregate is retained where raw receipt files are unavailable;
per-operation deduplication cannot be reconstructed from that aggregate alone.
Where raw cache receipts exist, they are read without rewriting the saved summary.
Live partial snapshots remain separate and are never added to finalized phase totals.

## Comparison time versus consumed resource time

The submitted OLMo continuation criterion remains unchanged: retained-update training,
full cumulative selection and one common SR subset preparation charge. Its cache,
shared prefix, evaluation, checkpoint, startup and discarded/replayed work remain
outside that particular comparison. These exclusions do not mean the resource cost
disappeared.

The all-attempt GPU table contains distinct measured operations, including real
retries. Restart reconciliation now also exposes `unique_recorded_gpu_seconds` and
`replayed_gpu_seconds`. These are separate from legacy raw-file sums and copy
subtraction fields, which remain available for historical interpretation. For a
complete retained history:

```text
unique recorded selection/training = retained selection/training + real replay
raw file sum = unique recorded operations + duplicated receipt copies
```

For example, two retained operations costing 8 GPU-seconds in total, three real
8-second replays, and one copied 4-second receipt give 32 unique GPU-seconds,
24 replay GPU-seconds and 36 raw-file GPU-seconds. The copied 4 seconds are not
another consumed allocation. Unknown interrupted intervals remain additional
unmeasured costs.

## Verification and use

The old reader returned 80 GPU-seconds for one 40-GPU-second operation whose receipt had
been copied. The new reader returns 40, retains a second real operation with a new
ID, and rejects conflicting copies. Tests include three public model status paths,
read-only artifact snapshots, session/phase conservation, shared costs, live partial
exclusion, zeros, interrupted starts, unit conversion and invalid data.

The calculation applies when the updated code reads existing status/results data;
recomputing these reports does not require restarting training. Existing running
coordinators do not acquire new launch-history fields through a source update.
No actual remote GPU performance, lease release or worker-registration recovery is
established by CPU tests. Missing historical measurements stay missing.
