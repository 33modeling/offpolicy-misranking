# Reproduction and measured costs

Code branch: `master` in
`33modeling/offpolicy-misranking`. The manuscript repository holds a link,
not another active implementation. No historical reward or time is changed.

## Python entry point

Run from the code repository root using the existing Pair/MBPP environment:

```bash
python scripts/run_srgc_rebuttal.py plan --check-inputs --allow-pending-cache
python scripts/run_srgc_rebuttal.py cache --bundle srgc_rebuttal/inputs/seed-5.json --cache-seed 5
python scripts/run_srgc_rebuttal.py --seed 5
python scripts/run_srgc_rebuttal.py --seed 5 --resume
python scripts/run_srgc_rebuttal.py costs
python scripts/run_srgc_rebuttal.py summary
```

The entry point selects `PAIR_PYTHON` for MATH or `SWITCH_PYTHON` for MBPP,
otherwise the existing `${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python`.
It reuses the operational OLMo compatibility check, local model loader and
offline Math-Verify bundle. Do not reinstall packages in the working experiment
environment. `requirements.txt` is only for a new isolated environment.
Large inputs, cache responses, checkpoints and results use group-volume
storage, even when the code checkout is on the user volume. The default is
`$OM_WORK/srgc-rebuttal`; see [CLUSTER.md](CLUSTER.md) for stopped-run migration.

Repeat cache generation and training for seeds 6 through 9. `run` (the default)
and `cache` start four local torchrun processes. `--task prefix` or
`--task random|sr|on_policy|switch` runs a single task. For automatic multi-node
scheduling, use `python scripts/run_srgc_rebuttal.py worker --dataset math`
on each allocated node; it builds missing caches automatically. MBPP uses
`worker --dataset mbpp` on each node with the included seed-5--9 inputs.
The shared-prefix and four-arm schedule is unchanged: selection every 25
updates, fresh training responses every update, endpoint at total update 275.
The 2026-09-28 protocol scores 40 On-policy candidates plus the next 40
unused SR prompts, at most 80 distinct prompts per refresh. SR-GC compares
the means of those 40-prompt sets, not the four-prompt training batches. Random and SR consume the
full candidate pool without replacement within a pass; see [README.md](README.md).

## What is timed

| Phase | Exclusive components |
| --- | --- |
| Cache build | Tokenization, generation, decode, reward verification, response receipt writes, export, separate startup |
| Startup | Tokenizer load, model/adapter load, backend setup |
| Preparation | Cached-SR ranking and selector setup; no fixed Random subset is constructed |
| Selection | 40-candidate/40-SR union and validation timed separately: generation, reward verification, forward, backward, projection, communication; cosine ranking; 40-vs-40 SR-GC arithmetic |
| Training | Fresh generation, reward verification, forward, backward, gradient reduction, clipping/optimizer, communication |
| Evaluation | Fresh generation and reward verification; communication |
| Checkpoint | Model/optimizer snapshot, write, read and restore |
| Invocation | Inclusive wall time from Python main entry through final synchronized completion, including imports and orchestration |

Each rank records prompt, response, generated-token, active backward-response
and zero-advantage-response counts. A skipped zero-advantage backward is not
charged as a performed backward. The scoring union reuses overlap; validation
is one reference, not A/B. SR-GC arithmetic reuses gradients and creates no
extra rollout. The existing protocol also scores the SR comparison set during
On-policy refreshes; those actual costs are retained, not silently subtracted.
On switching, scoring and SR-GC checks stop; fresh training continues.
Selection is not extrapolated by multiplying a one-time measurement by updates.

## Accounting rules

- GPU-seconds are allocated device time, not CUDA-kernel active time. A phase
  uses the maximum synchronized rank duration times the allocated GPU count.
  Convert to GPU-hours by dividing by 3600, never multiplying by GPUs again.
- Stage boundaries synchronize only the local device. There are no per-prompt
  cross-rank barriers; unequal shard lengths cannot deadlock the timer.
- Nested timers subtract child durations. Stage GPU-seconds sum rank-local
  exclusive wall time on GPU ranks. Do not multiply each stage's rank maximum
  by the world size, which would overcount different stragglers.
- Each phase includes an explicit `unattributed_and_wait` remainder. Rank-local
  wall seconds (including CPU work while a GPU is reserved), calls and raw rank
  receipts remain available. For replicated CPU checks, divide summed rank
  wall time by summed calls to obtain average per-call wall latency.
- Complete invocation receipts contain their phases, not extra work to add to
  phase totals. `experiment_accounting` reports their difference as orchestration
  and timer overhead. Node scheduler queue time and process teardown are outside
  this Python invocation boundary. No claim of full scheduler-billed time is made.
- The cache build and common prefix are separate shared costs, counted once
  in actual experiment spend. They are not charged four times to continuations.
  To quote a cold-start method cost, explicitly combine the applicable cache,
  one common prefix and that arm, stating reuse assumptions. Do not mix this
  counterfactual total with the actual four-arm experiment spend.
- Per-arm paired comparisons use selection + training + actual preparation.
  Startup, checkpoint and evaluation remain separate, alongside full continuation
  phase totals. With `--task all`, one model startup and prefix read are shared
  under `all`; with independent tasks they are recorded under the corresponding
  arm. Keep launch topology and GPU model consistent in comparisons.
- Receipts survive checkpoint rollback. Completed repeated work is counted;
  interrupted phases or invocations remain incomplete, with unknown totals
  represented as `null`. An absent phase is zero only in a completed measured
  scope, not evidence that a missing measurement was free.

## Output files

The group-storage `runs/additional-seeds/seed-N/` contains:

- `cost-receipts/{shared-prefix,random,sr,on_policy,switch,all}/`: durable
  phase receipts, exclusive sub-stages, workload counts and raw rank timings.
- `invocations/{task}/`: inclusive process receipts, separate from phases.
- `{arm}-endpoint.json`: reward, selector history and measured cost totals.

`inputs/seed-N.cache/` contains raw per-candidate cache receipts,
`cost-receipts/`, `invocations/` and `cost-summary.json`. Cache summaries bind
to the final input hash. A preexisting cache without complete measurement
receipts remains usable but is not assigned a measured zero generation cost.
`costs` permits inspection before all runs finish; cohort means stay `null`
until every planned seed has the corresponding completed measurements.
Cache generation also publishes `live-costs/` after every completed prompt.
These rank-local cumulative partial records retain generation, verification,
receipt-write and other elapsed stages before the whole cache phase finishes.
`cache_live_rank_costs` exposes them in the cost report. They are not additive
to finalized phase/invocation totals and do not turn interrupted work into a
complete measurement. Earlier records without these snapshots stay unchanged.
`summary` additionally requires all scientific endpoints and rejects mixed
input, plan, implementation or prefix identities.

No 7B GPU experiment or new scientific result is produced by CPU verification.
Fine-grained synchronization adds measurement overhead; all arms use the same
instrumentation, and total elapsed time includes that overhead.

Node-parallel tasks write the same per-seed receipts; their GPU-seconds add
across concurrent nodes, not their elapsed times on the calendar. Worker
heartbeats and task start/end records are orchestration records, never extra
selection/training charges. The final worker publishes both cohort reports.
