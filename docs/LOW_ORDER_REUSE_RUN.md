# Low-order reuse experiment

Implementation of the candidate in
[the Obsidian synthesis](OBSIDIAN_EXPERIMENT_SYNTHESIS_2026-09-12.md).
This is an experimental selector, not a validated improvement or novelty claim.
The user authorized implementation after the methodology review on 2026-09-12.

Before allocating GPUs, read the
[prior-art review and implementation rationale](LOW_ORDER_PRIOR_ART_AND_RATIONALE_2026-09-12.md).
Close precedents exist. The default three arms are a screening comparison,
not a complete novelty, partial-correction repair, or equal-cost evaluation.
No GPU run was launched locally; the full-study controls remain to be settled.

## Commands

Use the existing offpolicy-misranking checkout and environment. No second
repository, clone, venv, or modification of source runs is required.

```bash
git pull --ff-only
bash scripts/run_low_order.sh plan
bash scripts/run_low_order.sh check
```

`plan` only displays the protocol. `check` enumerates finite binary examples on
CPU and checks the score identities. Neither uses a GPU.

On an otherwise idle, allocated four-GPU node:

```bash
bash scripts/run_low_order.sh
```

Default: OLMo MATH d100, five seeds 0..4, eight cached responses, GRPO group
size eight, and three arms: `random`, `pair_u2`, `low_order`. Each arm resumes
the same source adapter and AdamW state for 100 additional GRPO updates.
Evaluation uses 300 disjoint questions with eight responses each. These are
chosen from the existing local MATH training pool after excluding the candidate
and validation questions of ALL source seeds. This is an independent test split
under the existing E5 protocol, not a claim of semantic deduplication or an
untouched external benchmark. An existing independent test JSON may instead
be supplied using `--eval-prompts`.

The command runs validation-gradient computation, numerical calibration,
candidate scoring, continuation training, and independent evaluation. No
candidate responses are generated during selection. Training and evaluation
DO generate responses. One before-policy evaluation is shared across arms
within each seed.

Other allocated nodes may run the same command against the same shared output
root. They skip leased points/arms. A failed task is attempted once per launch,
its successful artifacts remain, and other available tasks proceed. If all
remaining tasks are leased elsewhere, the launcher returns; it does not start
an unlimited retry loop or claim that the whole suite is complete.

```bash
bash scripts/run_low_order.sh status
bash scripts/run_low_order.sh live
bash scripts/run_low_order.sh stop
```

`status` shows all points, validation/scoring progress, every training arm, and
the shared baseline evaluation. `live` follows node launcher logs, including
worker progress every 15 seconds and failure excerpts; newly joining nodes
are included. `stop` targets this suite on the current physical node. A repeated
launch cleans up only this suite's previous local processes. It never kills
unrelated OLMo, Qwen, or E5 jobs. Ctrl-C terminates this controller's worker
process groups; checkpoints and completed scores are retained.

## Scope and configuration

The initial integration uses the existing E5 admission contract: completed
positive-drift MATH points, four-rank one-epoch GRPO, seeds 0..4, top 10%, and
`olmo_rlzero_math` prompt format. It is not a Qwen/MBPP launcher. The default
source is `$OM_OLMO3_ROOT` or the normal OLMo H100-v2 matrix root. Default output
is `$OM_WORK/runs/low-order-reuse-v1`; override it with `LOW_ORDER_ROOT`.

For a smaller first pass, freeze a separate one-seed suite. The launcher
accepts modes before options. Set `OM_WORK` to the existing shared work root
before using these explicit overrides:

```bash
LOW_ORDER_ROOT="$OM_WORK/runs/low-order-pilot" \
  bash scripts/run_low_order.sh prepare --seeds 0 --steps 20
LOW_ORDER_ROOT="$OM_WORK/runs/low-order-pilot" bash scripts/run_low_order.sh
```

`score` stops after scoring and freezing subsets; `train` resumes only the
frozen continuation/evaluation tasks. `prepare` validates and freezes inputs
without loading GPU models. Subsequent `run` without options resumes the
recorded settings, including custom seeds and budgets.

Additional preparation options: `--derivative autograd`, `--geometry identity`,
`--fd-step`, `--micro-batch`, `--eval-k`, `--test-count`, `--random-extra-steps`,
and `--arms random pair_u2 low_order passrate_beta`. Protocol changes require
a different output root. Do not pull scientific scoring changes into an active
suite and assume its cached scores still represent the same experiment.

## Computation and numerical checks

- The current adapter stays unmerged. Only LoRA parameters are trainable.
- The shared validation direction uses LOO reward gradients. Candidates use
  response-length-normalized derivatives to match the GRPO loss.
- Default geometry uses the source optimizer's frozen, bias-corrected RMS
  second moments. It is a linearized proxy, not the full nonlinear AdamW update.
- Default candidate derivatives use central finite differences. Four fixed
  mixed-reward validation responses are compared against autograd at the
  requested step and half-step on each worker. Both checks must pass; the
  smaller step is used. The parameter tensors are restored exactly on failure.
- Calibration is a local numerical check, not a uniform derivative guarantee.
  A failing probe does not silently switch methods. An explicit `autograd`
  suite provides the more expensive reference implementation.
- Full response importance ratios are not clipped or self-normalized. Overflow
  is reported, not converted into a plausible score. Pair aggregation uses
  stable log-space exclusion sums and O(K^2) scalar work.
- The reported sub-1% bound concerns ONLY the population GRPO normalization
  coefficient. It does not bound derivative error, variance, ranking mistakes,
  final reward, or the effect of optimizer clipping.

The finite backend uses a center likelihood pass and two perturbed passes per
candidate response group, plus behavior-policy likelihoods unless cached. It
also pays for validation backward passes and autograd calibration. The explicit
autograd backend computes one backward pass per candidate response. Both use
the existing chunked-logit implementation. No new `torch.compile` workers or
model downloads are introduced.

## Results and cost

Per-prompt scores and beta likelihoods are resumable. Frozen contracts bind
source files, direction, scoring code, selection, and downstream inputs.
Incomplete candidate scores never publish a top-k subset.

- `points/*/scores/p*.json`: scores, moments, ESS, probe details, derivatives,
  log ratios, and candidate work counts.
- `points/*/selection.json`: all frozen indices and scores, including the
  optional pass-rate control.
- `points/*/training/<arm>/`: existing GRPO checkpoints and independent
  evaluation shards with per-arm lineage contracts.
- `cost.jsonl`: phase start/completion, failures, elapsed time, allocated GPU
  seconds, host, and arm. Shared selection cost is not charged to random.
- `results.json`: observed per-seed test reward, change from the shared
  baseline, and difference from random when available. No bootstrap loop.

Equal updates are NOT equal total cost. Historical cache generation, shared
validation/scoring allocation, and hardware-specific runtime still need to be
included in the comparison. `matched_total_cost` remains false. To give random
additional training after measuring scoring overhead, preregister an explicit
`--random-extra-steps` in a new suite; this option does not automatically certify
equal GPU-hour budgets. No GPU throughput or benchmark superiority is claimed.

## Verification

Tests cover exact moment enumeration, the normalization bound, extreme weights,
variable response lengths, LoRA derivatives, parameter restoration, probe
failure, immutable inputs, restart reuse, per-arm continuation budgets, and
subprocess cleanup. Small CPU models include a real Transformers/PEFT LoRA
model. Full OLMo/H100 execution has not been performed locally.

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest -q \
  tests/test_low_order_reuse.py tests/test_low_order_backend.py \
  tests/test_low_order_experiment.py
```
