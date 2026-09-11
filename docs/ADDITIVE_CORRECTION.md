# Additive Correction And Terminal TayPO-2 Comparison

September 11, 2026. Experimental scoring extension, not a validated new method.
No existing matrix, E5 selector, GRPO loss, or registered artifact is changed.
The related-work and mathematical comparison is in
[TAYPO_COMPARISON.md](TAYPO_COMPARISON.md).

## Run From The Existing Checkout

```bash
bash scripts/run_additive.sh check
bash scripts/run_additive.sh plan
bash scripts/run_additive.sh
bash scripts/run_additive.sh status
bash scripts/run_additive.sh live
```

`check` runs deterministic CPU algebra checks. `plan`, `status`, and `live`
do not acquire GPUs or run environment setup. The bare command prepares and
then **runs GPU rescoring**, not GRPO training. Defaults are the existing OLMo
MATH d400 points for seeds 0, 1, 2. It requires an allocated four-GPU node and
the existing `.venv-cu126` environment. Do not launch on GPUs occupied by E5,
Qwen, or the original OLMo suite.

Several nodes can use the same command. A point is leased to one node; other
nodes skip it and process another point. Each node scores disjoint prompt
shards on its four GPUs. With three source points, at most three nodes can do
useful scoring simultaneously. There is no infinite waiting loop or automatic
repetition of a failed point in the same pass.

`prepare` validates and freezes inputs without GPU work. For explicit sources:

```bash
bash scripts/run_additive.sh prepare --runs /path/to/completed/point
```

To run a custom source after preparing it, supply the same
`--runs` argument to `run`, or use the matrix/seed/drift arguments consistently.
For example:

```bash
bash scripts/run_additive.sh run --runs /path/to/completed/point
```

Defaults can be overridden with `OM_OLMO3_ROOT`, `OM_WORK`, `ADDITIVE_ROOT`,
`VENV_DIR`, or `ADDITIVE_PYTHON`; no clone or separate code repository is needed.
`stop` stops this suite on the current physical node only. Repeating `run`
cleans up this suite's previous local controller and descendants, then resumes;
it never force-kills another experiment's node owner. Busy GPUs are reported.

## Exact Implemented Scores

Let `r_t` be the current-token ratio, `P_t` the product before it, and `S_t`
the product after it. The source run supplies the clipping cap `C`. Define
`c(w) = min(C, max(1/C, w))`.

- `gadd`: token weight `c(r_t P_t) + c(r_t S_t) - c(r_t)`.
- `tay2_terminal`: token weight
  `c(r_t) * (1 + sum_{u != t}(c(r_u) - 1))`.

Both multiply these possibly signed weights by the source leave-one-out,
unnormalized group advantage, then use the existing projected-gradient and
cosine scoring functions. The source R validation direction alone selects
prompts. A/B evaluation scores do not select the method or tune a parameter.

Clipping is applied **before composition**, not to the final weight. The
formula is explicit because these orders are not equivalent. Additive weights
can be negative and can exceed C; the TayPO-derived composite can grow with
response length even when each component is clipped. Neither is an importance
probability distribution. Negative terms are retained and diagnostics report
their frequency and maximum absolute weight. This is a scoring estimator,
not a change to the policy-training objective.

The raw identities in the analysis require `cap=None` and sufficient support
and moments. They do not establish bias or variance guarantees for these
component-clipped estimators. The CLI deliberately fixes the source cap rather
than searching for a cap using the eventual test results.

The single backward implementation of each method combines the token weights
before calling `prompt_gradient`. It does not compute three gradients and add
their cosine scores. Projection, layer range, ranking split, behavior responses,
advantages, candidate IDs, selection fraction and tie-breaking match the source.

## Artifacts, Recovery, And Cost

Outputs live under `$OM_WORK/runs/additive-correction-v1/points/<source-name>`:

- `experiment.json`: immutable source/code hashes, sampling contract, settings.
- `cache/beta-<prompt>.pt`: CPU behavior log-probability cache, bound to the contract.
- `scores/p<prompt>.json` and `.pt`: atomic scores and projected gradients.
- `scores_additive.json`: published only after exact candidate coverage.
- `subsets/subset-gadd.json`, `subset-tay2_terminal.json`: selected prompt datasets.
- `complete.json`: merged score and subset hashes.
- `comparison.json`, `comparison.csv`: independent A/B gain and descriptive
  agreement alongside the existing g00/g10/g01/g11, fresh, pass-rate and random
  selections. These are not downstream rewards or superiority tests.
- `progress-<shard>.json`, `logs/`: phase, prompt, host, progress and heartbeat.

Each model is loaded separately: behavior probabilities first, then the behavior
model is unloaded before the current model and its scoring gradients are loaded.
Completed prompts are skipped before either model loads. An interrupted prompt
is recomputed from its beginning; no half-gradient is considered complete.
Behavior log-probability caches survive interruption, so a failed current-model
pass need not repeat the whole behavior pass. Corrupt caches or changed inputs
are errors, not silently accepted results.

The heartbeat runs every 15 seconds, including model load and long prompts.
ETA is phase-local and based on finished prompts; lengths can differ. It is not
a promised cluster wall time. The shared `cost.jsonl` records allocated GPU
seconds, phase completion and failed attempts. Hard-killed unfinished events
remain unresolved, not zero cost. Historical response generation, verification,
and validation-gradient computation are not included in these rescoring totals.

No new response generation or verifier calls occur in this scoring command.
There are two gradient estimators per prompt, plus probability computation and
model startup. This is not a CPU-only run and not a claim of speedup over g11.

## What This Release Does Not Do

It does not launch downstream training or copy the reduced E5's outcomes into
a new comparison. The exported subsets are the inputs for a later, separately
frozen downstream extension. That extension must include `tay2_terminal`, not
only compare `gadd` against a weak or arbitrary comparator. Existing E5 results
may be reused as controls only after matching source checkpoint, subset, test
questions, training settings and evaluation settings; otherwise they are not
matched controls.

CPU tests cover the raw identity, the degree-two objective derivative, clipped
composition, signed weights, the combined backward, input binding, resumption,
shard coverage, failed-point continuation and shell CPU modes. Mock-model worker
tests are not CUDA tests. Real model loading, CUDA memory and throughput require
the remote cluster; no GPU execution or downstream advantage is claimed here.

Local verification on September 11, 2026: 72 tests passed across the two new test
modules and the existing method-choice, evidence-downstream, first-interval,
and E5 node-ownership regression modules. This includes a real tiny OLMo-3 CPU
model checking that one composite backward matches three component backwards
after projection. Ruff checks and shell syntax validation also passed. These
checks do not establish compatibility or memory usage on the remote GPU cluster.
