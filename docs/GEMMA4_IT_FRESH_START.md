# Gemma 4 12B IT fresh start

The normal `scripts/run_srgc_gemma4.sh` command now selects
`google/gemma-4-12B-it` at revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, rather than the PT checkpoint.
This is a different initial policy, recorded as instruction tuned in every plan.
No previous PT accuracy is relabeled as an IT measurement.

To cancel the old PT comparison and start with both initial caches empty:

```sh
git pull --ff-only
bash scripts/run_srgc_gemma4.sh all reset
```

Run `reset` once on one node. After that command initializes the shared root,
additional four-GPU nodes join with the ordinary command:

```sh
bash scripts/run_srgc_gemma4.sh all
```

The default new root is `$OM_WORK/srgc-rebuttal/gemma4-12b-it-v1`.
`reset` marks known previous Gemma PT queues for immediate stop, retaining their
original inputs, responses, checkpoints and timing records. Existing IT output
is moved to a timestamped sibling archive under an exclusive lifetime lease;
the new root has no cached rewards, rollouts, preflight receipts or checkpoints.
A reset refuses roots belonging to another model. It does not reset Qwen,
Llama or OLMo. Do not run `reset` independently on every node.

The IT snapshot is discovered in shared model storage, or fetched at its pinned
revision before GPU admission when absent. Explicit `SRGC_GEMMA_IT_MODEL_PATH`
(or the older generic `SRGC_GEMMA_MODEL_PATH`) must point to IT weights.
Unset a generic variable pointing to PT before starting the IT command.
Model files, tokenizer vocabulary and official chat template are checked against
pinned hashes. Existing PT weights are not renamed or overwritten.

Every task split uses the official tokenizer chat template, exactly one BOS and
all official stop IDs `[1, 106, 50]`. Thinking is explicitly off, matching the
Qwen/Llama comparison setting and the unchanged 2,048-token response budget.
The comparison retains temperature 1, top-p 1, top-k 0, eight responses, seeds
5–9, original cohorts and verifiers, 25 shared-prefix updates and 275 total
updates. The two official modality suppressions apply equally to generation
and normalized training log probabilities. These comparison settings differ
from the model card's native top-k/top-p and thinking-enabled benchmarks.

Both MATH and MBPP initial caches finish before MATH training. For each dataset,
the fixed eight-prompt/eight-response preflight must reach at least 25% success
and contain no empty responses before full caching or training. One correct draw
out of 64 no longer passes. This 25% floor is an operational guard against a
clearly degraded baseline, not an estimated model capability or a 70% target.
An unchanged failed diagnostic is read on CPU and pauses the run without
repeating expensive GPU admission. The floor is recorded in the receipt identity.
Preflight saves raw responses, token IDs, actual verifier rewards and truncation
counts. Its small-panel results are not full initial-cache accuracy. No 70%
accuracy assertion is inferred from a model-card score on another dataset.
Training retains rewards and checkpoints at least every 25 updates and exports
training-batch rewards separately from fixed held-out reward curves.

```sh
bash scripts/run_srgc_gemma4.sh status
bash scripts/run_srgc_gemma4.sh results
bash scripts/run_srgc_gemma4.sh stop --now
bash scripts/run_srgc_gemma4.sh resume
```

`status` and `results` remain CPU-only, with compact terminal output and JSON
export. `status` immediately prints the model and root being read, skips a full
recursive walk of rollout/checkpoint storage, and shows the preparation lease
and fixed-panel progress (including a saved PASS/PAUSED result verified against
the raw receipts). Known historical PT roots are listed separately with their
inspection command; their scores are never folded into IT status. A missing plan
is still reported as incomplete. Historical text-v3 PT inspection uses
`SRGC_GEMMA_IT_PROTOCOL=0 SRGC_GEMMA_TEXT_PROTOCOL=1` plus its original root;
original PT inspection uses `SRGC_GEMMA_LEGACY_PROTOCOL=1` plus its original root.
Use the experiment's pinned code for exact reproduction.

Primary references: [Google model card](https://ai.google.dev/gemma/docs/core/model_card_4),
[pinned IT repository](https://huggingface.co/google/gemma-4-12B-it/tree/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7).

## Actual-weight diagnosis before a full run

The local diagnostic uses the production `build_cache.generate_rewards` function,
the official IT tokenizer and the same MATH/MBPP verifiers. It verifies the entire
checkpoint against pinned hashes before loading, retains prompts, response token
IDs, binary rewards and source/data identities, and writes partial/error records
if interrupted. Existing output files cannot be overwritten.

```sh
python -m srgc_research.dispatch.gemma4_it.probe \
  --snapshot /path/to/gemma-4-12B-it --output /path/to/new-diagnostic.json \
  --profiles comparison,official --prompts 8 --responses 8 --tokens 2048
```

`comparison` retains temperature 1, top-k 0, top-p 1. `official` uses the pinned
generation configuration (temperature 1, top-k 64, top-p 0.95). Both retain the
official stop and modality-suppression tokens, thinking off, identical candidate
IDs, and the same per-candidate random seed. The `greedy` control uses one draw.
The optional `--gpu-offload` is a local memory workaround, not the H100 protocol.
Smaller prompt/response/token budgets must be reported as such. A truncated
answer is retained and counted; it is never silently removed from the denominator.

The [2026-10-22 recorded audit](evidence/gemma-it-audit-2026-10-22/manifest.json)
includes actual BF16 CPU generation from the pinned 12B IT weights. With the
comparison sampler and the full 2,048-token budget, the fixed MATH age regression
returned the correct answer 6 in 386 tokens, and the fixed MBPP inclusive-range
sum regression passed its original tests in 682 tokens. Both stopped normally.
Each has one sampled response; this deliberately chosen regression pair is not
an estimate of full-cohort accuracy. Separate 64-token greedy arithmetic/code
canaries also passed. No diagnostic performed optimizer updates.

Both matched native/comparison controls on a different MATH geometry question
hit the artificial 512-token cap and scored zero. Their subsequent MBPP draws
were interrupted without completed receipts and remain unmeasured. Different
question slices prevent attributing the full-budget results to the longer budget
or the sampler. Full initial-cache accuracy and held-out performance remain
unmeasured. All six completed saved responses were independently decoded and
rescored; data hashes and exact implementation digests were checked. See
[replay evidence](evidence/gemma-it-audit-2026-10-22/replay-verification.json) and
the adjacent `verify_probes.py` for reproduction without new generation.

The diagnostic panel is candidate data. Its score is neither fixed held-out
evaluation nor full initial-cache accuracy. Even a successful panel cannot
certify that the full cache will reach 70%. Measure the full cache separately.
`--canaries --profiles greedy --tokens 64` instead checks two explicitly labeled
synthetic arithmetic/code instructions. These positive controls are never pooled
with candidate or held-out scores. The read-only audit also exports the raw
preflight panel, recognizes every official stop token, and can rescore saved
responses to expose a mismatch between recorded rewards and the actual verifier.
For caching without training or model-chain transitions, use:

```sh
SRGC_CACHE_ONLY=1 SRGC_MODEL_CHAIN=0 bash scripts/run_srgc_gemma4.sh all
bash scripts/run_srgc_gemma4.sh audit --rescore 20 --output /path/outside-run/cache-audit.json
```
