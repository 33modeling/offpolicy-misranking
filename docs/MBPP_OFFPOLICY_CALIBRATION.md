# MBPP Off-Policy Calibration Follow-Up

## Purpose and Fixed Scope

Fill the missing MBPP off-policy score-calibration measurements for Figure 2.
This is not another policy-training run, continuation reward benchmark, or H
cost-to-target experiment. It does not replace the completed MBPP follow-up.

- Reuse the original OLMo-3 MBPP matrix at seeds 0, 1, 2 and checkpoints 0, 400.
- Six source points, with four estimators (`g00`, `g10`, `g01`, `g11`) per point:
  24 planned calibration rows. The scope matches the existing MATH subset.
- Reuse all 512 candidates and their eight stored behavior responses.
- Split responses by their original index into two groups of four; use
  leave-one-out advantages within each half and the saved ranking validation
  direction, through the existing `stale_splithalf.py` implementation.
- Reuse each source model, adapter, projection, gradient-layer, clipping,
  micro-batch, attention, LoRA, and prompt-format settings. No optimizer runs.
- Select 51 of 512 prompts. Report the measured cross-half gain, half-score
  correlation, and the existing Gaussian prediction. Predictions are not
  relabeled as measurements, and these 24 rows are not independent training runs.
- Recompute four full-group scores per shard (16 per point). Preserve their
  maximum differences from the original scores for review; exporting these
  differences does not by itself establish numerical equivalence.

## Bash Commands

Run on an idle allocation with four GPUs. All experiment options are fixed in
the launcher; the standard existing matrix path is used automatically.

```bash
git pull --ff-only
bash scripts/run_mbpp_offpolicy.sh
```

Read-only status and a single TXT containing current results:

```bash
bash scripts/run_mbpp_offpolicy.sh status
bash scripts/run_mbpp_offpolicy.sh results
```

The result is `~/mbpp-offpolicy-results.txt`. It includes partial coverage,
calibration rows, original input hashes, raw half scores, full-score checks,
and explicit missing/error states. It replaces an old export even if no point
is complete, so an older successful file cannot masquerade as new results.
`results` exits 1 for incomplete coverage but still writes the file.

One allocation is sufficient and processes the points in sequence; up to six
idle allocations can contend for the six point locks. Each point uses four
GPUs and cannot be claimed twice. A launcher makes one pass, skips points
already owned elsewhere, then exports current global coverage. Incomplete
coverage can therefore mean that other nodes still own work, not a failed
measurement. Re-running the same command resumes saved shards.

## Safety and Provenance

- Preflight checks all six existing source points before starting GPU work.
  Missing adapters, incomplete points, wrong dataset/checkpoint, or missing or
  duplicate behavior responses stop execution. Nothing falls back to training.
- Input/adapter hashes and scoring-source hashes are frozen before computation
  and checked again when reading results. Unbound old score files are preserved
  and refused, rather than silently accepted or overwritten.
- Repeated launch does not stop a running owner. Inherited force and Pair/RLOO
  cleanup settings are disabled. GPU workers retain the node and point locks
  even if their controller disappears; only the launcher's own children are
  signalled by its normal interrupt handler.
- `DONE` requires the input binding, complete finite half scores for all four
  estimators and all 512 candidates, matching scoring parameters, and all
  expected full-score consistency records. A held point lock reports `RUN`.
- Original rewards, checkpoints, rollout files and training state are not
  modified. Only new split-half scores, provenance, locks and logs are added
  beside their source point. No blanket GPU reset or process-name killing.

## Verification Record

Prepared after the completed MBPP follow-up was imported into manuscript
commit `582a361`. No GPU experiment was launched from the local development
machine. CPU tests exercise the existing gradient-scoring implementation with
tiny models, the new provenance/coverage checks, real Bash status/results and
already-complete resume, and partial export behavior. Cluster runtime and
scientific measurements still require the commands above on the user's server.

Final local checks: 68 calibration/scoring/export tests and 39 existing
node-ownership, shared-lock and queue-status tests passed (107 total).
Real Bash tests also verified four-worker GPU-index routing without using CUDA,
inherited worker lock descriptors, and refusal of a duplicate node launch.

The manuscript must retain empty MBPP off-policy calibration cells until these
new measurements are supplied and checked. Do not insert MBPP continuation
rewards into the calibration figure.
