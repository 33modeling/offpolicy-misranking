# Off-policy Misranking Backlog

Last updated: 2026-09-03 KST

This file records defects that can affect the confirmatory interpretation. A
checked source fix does not mark result regeneration complete.

## Pre-result audit fixes

- [x] `OM-2026-09-03-01` Remove the conditional retention bootstrap. The old
  analysis discarded draws with nonpositive fresh gain before forming a ratio.
  The v2 FIRST bootstrap now retains every draw and tests the paired contrast
  `stale_gain - retention_threshold * fresh_gain` after the separate positive
  fresh-gain gate.
- [x] `OM-2026-09-03-02` Match the primary fresh and stale candidate response
  counts. Oracle protocol v3 uses 8 responses for primary R, preserves the
  nested 16-response `r_high_budget` score as a descriptive sensitivity, and
  keeps A/B at 8 responses each.
- [x] `OM-2026-09-03-03` Implement Gaussian ceiling propagation. The v1 lookup
  supports only the registered MATH-500 `(400,40)` and MBPP `(512,51)` designs,
  propagates two-sided reliability endpoints, records its schema, and fails
  closed for other pool sizes.
- [x] `OM-2026-09-03-04` Weight distributed clip, ratio, and approximate-KL
  diagnostics by global response-token counts instead of averaging rank means.
- [x] `OM-2026-09-03-05` Record and enforce the first-pass on-policy invariant.
  New GRPO steps log the maximum absolute token log-ratio and abort above
  `5e-3` before the optimizer step.
- [x] `OM-2026-09-03-06` Scope the theorem to selectors measurable from the
  retained one-sided information and explicitly exclude claims about methods
  that observe the omitted ratio, compute `g11`, or collect current outcomes.
- [x] `OM-2026-09-03-07` State that ranking gradients are measured in the final
  dense merged-model layers while training updates LoRA coordinates, and state
  that one-epoch PPO-form GRPO has no effective clipping pass.
- [x] `OM-2026-09-03-08` Fix CUDA rollout recovery collapsing every failure to
  generation batch 1. Context failures now restart at the configured batch;
  only a current-attempt OOM uses the reduced recovery batch. The old policy
  produced observed d400 fresh-rollout times of 1,245--1,282 seconds per prompt
  at batch 1, roughly eight times the configured H100 batch-8 runtime.
- [x] `OM-2026-09-03-09` Split fresh-rollout runtime telemetry into generation,
  verifier, output-token throughput, length-cap count, and effective batch so a
  low-utilization recovery cannot be mislabeled as undifferentiated activity.

## Required before numerical freeze

- [ ] Recompute `scores_splithalf.json`, `scores_oracle.json`, and
  `oracle_protocol.json` from validated raw micro-groups under oracle protocol
  v3. Do not regenerate valid rollout JSONL files.
- [ ] Regenerate every `REGIME.json`, CSV, and report under regime schema v4
  with 10,000 FIRST replicates. Reject v3 report caches.
- [ ] Confirm every primary row uses `r` and that `r_high_budget` appears only
  as a descriptive sensitivity; record candidate response counts and realized
  token counts.
- [ ] Report Gaussian ceilings only for supported full-pool designs and label
  them model diagnostics unless iid, equal-noise, and conditional-independence
  assumptions survive the final residual checks.
- [x] Re-run the complete CPU suite, theory verifier, and clean LaTeX build:
  159 tests passed; both registered ceiling curves reproduced; the 20-page
  letter PDF has no undefined references, LaTeX errors, or overfull boxes.
- [ ] Run the target-cluster preflight on the committed revision used for final
  analysis.
- [ ] Stop the active batch-1 d400 recovery, deploy the corrected supervisor,
  and resume its durable `.partial` files at the configured batch. Verify the
  launcher reports batch 8 and that per-prompt runtime returns to the
  pre-recovery range before leaving the allocation unattended.

## Provenance note

These corrections were made before inserting or interpreting confirmatory
numbers. No local off-policy experiment process was running during the edit.
Existing policy updates and raw rollout artifacts are not relabeled; analysis
outputs must carry their generation revision and the new analysis revision.
