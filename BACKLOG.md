# Off-policy Misranking Backlog

Last updated: 2026-09-05 KST

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
  only an OOM in bytes appended to the failed stage logs during the current
  attempt uses a reduced recovery batch. The old policy produced observed d400
  fresh-rollout times of 1,245--1,282 seconds per prompt at batch 1, roughly
  eight times the configured H100 batch-8 runtime.
- [x] `OM-2026-09-03-09` Split fresh-rollout runtime telemetry into generation,
  verifier, output-token throughput, length-cap count, and effective batch so a
  low-utilization recovery cannot be mislabeled as undifferentiated activity.
- [x] `OM-2026-09-04-10` Replace mtime-only `STUCK` diagnosis with worker
  heartbeats and per-pipeline CPU/GPU telemetry. A quiet log now reports
  `COMPUTING`, `ALIVE`, `IDLE`, or `UNKNOWN` according to measured evidence;
  `STUCK` requires consecutive confirmed idle windows. Failed CPU/GPU probes
  suppress termination instead of being interpreted as zero utilization.

## 2026-09-06 full-code review fixes (four-area audit; details in docs/REVIEW_NOTE_2026-09-06.md)

- [x] `OM-2026-09-06-01` `run_matrix.sh` completion check demanded exactly `r/a/b`
  split-half keys while oracle protocol v3 writes `r_high_budget` too: every
  finished point was re-run as incomplete. Now requires exactly `r/r_high_budget/a/b`.
- [x] `OM-2026-09-06-02` `code_sandbox.py` rejected `typing`/`dataclasses`/`abc`/
  `enum` imports, `__name__` and single-underscore attributes, scoring correct
  MBPP solutions 0. Only dunder attributes and the blocked builtins remain banned.
- [x] `OM-2026-09-06-03` Math reward stripped every comma (`(3, 1)` vs `(3,1)` → 0,
  `12` vs `1,2` → 1) and the `\boxed` fallback ignored nested braces. Only
  thousands separators are dropped; `\dfrac`/`\text` are normalised; `_boxed` is used.
- [x] `OM-2026-09-06-04` `generate()` received no `eos_token_id`; the pinned
  Qwen3.5-9B config lists only `<|endoftext|>`, so `<|im_end|>` never stopped
  decoding. The resolved EOS set is passed explicitly (recorded in the manifest).
- [x] `OM-2026-09-06-05` `check_27b_fla.py` read kernels from the transformers
  module, which no longer re-exports them (AttributeError, launch abort).
- [x] `OM-2026-09-06-06` GRPO: LoRA init seeded from `--seed`; AdamW
  `weight_decay=0.0` stated explicitly (torch default 0.01 was unregistered);
  a loaded parent optimizer no longer overrides the registered lr; parent config
  compared on chain resume; NCCL timeout 2 h.
- [x] `OM-2026-09-06-07` Collection marker binds the analysis code digest and the
  REGIME.json schema/bootstrap count, so a v3/2,000-replicate cache is not current.
- [x] `OM-2026-09-06-08` `rlzero_status.py` let a watchdog echo in the worker log
  outrank confirmed-idle telemetry (PROGRESSING over STUCK).
- [x] `OM-2026-09-06-09` `make_tables.gate_numbers` read `report.json` without the
  input-hash check; `cleanup_run_processes.py` matched any `RUN_LABEL=v4-*`
  process outside its `--run-prefix`; dataset adoption rewrote an already
  adopted copy on a possibly read-only shared root.
- [x] `OM-2026-09-06-10` Launcher: `export X="$(helper)"` masked helper failures;
  the pipeline's own keepalive counted as pipeline activity (hard-stall kill
  unreachable); `fetch_datasets.sh` ignored manifest failures; stale-shard
  quarantine status was lost behind `tee`; Ctrl+C was reported as a log-writer
  failure; terminal filter matched words instead of tags.
- [ ] `OM-2026-09-06-11` 5/974 MBPP prompts are unsolvable by construction (two
  need `test_setup_code`, three use blocked dunder methods/`exit()`); constant
  zero reward → zero advantage, harmless to GRPO but wasted prompts. Fixing
  changes the registered dataset content hash; decide before the next matrix.
- [ ] `OM-2026-09-06-12` `make_tables` T3/T6 split micro-groups even/odd instead
  of the contiguous R/A/B partition (diagnostic tables only).

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
- [x] Re-run the complete CPU suite, theory verifier, shell syntax checks,
  Python byte compilation, and clean LaTeX build: 174 tests passed; both
  registered ceiling curves reproduced; the current 21-page letter PDF has no
  undefined references, LaTeX errors, or overfull boxes. Main text ends and
  references begin on page 9, so the post-result build must be rechecked against
  the nine-page limit.
- [ ] Run the target-cluster preflight on the committed revision used for final
  analysis.
- [ ] On the target cluster, confirm that no old batch-1 d400 recovery remains,
  deploy the corrected supervisor if needed, and resume durable `.partial`
  files at the configured batch. Verify that the launcher reports batch 8 and
  that per-prompt runtime returns to the pre-recovery range before leaving the
  allocation unattended. The 2026-09-05 local audit could not inspect this
  state because the shared group volume was not mounted.

## Provenance note

These corrections were made before inserting or interpreting confirmatory
numbers. No local off-policy experiment process was running during the edit.
Existing policy updates and raw rollout artifacts are not relabeled; analysis
outputs must carry their generation revision and the new analysis revision.
