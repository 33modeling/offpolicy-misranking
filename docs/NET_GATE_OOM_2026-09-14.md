# Net-Gate Autograd OOM Diagnosis

Source: `net-gate-errors-20260914T051745Z-e7MqM8.txt`, exported at
2026-09-14 05:17:45 UTC from code `5076cfb14037c483f2126391f525eff20aa96e9f`.
This is a failed-arm log export, not a complete suite-status or cost export.

## Confirmed Findings

- All three d100 development `selection_reduced` arms (seeds 0, 1, 2)
  report `autograd-score worker failed`. The autograd recovery code was reached.
- In every seed, shard 0 fails on candidate `prompt=356`. Its first attempt
  reached `scoring 89/100`; subsequent attempts start at `scoring 0/11` on that
  same prompt and fail again. Completed earlier scores are being reused.
- The retained log tails contain 4 CUDA OOM messages for seed 0, 3 for seed 1,
  and 4 for seed 2. These are counts within the export, not a complete ledger.
- Each failing allocation requests 998 MiB on an H100 with 79.11 GiB capacity.
  PyTorch has approximately 77.76--77.79 GiB allocated, with only about
  92--199 MiB free. Reserved-but-unused memory is approximately 0.4--0.5 GiB.
  This is not evidence of a fragmentation-only issue.
- Worker startup reports about 14.6 GB of model parameters/buffers and almost
  all device memory free before loading. The observed failure is therefore
  consistent with a per-input computation memory peak, not merely a model
  that cannot fit, nor proof of duplicate jobs occupying the same GPU.
- Seed 1 shards 1 and 3 report scoring complete. The parent stops unfinished
  sibling workers when another shard fails; this does not mean those siblings
  independently encountered CUDA errors.
- Old finite-difference calibration aborts are also present in `score-*.log`.
  They are historical records. The current failure entries name the separate
  `autograd-score-*.log` files and those logs identify CUDA OOM.

## Code Findings And Limits

`low_order_backend.load_current` disables the KV cache and puts the model in
evaluation mode, but does not enable transformer activation checkpointing.
`exact_directional` differentiates a full stored response, one response at a
time. `grads._token_logps_chunked` checkpoints the output-head chunks, not the
whole transformer stack. The ordinary GRPO trainer separately enables
non-reentrant gradient checkpointing; the scorer does not.

Transformer activation storage is a plausible repair target. The exported
log does not contain the inner OOM traceback, failing response token length,
or stage-specific allocator measurements, so the exact failed allocation
cannot be attributed to an individual operator from this file alone. It also
does not establish whether the long-input peak is in the forward log-prob
pass or the differentiable scoring pass. Do not assert a token length or a
memory leak as established fact.

More nodes will not increase the memory available to this individual model
replica. The four-worker scorer partitions prompts; it does not shard one
response's model/activation memory across the four GPUs. `GPU 0` in a masked
worker's CUDA message is its local device index, not evidence that workers on
different nodes share one physical GPU.

## Required Follow-Up

Reduce the scoring path's peak activation memory, preserving complete inputs,
the scoring formula, cached completed scores, validation direction, frozen
contracts and existing cost ledger. Do not truncate or skip prompt 356, alter
its score, reset failed-work costs, or silently disable numerical safeguards.
Any runtime repair must preserve dropout/evaluation semantics and be verified
against the existing exact score on a small model before a remote retry.

Unchanged retries demonstrably fail again on prompt 356 and consume budget.
There is no evidence in this file of a repaired successful H100 run. Total
remaining budget, complete suite progress and final benchmark rewards cannot
be reconstructed reliably from these log tails alone.

This commit records diagnosis only; it does not change the GPU execution path.
