# CUDA failure handling and memory fixes — 2026-10-06

The reported `torch/random.py ... CUDA error: unspecified launch failure`
is not sufficient to locate the first failed kernel. The changes below fix
reproduced code defects; they do not establish that the remote CUDA failure
was caused by any one of them. The local environment has CPU PyTorch only.

## Fixed defects

- A generation error could be replaced by a second error while restoring RNG
  state. Preserve the first exception and attach restoration failure details,
  prompt ID, rank, device, response count and sampling seed. Synchronize the
  generation device before restoring its RNG so asynchronous generation errors
  are attributed to that rollout. Seed only the CPU and the rank-owned GPU;
  `torch.manual_seed` previously touched all visible GPU generators.
- Dense per-prompt scoring gradients remained live while the next prompt's
  rollout allocated its KV cache. Release the buffers and loop-variable aliases
  before starting that rollout. Gradient arithmetic and prompt ordering remain
  unchanged.
- Ordinary OLMo checkpoint snapshots deep-copied optimizer tensors on their
  current device, unlike the Qwen-specific snapshot. Copy them directly to CPU
  with independent storage before the engine makes further snapshots.
- An OOM retry called `empty_cache` while the exception traceback still owned
  tensors from the failed rollout. Leave the exception scope and collect those
  references first. Retry only the existing one OOM attempt; an unspecified
  launch failure is propagated, not retried inside the damaged process.
- A CUDA synchronization error at stage exit left an open timing frame. Always
  unwind the frame while preserving the first error and leaving the interrupted
  phase receipt unfinished. Never publish unknown interrupted time as zero.

## Existing experiments and debugging

These core changes apply to the current package, identified as
`12cf5ef830ebfd92fa8a87ea62dc7df734cd9ceab57fbce18fc4b2548385f960`.
The pre-fix current package was `b5dd86cfbe7b3b50...`; its existing
checkpoints retain their original identity and cannot be silently resumed under
the new package. The previously supported
`f581eb043e89409e...` and `9435e80003f41e1f...` runtimes retain their exact original
backend and timer through byte-identical archived copies checked against the
existing manifest. Loading those frozen runtimes does **not** install the new
core fixes. The shared rollout wrapper receives the exception-lifetime fix.
No archived file, recorded input, checkpoint or result was overwritten, and
no experiment or GPU job was launched. Do not relabel new code as an old runtime
or disable the resume identity check to apply a patch to an existing run.

For the unresolved remote failure, collect the **first full traceback** and
preceding rollout/scoring log, failed command, node, dataset and seed. An isolated
diagnostic rerun of the same command can prefix `CUDA_LAUNCH_BLOCKING=1` to identify
the failing CUDA call. Use a new process after a launch failure. Do not leave the
flag enabled for GPU-time comparisons: it changes execution timing. This is the
[PyTorch CUDA debugging procedure](https://docs.pytorch.org/docs/stable/notes/cuda).

## Verification

Failure-injection tests cover RNG restoration masking, rank-local seeding,
asynchronous synchronization failure, OOM traceback lifetime and refusal to
retry fatal launch errors. Tiny CPU OLMo tests check actual generation,
selection gradients, optimizer updates and independent CPU snapshots. A weak
reference test fails against the exact old backend and passes against the new
one, confirming that scoring buffers are freed before the next rollout.

CPU distributed smoke tests passed with two Gloo ranks (global training update,
uneven scoring shards and failed-save recovery) and four Gloo ranks (all six
mechanism branches, checkpoint resume and carrier/optimizer restoration).
These are not H100/NCCL or production-kernel tests.

Final SRGC regression run: **687 passed**, plus **305 passing subtests**;
no failed or skipped tests (CPU PyTorch 2.13.0, Transformers 5.14.1, PEFT 0.20.0).
The saved-runtime tests were additionally rerun after making their archive
lookup independent of the optional torch import: **11 passed, 4 subtests**.
Targeted regression run: **37 passed**. Changed Python syntax, staged diff
whitespace, archived byte hashes and both supplied result JSON hashes passed.
The production GPU failure remains unverified without the remote traceback.
