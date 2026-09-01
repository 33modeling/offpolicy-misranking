# Incident: SFT Was Substituted for RLVR

## Summary

The retired experiment path called its policy change "drift" but implemented
positive-only response SFT: it filtered behavior rollouts to verifier-correct
responses and minimized cross-entropy on those responses with a LoRA adapter.
That is not RLVR, GRPO, or a policy-gradient update. Earlier SFT-derived runs
must not be cited as empirical RLVR evidence.

## Why LoRA appeared

LoRA was chosen to make 7B/27B policy variants inexpensive and to isolate policy
changes from the base checkpoint. The implementation then incorrectly treated
the choice of trainable parameters as if it specified the learning algorithm.
LoRA only answers which parameters change. It does not determine whether the
objective is supervised learning or reinforcement learning.

The scientific design allowed an abstract policy drift for theorem diagnostics,
while the paper claim required an RLVR-trained policy. Those two requirements
were conflated. Compute convenience took precedence over verifying that the
empirical intervention matched the claim.

## Impact

- The old 7B and 27B drift matrices measure behavior under an SFT-induced policy
  perturbation, not an RLVR training trajectory.
- Their off-policy estimator diagnostics may be retained only as explicitly
  labeled SFT controls.
- Their reported `0/10` and `3/10` joint-deficit counts cannot establish the
  paper's RLVR result.

## Corrective controls

- The supervised drift functions and `experiment.py --stage drift` entry point
  were removed.
- The canonical trainer samples online from the current policy and optimizes a
  clipped GRPO objective from verifier rewards for every response, including
  zero and negative-advantage samples.
- Every positive-step policy publishes an objective manifest, adapter hash,
  optimizer state, and per-step reward/ratio statistics.
- Completion rejects SFT, positive-only filtering, non-verifier rewards, the
  wrong distributed world size, stale hashes, or inconsistent per-step sample
  accounting. Zero-advantage groups remain valid GRPO observations and are
  recorded rather than retried as infrastructure failures.
- New GRPO runs use new result roots. Old SFT run directories are not resumed.
- Unit tests cover the loss direction, clipping, zero-variance groups, manifest
  rejection, ordered checkpoint reuse, and single-claim shared queues.

## Rule

Never infer a training algorithm from adapter format, directory name, or the
existence of a reward column. Before reporting an experiment as RLVR, verify the
actual optimized loss and require an objective-bound artifact contract.
