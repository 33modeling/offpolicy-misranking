# Methodology before implementation

Date: 2026-09-12 (Asia/Seoul)

Latest user direction: prioritize a cost-bounded measurement procedure that
uses a selector only when it establishes useful benefit over random and
otherwise returns to random promptly. The
[revised direction and unresolved design](LOW_ORDER_PRIOR_ART_AND_RATIONALE_2026-09-12.md#subsequent-direction-bounded-measurement-with-random-fallback)
separate inconclusive results from demonstrated lack of gain, require charging
diagnostic cost, and identify the existing runner's missing fallback control.
This is a design record, not an implemented or validated detector. Earlier
candidate-specific decisions below are retained for chronology.

Implementation update: after reviewing the follow-up specification, the user
explicitly requested code. That authorizes the isolated experimental
[implementation](LOW_ORDER_REUSE_RUN.md), not a main-method claim, a GPU launch
from this workstation, or a statement of benchmark superiority. The original
research-stage decision below is retained for chronology.

Pre-launch follow-up: the user requested a similarity review before running.
The [prior-art and rationale record](LOW_ORDER_PRIOR_ART_AND_RATIONALE_2026-09-12.md)
identifies close computational precedents, narrows the candidate contribution,
and records missing matched-target and cost controls. No GPU job was launched.

## Decision

No new main method is approved. Stop implementation until the acquisition
rule, its connection to the diagnosed error, its contribution beyond prior
work, and its full computation budget are defensible.

Follow-up: [Obsidian and experiment synthesis](OBSIDIAN_EXPERIMENT_SYNTHESIS_2026-09-12.md)
records a concrete low-order, learner-matched reuse-score candidate, its
derivation, the existing K=8/G=8 constraint, and unresolved novelty and cost
checks. It does not authorize implementation or change this gate.

The target is **higher independent benchmark performance at the same total
training-plus-selection cost**, or **lower total cost to reach a fixed benchmark
performance**, relative to random selection. Ranking agreement, estimator
variance, activation coverage, and alignment alone do not satisfy this target.

The assistant prematurely started two uncommitted files,
`src/activation_selection.py` and `src/activation_experiment.py`. Both were
removed after the user's instruction to establish the methodology first.
They were never tested, committed, pushed, or launched. Existing code,
concurrent author changes, GPU jobs, manuscript, and publication sites were
not modified. Earlier additive-scoring code remains exploratory, not an
approved remedy or a recommendation to launch another GPU study.

## Rejected activation proposal

The proposed rule weighted validation prompts by their failure rate, then
greedily selected training prompts that covered those validation prompts in
current-model activation space. Prompt-only features would avoid candidate
response generation and candidate backward passes.

This is not an adequate main-method proposal:

1. **Novelty is not established.** The existing Obsidian LEEPS note already
   identifies latent neighbors and historical outcomes as acquisition signals.
   Checking the original paper confirms its final-prompt-token representations,
   positive cosine similarities, and neighbor-based success estimates. Its
   objective and acquisition rule are not identical to weighted coverage, but
   adding failure weights and redundancy control does not by itself establish
   a substantive new contribution. [LEEPS, Methods](https://arxiv.org/html/2607.28077v1#S3)
2. **Failure is not trainability.** High error alone can favor prompts whose
   sampled responses all fail. In binary-reward GRPO, a homogeneous group has
   zero reward advantage. For independent Bernoulli outcomes with success
   probability p and group size G, the probability of a nonhomogeneous group
   is `1 - p^G - (1-p)^G`, not `1-p`. This does not make intermediate difficulty
   a new proposal either, nor does nonzero advantage establish useful transfer
   to the benchmark. [LEEPS, Preliminaries and Methods](https://arxiv.org/html/2607.28077v1)
3. **The causal connection is missing.** Replacing an off-policy gradient score
   with an activation score does not restore the omitted importance-ratio
   information. A different proxy is not a demonstrated repair of that error.
4. **The cost advantage is unmeasured.** Current activations require forward
   passes; validation rewards require generation and verification. Existing
   caches are not free in a cold-start or end-to-end accounting.

## What the existing finding does and does not imply

The theoretical issue concerns partially corrected, cross-policy scores.
It does not show that gradient selection in general is ineffective.
Nor does a rank reversal alone establish worse long-horizon LLM training.
The constructive argument must connect:

```
specific omitted information
  -> a wrong acquisition decision
  -> the proposed repair of that decision
  -> better downstream learning after charging for the repair
```

The local concurrent CFCS results were also reviewed, without editing or
staging that author's files. Their own conclusion does not support a reliable
cheaper-and-better method. A new name for another global correction mixture
would not address that negative result.

## Shortcuts that cannot be claimed as new

- Prefix reuse and selective continuation generation require a direct
  comparison with PROS and SPEC-RL, not a claim that reusing an answer prefix
  is itself new. [PROS, ICLR 2026 abstract](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4badb55ba9c18bccb9d4146be328a948-Abstract-Conference.html),
  [SPEC-RL author project](https://bingshuailiu.github.io/Spec-RL/).
- Combining logged and current-policy samples with multiple or defensive
  importance sampling is established. A ranking-specific allocation rule
  would need its own contribution and analysis; the mixture identity is not
  that contribution. [ICML 2019 policy-search sample transfer](https://proceedings.mlr.press/v97/tirinzoni19a.html),
  [Policy Gradient with Active Importance Sampling](https://arxiv.org/abs/2405.05630).
- Disagreement between two biased estimators is not automatically an error
  bound. It cannot certify that an apparently stable prompt is safe to reuse.

These are exclusion checks, not newly endorsed algorithms. The PROS check
used its official abstract; it is not a full-text novelty audit.

## Required method specification

Before implementation, write one concrete specification answering all of:

1. **Selection:** What exactly is scored or sampled, using which observable
   data, at which policy checkpoint? State the formula, inputs, and rule for
   selecting a prompt. Keep test outcomes outside this rule.
2. **Repair:** Which term or information missing from partial correction is
   recovered? Demonstrate the effect in the existing counterexample. State
   assumptions and failure cases; distinguish unbiased reward-gradient
   estimation from the actual normalized/clipped GRPO update.
3. **Contribution:** Identify the closest methods from the existing Obsidian
   research and their original papers. State the nontrivial difference, not
   merely a change of feature, terminology, benchmark, or application domain.
4. **Budget:** Count new and cached response generation, both policies'
   likelihood evaluation when needed, gradients or directional probes,
   activations, validation, verification, and training. Separate shared costs
   from method-specific costs and cached from cold-start regimes.
5. **Learning:** Explain why the repaired acquisition should improve the
   actual learner, not only the score estimate. Specify the downstream test
   that could falsify that explanation.

For a simple cost break-even calculation, let C_select be extra method cost
and C_step the cost of a random-baseline training update. An acquisition method
that saves delta_U updates needs `delta_U * C_step > C_select` just to save
compute, assuming comparable per-update costs and the same target performance.
Generating fewer scoring responses alone does not establish this inequality.

## Implementation gate

Do not implement a candidate merely because its score is easy to code. First
provide the specification above and report whether the novelty and budget
checks pass. Passing them justifies a small falsification experiment, not a
paper claim of superiority. No GPU speed estimate or new experiment launch is
authorized by this note. The method search remains unresolved.

## Local research consulted

Existing Obsidian notes, not a replacement literature collection:

- `papers/2607.28077-LEEPS-Explore-Exploit-Prompt-Sampling-arXiv-ko.md`
- `papers/2603.13201-NAIT-Neuron-Aware-Data-Selection-ICLR2026-ko.md`
- `papers/Single-Rollout-Hidden-State-Dynamics-for-Training-Free-RLVR-Data-Selection-ICML2026-ko.md`
- `papers/2602.10388-Less-Is-Enough-ICML2026-ko.md`
- `papers/Gradient-Aware-Scheduling-Async-RL-ICML2026-ko.md`
- `papers/2608.29252-DIEM-Dynamic-Gradient-Alignment-Batch-Reweighting-arXiv-ko.md`

The last two notes informed the questions to ask, not a verified full-text
novelty clearance for a new algorithm.
