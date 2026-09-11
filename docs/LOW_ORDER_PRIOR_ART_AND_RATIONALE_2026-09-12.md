# Low-order reuse: prior art, implementation rationale, and launch decision

Date: 2026-09-12 (Asia/Seoul)

## Current direction: a gate before selection

The user's subsequent clarification supersedes the operating sketch below:
compare random and selection training OFFLINE to develop a predictor; in
deployment place that predictor BEFORE expensive selection. Do not run paired
shadow trainings at each gate check. The user permits replacing the earlier
manuscript argument and theory to support this direction.

The current specification is
[Pre-selection gate: experiment and implementation design](SELECTION_GATE_DESIGN_2026-09-12.md).
The [mathematical draft](https://github.com/33modeling/offpolicy-misranking-paper-v2/blob/main/research/2026-09-12/SELECTION_GATE_THEORY.md)
derives the cost-adjusted objective, decision-error loss, a conditional
whole-controller certificate, and the same-state stopping identity. None
proves that the proposed inexpensive features predict RLVR learning benefit.

The first controller design uses bounded checks and absorbing random fallback;
it does not terminate healthy training. The gate is not implemented. Existing
fixed-arm code and GPU jobs are unchanged. The following earlier paired-pilot
sketch is retained as SUPERSEDED history, not current runtime instructions.

## Superseded sketch: bounded measurement with random fallback

After this review, the user proposed a different primary contribution:
measure whether selection is worth using, and quickly return to uniform
random prompt selection when it is not. This supersedes treating the
low-order estimator as the required main method. It remains one candidate
selector with which to test the measurement procedure.

The proposed operating rule is **random by default; use the selector only
when a bounded diagnostic establishes sufficient benefit**. Failure to
establish benefit is an operational reason to use random, not proof that
the selector is worse. A near tie may require substantial measurement to
resolve; the procedure must be allowed to abstain rather than keep sampling.

The research target is the final independent reward of the entire gated
training procedure at a fixed total compute budget, compared with always
random and always using the selector. Charge scoring, diagnostic evaluation,
any pilot training, discarded branches, and later rechecks to that budget.
Returning to random does not recover already-spent compute or undo earlier
selector-induced parameter changes.

A concrete staged design to review before implementation:

1. Freeze a diagnostic budget cap, the candidate selector, a useful-benefit
   margin, and a recheck schedule. Do not tune these on final benchmark results.
2. Screen existing cache support, reward contrast, and numerical validity
   first. These checks can reject an unusable score; passing them cannot
   establish downstream superiority. Overlap is not the acceptance criterion.
3. For a direct learning check, start short selector and random continuations
   from the same checkpoint and optimizer state, using the same compute
   accounting. Allow random to spend saved selection compute on training.
   Evaluate fixed resulting policies on independent diagnostic prompts in
   small batches. Both branches and evaluation are real overhead; this is
   not a claim that two pilot trainings are inexpensive.
4. Use predeclared sequential error control, not ordinary confidence
   intervals repeatedly inspected until favorable. Accept the selector only
   if the lower bound on the defined short-horizon reward difference exceeds
   the useful-benefit margin. Stop for futility when its upper bound does not
   exceed that margin. If the budget expires first, use random and label the
   result inconclusive. Without calibrated score-to-reward evidence, an
   alignment interval cannot replace this reward comparison.
5. Continue the selected branch while preserving artifacts. On a later
   fallback, stop expensive selection and sample prompts uniformly from the
   eligible pool; do not claim this restores the counterfactual always-random
   trajectory. Bound the number and total cost of rechecks to prevent a
   diagnose-switch-diagnose loop.

Sequential evaluation of fixed trained policies can quantify uncertainty
from diagnostic sampling. It does not cover training-seed variation, new
checkpoints, or long-horizon performance. Those require separate validation.
Do not pool observations from changing policies as if their reward difference
were a single stationary mean, or reuse diagnostic prompts for final reporting
as an untouched benchmark. More diagnostic responses are not more independent
training runs.

Record distinguishable decisions such as `use_selector`, `no_useful_gain`,
`inconclusive_budget`, and `invalid_measurement`. Invalid measurements must
not become fabricated negative rewards. A broken verifier or corrupt training
input is not fixed by random selection; the existing task-failure handling
must still report it and may yield to other valid work.

Baseline fallback itself is established: [conservative contextual bandits](https://papers.nips.cc/paper_files/paper/2017/hash/bdc4626aa1d1df8e14d80d345b2a442d-Abstract.html)
constrain performance relative to a baseline, and
[SPIBB](https://proceedings.mlr.press/v97/laroche19a.html) retains baseline
behavior in uncertain regions. Their baseline is an action policy, not a
prompt sampler coupled to an evolving GRPO learner; their guarantees do not
automatically apply here. [Time-uniform confidence sequences](https://arxiv.org/abs/1810.08240)
provide established sequential measurement tools, not a new contribution by
themselves. These primary abstracts were checked as a first overlap screen,
not a complete novelty audit of the revised direction.

The contribution to establish is **an inexpensive, learning-relevant trigger
that avoids wasted selection cost without discarding valuable selection too
often**, validated across both beneficial and unhelpful settings. Report
diagnostic cost, time to decision, harmful acceptances, missed useful cases,
and final reward versus total cost. A detector that always chooses random is
not a successful solution merely because it avoids selection overhead.

Implementation status: `method_choice.py` selects from existing score
estimates; `low_order_experiment.py` runs fixed comparison arms. Neither is
this sequential fallback controller. In particular, the low-order runner
currently waits for complete scores before preparing its random training
subset. A future controller must make random training eligible independently
of successful selector scoring. No automatic-switching code, default launcher
change, GPU job, or manuscript claim was introduced with this direction note.

### Superseded paired-pilot timing sketch

This subsection predates the user's clarification above. Its expensive paired
continuations belong in offline research, not in the deployed pre-selection
gate. Use the linked current design for runtime timing and stopping behavior.

The user's next refinement is to make the procedure specify the timing and
size of measurement, not just produce a final selector/random label. Its
output should identify the current action, its evidence and cost, and the
earliest eligible next check. This remains a proposed protocol, not an
implemented controller or a demonstrated optimal measurement schedule.

**Start and recheck.** Start the first check before paying for a new selector
deployment. Use existing cache metadata and ordinary training telemetry to
screen feasibility before allocating new scoring work. After a decision,
keep it for a predeclared block of training; do not evaluate after every
update. A later check is eligible only at a block boundary with remaining
diagnostic budget. A changed dataset, verifier, or selector invalidates the
old decision's scope and returns the controller to unverified/random mode;
it does not authorize an unlimited new budget. Drift or ESS may trigger a
review, but neither establishes that the selector has lost learning value.

For the first implementation, use fixed block boundaries and a maximum
number of checks. An adaptive schedule that learns when a check is worth its
cost is a later research extension, not something the current code supplies.
If even a minimally informative paired pilot cannot fit the reserved budget,
skip it and train randomly; do not describe this as a negative experiment.

**Observe.** For a given check, fix the source checkpoint, optimizer state,
candidate pool, selector, comparison horizon, and cost ledger. Construct the
paired continuation comparison described above. Freeze the resulting policy
pair during sequential evaluation. On each independent diagnostic prompt,
observe both policies' verifier rewards under the same evaluation protocol
and use their paired difference. Additional evaluation batches add evidence
about that pair, not new training seeds. Keep the diagnostic set separate
from the selector's fitting samples and the final reported benchmark.

Choose a positive minimum useful gain `delta_min` before observing outcomes.
It is a practical short-horizon reward margin at the stated budget, not an
automatic conversion of GPU seconds into accuracy. Track its time-uniform
uncertainty interval `[L_n, U_n]`, prompt/response counts, and accumulated
selection, pilot, and evaluation costs.

| Condition | Measurement action | Training action |
| --- | --- | --- |
| `L_n > delta_min` | Stop: useful short-horizon gain supported for this policy pair. | Use the selector for the next fixed block. |
| `U_n <= delta_min` | Stop: the stated useful-gain threshold is not supported by the upper bound. | Use random; this need not mean the selector is strictly worse. |
| Interval crosses the margin and the next batch fits the remaining cap | Collect one more predeclared batch of independent diagnostic prompts. | Do not expand the pilot-training horizon to chase a favorable result. |
| Interval crosses the margin but any time, token, GPU-cost, or check-count cap is reached | Stop: inconclusive at the allowed budget. | Use random; retain the incomplete-evidence reason. |
| Scoring-specific numerical or lineage validation fails | Stop this check and report the actual failure. | Use random only if common training inputs and verifier are valid. |

The diagnostic must have a hard wall-time limit as well as a compute ledger
so a stalled subprocess cannot keep the decision pending indefinitely.
Reserve room for the next batch before dispatch and log any overshoot or
partial work. Error control across multiple checks must be specified, for
example by a predeclared allocation of the total error probability; restarting
an independent nominal 95% test at every checkpoint is not a global guarantee.

**End selection, not learning.** A random decision stops further selector
scoring during that block. Continue GRPO from the designated usable checkpoint
with uniform prompt sampling; do not terminate a healthy training job or
restart it from the beginning. The original total training budget controls
when learning ends. The controller cannot promise that this continuation is
identical to an always-random trajectory.

The study must measure whether this timing rule finds useful selection early
enough to repay its own overhead. Report it against always-random,
always-selector, and fixed-period checking at matched total cost. Very small
effects may remain unresolved cheaply; bounded abstention is part of the
design, not evidence of successful discrimination.

## Decision

**Related methods exist, including close precedents for the computational
backend. The implementation is a hypothesis-testing harness, not a cleared
novel method or an established improvement. No GPU experiment was launched.**

The user requested code after the
[method specification](OBSIDIAN_EXPERIMENT_SYNTHESIS_2026-09-12.md), then asked
for this comparison before running it. Keep the implementation, but do not
interpret its availability as a recommendation to launch the default
five-seed training suite. First settle the matched-target controls and cost
protocol below. This is a research decision, not a new runtime lock; no
running OLMo, Qwen, or E5 process is changed.

The defensible question is narrow:

> Can a score estimated from eight previously generated responses select
> prompts that improve subsequent current-policy GRPO training, after paying
> for scoring, better than simpler selectors using the same resources?

This is not a proposal to use overlap as a selection rule. It is not a new
off-policy training loss. The selected objects are prompts; the learner
generates new responses on those prompts during continuation training.

## Existing research consulted

This review starts from the existing Obsidian vault, snapshot
`6c3874c0db756d8e61eedd5f539666fcaf0d3c81`, rather than replacing its research.

- `Data-Selection-Papers-ICLR2026-ko.md`: includes Train on Validation (ToV).
- `Data-Selection-Papers-ICML2026-ko.md` and the OPUS/GIST paper notes.
- `Data-Selection-Papers-ACL2026-ko.md`: includes For-Value.
- `papers/2510.26491-CROPI-Data-Efficient-RLVR-arXiv-ko.md`.
- `papers/2602.21492-GradAlign-Gradient-Aligned-Data-Selection-COLM2026-ko.md`.
- `papers/2506.11480-LearnAlign-Gradient-Alignment-Data-Selection-ACL2026-ko.md`.
- `When-Does-LLM-Unlearning-Fail-Related-Work-ko.md`, section 3: already points
  to Mirrored Influence and forward-based influence estimation.
- `RLVR-Methods-First-New-Direction-2026-09-12-ko.md`: a distinct,
  variance-corrected batch-matching proposal, discussed separately below.
- [Earlier TayPO comparison](TAYPO_COMPARISON.md) and
  [methodology gate](METHOD_DECISION_2026-09-12.md).

The paper repository's `RELATED_WORK_SURVEY_2026-09-12.md` was also read.
Its statements that low split-half overlap implies performance
indistinguishable from random, or that a reliability law applies unchanged to
arbitrary prompt weights, are not accepted as demonstrated results here.
The required assumptions and downstream comparisons are separate questions.

The targeted primary-source checks below cover methods, selected theorems,
and relevant appendices, not a full reproduction or proof audit of every
paper. Conference labels in a local filename are not independently verified
acceptance evidence. Links identify the versions actually inspected. Some
OpenReview pages returned browser challenges; accessible arXiv versions were
used instead, and uninspected revision changes remain outside this review.

## Closest prior art

### Selection objective and implementation

| Work and inspected material | Established overlap | Boundary for this code |
| --- | --- | --- |
| [Off-policy influence guidance, v2, sections 3-4](https://arxiv.org/html/2510.26491v2#S3) | Uses cached responses to score prompts for the current policy, then trains on a selected subset. | This is the closest workflow precedent. Reusing answers for selection is not new. Our full response ratios, learner coordinates, and moment estimator change the scoring rule; they do not establish superiority. |
| [GradAlign, v1, sections 4 and 7](https://arxiv.org/html/2602.21492v1#S4) | Selects RL prompts using alignment with a trusted validation gradient; discusses inner products, cosine scores, and gradient noise. | Validation-directed RL selection is occupied. Our fixed-checkpoint cached-response estimator differs from its on-policy refresh strategy. Changing cosine to a signed directional score is not sufficient novelty. |
| [LearnAlign, v1, section 4](https://arxiv.org/html/2506.11480v1#S4) | Combines gradient cosine with success-rate factors of the form p(1-p). | Combining reward contrast, difficulty, and gradient information is not new. Our population GRPO normalization factor and disjoint-product estimator are not the same formula, but must earn their extra complexity. |
| [OPUS, v1, sections 4-5](https://arxiv.org/html/2602.05400v1#S4) | Scores optimizer-induced updates; its AdamW approximation freezes RMS geometry. It also includes a batch redundancy term. | `make_direction` uses an established diagonal approximation. We do not implement OPUS's full selection algorithm or its batch interactions. Optimizer awareness is a control, not our contribution. |
| [GIST, v1, sections 3-4](https://arxiv.org/html/2602.18584v1#S3) | Uses target-gradient geometry and spectral structure for instruction-data selection; questions diagonal approximations. | Unmerged LoRA coordinates do not make a selector novel or make frozen Adam geometry exact. We do not implement GIST's non-diagonal spectral method. |
| [Train on Validation, v1, sections 1.1 and 2.1](https://arxiv.org/html/2510.00386v1#S1.SS1) | Reverses training/validation roles: target-side updates and candidate forward losses approximate influence without candidate gradients. | This is a close computational precedent. Our central directional probe is a local version of the same symmetry, not a new way to obtain alignment. Its GRPO reward/ratio aggregation differs from ToV's supervised token-loss scoring. |
| [Mirrored Influence, v1, sections 1-3](https://arxiv.org/html/2402.08922v1) | Moves expensive updates to the smaller target set and measures training-side changes through forward passes. | Target-side perturbation plus candidate forward evaluation predates this code. Its attribution applications and perturbation procedure are not identical to our fixed-direction score. |
| [TACS, v1, sections 3-4](https://arxiv.org/html/2605.09404v1#S3) | Uses a validation-induced, capacity-limited trajectory and normalized candidate loss changes. | A reusable target-induced path is also prior art. This code uses a local derivative, not the full trajectory method; it cannot claim to solve reference-path bias. |
| [For-Value, v1, section 4](https://arxiv.org/html/2508.10180v1#S4), [ACL record](https://aclanthology.org/2026.acl-long.664/) | Uses token representations and prediction-error interactions for forward-only valuation under a stated feature model. | Forward-based influence scoring is not new. This is a different proxy from a LoRA directional derivative and is a possible efficiency comparator, not an exact interchangeable estimator. |

ToV, Mirrored Influence, and For-Value were already present in other vault
indexes or research notes, but were missing from the preceding low-order
synthesis's comparison table. This review closes that specific omission.
TACS was additionally inspected through primary-source search.

### Group statistics and correction

| Work and inspected material | Established overlap | Boundary for this code |
| --- | --- | --- |
| [GRPO U-statistic analysis, v3, section 4 and Appendix A](https://arxiv.org/html/2603.01162v3#A1) | Establishes the pair representation and studies sampling error. Appendix A addresses random standardization and importance sampling, including an asymptotic second-order representation. | Neither GRPO-as-U-statistics nor the existence of normalization/correction issues is new. The implementation instead approximates a specified finite-G population coefficient and estimates its moments. |
| [PAIR, v1, sections 3-5 and Appendix B](https://arxiv.org/html/2608.11368v1#S3) | Estimates pair terms using joint inclusion correction after adaptive on-policy suffix continuation. Its exact regime excludes random reward standardization and active clipping. | PAIR corrects which endpoints are observed; our weights correct a change of response-generating policy. These are different sampling mechanisms, but pairwise correction and using all observed contrasts are established. |
| [Group-standard-deviation identity, v1, Theorem 1](https://arxiv.org/html/2607.00152v1) | Expresses binary-reward GRPO using correct/incorrect response contrasts and the finite group's reward standard deviation. | Conditioning on binary group composition builds on this known structure. Retaining the trainer's epsilon and length normalization is target matching, not by itself a novel identity. |
| [TayPO, ICML 2020, sections 2-4](https://proceedings.mlr.press/v119/tang20d/tang20d.pdf) | Expands policy value around a behavior policy and trades approximation order against estimation difficulty. | Our approximation is in the finite-group normalization factor, not a truncation of token-ratio deviations or policy value. Full sequence ratios remain inside retained terms. TayPO's improvement/remainder guarantees do not transfer. |
| [Conditional IS, AISTATS 2020, official abstract](https://proceedings.mlr.press/v108/rowland20b.html) | Relates conditioning to off-policy estimation and variance reduction. | Conditioning away group composition is not a new general principle. We have not established that our subsequent low-order approximation is a conditional expectation of the full estimator, or inherits variance dominance. |

The symbols `U2` and `U4` in this repository mean two-response and
four-response statistics. They must not be confused with TayPO's expansion
terms. The published method PAIR is also not a name for our `pair_u2` arm;
that arm is a plain full-ratio pairwise covariance control.

### Additional boundaries

- [DIEM, v1, section 4, posted 2026-08-29](https://arxiv.org/html/2608.29252v1#S4)
  measures alignment with the current batch gradient and reweights examples
  while constraining update magnitude. It is recent, directly relevant RL
  prior art, but uses a different reference direction and acts within the
  training batch. Dynamic alignment weighting is not a new contribution here.
- [Asymmetric Prompt Weighting, v1, sections 3 and 5](https://arxiv.org/html/2602.11128v1#S3)
  studies reward-based prompt weights and how their value depends on the
  learning/cost regime. It is not a pre-generation prompt selector. Preserving
  GRPO's weighting is not proof that GRPO is the best use of a rollout budget.
- [Mu-GRPO, v1, section 3](https://arxiv.org/html/2605.17570v1#S3)
  stabilizes stale-response training through staged reuse and update controls.
  Its existence rules out describing our motivation as "off-policy data cannot
  train a useful model." Our trainer still samples current-policy responses.

## Why this implementation was written

The prior 40-point study concerns score measurement, policy drift, and
selection agreement. It does not establish that gradient-based selection
generally hurts benchmark performance. The
[evidence inventory](OBSIDIAN_EXPERIMENT_SYNTHESIS_2026-09-12.md#evidence-actually-available)
also records a measurement-to-training mismatch in parameter coordinates,
response length weighting, reward normalization, and optimizer geometry.
The purpose of this implementation is to test a particular repair candidate
against actual continuation learning, not to turn low overlap into a claim
of downstream failure.

| Implementation choice | Reason | Limit |
| --- | --- | --- |
| Same prompt pool and cached K=8 responses | Test whether existing answers can support acquisition without a new candidate-generation round. | Historical generation is not free in cold-start accounting. |
| Unmerged trainable LoRA parameters | Score in coordinates actually updated by the learner. | Coordinate matching alone may explain any gain. |
| Candidate mean-token derivatives | Match the response-length convention of the current GRPO loss. | This is not the same vector as an unnormalized expected-reward gradient. |
| Validation reward gradient using token sums | Define the target direction by expected verifier reward, not by reproducing a noisy ranking. | A finite validation sample can still be noisy or unrepresentative. |
| Full response likelihood ratios | Retain response-distribution information omitted by partial token correction. | Support, correct likelihoods, and manageable variance are still required. |
| Two- and four-response moments | Estimate a bounded approximation to G=8 group normalization without fitting unknown current success probability. | No universal variance or rank-preservation guarantee. |
| Checked central differences, with an autograd reference | Avoid storing every candidate gradient while measuring the same directional target. | This is established numerical machinery; validation backward passes and calibration remain expensive. |
| Existing GRPO continuation and independent evaluation | Test whether selection changes learning, keeping source runs immutable. | Equal updates are not equal total cost; the default test split is not an untouched external benchmark. |

### The exact hypothesis being tested

For a fixed prompt, let p be current-policy success probability and define
`z(y) = grad_LoRA log pi(y|x) / T(y)`. Under independent binary-reward
responses, the expected initial unclipped GRPO ascent vector can be written
as `f_G(p) * C`, where `C = E_pi[r*z] - p*E_pi[z]`.
The explicit finite sum for `f_G` is in the earlier specification.

For the existing G=8 and epsilon=0.0001, the implementation approximates
that known coefficient by `a - b*p*(1-p)`. Its deterministic absolute
coefficient error is below 0.018162, and its relative coefficient error is
below 0.979%. These are algebraic bounds, not measured LLM results.

With a fixed validation/optimizer direction d and full response ratios w:

```text
q_j = d^T z(y_j)
h_jl = 0.5*w_j*w_l*(r_j-r_l)*(q_j-q_l)
v_jl = 0.5*w_j*w_l*(r_j-r_l)^2
U2 = mean of h_jl over unordered pairs
U4 = mean of h_jl*v_mn over disjoint pairs, with pair roles distinguished
score = a*U2 - b*U4
```

Distinct response indices are necessary: multiplying two estimates that
reuse an endpoint would not generally estimate the product of their
expectations. With K=G=8, an exact eight-response group statistic has only
one complete group. The lower-order moments allow multiple combinations
within that cache, but those combinations are dependent and are not extra
independent observations. Pair aggregation needs O(K^2) scalar arithmetic;
model likelihood and derivative passes dominate the unresolved cost.

Conditional on fixed policies and a fixed direction independent of the
candidate cache, exact ratios and derivatives give
`E[U2]=d^T C` and `E[U4]=p*(1-p)*d^T C`. The actual finite-difference backend
adds numerical approximation error. Input manifests check lineage; they do
not by themselves prove stochastic independence or absence of pretraining
contamination. Clipping, unknown support, data-dependent policy fitting, and
later nonlinear AdamW updates are not covered by these identities.

The potentially distinct element is therefore the **specific finite-group,
cross-policy moment approximation used for prompt acquisition**. In the
sources inspected, no identical `a*U2-b*U4` acquisition rule was identified.
That is a scoped search result, not evidence that the combination is globally
new or sufficiently substantive for a methods paper. No claim of first use
of importance sampling, pair statistics, optimizer matching, or forward-based
influence evaluation is defensible.

### Not the other Obsidian proposal

The vault's methods-first note proposes variance-corrected **batch matching**:
it estimates inter-prompt gradient interactions and distinguishes scoring
replicates from future training replicates. This implementation does not do
that. It ranks individual prompts with a scalar score; it does not optimize
a Gram matrix, batch redundancy, or training-variance penalty. We implemented
the low-order specification linked at the top, not every candidate in the
vault. These alternatives must not be described as the same algorithm.

## What the current arms can establish

| Comparison | Question it can answer | What it cannot establish |
| --- | --- | --- |
| `low_order` vs `pair_u2` | Does the finite-group normalization term help beyond the plain full-ratio covariance score? | Superiority to a published selector or repair of partial correction. |
| `low_order` vs equal-update `random` | Does this selected subset improve continuation under the same update count? | Compute efficiency or the reason for any gain. |
| Optional `passrate_beta` | Does the method beat selecting by the largest cached success fraction? | Superiority to intermediate-difficulty sampling, MoPPS, or Prompt Replay. Those are not this arm's implemented rule. |
| Identity vs frozen-RMS geometry in separate suites | Is the optimizer approximation responsible for a difference? | The correctness of a full AdamW utility prediction. |

The default three-arm suite is thus a screening experiment. It is **not** a
complete test of the paper's proposed causal chain. Existing g00/g11 outputs
use different measurement conventions; treating them as matched controls
would confound correction with coordinate, normalization, and score changes.

Before a full study, specify and add only the controls needed for its claim:

1. A same-backend partial-correction control and an exact-G target audit on a
   tractable frozen model. Show the omitted-information failure under this
   learner-matched target; the earlier theorem uses a different score.
2. An independent current-policy directional reference, with generation and
   derivatives charged, to distinguish a poor estimator from a poor target.
3. A strong low-cost reward-history selector, not merely the easiest-prompts
   control; fix its rule without using final test outcomes.
4. A ToV/Mirrored-Influence-style adaptation if claiming computational novelty
   or efficiency over forward influence methods. State its reward objective
   and correction explicitly; do not label an adaptation a reproduction.
5. Random training with the saved selection budget, and benchmark evaluation
   outside the selector's validation set. Default E5 holdout results alone
   do not establish broad cross-benchmark generalization.

These controls are requirements identified by this review, **not silently
implemented or already completed experiments**. A one-seed pilot can check
numerics and runtime after the protocol is settled, but cannot establish
benchmark superiority. Do not spend all five seeds to answer a question the
default arms cannot identify.

## Cost and stop conditions

The default finite backend requires current-policy center likelihoods plus
two perturbed forward evaluations, and behavior likelihoods when not cached.
It also requires validation gradients and exact calibration probes. Training
and independent evaluation generate new responses. No H100 timing was
measured; avoiding candidate backward passes is not a measured speedup.

Keep two accounting regimes separate: existing-cache marginal cost and
cold-start cost including cache creation. In a stand-alone method comparison,
charge each selector the validation/scoring work it would require; sharing
one scoring run across experimental arms does not divide that deployment
cost by the number of arms. `cost.jsonl` is an input to this accounting, not
an automatic equal-budget certificate. `matched_total_cost` remains false.

If scoring costs C_select and a random training update costs C_step, saving
delta_U updates only saves compute when `delta_U*C_step > C_select`, at a
comparable final reward. The existing `--random-extra-steps` option implements
an explicitly chosen control budget, not an automatic budget match.

Stop or simplify the candidate if:

- U4 adds noise without a reproducible downstream benefit over U2.
- Full ratios are dominated by a few responses or fail numerical checks.
- Coordinate/geometry matching alone accounts for the observed gain.
- Gains disappear against stronger low-cost selection or cost-matched random.
- Better directional scores do not improve independently measured learning.

Do not hide these outcomes with clipping, additional fresh samples, or a
changed test set while retaining the same estimator claim.

## Implementation and verification record

- [Scalar moments and algebra audit](../src/low_order_reuse.py).
- [LoRA derivative backend and calibration](../src/low_order_backend.py).
- [Resumable scoring and existing-engine training integration](../src/low_order_experiment.py).
- [Shell entrypoint](../scripts/run_low_order.sh) and [operating guide](LOW_ORDER_REUSE_RUN.md).
- CPU-only focused and regression tests: **137 passed in 28.47 seconds**.
  Includes a small real Transformers/PEFT LoRA model, exact finite moment
  enumeration, restart/contract tests, and subprocess failure cleanup.
- No OLMo/H100 scoring, continuation training, benchmark evaluation, manuscript
  edit, website deployment, or cluster process restart was performed.

Search terms included GRPO U-statistics, finite-group normalization,
off-policy polynomial correction, directional-derivative data selection, and
forward-only influence. Date-restricted searches were also attempted for
2026-08-12 through 2026-09-12; search-engine dates and coverage are incomplete.
The directly checked DIEM version is within that interval; PAIR's 2026-08-11
version is just outside it. Neither search results nor this inventory certify
an absence of overlapping work.
