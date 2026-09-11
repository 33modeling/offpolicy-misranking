# Connecting the Obsidian research to the current experiments

Date: 2026-09-12 (Asia/Seoul)

Subsequent user instruction authorized implementation. The
[execution guide](LOW_ORDER_REUSE_RUN.md) records the code and verification.
The research-stage status and claims below are preserved; implementation does
not establish novelty, benchmark superiority, or GPU efficiency.

The subsequent [pre-launch comparison](LOW_ORDER_PRIOR_ART_AND_RATIONALE_2026-09-12.md)
adds ToV, Mirrored Influence, For-Value, and TACS to the computational-prior-art
assessment. The first three were already in other vault indexes or notes;
their omission from this synthesis did not imply an absence of precedent.

Status: research specification, not an implemented or validated main method.
No experiment code, running job, manuscript, or publication site was changed.
This extends [the methodology gate](METHOD_DECISION_2026-09-12.md); it does not
override it. Literature novelty and equal-total-cost benchmark superiority are
not established.

## Decision in plain language

Investigate a **training-matched, low-order reuse score**. Use the eight cached
responses to estimate which prompts would produce a useful current-policy
GRPO gradient. Recover the current-policy distribution using full response
likelihood ratios, use all correct/incorrect response contrasts, and account
for the actual eight-response reward normalization with a bounded approximation.
Rank prompts by a directional score in the learner's trainable coordinates.

The selected objects are **prompts from the same candidate pool**. Cached
responses supply their selection scores. Subsequent training still generates
new responses on the selected prompts. This proposal does not change the
trainer into an off-policy replay algorithm.

The useful connection is not another activation heuristic or another mixture
of g00/g10/g01/g11. It is that our measurement gradient, the stochastic GRPO
loss gradient, and an AdamW update are different objects. An acquisition method
must state which one it predicts. The new candidate targets the second and
uses an explicitly approximate mapping to the third.

The proposed rule has a concrete expression below and requires no unknown
current success probability. Its small normalization-approximation error is
an algebraic property, **not a bound on selection error or benchmark loss**.

## Evidence actually available

Local checkout snapshots inspected:

| Repository | Commit | Role |
| --- | --- | --- |
| offpolicy-misranking | a1e6dc6 | Scoring, trainer, configuration, method gate |
| obsidian | 6c3874c | Existing paper notes and conference indexes |
| offpolicy-misranking-paper-v2 | 9b4c8fc | Registered result summaries and reliability analysis |
| transfer | e8676d5 | Locally available transferred logs |

The transfer remote returned `Repository not found` during this review. Its
newest remote state was not verified. Concurrent untracked CFCS and reference
control files were read where relevant, but were not edited or staged.

During final review, the concurrent `ae94822` commit added reference-control
code, its shell wrapper, and tests. It did not change the scoring, trainer,
rollout, or H100 configuration inspected here. That control compares the
reproducibility of coarser reward-based rankings; it is not an implementation
of the candidate below and has not supplied downstream evidence in this review.

### The 40-point OLMo matrix

Source: paper repository, `evidence/2026-09-10/observed-results.json` and
`observed-points.csv`, snapshot `2026-09-10T08:31:22Z`.

| Dataset | d0 A/B top-k overlap | d400 normalized ESS |
| --- | ---: | ---: |
| MATH-500 | 0.1350 | 0.2676 |
| MBPP | 0.1020 | 0.0354 |

Each entry averages five seeds. Chance overlap is approximately 0.10. These
are rounded status-derived measurements, not downstream test accuracies. The
artifact explicitly records `selection_gains_available=false` and
`final_matrix_verified=false`: all 40 rows are present, but raw score artifacts
and registered confidence-bound decisions were not locally validated.

The low d0 overlap motivates investigating sampling noise. It does not prove
that gradient selection fails to improve learning, or identify which noise
source dominates. Candidate and validation samples both change between A/B.
The ESS observation warns against claiming that full importance weighting
will cheaply rescue arbitrarily old MBPP responses.

The separate `evidence/2026-09-09/reliability-budget-d0.json` reuses existing
pools, includes five MATH seeds but only three MBPP seeds, and fixes validation
half-size. Its larger-budget projections are not new measurements. It does
not establish an irreducible noise floor.

### Existing synthetic continuation results

Source: `outputs/cfcs-anchor-study-20260911/summary.json`, `results.csv`, and
`config.json`. At parameter drift 4 and cached K=8, twenty seeds give:

| Selector | Mean held-out reward increase, percentage points |
| --- | ---: |
| Uniform | 14.73 |
| Pass-rate | 16.10 |
| g00 | 22.31 |
| Clipped g11 | 19.62 |
| CFCS | 19.65 |
| Anchor CFCS | 20.39 |
| Independent fresh directional score | 25.54 |

This is an autoregressive Markov environment, not an LLM benchmark. Each arm
has 64 continuation updates; total computation is not matched. The full grid
contains ten conditions, not just this row. Anchor CFCS loses 1.92 percentage
points to g00 here, while g00 exceeds uniform by 7.58 points. Therefore neither
"gradient selection is useless" nor "our correction mixture solves it" is
supported. Full fresh scoring suggests remaining utility in this toy setting,
but its additional generation is not free.

No completed, independent LLM downstream arm results were located for this
review. Files under pytest temporary directories are fixtures, not evidence.

## A concrete measurement-to-training gap

These are code-contract observations, not a claim that the registered matrix
was implemented incorrectly. Its estimators intentionally share a proxy.

| Component | Existing measurement | Actual GRPO learner |
| --- | --- | --- |
| Parameters | Ordinary weights in the last decoder blocks and final norm, after adapter merge | Unmerged trainable LoRA parameters |
| Advantage | Leave-one-out reward centering | Within-group population-standard-deviation normalization |
| Token aggregation | Sum of response-token score gradients | Mean over each response's tokens |
| Geometry | Projected gradient cosine | AdamW, with gradient clipping and optimizer state |
| Group size | Four-response scoring micro-groups | Eight-response training groups |

Code anchors at the snapshot above:

- `src/grads.py:140`, `grad_params`, and `prompt_gradient` below it.
- `src/rollout.py`, `load_policy`: adapter `merge_and_unload` for scoring.
- `src/experiment.py:128`, `score_oracle_microgroups`: candidate cosine and
  independently partitioned validation directions.
- `src/experiment.py:338-393`: LOO advantages and micro-group gradients.
- `src/train_policy_grpo.py:96`: `standardized_group_advantages`.
- `src/train_policy_grpo.py:140`: `clipped_grpo_loss`, including length means.
- `src/train_policy_grpo.py:788-829`: trainable LoRA and AdamW state.
- `configs/olmo3_rlzero_h100.json`: cached K=8, fresh K=32, micro-group=4,
  training group G=8. Other inspected model configurations also use K=8/G=8.

LOO centering and group-mean centering differ only by a constant at fixed
group size. The substantive differences are random standardization, response
length weighting, parameter coordinates, and the optimizer. These must not
be conflated with policy-distribution correction.

## What the existing research contributes

This was a targeted review of the existing indexes and relevant notes, not a
claim to have read every paper in the vault. In particular, the GIST and OPUS
notes are marked `abstract-reviewed`; their primary methods were inspected
for this review rather than treating those notes as full-paper audits.

| Existing Obsidian note or newly checked source | Connection and novelty boundary |
| --- | --- |
| GIST, `papers/GIST-Targeted-Data-Selection-for-Instruction-Tuning-via-Coupled-Optimization-Geometry-ICML2026-ko.md` | LoRA geometry and validation-gradient spectral structure matter. Merely switching to LoRA gradients is not new. [Methods](https://arxiv.org/html/2602.18584v1) |
| OPUS, `papers/OPUS-Towards-Efficient-and-Principled-Data-Selection-in-Large-Language-Model-Pre-training-ICML2026-ko.md` | Optimizer-induced updates, including an approximate AdamW mapping, are an established selection target. [Methods](https://arxiv.org/html/2602.05400v1) |
| GradAlign, `papers/2602.21492-GradAlign-Gradient-Aligned-Data-Selection-COLM2026-ko.md` | Validation-directed RL prompt selection is already established. Its cosine choice and normalized-advantage discussion are relevant controls, not concepts to rename. [Sections 4 and 7](https://arxiv.org/html/2602.21492v1) |
| Conditional IS, `papers/Conditional-Importance-Sampling-Off-Policy-Learning-AISTATS2020-ko.md` | Conditioning provides a way to distinguish necessary correction information from removable randomness. This is an established principle, not our new theorem. Primary abstract and local note inspected. [AISTATS](https://proceedings.mlr.press/v108/rowland20b.html) |
| Mu-GRPO, `papers/2605.17570-Mu-GRPO-Off-Policy-GRPO-arXiv-ko.md` | Successful stale-response training is possible; correction fidelity and useful training are not equivalent. Its stage/clipping design is different from prompt acquisition. [Paper](https://arxiv.org/html/2605.17570v1) |
| LEEPS, `papers/2607.28077-LEEPS-Explore-Exploit-Prompt-Sampling-arXiv-ko.md` | Activation neighbors plus historical outcomes already inform acquisition. An activation substitution is not sufficient novelty. [Methods](https://arxiv.org/html/2607.28077v1#S3) |
| CurES, located through the ICLR data-selection index | Difficulty estimation and adaptive group budgets are already studied. Simply fitting p and allocating more responses is not a new method. [Sections 3-4](https://arxiv.org/html/2510.01037v1) |
| GRPO U-statistic analysis, newly checked | The pairwise LOO identity is established. Appendix A also treats normalization and IS, using an asymptotic second-order representation for the standardized case. Do not claim prior work entirely ignores normalization or off-policy data. [v3, Appendix A](https://arxiv.org/pdf/2603.01162v3) |
| Group-standard-deviation identity, newly checked | Binary finite-group GRPO identities and difficulty weighting are already studied. These are ingredients, not a new discovery. [Theorem 1](https://arxiv.org/html/2607.00152v1) |
| PAIR, newly checked; no matching vault filename found | Uses pair statistics and joint inclusion correction for adaptive on-policy suffix continuation. Its exact claim excludes random standardization. Our target is cached cross-policy prompt scoring, not suffix allocation; pairwise correction alone is already occupied. [Sections 3-5 and Appendix B](https://arxiv.org/html/2608.11368v1) |

Additional vault notes consulted to screen alternatives: Two-Stage
Optimizer-Aware Online Data Selection, Training-Trajectory-Aware Token
Selection, PUST, BLISS, Active Learning with Low-Rank Structure, POPO, and
Doubly Robust Policy Gradient. Their presence does not mean all their
full texts were audited. None supplies evidence that our candidate already
outperforms random at equal total cost.

## Candidate specification

### 1. Target and assumptions

Fix a current policy pi, behavior policy beta, prompt x, and trainable LoRA
coordinates psi. For a response y, define binary verifier reward r(y), length
T(y), and the length-normalized score vector

```
z(y) = (1 / T(y)) * grad_psi log pi(y | x).
```

For G independent current-policy responses, q is their mean reward and
sigma=sqrt(q*(1-q)). The target is the expected initial, unclipped GRPO
ascent loss gradient:

```
g_G(x) = E_pi^G [ (1/G) sum_j ((r_j-q)/(sigma+epsilon)) z(y_j) ].
```

At the first evaluation of a new on-policy training group, the PPO ratios
are one. This expression does not describe later optimizer epochs, an active
clipping boundary, KL penalties, or the nonlinear AdamW update exactly.

Assumptions: fixed policies; independent responses; binary, consistently
verified rewards; correct generation probabilities including EOS/truncation;
beta support covering pi; and a scoring cache independent of fitting the
policy/direction under the claimed conditional analysis. Actual temperature,
top-p, checkpoint, and optimizer manifests must be checked before application.

### 2. Average over group composition analytically

Let p=Pr_pi(r=1) and define the vector covariance

```
C(x) = E_pi[r*z] - p*E_pi[z].
```

Conditioning on the number of successes in a group gives

```
g_G(x) = f_G(p) * C(x)

f_G(p) = (G-1)/G * sum_{l=0}^{G-2} binom(G-2,l)
         * p^l * (1-p)^(G-2-l)
         / (sqrt(((l+1)/G)*(1-(l+1)/G)) + epsilon).
```

Derivation: conditional on m successes, the group's expected gradient is
`q*(1-q)/(sqrt(q*(1-q))+epsilon) * (E[z|r=1]-E[z|r=0])`.
Average m under Binomial(G,p), factor out p*(1-p), and use
`binom(G,m)*m*(G-m)/G^2 = (G-1)/G * binom(G-2,m-1)`.
At p=0 or 1 the covariance and gradient vanish; use the continuous extension
of f. Length normalization is retained, so E_pi[z] need not be zero.

This identity builds on existing group-normalization analyses. It is not
presented as a newly discovered GRPO identity.

### 3. A controlled low-order approximation, not an estimated success rate

For the current G=8, epsilon=0.0001, write t=p*(1-p), in [0,1/4]. The preceding
finite sum is the cubic polynomial

```
f_8 = 2.644951552889 - 3.748153017867*t
      + 2.423623358082*t^2 - 0.264358267015*t^3.
```

Its second derivative is positive throughout this interval. Take its secant
line and subtract half the maximum secant error. This yields

```
f_tilde(p) = a - b*p*(1-p)
a = 2.626790565102
b = 3.158769570035
sup_{p in [0,1]} |f_tilde(p)-f_8(p)| < 0.018162.
```

The maximum secant error occurs at t approximately 0.124111638725. Since
`min f_8 > 1.85525`, the relative coefficient error is below 0.979%.
This is a deterministic approximation bound from G and epsilon, not a fit
to experiment results and not a confidence interval. It applies to the
population gradient coefficient only, not to finite-cache estimates,
validation direction error, rankings, AdamW updates, or final accuracy.

An exact eight-way importance-weighted group statistic would have no
regrouping benefit at the existing K=G=8. The low-order approximation instead
uses two- and four-response terms, allowing multiple contrasts within those
same eight observations without changing the trainer's G to four.

### 4. Estimate the score from the existing responses

Let v be a validation reward-gradient direction in the same LoRA coordinates,
held fixed independently of candidate responses. Let M be a frozen linearized
optimizer mapping, and d=M^T*v. Use v itself for the identity-mapping control.
An AdamW mapping must be labeled approximate: its new second moment and
global gradient clipping depend on the candidate batch. OPUS-style optimizer
awareness is a control, not the novelty claim.

For each of the K cached responses compute

```
w_j = pi(y_j | x) / beta(y_j | x)      # full response ratio, not token-only
z_j = d^T * z(y_j)                     # scalar directional derivative
h_jl = 0.5*w_j*w_l*(r_j-r_l)*(z_j-z_l)
v_jl = 0.5*w_j*w_l*(r_j-r_l)^2.

U2 = average h_jl over all j<l
U4 = average h_jl*v_mn over all disjoint unordered pairs (j,l), (m,n),
     with the first and second pair roles distinguished

score(x) = a*U2 - b*U4.
```

For independent beta responses, ordinary importance sampling gives
`E[U2]=d^T*C` and `E[U4]=p*(1-p)*d^T*C`. Thus the score is unbiased for the
specified approximate directional target, **not** exactly unbiased for full
GRPO. No p estimate, fitted success predictor, or fresh candidate response is
needed. All reward and importance coefficients are detached; differentiating
them again would change the estimator.

Choose the top k prompts by this signed score, with a fixed tie rule. Keep
validation/test splits separate. A provisional study should freeze selection
once at the existing checkpoint; do not invent an uncharged online rescoring
schedule or optimize a threshold on held-out test results.

There is no combinatorial model computation. At K=8 there are 28 pairs.
Let V=sum(v_jl) and I_j=sum_{l != j}(v_jl). Then

```
U4 = sum_{j<l} h_jl*(V-I_j-I_l+v_jl)
     / (binom(K,2)*binom(K-2,2)).
```

This costs O(K^2) scalar arithmetic after obtaining response derivatives.
The pairs are dependent; 28 pairs are not 28 independent responses.

### 5. What this repairs and what it does not

The score restores full response-distribution information in its retained
terms, rather than silently omitting prefix/suffix ratios. It also makes the
GRPO normalization approximation explicit and quantitatively bounded.
Unlike a Taylor expansion in policy drift, the approximation here is of the
known finite-group normalization factor. Standard U-statistics, covariance
identities, and polynomial approximation are not themselves novel.

The candidate does NOT inherit the existing partial-correction theorem's
target automatically: that theorem concerns a different gradient score.
Any claimed repair must be demonstrated under the same newly specified
learner-matched target in both counterexample and practical comparison.

Full response weights can still have extreme variance. Four-way products
can be unstable; subtracting U4 can also amplify noise. Lower polynomial
degree does not universally guarantee lower variance or lower ranking regret.
Clipping or self-normalizing weights changes the stated expectation and must
not be slipped into an implementation while retaining its guarantee.
All-correct/all-incorrect caches give zero contrast and cannot reveal missing
outcomes. Nothing here guarantees rescuing the MBPP d400 cache.

A finite enumeration of a two-outcome response distribution checked the U2
and U4 expectation identities at K=8 and the coefficient calculation. It also
exhibited slightly HIGHER variance than exact on-policy group scoring at
success probabilities 0.1 and 0.8. These are arithmetic sanity checks in the
tool session, not benchmark experiments, GPU tests, or evidence of superiority.
The check enumerated all 256 binary caches with scalar score
`z=(r-p)/(1 if r=1 else 3)` at (pi success, beta success) pairs
`(0.1,0.1), (0.3,0.3), (0.5,0.5), (0.8,0.8), (0.6,0.3), (0.1,0.8)`.
Both moment identities and the exact full-group expectation agreed with
their analytic values to absolute tolerance 1e-10. This is not an independent
audit of the derivation or a numerical validation of an LLM backend.

## Cost and the decisive comparison

One shared validation gradient is still expensive and noisy. Obtaining z_j
in unmerged LoRA coordinates requires model work; existing projected
micro-group tensors cannot reconstruct it. A directional JVP or central
finite difference could avoid storing every candidate gradient. A finite
difference backend needs two perturbed teacher-forced passes plus an exact
center pass for pi likelihoods unless already available. Beta likelihoods
must also be available or computed. Low-precision finite differences need
numerical validation. No GPU speed or memory saving has been measured.

Count candidate generation already incurred in a cold-start comparison,
cached likelihood evaluation, validation generation/backward, verification,
all directional passes, checkpoint handling, and subsequent training.
K=8 cached responses do not become free merely because this score reuses them.
Pair arithmetic is cheap; candidate transformer passes are the main concern.

Random selection has almost no selection overhead. At the same final
performance, if the proposal costs C_select and saves delta_U training
updates of cost C_step, it needs `delta_U*C_step > C_select` to save compute.
At a fixed total GPU-hour budget, the random arm must be allowed to spend
the saved scoring budget on more training. Equal update counts are not enough.

The minimum informative comparisons, after the method gate is passed, are:

1. Random plus extra training under the same total cost.
2. A cheap difficulty selector, using the same available reward history.
3. Existing g00 and full-correction controls with identical coordinates and
   directional backend, so their comparison does not confound correction
   with the measurement-to-training mismatch.
4. Plain full-ratio U2: essential to show the bounded normalization term earns
   its complexity over an already known pairwise covariance estimator.
5. Full and low-order expected-GRPO targets on a tractable frozen-policy audit;
   then independent fresh learner-matched scoring as a charged control.

Measure independent benchmark reward against cumulative total cost, not
only overlap or alignment. Reuse the existing matched-training harness after
auditing its manifests; do not create another repository or parallel runner.

## Research gate and falsifiers

The candidate is worth a focused mathematical review, not a large GPU launch.
Its potentially distinct contribution is the combination of a finite-G
normalization approximation with an explicit error bound, full cross-policy
moment correction, and inexpensive prompt acquisition in the learner's
coordinates. The search did not establish that this combination is globally
new. PAIR, GRPO U-statistic theory, and group-normalization work are close
enough to require a precise comparison before claiming novelty.

Stop or simplify the proposal if any of the following holds:

- LoRA/length/optimizer matching alone explains the improvement: report that
  finding, but do not attribute it to the new estimator.
- Plain U2 performs as well at less cost: the normalization approximation has
  not earned a main-method role.
- Full ratios make the current eight-response estimates unusable: do not
  silently add fresh pilots or clipping and call the method unchanged.
- Better score estimation does not improve independently measured learning.
- The random baseline catches up when allowed to spend the scoring budget.

Before implementation, separately review the derivation, the nearest prior
methods, and the available raw response/optimizer artifacts. Existing aggregate
40-point logs cannot answer these questions. This record supplies a concrete
method candidate and failure conditions; it does not declare the paper's
solution complete.

## Artifact fingerprints

SHA-256 of the local evidence read, so future uploads can be distinguished:

```
paper/evidence/2026-09-10/observed-results.json
7b363ba4b413fbfb64f4c81f940a857f5a8479783046ae1430cfb443e7632c3e
paper/evidence/2026-09-09/reliability-budget-d0.json
386c2330f6c5aed2fbfb51cbfed07e3a952f4f05d6cf20520d6c1d2836981ab8
outputs/cfcs-anchor-study-20260911/summary.json
f45baa86bb922b64f8f1f61346b36122d8cd0988832da376a772594cc10171b5
outputs/cfcs-anchor-study-20260911/config.json
af5711e843630ff0b8090005f346e0a080a3e19cb1a41e2a55eb3acbceeec68c
outputs/cfcs-anchor-study-20260911/results.csv
cb4bf1f705917d679e85d6f324949e0fb7d8e8eaec0467f59b3f99c481b43ca8
```
