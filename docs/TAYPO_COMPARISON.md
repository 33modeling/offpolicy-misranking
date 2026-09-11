# TayPO And The Proposed Additive Scoring Estimator

September 11, 2026. This is a related-work assessment and an independent
derivation, not a novelty certification or a result from the GPU cluster.

## Bottom Line

The additive estimator is **not generally identical to TayPO-2**, but its
leading correction is already present in a terminal-reward specialization of
TayPO-2. It is inappropriate to claim that low-order off-policy correction,
canceling leading bias, or using old trajectories for this purpose is new.
The additive construction retains additional within-prefix and within-suffix
products. Whether retaining those terms improves data selection is an empirical
question. It may instead increase error or variance.

The implementation therefore includes both `gadd` and `tay2_terminal`. The
latter is explicitly a terminal-reward, group-advantage, clipped adaptation
used as a scoring comparator, not a reproduction of the published Atari agent.

## What TayPO Establishes

Tang, Valko, and Munos introduced Taylor Expansion Policy Optimization at
ICML 2020. They expand policy value around a behavior policy using powers of
ratio deviations, connecting low-order surrogate optimization with off-policy
evaluation. First-order terms recover an idealized familiar policy surrogate;
second-order terms add corrections involving pairs of distinct times. TayPO-2
optimizes the sum of those terms. Higher order brings an approximation/variance
trade-off, not automatic improvement. Their value guarantee includes a remainder
and a uniform policy-distance condition; it is not certification by empirical KL
or ESS. See the [official record](https://proceedings.mlr.press/v119/tang20d.html)
and [main paper, Sections 2-4, Eq. 5-11](https://proceedings.mlr.press/v119/tang20d/tang20d.pdf).

The [supplement, H.4](https://proceedings.mlr.press/v119/tang20d/tang20d-supp.pdf)
describes a truncated pair-enumeration implementation with quadratic trajectory
work, while sampled pairs can reduce that work. The experiments concern policy
and value learning, not validation-gradient prompt ranking. These practical
choices must be distinguished from the exact population identities.

## A Common Finite-Horizon Setting

The following is our derivation to compare the formulas in the paper's setting.
Consider a fixed prompt, finite response trajectory, terminal reward `R`, fixed
behavior policy `beta`, and target policy `pi_theta`. Assume adequate support
and integrable score terms. No clipping, length normalization, or differentiable
reward is used here. For a sampled trajectory define

```text
r_t = pi_theta(a_t | h_t) / beta(a_t | h_t)
d_t = r_t - 1
P_t = product_{u<t} r_u
S_t = product_{u>t} r_u
z_t = grad_theta log pi_theta(a_t | h_t)
```

The response probability ratio is `product_t r_t`. Expand this product by
degree in the individual `d_t`. The degree-two terminal-reward surrogate is

```text
F_2(theta; trajectory) = R * (1 + sum_t d_t + sum_{i<j} d_i*d_j).
```

Holding the sampled trajectory and behavior policy fixed while differentiating
gives

```text
grad F_2 = R * sum_t r_t * (1 + sum_{u!=t} d_u) * z_t.
```

Thus our unclipped `tay2_terminal` comparator uses token coefficient

```text
w_tay2,t = r_t * (1 + sum_{u!=t}(r_u - 1)).
```

This is the terminal-return specialization of the degree-two ratio expansion.
It is not a claim that the discounted Q-function implementation, state-dependent
value baselines, group-standardized GRPO, and this estimator are interchangeable.
The CPU test differentiates `F_2` with autograd and checks the detached-weight
implementation against that derivative.

## What The Additive Construction Keeps And Drops

The proposed raw combination is

```text
g_add = g10 + g01 - g00
w_add,t = r_t * (P_t + S_t - 1)
w_full,t - w_add,t = r_t * (P_t - 1) * (S_t - 1).
```

Expand `P_t` and `S_t` separately in token deviations:

```text
w_add,t / r_t
  = 1 + sum_{u!=t} d_u
    + sum_{i<j<t} d_i*d_j + sum_{t<i<j} d_i*d_j
    + higher-degree terms entirely before t or entirely after t.
```

The first line is exactly `w_tay2,t / r_t`. The additive construction also
retains same-side terms of all orders. It omits terms involving at least one
token before `t` and one after `t`. It is therefore not a fixed-degree Taylor
truncation in the individual token deviations.

For one or two response tokens, no position has both a nonempty prefix and a
nonempty suffix, so the raw additive, degree-two, and full coefficients agree.
For three tokens with ratios `(2, 3, 4)`, they differ:

| Coefficients | Token 1 | Token 2 | Token 3 |
| --- | ---: | ---: | ---: |
| Full | 24 | 24 | 24 |
| Additive | 24 | 15 | 24 |
| Degree-two terminal | 12 | 15 | 16 |

These are algebraic coefficients, not measured rewards. All ratios are positive
and can be realized by choosing behavior probabilities sufficiently small.

Solving the manuscript's two-token counterexample consequently does **not**
distinguish the additive construction from this TayPO-derived comparator.
Both remove that particular omission in the unclipped setting.

## Relation To The Manuscript's Impossibility Result

The impossibility claim restricts the selector's information: it observes the
partially corrected scores without the omitted ratios or new current-policy
outcomes. Both composite estimators here access token ratios on both sides of
the scored token. They are outside that restricted information class. Their
existence therefore does not contradict the impossibility result, but that
result cannot establish that off-policy reuse in general is impossible.

Correcting a population gradient and reliably ranking prompts from a finite
response pool are also different tasks. A Taylor correction does not by itself
settle the latter: sample variance, cosine normalization and validation noise
still matter. Those issues must be tested rather than presented as a benefit
already established by the new formula.

## A Further Difference: Not Generally An Objective Gradient

On a three-token reward-bearing path, the additive vector field in independent
ratio coordinates has coefficients

```text
F_1 = r_2*r_3
F_2 = r_1+r_3-1
F_3 = r_1*r_2.
```

Here `partial F_1 / partial r_2 = r_3`, whereas
`partial F_2 / partial r_1 = 1`. Unless `r_3=1`, the mixed partials differ.
Thus this sample-level field is not generally the derivative of one scalar
surrogate. The example can be isolated using a terminal reward that is one
only on this path. This does not invalidate its use as a gradient *estimator*
for scoring, but it rules out importing TayPO's objective-improvement argument.

Our code detaches all composite coefficients and computes a weighted sum of
current-policy log-probability gradients. It does not backpropagate through the
coefficients a second time, nor use the resulting field as a new GRPO loss.

## Limits Of The Bias Argument

If `|P_t-1| <= a` and `|S_t-1| <= b` uniformly, the raw additive bias has norm
at most `a*b*E_beta[sum_t |r_t R| * ||z_t||]`. This is an absolute-error bound
under explicit conditions. Small average KL alone does not supply those
cumulative-ratio bounds, especially on long responses. Pointwise cancellations
can also make a partial estimator more accurate than the additive one.

Both additive and degree-two terminal coefficients agree with full correction
to first order in small token deviations at fixed finite horizon. Claiming
second-order residual error therefore does not distinguish the proposal from
the comparator. Constants can grow with horizon, and no variance dominance
follows. Full correction may still be better; more terms are not always better.

The theory also concerns gradient error, not cosine error without conditions.
For a fixed unit validation direction and nonvanishing target gradient norm,
normalization has a local error bound. Near-zero norms can amplify error;
uncertainty in the shared validation direction adds another issue. Existing
finite-budget cosine bias is not removed by either construction.

Likewise, a small absolute score error can still reverse nearly tied prompts.
The original small-KL counterexample has shrinking score magnitudes. It is not
in conflict with a small absolute value-approximation bound and does not establish
fixed positive downstream damage as policy distance tends to zero.

## Practical Clipping And Group Advantages

The new experiment deliberately inherits the source clipping cap. Its `gadd`
is exactly the linear combination of the source **component-clipped gradients**.
The terminal comparator uses clipped current-token ratios in its degree-two
coefficient. Neither equals the raw formula after clipping activates. Their
remainder contains additional clipping error. Even with two tokens, agreement
of raw expressions is not a guarantee of unbiasedness for clipped estimates.

The scoring code uses unnormalized leave-one-out group advantages, as does the
existing experiment. It is not silently substituted for arbitrary normalized
GRPO advantages in the derivation. Any group-standardized version needs its own
argument. In all cases finite sample variance remains to be measured.

## Cost And Experimental Decision

Both scoring rules use existing response tokens and rewards. With current and
behavior token log probabilities available, both coefficient calculations are
linear in response length. The terminal specialization permits a prefix/suffix
sum implementation for the degree-two comparator; it is therefore misleading to
claim an inherent quadratic-versus-linear speed advantage over TayPO here.

Each rule can use one weighted backward per gradient micro-batch, followed by
the existing projection and cosine. One method does not require three separate
backwards. Benchmarking both does require two gradient passes. Model loading,
probability computation, and gradient passes are real GPU costs. The current
source artifact saves scores and norms, not all component vectors, so this is
not a zero-cost CPU recombination of those files.

The minimum useful comparison is against the terminal TayPO-derived estimator,
the actual basic reuse estimator, and full correction, with the same tokens,
rewards, layer range, projection, validation direction, and clipping disclosure.
Independent downstream reward is needed before claiming a useful selection
improvement. Higher overlap or winning on the two-token construction is not
sufficient evidence.

Recommendation: retain the additive construction as an **experimental variant**,
include the TayPO-derived comparator now, and do not expand the manuscript's
novelty claim until a distinct mechanism and benefit survive that comparison.
No GPU training is launched by this release; the executable scorer is a first
screen, not a substitute for downstream validation.

## Verification Artifacts

- `src/additive_correction.py`: raw and explicitly clipped coefficients.
- `tests/test_additive_correction.py`: exact identities, autograd equivalence,
  signed weights, clipping effects and overflow rejection.
- `bash scripts/run_additive.sh check`: deterministic algebra audit.
- `docs/ADDITIVE_CORRECTION.md`: cluster scoring and recovery commands.

No unpublished author code, literature-wide absence claim, empirical novelty
claim, or reproduced Atari score is asserted in this note.
