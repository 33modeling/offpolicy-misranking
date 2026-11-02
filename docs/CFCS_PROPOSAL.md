# Cross-Fitted Correction Selection

Date: 2026-09-11. Exploratory algorithm proposal, not a manuscript result or
retrospective registration. Existing experiments and their claims are unchanged.

Implemented and tested. Both exploratory grids are complete; see
[the results and adoption decision](CFCS_RESULTS_2026-09-11.md). The proposed
mixture does NOT yet establish an advantage over the cheap g00 baseline.

## Concrete algorithm

Freeze a validation-update direction v from a small, independent current-policy
validation batch. For each stored behavior trajectory, compute current log
probabilities and the directional token derivative d_t = v' grad log pi_t.
The prototype uses central differences on the same stored tokens: evaluate
theta + h v and theta - h v, with a separate exact center pass for ratios.
There is no candidate-gradient normalization. The target is directional
alignment, not population cosine and not downstream reward. With a small SGD
step this relates to first-order validation improvement; Adam/GRPO finite-step
improvement is not an identity and must be tested.

The four endpoint sample scores use the existing g00/g10/g01/g11 ratio rules.
Divide behavior responses into disjoint groups of two and compute a LOO
advantage within each group. K >= 8 supplies four independent group estimates.
All endpoints use the same trajectories and token derivatives.

For each prompt, decompose the endpoint scores as

```
b = g00
p = g10 - g00
s = g01 - g00
h = g11 - g10 - g01 + g00
score(lambda) = b + lambda_p p + lambda_s s + lambda_h h
```

For raw weights, the interaction has token weight r_t(P_t-1)(S_t-1).
The additive endpoint is lambda=(1,1,0); full correction is (1,1,1).
The implementation clips endpoint weights BEFORE differencing: clipped-full
is its calibration target. Clipped identities cannot certify the true gradient.
The independently existing `src/additive_correction.py` is not modified or
claimed as newly authored here.

Fit shared coefficients from one group half; score the other half, then swap.
Let M estimate between-prompt covariance of the three population corrections,
after subtracting the estimated sampling-noise covariance and projecting to
the positive semidefinite cone. Let V be correction-noise covariance in the
scoring-half mean and c its covariance with the base-score noise. Solve

```
min_{0 <= lambda <= 1}
    (lambda-1)' M (lambda-1) + lambda' V lambda + 2 c' lambda.
```

This is the plug-in expected squared error of CENTERED scores relative to
the clipped-full population scores, up to a lambda-independent term.
Centering removes pool-wide shifts that cannot affect selection. The solver
enumerates all 27 active faces of the three-dimensional box. A shared fit
uses evidence across prompts instead of attempting a covariance estimate
separately from eight responses for every prompt. Cross-fitting separates
coefficient calibration and score evaluation; it does not make the final
adaptive estimator unbiased or yield a confidence interval.

Select the top k scores with existing seeded random tie breaking. Optional
uniform exploration reserves part of k; the first study uses no exploration
for ANY arm, isolating the correction rule. Independent probe sampling is
needed in a repeated online deployment to revisit unseen/zero-signal prompts.

## Why this follows from the analysis

- Partial corrections omit different components: estimate all increments.
- Full products have correlated sampling noise: include their covariance.
- Shared score offsets do not change which examples help: fit centered risk.
- Candidate cosine introduces normalization bias: use a linear directional
  score and state the changed estimand explicitly.
- Four backward passes per candidate are costly: all endpoint weights combine
  with the SAME scalar directional token derivatives from forward evaluations.

The new proposal is this specific correction decomposition plus cross-fitted,
selection-relevant covariance shrinkage and a common-direction scorer. Basic
alignment, finite differences, shrinkage, and cross-fitting are prior art.
Novelty beyond this particular construction has NOT been established.

Relevant prior art:

- GradAlign: https://arxiv.org/abs/2602.21492
- CROPI: https://aclanthology.org/2026.acl-long.2141/
- Taylor Expansion Policy Optimization: https://proceedings.mlr.press/v119/tang20d.html
- Doubly robust OPE with shrinkage: https://proceedings.mlr.press/v119/su20a.html
- GPS: https://arxiv.org/abs/2602.01970
- Z0-Inf and other forward influence approximations require a dedicated audit.

## First falsification study

Use the existing synthetic verifiable Markov environment, not an artificial
score-only example. Default grid fixed before the first run: drift
0/0.5/1/2/4, behavior responses 8/32, seeds 1000..1019, 160 candidates,
64 further sampled GRPO-style updates. Validation direction uses 32 disjoint
validation prompts; 32 OTHER prompts measure exact terminal test reward.
The exact gradient is only used by the evaluator, never by CFCS.

Compare g00/g10/g01/g11, additive, CFCS without cross-fitting, historical
pass-rate selection, uniform, fresh directional and fresh cosine selection.
Every arm gets the same sampled continuation update budget. This is NOT an
equal-total-cost study. Record selection generation, forward evaluations,
gradient-token counts, training generation, and calibration CPU time separately.
The real GPU cost advantage remains unmeasured.

Report every drift, budget, seed, negative outcome, and paired standard error.
The broad synthetic goal is to improve reward and/or directional regret over
strong reuse endpoints without fresh candidate generation. A failure against
g00 is a real negative result, not a reason to hide that arm or tune test seeds.
Small smoke runs use seeds 10.. and are not mixed with the fixed study grid.

## Commands

```
python3 -m pytest -q tests/test_correction_selector.py
python3 src/correction_selector_study.py --output-dir outputs/cfcs-study-20260911
python3 src/correction_selector.py outputs/cfcs-study-20260911/example_input.npz \
  --k 16 --output outputs/cfcs-selection-20260911.json
```

The NPZ interface accepts `endpoint_groups` of shape (N,G,4) in the declared
endpoint order and optional unique `prompt_ids`. These must be directional
group scores, not the existing aggregate cosine JSON artifacts. A live model
adapter must supply token derivatives and independent groups. No real-model
launcher or canonical manuscript is changed by these commands.

## Small fresh-anchor variant

The first exploratory study showed CFCS often improved over full correction
but did not consistently improve over g00. A stale-only fit cannot identify
common bias in its own calibration target. The follow-up adds fresh information
explicitly: uniformly choose 10% of candidate prompts (minimum eight), obtain
32 current-policy responses on each, and project their group gradients on
the SAME independently frozen validation direction.

For each stale group fold, fit the same three bounded coefficients to these
fresh scores y on anchor prompts A:

```
min_{0 <= lambda <= 1, intercept}
    sum_{i in A} (b_i + lambda' c_i + intercept - y_i)^2
```

The intercept is eliminated by centering and cannot affect the final ranking.
Score the other stale half, then swap. The anchors calibrate which correction
components to retain across the pool; they do not become evaluation labels or
direct fresh replacements for their own candidate scores. This fit is a
bounded calibration of the specific 2x2 decomposition, not a new general
regression principle. It still needs a prior-art audit.

An `anchor_best` control gets the SAME fresh anchors and chooses among g00,
g10, g01, g11, and the additive endpoint under the same centered squared-error
criterion. This tests whether continuously mixing corrections adds value over
simply choosing an existing estimator. Plain g00 remains a mandatory control.

The follow-up uses the same drift/budget/continuation settings, fresh seeds
2000..2019, and includes ALL original arms plus both anchored variants. These
settings are fixed before that run. Source snapshots and hashes are recorded.
At N=160 this spends 16*32=512 fresh candidate responses versus 1280 or 5120
for the full fresh controls at K=8 or K=32. Those are 60%/90% reductions in
fresh CANDIDATE generation only. Validation, stale preprocessing, gradient
work, training, and cold-start behavior generation still count.

```
python3 src/correction_selector_study.py --output-dir outputs/cfcs-anchor-study-20260911 --seed-start 2000
python3 src/correction_selector.py outputs/cfcs-anchor-study-20260911/example_input.npz \
  --anchors outputs/cfcs-anchor-study-20260911/example_anchors.npz \
  --k 16 --output outputs/cfcs-anchor-selection-20260911.json
```

## Failure modes and actual completion boundary

No stale-only selector can recover outcomes missing from its observations.
All-zero advantages provide no information; optional exploration is not a
certificate. Small groups, unreliable validation gradients, product clipping,
pool heterogeneity, or inaccurate finite differences may defeat the fit.
Forward-only candidate scoring trades memory for multiple inference passes;
it is not free. The common direction must use the same parameter scope and
dropout-disabled deterministic evaluation. A real backend needs finite-
difference step sensitivity and measured GPU accounting before any efficiency
claim. GPS and published CROPI/GradAlign require actual reproductions for a
paper-level comparison; endpoint proxies do not count as those reproductions.
