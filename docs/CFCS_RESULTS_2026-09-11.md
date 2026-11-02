# CFCS: implementation and falsification results

## Decision

The new selector is implemented, executable, and tested. It is NOT yet a
validated cheaper-and-better method, and these results do not justify adopting
it as the main algorithmic contribution of the paper. The fresh-anchor variant
sometimes improves downstream reward, but loses to g00 under large policy
drift. Its advantage over simply choosing an existing endpoint is inconclusive.
Do not advertise the candidate-generation reduction as a GPU speedup or a
Pareto improvement: quality and other computation costs also matter.

## What was built

CFCS learns how much prefix, suffix, and interaction correction to apply:

```
s_i = g00_i + lp (g10_i - g00_i) + ls (g01_i - g00_i)
            + lh (g11_i - g10_i - g01_i + g00_i),  0 <= lp,ls,lh <= 1.
```

Each g is a scalar projection onto the same independent validation direction,
not a candidate cosine. Stale-only CFCS estimates signal/noise covariance;
anchor CFCS fits the coefficients using fresh responses on a small random
pilot. Disjoint response-group folds separate fitting from scoring.

The specific construction is experimental. Alignment gain, finite differences,
shrinkage, cross-fitting, and least squares are existing ideas. No claim of
globally new mathematics or a completed prior-art audit is made. Details and
primary-source references are in [the proposal](CFCS_PROPOSAL.md).

## Evaluation protocol

- Existing synthetic autoregressive Markov environment with verifiable binary
  terminal rewards; NOT MATH, MBPP, or a real language model.
- 160 candidate prompts; select 16 once, then train for 64 sampled updates.
- 32 validation prompts estimate the direction. Another disjoint 32 measure
  exact expected test reward before and after sampled training.
- Five drift scales and two stale-response budgets; 20 seeds per condition.
  Drift is parameter perturbation scale, not measured KL.
- Exact candidate gradients are evaluator-only. Replacing them with zeros in
  a test does not change any arm's final training reward.
- Fresh controls estimate scores from current-policy responses; no oracle
  selection control is passed off as a practical baseline.
- Equal continuation updates, NOT equal total GPU or token cost.
- Each grid was fixed before its run. The anchor variant was designed after
  inspecting the stale-only run; it is an exploratory follow-up with new seeds,
  not a preregistered confirmatory experiment.

## Full follow-up grid

Seeds 2000..2019. Entries are mean held-out reward increase in percentage
points after 64 updates. `Full` is clipped g11. `Anchor-best` uses exactly the
same fresh pilot to choose among g00, g10, g01, g11, and additive. `Fresh` uses
current-policy directional scoring for every candidate. These are endpoint
controls, not reproductions of GPS, GradAlign, or CROPI.

| Drift | Stale K | g00 | Full | CFCS | Anchor CFCS | Anchor-best | Fresh |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | 24.45 | 24.45 | 24.45 | 24.45 | 24.45 | 25.54 |
| 0 | 32 | 24.91 | 24.91 | 24.91 | 24.91 | 24.91 | 25.81 |
| 0.5 | 8 | 24.05 | 23.53 | 24.18 | 23.93 | 23.73 | 25.54 |
| 0.5 | 32 | 25.50 | 25.90 | 25.36 | 25.79 | 25.56 | 25.81 |
| 1 | 8 | 23.07 | 22.78 | 23.45 | 23.53 | 23.33 | 25.54 |
| 1 | 32 | 24.38 | 24.44 | 25.14 | 25.12 | 24.76 | 25.81 |
| 2 | 8 | 22.68 | 20.65 | 22.59 | 21.85 | 21.75 | 25.54 |
| 2 | 32 | 25.30 | 23.08 | 24.94 | 24.68 | 25.15 | 25.81 |
| 4 | 8 | 22.31 | 19.62 | 19.65 | 20.39 | 19.92 | 25.54 |
| 4 | 32 | 24.13 | 21.06 | 23.16 | 23.54 | 22.68 | 25.81 |

Fresh controls do not depend on behavior drift; their repeated values across
drifts are intentional, not additional independent observations.

Paired differences across the same 20 seeds, in percentage points. The number
after +/- is ONE standard error, not a confidence interval. No correction for
multiple comparisons is applied; these numbers do not establish superiority.

| Drift | K | Anchor CFCS minus g00 | Anchor CFCS minus anchor-best |
| ---: | ---: | ---: | ---: |
| 0 | 8 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| 0 | 32 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| 0.5 | 8 | -0.125 +/- 0.354 | 0.201 +/- 0.224 |
| 0.5 | 32 | 0.288 +/- 0.216 | 0.230 +/- 0.199 |
| 1 | 8 | 0.460 +/- 0.484 | 0.199 +/- 0.398 |
| 1 | 32 | 0.737 +/- 0.432 | 0.356 +/- 0.373 |
| 2 | 8 | -0.835 +/- 0.594 | 0.100 +/- 0.331 |
| 2 | 32 | -0.623 +/- 0.481 | -0.474 +/- 0.454 |
| 4 | 8 | -1.923 +/- 0.573 | 0.465 +/- 0.540 |
| 4 | 32 | -0.592 +/- 0.479 | 0.860 +/- 0.568 |

The clearest practically relevant result is a failure: at drift 4, K=8,
anchor CFCS loses 1.92 points to cheaper g00. At drift 1, K=32, its apparent
0.74-point gain is smaller than twice its paired standard error. There is no
basis for declaring the new mixture generally superior.

## Earlier stale-only experiment

Seeds 1000..1019; same grid and continuation protocol, before anchors were
introduced. CFCS minus g00, percentage points, for EVERY condition:

| Drift | K=8 | K=32 |
| ---: | ---: | ---: |
| 0 | 0.000 | 0.000 |
| 0.5 | -0.145 | -0.026 |
| 1 | 0.207 | -0.492 |
| 2 | -0.156 | -0.883 |
| 4 | -1.387 | -1.005 |

This first run also failed the strong cheap-baseline test. The follow-up must
not erase that negative result. The original run retains its raw records; it
predates automatic source snapshots, so use its recorded configuration rather
than assuming the current script is byte-identical to the original run.

## Cost accounting

At N=160, anchor CFCS uses 16 pilot prompts with 32 fresh responses each:
512 fresh candidate responses. Full fresh scoring uses 1280 responses at K=8
or 5120 at K=32: 60% or 90% fewer fresh CANDIDATE responses for anchor CFCS.
Plain CFCS and g00 use no fresh candidate responses.

For each non-pass-rate stale scorer, all four endpoints share one directional
derivative computation: an exact center and two perturbed forward evaluations
over stored tokens. Three passes are charged, not one. The anchor additionally
requires fresh pilot generation and gradient computation. The common validation
direction uses 1024 fresh responses; it is not free. All training uses 2048
fresh responses per arm. With cached behavior data and all this generation
included, anchor versus fresh response-count savings are approximately 17.6%
(K=8) and 56.2% (K=32), before accounting for forward/backward cost. If the
behavior cache must be generated now, that generation also counts and can
remove the claimed saving entirely.

Token counts in `results.csv` separately include behavior generation, fresh
candidate generation, validation generation, candidate forward evaluations,
gradient-token work, and training. Synthetic calibration CPU timing does not
measure model overhead. No wall-clock GPU saving, peak-memory saving, or
equal-total-cost reward advantage has been measured. The cheap backend is
also available to endpoint baselines; it is not an exclusive CFCS advantage.

## Artifacts and reproduction

- Selector: `src/correction_selector.py`.
- Study: `src/correction_selector_study.py`.
- Tests: `tests/test_correction_selector.py`.
- Original run: `outputs/cfcs-study-20260911/` (200 cells, 11 arms).
- Follow-up: `outputs/cfcs-anchor-study-20260911/` (200 cells, 13 arms).
- Follow-up `source/` and `config.json` preserve source snapshots and SHA-256
  hashes. `results.csv`, `summary.json`, and `folds.json` retain all arms,
  seeds, metrics, and learned coefficients. The source snapshot predates only
  the CLI anchor-input provenance field added after the run; the scoring and
  experiment logic have not been changed after inspecting these results.

```sh
python3 -m pytest -q tests/test_correction_selector.py
python3 src/correction_selector.py \
  outputs/cfcs-anchor-study-20260911/example_input.npz \
  --anchors outputs/cfcs-anchor-study-20260911/example_anchors.npz \
  --k 16 --output outputs/cfcs-anchor-selection-20260911.json
```

The selector refuses to overwrite an existing output. Running the study with
a NEW output directory and `--seed-start 2000` reproduces the follow-up. The
exported example is the first, zero-drift cell and illustrates the interface,
not an efficacy demonstration. Nonzero-drift coefficients are in `folds.json`.

Verification completed: 16 tests passed; Ruff checks passed; both standalone
CLI modes emitted 16 unique prompt IDs. All 2600 follow-up method/seed/cell
records are unique, and all archived source hashes match. The largest synthetic
central-difference error against the analytic token derivative was 2.58e-8;
this float64 toy check does not establish stability in low-precision LLMs.

## What the result does and does not imply

Shrinking correction noise is implementable at low pilot cost, but matching
centered score MSE does not guarantee a better selected set or greater finite-
step reward. A global three-coefficient fit can also miss heterogeneous prompt
bias. These are plausible explanations, not causal findings from this study.

Cross-fitting cannot restore missing support or recover a good direction from
uninformative validation samples. A fresh pilot helps observe bias but does
not by itself solve it. A publishable claim still requires real-model adapters,
finite-difference precision checks, comparable reproduced strong baselines,
and measured equal-total-budget downstream gains. None is claimed complete.
