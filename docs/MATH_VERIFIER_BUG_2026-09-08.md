# MATH-500 verifier bug (found 2026-09-08, during the running matrix)

Recorded before any drift result of the main matrix was read, as
`EXPERIMENT_PLAN.md` §9 requires for anything that touches scoring.

## The bug

`src/data.py::_math_reward` decides a MATH-500 answer that is neither an exact
nor a numeric match by calling `math_verify.parse(prediction)` on **bare text**.
On bare text the numeric extractor reads `2x` as the number 2 and leaves roots
and symbolic expressions unextracted. Parsing the same string as mathematics,
`parse("$" + expression + "$", extraction_config=[LatexExtractionConfig()])`,
is what the verifier was meant to do. Concretely:

| prediction | gold | pinned verifier | corrected | |
|---|---|---|---|---|
| `2` | `2x` | 1 | 0 | a wrong answer credited |
| `\sqrt{2}` | `\sqrt{2}` | 0 | 1 | a right answer rejected |
| `x+1` | `1+x` | 0 | 1 | a right answer rejected |
| `0.5` | `\frac{1}{2}` | 1 | 1 | unchanged |

The bug is present in the pinned generation commit `0e4cd412` (the code every
point of the OLMo-3 h100-v2 matrix is generated and scored with) and in every
`master` commit up to and including `6f5f151`. The corrected verifier exists as
an uncommitted working-tree edit of `src/data.py` with
`tests/test_math_expression_reward.py` (10 tests, passing); it is not part of
any launched run. MBPP is unaffected: it is scored by executing the code.

## Who is affected

- **OLMo-3 main matrix, every MATH-500 point** (5 families x 4 points): rewards
  of behavior, fresh and validation rollouts, and therefore the GRPO advantages
  the d25/d100/d400 policies were trained with, and the per-prompt scores.
- **Qwen3.5-9B replication** if it is launched from a commit without the fix
  (`scripts/run_qwen35_9b.sh`, launched 2026-09-08 afternoon from `master`).

## Measured size (2026-09-08T05:01Z, `scripts/check_math_reward.sh`)

Every stored MATH-500 response of the eight finished points was re-decoded and
scored with both verifiers. The pinned verifier reproduced the stored `reward`
field on all 128,000 rows (0 mismatches), so the numbers are trustworthy.

| | count | share |
|---|---|---|
| rows checked | 128,000 | 8 finished points, 3,200 behavior + 12,800 fresh each |
| right answer scored 0, corrected to 1 | 2,066 | 1.61 % |
| wrong answer scored 1, corrected to 0 | 1,063 | 0.83 % |
| prompt groups with all-equal rewards (no gradient), pinned | 2,031 / 6,400 | |
| same, corrected | 1,798 / 6,400 | 233 groups regain a gradient (11.5 % of the flat ones) |

Per file the rate is stable across seeds and drift points (0->1 between 1.3 %
and 1.8 % on fresh rollouts, 1.6-1.8 % on behavior rollouts). Seven
`Timeout during comparison` lines came from math-verify's own comparison
timeout; those rows score 0 under both verifiers and are not counted as flips.

Full output: `transfer/offpolicy-misranking/math-reward-check-olmo3-1025-7b-base-rlzero-grpo-h100-v2-20260908T050111Z.txt`.

## Why it matters for the paper

The measurement compares how a stale (behavior) sample and a fresh sample rank
the same prompts. Both sides use the same verifier, so the bug does not favour
either side; it adds label noise to both. Two effects are real:

1. Prompts whose K rewards are all equal carry no ranking signal. The bug keeps
   11.5 % more of them flat than the corrected verifier would. This is a
   plausible contributor to the weak d0 reliability seen on math500/s0
   (precision 0.275 against a split-half floor of 0.20).
2. The d25/d100/d400 policies were trained with the noisy rewards. That is
   part of the treatment as run and is not undone by rescoring.

## Decision (taken 2026-09-08, before any drift result was inspected)

1. **The running matrix is not changed.** All 40 points finish under the pinned
   verifier so every point is scored by one rule.
2. **After the matrix completes, every point is rescored with the corrected
   verifier** from the stored responses: no regeneration, no retraining. The
   rescored artifacts are written beside the originals, never over them, so
   both scorings can be reported. The analysis is read from the corrected
   scoring; the pinned scoring is reported as a robustness check.
3. If the two scorings lead to the same regime decision, that agreement is
   reported. If they differ, the corrected scoring is primary and the
   difference is reported as a verifier sensitivity, per §9's
   "MATH-500 and MBPP disagree -> domain/verifier-dependent boundary".
4. The corrected verifier is committed to `master` for every matrix launched
   after this date. A matrix already generating under the pinned verifier
   keeps it until rescored.
5. No threshold, drift point, seed or subset changes. This record is the only
   registered change, and it is a scoring correction, not a design change.

## Status of the pieces

| piece | state |
|---|---|
| measurement tool `scripts/check_math_reward.sh` / `src/measure_math_reward.py` | committed `4a1732e`, `6f5f151` |
| corrected verifier (`src/data.py`) | committed `57e2e43`; every matrix launched from that commit on uses it |
| rescoring path `scripts/rescore_math500.sh` / `src/rescore_rollouts.py` | written 2026-09-08 evening: rewrites `reward` in the stored rollouts (pinned value kept as `reward_pinned`), reseals manifests (pinned hashes in `<prefix>.rescore.json`), moves gradients/scores/report/DONE into `pinned-scoring/<stamp>/`; a normal worker then recomputes them. Verifies the whole family read-only before touching a file. Dry run by default. |
