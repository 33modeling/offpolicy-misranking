# Cached SR versus current gradient

Run on the machine holding the original experiment artifacts:

```bash
git pull --ff-only origin master
bash scripts/run_cached_sr_gradient_audit.sh
```

The default scans `$OM_WORK/runs` (`OM_WORK` defaults to
`/group-volume/minsoo3.kim/offpolicy-misranking`). Explicit scoring directories
or run roots can be passed as positional arguments. It follows source symlinks
once, deduplicates canonical paths, and skips policy/checkpoint directories.
It includes saved MATH/MBPP scoring points at all available stages, not only d0.
It does not evaluate a checkpoint or create a scoring point where none exists.

Reports go to a separate timestamped `exports/sr-gradient-*/` directory.
Send `data.json`; `comparison.csv` is the compact table. Inputs are read-only,
CUDA is disabled, existing output directories are rejected. No model generation,
backward pass, training, regression, or continuation outcome is used. Exit code
2 means at least one point was skipped (or no valid point), with reasons saved.

## Exactly what is calculated

For each saved policy state, reconstruct the existing top-k on-policy selector
from the first quarter of candidate LOO gradient groups and first half of
validation gradient groups. The cached SR selector uses the original behavior
rewards, ranking by `-|p - 0.5|`. Selection/tie seeds match `downstream_compare`.
Random is both a seeded top-k comparator and an exact uniform-selection
expectation (the pool mean for the additive statistic).

Freeze these selections. Use the final two quarters A/B of candidate and
validation gradients to compute the mean projected dot product on each set:

`D_m = mean_{i in S_m} [ (<Pg_i,A, Pv_A> + <Pg_i,B, Pv_B>) / 2 ]`.

`D_on - D_SR > 0` means higher estimated directional utility for on-policy,
not positive H. The dot product retains magnitudes that cosine discards.
Report A/B separately, mean cosine, overlap, all selected IDs, per-prompt
statistics, source metadata and SHA256 hashes. A/B are not a confidence interval.
Selection is not tuned against A/B or later learning outcomes. The diagnostic
is retrospective reaggregation of same-state measurements, not evidence that
the quantities were actually used prospectively. Input provenance and policy
lineage are reported, not independently certified by this tool.

## Why H is deliberately null

The stored vectors are projected LOO, sequence-summed gradients on scoring
layers, not GRPO/Adam parameter updates. Aggregate LOO vectors cannot generally
recover GRPO's response-length weighting and group-standardization, particularly
when scoring and training group sizes differ. Neither cosine nor these raw dot
products is a calibrated reward increment. Projection error is unquantified.
Do not insert them into the target-cost H formula as reward per update.

The proposed local formula additionally needs valid GRPO/Adam reward increments,
current reward, a preset target, and prospective costs. No learning-rate factor,
missing cost, target, or H label is invented here. `h_gpu_seconds` stays null and
`decision` stays `not_estimated`. No paper claims/tables are updated by this tool.

Reaggregation is cheap because the original scoring artifacts already exist;
original full-pool scoring was not free. A deployment using only one selected
batch does not automatically have gradients for the unselected SR prompts.
This diagnostic therefore does not establish a cheap single-selector H method.
