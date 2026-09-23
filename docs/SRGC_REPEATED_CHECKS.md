# Repeated SR-GC Checks

While On is active, evaluate the existing SR-GC contrast at a fixed interval
(default: 25 additional updates). D >= 0 retains On for the next interval.
The first D < 0 selects SR permanently; there is no SR-to-On return and no
further gradient measurement after that decision.

## Existing Data

Each `sSEED-tSTART` is processed independently, using its saved fixed-On
checkpoint path. The initial frozen SR-GC measurement is reused. Subsequent
checks compare the same On and initial-cache SR subsets using current-policy
A/B gradient projections and the unchanged zero threshold. No reward curves,
future endpoints, target costs, fitted predictor, or other starting states
enter D. A missing scheduled checkpoint or gradient blocks later checks;
it never counts as an On decision.

The additional script does not rerun training or alter any Pair checkpoint,
cost ledger, receipt, original decision, or queue task. New projection files
are isolated in `PAIR_ROOT/sr-gc-repeat/every-25/sSEED-tSTART/step-STEP/`.
An absorbing-rule replay is not a measured post-switch learning curve. The
existing independently trained SR arm must not be spliced onto the On arm
and relabeled as an executed adaptive trajectory.

## Commands

Normal result export now recomputes the saved contrasts and includes the
`srgc_repeated` JSON section and a checkpoint-by-checkpoint text table:

```bash
bash scripts/run_selector_pair_results.sh
```

It is read-only and never launches GPU work. To generate any missing A/B
projections from the saved weights, on a separately available four-GPU
allocation (no policy training):

```bash
bash scripts/run_selector_pair_srgc_repeat.sh measure --interval 25
bash scripts/run_selector_pair_results.sh --srgc-interval 25
```

The same score worker and A/B definition are retained. The union of the two
already selected subsets is scored, without reranking the full candidate
pool. Existing sealed shards are reused. New generation/backward costs are
metered separately as research measurements; only reaggregation is free of
new GPU work. The per-checkpoint measurement cap is 14,400 allocated GPU
seconds by default, configurable with `--max-gpu-seconds-per-checkpoint`.

Normal export: `~/selector-pair-results.txt`.
Standalone diagnostic: `~/selector-pair-srgc-repeat-results.txt`.
Missing inputs and invalid bindings are explicitly included in the export.
The ongoing one-decision Adaptive queue is intentionally not redefined or
restarted by this additional analysis script.

## Figure 2 Contract

Use `decisions[].updates` for repeated checks and `first_sr_updates` for the
first negative-D switch marker on the corresponding panel. Label the black
marker as the SR-GC trigger, not the measured reward crossing. Display all
recorded On decisions leading up to it; after SR, retain the SR state without
inventing additional D values. Keep observed fixed-arm crossing brackets as
separate evidence. Never turn a pending check into an On label or fabricate
adaptive rewards from the selected fixed-arm endpoints.
