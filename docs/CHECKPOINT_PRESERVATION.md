# Checkpoint preservation

New trainer processes using this revision retain every published
`policy/checkpoint-NNNNNN` directory, including weights, optimizer state,
statistics and checkpoint metadata. Saving a later checkpoint and publishing
the final policy no longer remove earlier checkpoints. A conflicting existing
checkpoint path stops the write for review instead of deleting saved work.
Unpublished temporary save directories can still be cleaned up on retry.

The shared GRPO writer covers GRPO, Dr.GRPO, RLOO and the budgeted gate/MOPPS
drivers. Checkpoint cadence, learner updates, optimizer settings and evaluation
settings are unchanged. Storage use grows with the number of saved checkpoints;
there is no automatic disk quota or pruning fallback.

The Pair/convergence entry publishes its lightweight `curve-checkpoints/step-N`
copy immediately after each checkpoint save, including the Pair cost receipt.
It retains the full original checkpoint as well. Curve evaluation therefore
does not depend on the old deletion hook being triggered.

## Deployment boundary

This source change does not modify or restart a process already running on the
server, recover deleted checkpoints, or reevaluate saved models. Already loaded
old trainers retain their old behavior until restarted under the new code.

Existing scientific runs may pin trainer and curve-entry source hashes.
Those checks remain enforced: this revision does not rewrite their manifests,
silently accept changed hashes, or claim that an existing frozen Pair run has
been migrated. Do not pull into an active frozen-run checkout or restart that
run under changed code without a separate reviewed deployment/migration.
Use this revision in a separate checkout for newly prepared runs meanwhile.

## Verification

```bash
PYTHONPATH=src python3 -m pytest -q tests/test_checkpoint_preservation_cpu.py
```

The CPU tests execute the actual checkpoint I/O functions extracted from the
trainer, substituting only model/optimizer serialization fixtures. They test
all-checkpoint retention, valid resume selection, failed writes, conflicting
paths, final-publication guards, and Pair archive/cost-receipt preservation.
They do not launch distributed GPU training or certify migration of frozen runs.
