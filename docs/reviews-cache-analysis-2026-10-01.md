# Cache analysis and worker regression checks

Scope: `scripts/srgc_cache_analysis.py` and the basic single/multi-plan worker
wrapper. Frozen `srgc_rebuttal/*.py`, input bundles, checkpoints and results
were not changed. No remote GPU experiment was launched or recovered.

Fixed and covered by tests:

- Reject fractional/non-numeric cache rewards instead of truncating them.
- Reject duplicate history steps, wrong seed/arm metadata and unknown prompt IDs.
- Separate predicted schedules, observed continuation updates and excluded prefix.
- Report SR comparison coverage; no overlapping history is not a successful match.
- Treat missing counts and ambiguous retry receipts as unknown, never as zero.
- Weight signal summaries by observed updates; expose both metric denominators.
- Infer the input seed from provenance and reject conflicting seed arguments.
- Return nonzero for skipped cache inputs or observed SR mismatches.
- Park before GPU admission when GPU ownership is busy, without claiming a task.
- Propagate interrupts and distinguish ordinary child exceptions from exit 130.

Verification: `/tmp/v6-srgc-test-env/bin/python -m unittest discover -s
srgc_rebuttal/tests -q`: 367 tests, no failures, 9 environment-dependent skips.
Cache schedule replay was checked against ToyBackend Engine training histories.
Worker tests include concurrent queue draining, exclusive task leases and idle
ownership handling. These are CPU/fixture checks, not a real cluster validation.

Training `sample_reward` is not independent evaluation. Base-arm progress JSON
is published every 25 updates; the latest checkpoint can be newer. This change
does not introduce intermediate evaluation or reconstruct old checkpoints.
