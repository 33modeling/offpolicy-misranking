# v4 progress bundle received 2026-08-24

## Provenance

- Source archive: `/home/kms/Downloads/0824.zip`
- Archive SHA-256: `91503bcc430975a7a0a2c86194b20e6ad1b8d80630ddd819a52efaeb13a75b67`
- Archive integrity: `unzip -t` passed for all ten entries
- Status: **partial progress bundle, not a frozen result set**
- Matrix completion: 15/20 runs overall; 10/10 7B and 5/10 27B

The files in this directory are an unmodified copy of the extracted bundle.
They are retained for provenance and must not be edited into a complete matrix.

## Artifact hashes

| file | SHA-256 |
|---|---|
| `FRONTIER-v2.md` | `9482a2ae4182b59c3861e0bf3bb3c6bb88410e72c95d081036af3a7359d7c9db` |
| `FRONTIER-v3.md` | `2e9253d601fd74b45148a6666f022c9a04e10af2e3025a16f627d793f04aeead` |
| `HARVEST_FAILURES.md` | `5423432afff4dd55b5c7d9b1ccda49efd1cf350e5c406455794b2071c53460a0` |
| `KCURVE.md` | `ff546f83f3519fc3c4d5b4847966cf83cee40c79d14cfa36b5ba386044557523` |
| `KCURVE_ALL.md` | `39a6e21813209873ccbb5ad6487e5e950989bbe4f40b725e831e17002de85587` |
| `READOUT.md` | `e7c3b2c6f6e7d6609515f874bba9ab00f3d319d58bfc2d66b98d82011dd4efe0` |
| `REVERSAL.md` | `193edeb41365f8266b90b88488c82dc67cb4fc322626ffed7406ff2d20a9c8e0` |
| `STATS.md` | `c865c7153f58456510756187bbb7edf7ceeecfbb251c02adc238879d6faf5902` |
| `TABLES-v2.md` | `f8a0394872b793db9f813d046345d88509fb940211adb37649561cdb60026e2e` |
| `TABLES-v3.md` | `e6da14bed35681d267d3586afbfc86f84702663710e0b50b6bde542022fe84ef` |

## Completion audit

The bundle's own `HARVEST_FAILURES.md` reports
`v4-matrix:incomplete-20-artifacts`. The reportable 27B runs are:

- `v4-27b-s0-math500`
- `v4-27b-s1`
- `v4-27b-s1-math500`
- `v4-27b-s2-math500`
- `v4-27b-s3-math500`

The missing 27B runs are:

- `v4-27b-s0` (GSM8K)
- `v4-27b-s2` (GSM8K)
- `v4-27b-s3` (GSM8K)
- `v4-27b-s4` (GSM8K)
- `v4-27b-s4-math500`

`TABLES-v2.md`, `TABLES-v3.md`, `FRONTIER-v2.md`, and `FRONTIER-v3.md`
are historical outputs included by the harvester. They are not corrected v4
confirmatory tables and must not be merged with the v4 rows.

## Provisional reading

- The automated joint one-sided threshold fires for 2/4 completed 27B
  MATH-500 runs and 0/1 completed 27B GSM8K run. This is not a five-seed vote.
- Within each completed 27B run, selector bootstrap intervals are broad and
  overlap. No one-sided-vs-`g11` McNemar comparison reaches `p < 0.05`.
- The significant `g00`-vs-`g11` McNemar results in two MATH-500 runs compare
  no correction with full correction; they do not establish one-sided harm.
- No 27B hybrid result is available, so recovery claim C1' is untested.
- Across all 15 completed runs, pooled reversal rates are near the oracle
  self-disagreement anchor. The pool repeats prompt sets across seeds and is
  not an independent sample, so it does not support a pooled p-value.
- The complete 7B block triggers the automated one-sided threshold in 0/5
  GSM8K and 0/5 MATH-500 runs.

Accordingly, this bundle is progress evidence only. Freeze manuscript numbers
after all 20 runs pass contract and lineage validation and the final reports
are regenerated from one complete matrix.
