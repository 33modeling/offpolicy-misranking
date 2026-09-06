# Follow-up: does the misranking regime map generalize? (design, 2026-09-06)

Status: designed and wired (configs + launcher profiles), **not launched**. Main
experiments first: OLMo-3 7B RL-Zero (running) and Qwen3.5-9B replication.

## Question

The main result maps, for one model family and two verifiable domains, where
stale one-sided importance corrections misrank prompts relative to the fresh
oracle (regime map over drift 0/25/100/400). The follow-up asks whether that map
is a property of the estimators (robust) or of the model/data (fragile), along
four axes that the existing code can already run without new machinery.

## Axes and registered matrices

| Axis | Matrix (config) | Launch | What changes | Cost estimate* |
|---|---|---|---|---|
| A. Scale, same family/data | `qwen35_2b_grpo.json`, `qwen35_4b_grpo.json` | `bash scripts/run_followup.sh qwen35_2b` / `qwen35_4b` | Qwen3.5 2B/4B post-trained; MATH-500/MBPP; seeds 0-2; GRPO 0/25/100/400 | 2B ~0.4 d/family, 4B ~0.7 d/family (batch 32/16) → 6 families each |
| B. Domain, same model | `olmo3_domains_grpo.json` | `bash scripts/run_followup.sh olmo3_domains` | OLMo-3 7B base on KK (logic), ARC-Challenge (science MC), MMLU-Pro non-math (knowledge); 512 tokens; `verifiable_completion` `####` template | ~0.8 d/family → 9 families |
| C. Objective | `RLVR_METHOD=dr_grpo` / `rloo` with any matrix | env on the launcher | Dr.GRPO (no std normalization) and RLOO (LOO baseline) — does misranking depend on GRPO's group standardization? | same as host matrix |
| D. Drift resolution | copy of a matrix with `drifts: [0,5,10,25]` | new config (hash) | where does misranking set in? | ~0.5× (short chains) |

*Per 4×H100 node, using the measured OLMo rate (~2048-token fresh rollouts dominate). Fill in from `grpo_stats.jsonl`/`tok/s=` once the 9B run reports; 2B/4B/512-token domains are far cheaper than 9B/2048.

Already registered before this note (other models, same domains): `generalization_{logic,science,knowledge}.json` with OLMoE-1B-7B base and Qwen2.5-14B base (`legacy` profile). Axis B adds the paper's main model to those domains so the domain comparison is within-model.

## Protocol (unchanged from the main run)

Behavior K=8 / fresh K=32 / val K=8; 4-rollout micro-groups; R/A/B split
(rank on R, held-out reference = mean of A and B scalar scores, A-vs-B top-k
agreement = reliability floor); CountSketch 4096; final 4 decoder blocks; top
10%; 10,000 FIRST bootstrap replicates for regime labels. One family = one
behavior pool + the continuous checkpoint chain; config hash = contract.

## Pre-registered readouts

1. **Regime label agreement**: fraction of (dataset, drift) cells whose label
   (one-sided-deficit / no-deficit / no-signal) matches the OLMo-3 main map.
2. **Deficit magnitude vs scale** (axis A): Δ(precision_onesided − floor) as a
   function of parameters at fixed drift; monotone trend or not.
3. **Domain dependence** (axis B): same statistic across KK/ARC/MMLU vs
   MATH/MBPP; short-answer domains have tiny trajectory KL — prediction:
   misranking shrinks, "no-signal" cells grow.
4. **Objective dependence** (axis C): GRPO vs Dr.GRPO vs RLOO on identical
   behavior pools (reuse d0 pools; only positive-drift points rerun).
5. **Onset** (axis D): smallest drift with a confirmed deficit.

Negative results are reported as such; no cell is dropped.

## Order of execution given the deadline

1. Axis A with 2B first (cheapest, and the 2B/4B/9B snapshots are already on
   the volume) — gives a within-family scale curve alongside the 9B replication.
2. Axis B (OLMo-3 on KK/ARC/MMLU) — reuses the running model and its verifier
   qualifications; 512-token generations.
3. Axis C on one dataset (MATH-500) only if time remains; axis D is appendix.

## Operational notes

- Same bare-command ergonomics: `bash scripts/run_followup.sh <profile>` pulls,
  cleans up, adopts uploaded snapshots (any folder name, trust-local check),
  and prints `DIAGNOSIS/ACTION` on failure. Needs an idle 4×H100 node (the OLMo
  primary owns its node's GPUs).
- 2B/4B pinned revisions are the Hub `main` commits on 2026-09-06
  (`15852e8c…`, `851bf6e8…`); uploaded folders are validated for loadability.
- Results roots: `$OM_WORK/results/{qwen35-2b,qwen35-4b}-posttrained-math-code-grpo-v1/`,
  `$OM_WORK/results/olmo3-7b-base-domains-grpo-v1/`.
