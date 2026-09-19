# Does the gate switch to random at the right learning state?

Date: 2026-09-14
Status: implemented in an isolated entry point; GPU validation/results pending.
Manuscript target: v4. Preserve running net-gain, E5 and Qwen experiments.

## Claim under test

Given a model that has been learning on selected prompts, the gate decides
whether to continue score-based selection or switch to uniform random prompts
for the remaining training allocation. The claim concerns that decision, not
merely the fact that scoring costs compute.

The experiment must identify BOTH types of error: switching when continuing
selection would help, and retaining selection when switching would help.
Decision accuracy alone is inadequate: measure the final reward lost by each
error and the actual gated training outcome.

## Correct common history

Create a selected-training prefix for each seed. The primary selector is the
paper's on-policy gradient selector (gradients computed under the current
policy; frozen key `fresh_r`), with its exact score/grouping contract; do
not silently substitute low_order, pair_u2 or a different gradient estimator.
Use the same candidate pool, subset fraction, learner and verifier as the
existing continuation study. The prefix uses one initial selected subset.

Save adapters, optimizer state, RNG/sampling configuration and training logs
at 25, 50 and 100 prefix updates. These three decision opportunities are
prespecified. An existing selected-prefix checkpoint can be reused only after
its actual selection history and checkpoint contract are verified.

Generic GRPO matrix checkpoints alone do not certify a selected-training
prefix. The ongoing net-gain fixed-budget results remain useful evidence but
must not be renamed as this switch experiment.

## Paired branches at each checkpoint

At each checkpoint, freeze the gate's feature record, model hash and decision
before launching any continuation or inspecting its evaluation rewards.

1. CONTINUE: compute current selection scores, select a subset and continue
   training on it. Charge the new score calculation to the remaining budget.
2. SWITCH: bypass new scores, draw the same number of prompts uniformly,
   and train with the same learner from exactly the same parent state.
3. GATE: execute the previously saved continue/switch decision, with its
   actual diagnostic and scoring charges. Do not obtain its result by simply
   copying the winning control reward.

Two additional audit controls, CONTINUE_D and SWITCH_D, pay the same one-time
diagnostic charge as GATE before taking their fixed actions. The actual
diagnostic scan is shared once at each state, but its charge is assigned once
to each applicable counterfactual budget. These controls determine whether
the gate chose correctly after observing the diagnostic. The diagnostic-free
CONTINUE and SWITCH controls determine whether the complete gate is useful.

In the primary cost-inclusive protocol, all branches at a checkpoint have the
same remaining total GPU-second cap.
Freeze that cap before comparing outcomes. Final test questions and verifier
are identical and independent of ranking/diagnostic inputs. Evaluation has
a separate common reporting budget. Record actual completed updates and all
diagnostic/scoring/training/retry charges. Preserve unfavorable and failed
conditions rather than dropping them from the report.

The main MBPP protocol is **On-policy · 선택비용 별도** (`quality`:
`fresh_r` / `matched` / `convergence`, or selector/accounting/gate). It applies
the manuscript's **Separating learning quality from selection cost** design:
selection is metered separately on the reporting ledger, outside a common
diagnostic/training allocation, so scoring alone does not shorten selected-data
training. Equal allocation does not promise equal completed updates or equal
total GPU cost. Its convergence label still charges separately recorded selection
in random-training update units; three intermediate checkpoints use four responses
per question. This is a different accounting protocol from the MATH total-budget
comparison, not evidence obtained by substituting a MATH budget number.

The existing quality root, trained checkpoints and validated results are reused.
The main plan is **48 continuations: 18 development plus 30 held-out** from the
certified MBPP prefix states. Report Full selection / Full random / Gate policy
final held-out reward, completed updates and selection/diagnosis/training/evaluation
GPU costs separately, retaining the additional diagnostic-paid controls.
Their sum measures actual total compute; a learning-quality gain alone is not
a total-cost benefit or a test of a direct on-policy-to-difficulty switch.

Older `fresh`, `difficulty` and `long` roots remain observable and explicitly
runnable but are not scheduled by the default `all` request. Nothing deletes
or renames their files or kills their in-flight work. Existing keys and frozen
contracts are preserved; see [the MBPP run guide](MBPP_SELECTION_RUN.md).
This scope update does not establish new GPU results.
Budget exhaustion without a valid evaluated result is reported as incomplete,
never converted to reward zero or `DONE`. Retain its actual costs and explain
the missing result instead of treating it as a measured negative reward.

This tests whether to renew selection now or switch to random. CONTINUE
explicitly renews scoring; merely training longer on a cached fixed subset
would test a different question and omit the avoided scoring stage.

## One gate evaluation per continuation

The 25/50/100 checkpoints define separate paired experiments. Each GATE branch
is measured once and holds its action for the remainder of its continuation.
They are not repeated gate calls along one deployed trajectory. The diagnostic
uses only current/past information and a declared cap, not new trial training.

This design supports a state-specific switching decision, not a unique global
optimal switch time. Do not infer a first-crossing policy from the control
checkpoints: later states on a switched trajectory would differ. A claim about
an optimal time along a single closed-loop trajectory requires a separate
prespecified sequential experiment and its repeated diagnostic accounting.

## Frozen candidate gate

Fit a four-input ridge action-value predictor on development trajectories,
with alpha=1 and development-only standardization. Inputs are the recent
reward mean, recent active-group fraction, cached per-prompt success-rate
spread, and log(1+prefix update count). The recent window is the last 20
completed prefix updates. Each trajectory has unit total weight across its
checkpoint observations. No test reward, future log row, trial continuation,
or current candidate gradient is an input to the gate.

The target is the remaining-training advantage Delta defined below. At the
decision state, CONTINUE if the frozen prediction exceeds zero; otherwise
SWITCH. The diagnostic performs one bounded scan of existing cache/logs.
Its time cap is the smaller of 30 wall seconds and 1% of remaining GPU budget
divided by allocated GPU count. A timeout/invalid measurement produces a
recorded fallback, not a favorable statistical decision.

This is a prespecified candidate, not a proven new estimator. The held-out
comparison must establish whether state inputs beat the checkpoint-only rule.
Do not adjust alpha, threshold, features or fallback behavior to manufacture
successful transitions on held-out data.

## Split and staged allocation

- Development: seeds 0/1/2, three checkpoint states per seed. Run CONTINUE_D and
  SWITCH_D to form matched-state action-value labels: 18 continuations.
- Freeze the gate, feature definitions, inference budget, practical reward
  margin and tie/failure rule before held-out continuation results.
- Initial independent check: seeds 3/4 at the same three checkpoints, all
  five branches: 30 continuations. This is two independent trajectories,
  not 30 independent replicates or a definitive population-level result.
- Stronger final replication: expand the prespecified held-out set to five
  independent seeds (3 through 7), if resources allow, without choosing which
  seeds to retain based on their rewards. That is 75 held-out continuations
  total, including the initial 30, not 75 additional ones.

Prefix generation is additional shared research work unless an accepted
checkpoint already exists. The five-seed final design requires eight prefixes
including development. Do not present these runs as free gate deployment.
Checkpoint observations from one prefix remain clustered by seed throughout
analysis. Fit/calibration compute is reported separately and amortization over
future uses must be explicit; the gate does not perform these branch trials
at deployment time.

Existing observations may guide which scientific question to register, but
held-out switching outcomes must not determine feature selection, margin,
checkpoint placement, or the chosen favorable subset of seeds.

## Primary measurements

For a checkpoint state h and declared remaining budget B, define

    Delta(h) = J_CONTINUE_D(h,B) - J_SWITCH_D(h,B).

Both pay diagnosis; CONTINUE_D additionally pays scoring. The matched-state
contrast labels the observed advantage of retaining selection. Record:

- Wrong switch: gate chooses SWITCH where the observed Delta is positive.
  Lost continuation reward is max(Delta,0).
- Wrong retention: gate chooses CONTINUE where observed Delta is negative.
  Lost continuation reward is max(-Delta,0).
- Reward-weighted decision regret: max(J_CONTINUE_D,J_SWITCH_D) minus the reward
  of the control corresponding to the gate's chosen action.
- Executed system gain: actual J_GATE-J_SWITCH and J_GATE-J_CONTINUE, including
  the diagnostic charge. Good action agreement alone does not establish this.
- Runtime failures, unsupported-state fallback and ties are reported separately
  from a model's intended statistical decision.

The paired-control label is an observed contrast, not a noise-free oracle.
Report per-seed paired question-level contrasts and uncertainty using the
existing evaluation protocol. A prespecified practical-equivalence margin can
identify near ties; a point estimate's sign is not a significance declaration.
Do not count checkpoints or individual response samples as independent seeds.

The rule is learned on matched post-diagnostic actions. The diagnostic-free
SWITCH control assesses the entire gate. Never use a free fallback as the
action label while executing a paid one. Freeze the same remaining total cap
and the measured diagnostic charge before either paid control proceeds.

## Controls for the gate, not just the selector

Compare its held-out decisions with ALWAYS CONTINUE, ALWAYS SWITCH and a
checkpoint-only rule fit on development data. These simpler policies test
whether model-state diagnostics add value beyond a fixed checkpoint rule.
An old reliability-only gate is a useful ablation only if its score definition
and diagnostic cost can be reproduced; do not infer it from unrelated caches.

If all observed states favor random, report that result. It can support
avoiding selection in those states, but cannot establish recognition of a
useful selector or a meaningful transition boundary. Do not manufacture a
favorable crossing by adding only conditions chosen after viewing test results.

## Main figures

Figure 1: checkpoint (25/50/100) versus Delta in reward percentage points,
with a horizontal zero line. Show seed-level paired intervals. Overlay the
gate's frozen CONTINUE/SWITCH decision using distinct markers. A reader should
see where the gate rejects beneficial selection or retains harmful selection.

Figure 2: actual GATE, CONTINUE and SWITCH final held-out rewards at each
checkpoint, with a lower panel for gate-minus-control contrasts. This is the
training-effectiveness result; it cannot be replaced with score correlation.

Figure 3: reward lost by wrong switching and wrong retention, plus the
checkpoint-only baseline. Show diagnostic GPU time and total charged time in
a supporting panel so that overhead is visible without becoming the main claim.

Use tall, aligned panels and visible per-seed marks. Never draw independent
branched continuations as a single executed reward trajectory.

## Implementation boundary

No new GPU job has been launched locally. A draft based only on the old generic
checkpoint runner was withdrawn before commit: it did not establish a
selected-training prefix. The replacement is `scripts/run_selection_switch.sh`;
see [the implementation and run guide](SELECTION_SWITCH_RUN.md). Existing E5,
Qwen and net-gain entry points/results are unchanged. Before full GPU launch:

- Audit reusable selected-prefix checkpoints and exact on-policy-selector code.
- Freeze the diagnostic-paid control design and remaining-budget convention.
- Bind the selector, prior selection history, checkpoint, optimizer, model,
  decision timestamp/hash, evaluation identities and cost ledgers.
- Test decision-before-outcome leakage, branch matching, budget accounting,
  resume, ownership and failed-score fallback on CPU.
- Run a bounded representative four-GPU smoke test, then estimate the remaining
  wall time including scoring, training, evaluation and failure overhead.

## Prior-work boundary

The registered low-overhead comparison is **MoPPS** (Qu et al.,
*Can Prompt Difficulty be Online Predicted for Accelerating RL Finetuning of
Reasoning Models?*, KDD 2026;
[paper](https://arxiv.org/abs/2507.04632v5)). Its online reward-history selection
is tested in the [separate MoPPS extension](MOPPS_COMPARISON.md): 12 held-out
continuations, original checkpoints/budgets, and a matched online-random
control. The primary comparison is **executed GATE versus reward-based MoPPS**,
including diagnostic cost, reported as GATE-minus-MoPPS final reward. Random
and on-policy selection (`fresh_r`) are secondary controls. The live on-policy
gate study is unchanged. This is not the cached
`passrate_beta` baseline and not a claim to reproduce MoPPS's full training
system. GPU results remain pending.

Scoring/training cost tradeoffs are already studied in
[Yin and Rush, ICLR 2025](https://arxiv.org/abs/2410.16208), and budget-aware
selection optimization in [Wan et al.](https://arxiv.org/abs/2510.16806).
Cost-sensitive deferral is also established; see
[Mozannar and Sontag, ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html).
These primary-source abstracts were checked, not a complete novelty audit.

The proposed contribution must be a useful state-dependent switch decision
with independent training evidence, not a renamed cost comparison, a regression
model alone, or an unsupported guarantee of optimal switching or acceptance.
