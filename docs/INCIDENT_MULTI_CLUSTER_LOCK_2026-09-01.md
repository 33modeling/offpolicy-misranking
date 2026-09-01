# Multi-Cluster Admission Lock Incident

Date: 2026-09-01 KST

## Impact

Three independent four-H100 clusters launched the same canonical command against
one `GROUP_VOLUME`. Only one entered the experiment queue; the other two exited
before claiming work. This wasted scheduled cluster time and delayed the primary
matrix.

## Root cause

`run_rlvr.sh` used `hostname` to name a process-lifetime admission lock and put
that lock below `$OM_WORK/locks` on the shared volume. The three cloned cluster
environments reported the same hostname. They were therefore incorrectly
treated as duplicate processes on one physical node.

The seed/dataset family queue itself was not the defect. Its nonblocking shared
locks correctly assign one complete checkpoint chain to one worker and release
claims after process death. The invalid shared launcher lock prevented two
workers from ever reaching that queue.

## Correction

Revision `fb48565dda10d35aa58452b0b37696ae122558aa` makes the lock boundary
explicit:

- `primary.lock` and `additional-suite.lock` live in node-local
  `/tmp/offpolicy-misranking-$UID` by default.
- A configured `OM_LOCAL_LOCK_DIR` below `GROUP_VOLUME` is rejected.
- Shared storage retains only seed/dataset family locks, collection locks,
  immutable contracts, artifacts, logs, and the final harvest lock.
- Each launcher receives a random worker suffix, preventing same-hostname log
  and smoke-marker collisions.
- Additional experiments wait on the same node-local primary lock before using
  that node's GPUs.

## Current `295dfea` run recovery

Do not pull or edit the checkout of the worker that is still running. To join
the two stopped clusters to the same `295dfea` experiment without changing the
recorded source revision, run the following from their existing clean checkouts,
using a different label on each cluster:

```bash
# stopped cluster 2
bash -c 'hostname(){ printf "rlvr-cluster-2\n"; }; export -f hostname; exec bash scripts/run_rlvr.sh'

# stopped cluster 3
bash -c 'hostname(){ printf "rlvr-cluster-3\n"; }; export -f hostname; exec bash scripts/run_rlvr.sh'
```

This confines the hostname override to the launcher process, creates distinct
legacy admission-lock names, and leaves the checkout and Git provenance clean.
Both workers then enter the existing shared family queue. Future launches should
pull `fb48565` or later and use the ordinary command with no override.

Do not pull the checkout of a process that is still running. Revision
`a223ee31bd0542cbe6c35af66cfb3549d0a8c40e` covers the later stopped-process
case: after a worker exits, a newer supervisor may be pulled and launched with
the ordinary command. The matrix-level `generation.git` marker is created
without requiring a pre-existing snapshot or manifest. If partial artifacts
record an older commit, the supervisor automatically restores that commit in a
node-local detached worktree and resumes only the missing or invalid points.
Mixed provenance fails before a family claim.

## Prompt mismatch retry incident

Revision `c932ca0da2c2eb34e5defd914bb4a684bdf5d982` fixes a second failure found
during live recovery. A positive-drift point regenerated `prompts.json` through
the node's current dataset path before comparing it with its d0 behavior source.
If two nodes resolved different local copies, the comparison failed
deterministically, but the point retry loop and launcher restart loop treated it
as transient and repeatedly reacquired the GPUs.

Positive-drift points now take the exact prompt bytes from d0. When resuming an
older generation commit, the current supervisor creates a hash-addressed local
loader snapshot whose output exactly reconstructs d0, then passes that path to
the old pipeline without changing its Git provenance. A pre-existing mismatched
target is preserved under quarantine and regenerated once. Base or
qualification prompt mismatches exit with code 43, which both launchers treat
as non-retryable.

Follow-up revision `ccc2807568b270a3efc92933c45aaccff79fe60f` added native
source-derived snapshots for MBPP, Knights-and-Knaves, and ARC-Challenge in
addition to GSM8K and MATH-500. Its inverse-shuffle seed change was incorrect:
`experiment.py` intentionally fixes candidate-pool splitting at seed 0 across
all experiment seeds. Passing the policy/rollout family seed reconstructed a
different prompt order for seeds 1 and 2 and caused deterministic d25 prompt
mismatches. The current correction always reconstructs with split seed 0 and
tests the exact loader path used by `experiment.py`.

## Offline input and rollout independence incident

Compute nodes are security-isolated. Environment-only offline flags did not
make the loader contract sufficiently explicit: Transformers and PEFT calls
did not pass `local_files_only=True`, and inherited Hub token variables remained
visible. Compute launchers now remove inherited token variables, force all
three Hugging Face offline flags, disable implicit tokens, and open the base
model, tokenizer, and adapters in local-files-only mode. Dataset Hub fallbacks
are rejected unless an explicit preparation process sets `OM_ONLINE=1`.
Snapshot and reward-runtime qualification runs before a GPU training process.

A separate audit found that behavior and fresh generation stages initialized
the same RNG stream. This contradicted the independent Monte Carlo protocol and
made resume output depend on the interruption boundary. New generation uses
disjoint behavior, fresh-train, and fresh-validation seed domains; fresh seeds
also include policy drift. A per-prompt seed derived from the global prompt ID
makes output stable under interruption and resharding. Manifests bind this RNG
scheme, and registered additional runs reject missing or mismatched bindings.
Legacy rollout manifests without the binding are not eligible for the new
registered extension.

## OLMo RL-Zero dependency bootstrap incident

The OLMo launcher declared `math-verify==0.9.0` in `requirements.txt` but
assumed every existing shared cluster venv had already been reprovisioned.
`git pull && bash scripts/run_olmo3_rlzero.sh run` therefore failed with
`ModuleNotFoundError: math_verify` on a stale venv, before the experiment could
enter its queue.

The validation mistake was specific and avoidable: the full suite was run only
inside the local CUDA venv where `math_verify` was already installed. That
proved the verifier behavior but did not test the launcher's stated fresh/stale
venv contract. Reporting the launcher ready on that evidence was incorrect.

The correction vendors the complete pure-Python dependency closure:
math-verify 0.9.0, latex2sympy2-extended 1.11.0, ANTLR 4.13.2, SymPy 1.14.0,
and mpmath 1.3.0. Their SHA-256 values are fixed in
`src/bootstrap_math_verify.py`. Before any GPU admission, the launcher
atomically extracts them under `$OM_WORK/runtime-deps`, prepends that directory
to `PYTHONPATH`, asserts that all five modules came from the bundle, and runs a
symbolic `1/2 == 0.5` verification. No pip, Hub token, network, or shared-venv
mutation is involved.

The bundle path is preserved when the supervisor switches to a commit-pinned
generation worktree and when it launches each family pipeline. A regression
fixture rejects any family command that overwrites `PYTHONPATH` and drops the
bundle after preflight.

Regression policy: a one-command launcher is not considered verified merely
because its dependencies appear in `requirements.txt`. Each nonstandard
runtime dependency must have a launch-path test from an empty bundle cache,
must prove its imported module path and functional smoke case, and must fail
before GPU admission when its checked artifact is absent or corrupt.

The same regression run exposed a queue-scheduling edge case: with synthetic
families completing in milliseconds, the first worker could release one family
lock and immediately acquire the next before another ready worker was
scheduled. The launcher now yields for one second after each completed family.
This is negligible relative to real family runtime and gives waiting clusters
a deterministic opportunity to claim independent work.

A subsequent dataset-adoption failure exposed another invalid assumption. The
first adoption implementation recognized flexible file formats but still
favored a small list of folder names and compared an order-sensitive loader
fingerprint. A separately uploaded official dataset can live under any folder
name, can enumerate the same rows in a different order, and original MATH data
may contain `solution` rather than a derived `answer` field. The resolver now
recursively examines candidate JSONL, parquet, and HF saved-dataset locations,
accepts only the exact official row multiset, and supports boxed-answer
extraction. Regression coverage uses an unrelated three-level folder name,
reversed row order, and `question/solution` MATH rows.

The MBPP qualification then failed on a compute image where bubblewrap was
missing or user-namespace creation was prohibited. Treating one host sandbox
binary as universally available made a valid preloaded dataset unusable. The
code verifier now probes bubblewrap functionality rather than mere executable
presence and falls back, for MBPP function/assert evaluation only, to an
isolated Python subprocess. The fallback applies the existing CPU, address
space, output-file, descriptor, and process limits and additionally rejects
filesystem builtins, private attributes, `sys.modules`, import escape paths,
process control, and dynamic introspection before executing candidate code.
APPS-style arbitrary stdin programs remain bubblewrap-only. Regression tests
force the fallback backend and verify both a valid function and attempted host
file/module access.

Because the OLMo launcher writes `generation.git` before GPU preflight, failed
preflight attempts could otherwise keep selecting code from before the sandbox
fix after a pull. The launcher may now advance that marker only when there is no
`run_config.json` and no `family-*` directory. Once any family has been created,
the original immutable generation commit remains mandatory.

## OLMo live-launch validation failure and repeated aborts

### Impact

The canonical OLMo command repeatedly aborted on the real cluster after being
reported ready. Failures occurred around GPU-idle admission, pinned worktree
recovery, and the `run_matrix.sh` clean-checkout guard. This delayed the main
experiment and consumed allocated cluster time. The exact lost GPU-hours and
cost were not captured, so this record must not invent a number.

### What went wrong

The implementation and validation work made the following mistakes in order:

1. The first performance diagnosis counted rollout samples but did not first
   account for the cluster's utilization-based GPU reclamation rule. The
   existing keepalive began inside `run_point.sh`, after model/data checks and
   signal qualification, leaving an unprotected launch interval.
2. Revision `95a4836` moved keepalive ownership to the top-level worker, but the
   first implementation trusted a `duty_percent` argument that the old worker
   ignored and did near-continuous tiny-kernel work. It also did not account for
   an older commit-pinned `run_point.sh` starting its own keepalive.
3. Revision `e3987f0` prevented new point code from duplicating the supervisor
   keepalive. During that correction, a supervisor keepalive was initially
   restarted inside a family-lock subshell and inherited lock file descriptors.
   The final version closes descriptors 8 and 9 and restarts only after the
   family-lock subshell returns. Older pinned points pause the supervisor while
   their own point-local keepalive is active.
4. Revision `a8340c3` changed `git worktree add` to use `-f`, but that was an
   incomplete fix. It handled a missing registered path but not a locked
   registration or an existing non-worktree directory. Reporting that change
   as sufficient before testing those states was incorrect.
5. Revision `e9a723a` added recovery for all three cache states: wrong/dirty
   checkout, locked or missing registration, and an existing invalid path.
   Invalid node-local pipeline caches are preserved under `.stale-*`, registry
   entries are pruned, and the detached checkout is recreated with `-f -f`.
   Experiment artifacts under `OM_WORK/runs` are not moved or deleted.
6. The launcher still invoked the old pinned commit's `run_matrix.sh`, so the
   new recovery logic did not govern the failing orchestration path. Revision
   `8068205` separates responsibilities: the current supervisor runs
   `run_matrix.sh`, while generation remains bound to the immutable pinned
   `run_point.sh` and source checkout. A suite-bound generation commit now
   triggers automatic checkout repair instead of the user-facing
   `use OM_PIPELINE_REPO=<clean checkout ...>` abort.
7. The family supervisor still exited after three failed attempts and printed
   an instruction to rerun the same command. Its `EXIT` trap then stopped the
   supervisor keepalive, so an unattended transient failure could release the
   GPU allocation even though all partial artifacts were restartable. The
   corrected loop preserves artifacts, releases only the failed family lock,
   continues other families, and retries failures after a bounded delay without
   exiting the worker. After family execution begins, normal exit is reserved
   for an external signal or a completed matrix.
8. The point-launch environment in `run_matrix.sh` replaced `PYTHONPATH` with
   only the pinned generation checkout's `src` directory. That discarded the
   bundled `math_verify` directory bootstrapped by the OLMo launcher, so every
   point failed its verifier import and the new family supervisor repeated the
   same deterministic failure. Point launches now prepend the pinned `src`
   directory while preserving the inherited runtime dependency path. A
   regression test rejects the previous replacement form.

### Validation failure

The reported `4 passed` and later `6 passed` results were local simulations,
not real H100-cluster validation. `test_olmo3_launcher.py` used fake
`nvidia-smi`, fake model/data Python entry points, and initially a fake
`run_matrix.sh`. `test_regime_queue.py` exercised the real matrix supervisor but
used a synthetic point script and temporary storage. Because each side mocked
the boundary where the other test ended, neither test initially executed the
actual `run_olmo3_rlzero.sh -> run_matrix.sh -> run_point.sh` composition. The
mock results were useful unit evidence but were overstated as launch readiness.

The audit host also lacked the cluster's mounted group volume and four H100s.
No claim of cluster success may be based on these fixtures. The current code has
shell/static and simulated recovery coverage, including a locked registration
whose path is replaced by an invalid directory, but live correction remains
unverified until the canonical command passes on the target cluster.

### Mandatory regression policy

- A canonical launcher is not ready when only component mocks pass. Its exact
  launcher, worktree, matrix, point, queue, cleanup, and resume composition must
  run in one integration scenario.
- Worktree recovery must cover clean reuse, missing registered paths, locked
  missing paths, existing invalid directories, dirty/wrong-HEAD caches, and a
  pull followed by partial-run resume. Cache repair must preserve invalid bytes
  under quarantine rather than delete them.
- Background helpers must close inherited lock descriptors. Tests must prove a
  helper cannot keep a node or family lock alive after its parent exits.
- Keepalive tests must distinguish preflight, rollout generation, CPU verifier,
  point transition, retry, queue wait, and cleanup. Duplicate keepalive
  processes are a failure. Duty limits must be measured, not merely accepted as
  an unused argument.
- Every test report must label evidence as unit, simulated integration, local
  real-model, or target-cluster runtime. Only the last category validates H100
  scheduling, shared-volume behavior, CUDA execution, and utilization-based
  reclamation.
- Before allocating the main experiment, capture the exact commit, command,
  worktree HEAD, matrix generation commit, four GPU process/utilization rows,
  first completed rollout shard, and restart evidence from the target cluster.
- If target-cluster access is unavailable, report that limitation and leave the
  launch gate unverified. Do not replace missing runtime evidence with a pass
  count from mocks.

## Verification

- Full suite: 86 passed.
- Three launchers with one cloned hostname all entered both matrix phases.
- Three shared-queue workers all claimed at least one family; every cell ran
  once, aside from one intentionally killed point that was retried exactly once.
- The launcher and queue pair passed five consecutive repetitions.
- A duplicate launcher sharing one local lock was rejected.
- A node-lock directory below the shared volume was rejected.
- A three-worker empty matrix selected one atomic generation commit; an
  interrupted matrix resumed after a simulated pull using the recorded commit.
- Malformed markers, mixed run-config commits, and an explicitly wrong resume
  checkout were rejected before new work was claimed.
- Source-derived prompt snapshots reproduce all five registered loaders' exact
  fixed-seed train/validation order across experiment seeds 0-2.
- A simulated mismatched point is quarantined and rebuilt once, while a
  permanent contract error is launched only once.
- Python compilation, fatal Ruff rules, shell syntax, JSON parsing, dependency
  integrity, and whitespace checks passed.
- The post-correction suite passes 103 tests. A crash/resume fixture produces
  byte-identical rollout rows to an uninterrupted run, and a manifest fixture
  rejects shared behavior/fresh RNG domains.
- A real local Qwen2.5-0.5B snapshot loaded successfully with invalid inherited
  token values, a closed Hub endpoint, and all offline flags enabled.

Actual H100 scheduling remains cluster-runtime evidence to capture from the
three console logs; this audit host has no access to that cluster volume.
