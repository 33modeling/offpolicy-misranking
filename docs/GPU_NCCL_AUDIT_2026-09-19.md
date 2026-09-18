# GPU and NCCL Runtime Audit - 2026-09-19

## Scope and Verdict

Repository baseline: `e0f091d` (`master`). Full CPU suite, Python fatal-error
lint, shell syntax checks, runtime ownership/admission review, and local CUDA
probes were exercised. This is **not a clean bill of health**: outstanding
MoPPS resume and test-contract failures are recorded below.

The working tree contained user changes before this audit and changed during
the first full run. Only this audit's launcher/preflight changes and regression
tests are included in its commit. Concurrent MBPP cooldown/restart changes and
the existing evidence exporter edits are not included. The staged code is also
tested separately in an exported snapshot to avoid depending on those edits.

No remote experiment was stopped, restarted, or modified. No training equation,
selection rule, frozen scientific hash allowlist, policy, or cost ledger was
changed by this audit. Remote H100/NVSwitch health remains unverified.

## Fixed Findings

1. **P1: cleanup could target unrelated GPU jobs.**
   `run_experiments.sh` accepted a GPU process based on UID alone, and root
   sweeps did not distinguish allocations sharing a work directory. Cleanup now
   requires an exact node identity, an experiment command, and the work/root
   marker. An outer keepalive requires its work marker. Unverified holders are
   reported and left running. Command matching no longer truncates long paths.

2. **P1: stale PID files and root stop commands could terminate the wrong job.**
   The three launchers now validate command and environment identity before
   accepting a stored PID. Switch/MoPPS stop and duplicate-worker scans require
   exact root and node markers. Their unscoped `queue_status --kill-orphans`
   calls were removed; scoped sweeps remain. Sweeps exclude the caller's actual
   process group. CPU regressions use disposable sleep processes, not GPU jobs.

3. **P1: a rejected node could enter the other queue immediately.**
   A Switch exit of 75 (busy), 78 (admission failed), or 79 (cooldown) now skips
   MoPPS for that pass. A MoPPS admission failure/cooldown now also prevents the
   READY-work poll from short-circuiting the hold. Logs distinguish this skip
   from an experiment that is complete or not prepared.

4. **P2: GPU-query failure looked like a free GPU.**
   `gpus_free` previously lost the exit status of process substitution and
   accepted an empty response. Missing `nvidia-smi`, timeout, nonzero exit, empty
   output, and nonnumeric memory now return failure. Node identity and cleanup
   probes have TERM/KILL time limits. This fixes cleanup's verdict; the tiny
   NCCL admission probe remains mandatory before training.

5. **P2: NCCL fallback composition stopped prematurely.**
   CUDA 802 followed by an eligible legacy host-allocation failure could not
   take the host workaround because it was restricted to the baseline attempt.
   It can now run once after a fabric workaround, preserving earlier overrides.
   Explicit operator settings remain authoritative. A five-failure mixed-error
   regression verifies bounded retries, unique evidence directories, and final
   rejection, not unconditional admission.

The initial eight new regression cases produced seven failures on the old
code. Their fixes were then checked with an expanded runtime suite.

## Verification

Local GPU: RTX 3050, 6 GiB; driver 595.84; PyTorch 2.7.1+cu126;
CUDA runtime 12.6; NCCL 2.26.2. No compute job held the GPU before the probes.

- Initial full CPU run: **1,696 passed, 84 failed, 16 skipped**, 671 seconds.
- Missing gate dependencies accounted for 23 failures. Installed the existing
  `requirements-gate.txt` into the local test venv; no Torch/CUDA upgrade.
- Rechecked all seven remaining initially failing modules other than the
  separately tested launcher module: **197 passed, 40 failed**.
- The 19 runtime-fixture failures in the initial run were caused by a concurrent
  new `node_fault_state.py` dependency missing from fixture copies. They passed
  after the concurrent fixture update; that update is not this audit's change.
- Launcher/NCCL/worker cleanup tests on the working tree: **99 passed,
  11 skipped** before adding the final mixed-error bound regression.
- Actual local CUDA admission tests: **3 passed**, including DDP update and
  collectives, explicit legacy host allocation, and invalid-rank rejection.
- Expanded commit-snapshot CUDA selection: **30 passed**, including **15 actual
  GPU cases** and 15 configuration/classifier cases. GPU cases include resistant
  descendant cleanup, SIGINT/SIGTERM teardown, cost completion, lock release,
  restart, and eager/SDPA KV-cache gradient checks. No compute process remained
  in `nvidia-smi` after completion.
- `ruff check src scripts tests --select E9,F821,F822,F823`: passed.
- `bash -n` on all `scripts/*.sh`: passed. `git diff --cached --check`: passed.

The first isolated CPU snapshot run was 129 passed, 1 failed, 11 skipped. The
new process-identity fixture failed once and passed in isolation; its fixed
sleep was replaced with an explicit `/proc` exec-readiness check before reading
process identity. The final isolated snapshot run passed: **130 passed,
11 skipped**, 111 seconds. This includes the pinned-runtime integration tests.
The 11 skips are opt-in CUDA cases covered by the separate GPU run.
The tested staged source tree, before adding these audit documents, was
`9623bdcb04baaa221609f1ab84df7e3d8a0060f3`.
Machine-readable run summaries are in `docs/audits/gpu-nccl-2026-09-19.json`.
The complete local JUnit files are `/tmp/offpolicy-*-audit-20260919.xml` and
`/tmp/offpolicy-failure-recheck-20260919.xml`.

## Outstanding Findings

### P1: Historical MoPPS Resume Is Blocked

`src/mopps_comparison_gpu.py:protocol` rejects known historical MoPPS manifests
after shared runtime publication fixes changed `net_gain_gate_gpu.py` and
`train_selection_gate_grpo.py`. Those files are not in its permitted-change set,
although Switch has an explicit reviewed migration for them.

Independent reproduction used actual `git show 47339ca:<file>` bytes for every
file in `mopps_comparison_gpu.CODE`. Their combined fingerprint exactly matched
`PRE_DATASET_CODE`:
`c12c6e7956f4a61e648c29b3da2762bc71fc0713b68bd77012aed59004ecdddd`.
Calling `protocol()` on a temporary manifest with those hashes raised
`ValueError: frozen MoPPS experiment changed: unreviewed code hashes`.
This reproduction did not touch a real experiment root.

The 28 failing MoPPS tests first fail even earlier: predecessor fixtures combine
current shared files with historical hashes and no longer reconstruct the old
fingerprints. Updating fixtures alone would not fix the independently reproduced
runtime rejection. This requires an exact, reviewed MoPPS migration and receipt
preservation tests. It remains **unfixed**; no broad hash-check bypass was added.

### Other Remaining Test Failures

- 11 `test_status_reward_audit.py` cases assert the former Qwen full-grid output
  against the current compact default view. A reproduced blocked-job case still
  reports `state DEGRADED`, a blocked count, and the actual runtime error. These
  are not evidence of 11 independent training failures; the output contract
  needs reconciliation.
- One test in the pre-existing, uncommitted evidence-exporter changes parses a
  blank line as JSON. Those edits were left untouched and are not included here.
- The CPU run skipped opt-in GPU cases and one plot test due to missing
`matplotlib`. A full multi-GPU model training run was not performed.

## Reproduction Commands

Run the full CPU suite with the local experiment Python:

```sh
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts:tests \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .work/.venv-cu126/bin/python -B -m pytest -q -ra -p no:cacheprovider
```

The local GPU checks below are opt-in and require an idle GPU 0. They start and
terminate their own tiny workers, not an existing experiment:

```sh
env SWITCH_TEST_CUDA=0 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:scripts:tests \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .work/.venv-cu126/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_selection_nccl_preflight.py tests/test_selection_worker_shutdown.py \
  tests/test_net_gate_memory_math.py -k cuda
```

## NCCL Interpretation and Next Evidence

The stored September 15 report contains four-rank CUDA 802 failures as well as
passing admissions on another node. It is historical evidence, not the latest
running job's diagnosis. CUDA 802 alone does not identify which driver/fabric
component failed. Current failing-node `node-preflight/*/admission.json`, rank
JSON/log files, and launcher logs are still needed to diagnose today's cluster.

The local single-GPU checks cannot validate H100 inter-GPU P2P, NVLS, NVSwitch,
or InfiniBand. No cluster-wide transport disable, driver reinstall, or Fabric
Manager restart was attempted.

NVIDIA's documented legacy cuMem-host fallback and 2.26.5 behavior:
https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/user-guide/docs/troubleshooting.html

PyTorch's eager NCCL initialization via `device_id`:
https://docs.pytorch.org/docs/stable/distributed.html
