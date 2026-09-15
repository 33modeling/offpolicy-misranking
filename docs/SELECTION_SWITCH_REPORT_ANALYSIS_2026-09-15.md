# Switch and MoPPS report analysis: 2026-09-15

## Evidence and scope

Read-only inputs supplied in `/home/kms`:

| Report | Export time (UTC) | Export checkout | SHA-256 |
| --- | --- | --- | --- |
| `switch-why-20260915T100511Z-9ZQFr8.txt` | 10:05:11 | `bdd727e` | `8aabd6319a205e1ce3b9e0693ef4d80a9125df06f2ad6fa5aa286754b8b9bc99` |
| `mopps-why-20260915T100615Z-Qn04Kw.txt` | 10:06:15 | `68c40df` | `e484fcbc5d87cedc97c5cde852a5e0c1ceb85722d6fe5ec5cea4be69f616e80b` |

Original files were not changed or uploaded. These are time-bounded exports,
not a live cluster inspection or atomic snapshot. No secure-cluster access was
attempted. Export checkout, pinned worker revision and historical log revision
must not be conflated.

## Findings

### CUDA 802, not just the ChildFailedError wrapper

The rank error is `Cuda failure 802 'system not yet initialized'`, with NCCL
2.26.2. On newly pinned `bdd727e` workers it occurs during the tiny admission
probe's `dist.init_process_group`, before any training branch is claimed.
Both the baseline and the `NCCL_CUMEM_HOST_ENABLE=0` fallback fail identically.

| Host | Evidence in these exports | Meaning at that time |
| --- | --- | --- |
| `run282224-wss-4` | Switch has five probe directories, each with two failed attempts | Repeated admission failure, not a claimable healthy worker |
| `run282225-wss-5` | MoPPS has two failed admission records, each with two attempts | Same 802 failure after the latest cache fix |
| `run282226-wss-6` | MoPPS has three passing admission records with all four ranks | Communication and tiny DDP passed; old branch failures still need explicit retry |

MoPPS report lines 269, 760, 1251, 1497 and 1743 locate these admission
records. The Switch export previously omitted admission/rank JSON; its probe
logs and cost journals provide the failed-attempt evidence instead.

The node-6 launcher log tail at MoPPS line 9113 reprints a trainer failure
whose embedded hostname is `run282128-wss-4` and runtime is `f258c46`.
That reprinted failure does not establish a new node-6 NCCL failure. The three
fresh passing records are separate evidence.

NVIDIA documents CUDA 802 as a system-readiness problem and identifies driver
initialization and Fabric Manager on NVSwitch systems as checks. It distinguishes
802 from the driver-combination error 803. This makes host driver/library and
fabric inspection appropriate; these reports do **not** prove that Fabric Manager
is absent or identify one faulty component. Some earlier trainers successfully
loaded the model before collective initialization failed.
[NVIDIA CUDA driver troubleshooting](https://docs.nvidia.com/nim/large-language-models/2.0.2/troubleshooting/cuda-driver.html).

Do not blindly reinstall PyTorch/NCCL, disable P2P/IB/NVLS or restart services
on active nodes. Give the administrator the failing node identities and the
probe records. Useful read-only host checks, where available, include
`nvidia-smi -q`, `nvidia-smi topo -m`, `systemctl status nvidia-fabricmanager`
and `nv-fabricmanager --version`; container service visibility alone is not
proof about the host service. Check the actually loaded `libcuda.so` as well.
Re-run admission after the administrator resolves or clears the node issue.

### Work is partially running; Gate and MoPPS have different blockers

The Switch report records two active controllers. One minute later the parent
status in the MoPPS report records three, with fresh heartbeats:

- `run281932-wss-1`: `s1/t50 SEL`, fresh-r candidate scoring.
- `run282223-wss-2`: `s0/t100 RND`, training.
- `run280018-first-qw-3`: `s1/t25 RND`, training by 10:06:16 UTC.

At that later snapshot: prefixes 12/15, development 6/18, held-out test 0/30.
Gate correctly awaits 12 unpublished development branches. Seed 1 prefix 100
is ready; seed 3 and 4 prefix 100 failed. Do not stop healthy progressing jobs
merely to install this launcher fix. A pull does not replace their pinned code.

MoPPS has eight historical failed continuations (seeds 3/4, steps 25/50,
both arms), plus four blocked continuations waiting on the failed step-100
prefixes. `run` does not silently spend compute retrying historical failures.
After diagnosis, `retry` is the explicit authorization for that work. No MoPPS
completion or Gate-versus-MoPPS scientific outcome is present in these reports.

### Four development costs cannot be reconstructed from these files

No matching completed receipt was exported for the following deployment
events. Each directory is relative to `selection-switch-v1`:

| Directory | Event ID | Host / owner PID | Start epoch (UTC) | Last recorded seconds, lower bound only |
| --- | --- | --- | --- | --- |
| `states/s0-t25/points/view-25/selection_reduced` | `9643f4b7a94841f282f07ad848d7a91a` | `run281793-wss-1` / 614 | 1789394324.028645 | 0.013336427509784698 |
| `states/s0-t25/points/view-25/random_reduced` | `903d7da63bde4dc79692bf4ec3c7f98d` | `run281729-wss-2` / 713 | 1789389854.0817823 | 1680.9683785773814 |
| `states/s0-t100/points/view-100/selection_reduced` | `5db41e8e41334ec296868da9208176b8` | `run281934-wss-3` / 183411 | 1789464713.9706445 | 660.4382969997823 |
| `states/s2-t50/points/view-50/random_reduced` | `7ea8e38c6ab646919bb382f65745c4b9` | `run281816-wss-3` / 577 | 1789397741.409775 | 1215.7849286608398 |

All four allocations are four GPUs. The heartbeat duration is not a finish
time. It must not be passed as an invented exact `--seconds` value, zeroed,
deleted, or replaced by a fresh budget. Obtain a matching atomic finish receipt
or the confirmed termination duration and its scheduler/operator evidence.
The existing recovery command validates the event/allocation and acquires its
owner lock before appending evidence; it does not rewrite the original ledger.

```bash
bash scripts/run_selection_switch.sh recover-cost
```

For each confirmed stopped event, using the directory and ID above:

```bash
bash scripts/run_selection_switch.sh recover-cost \
  --directory RELATIVE_DIRECTORY --event-id EVENT_ID \
  --seconds ACTUAL_ELAPSED_SECONDS --reason 'termination evidence reference'
```

If the exact finish receipt is present on the cluster, omit `--seconds` and
`--reason` and let the tool verify that receipt. Do not recover an active event.
The concurrently added `--stale` command is retained, but now auto-recovers
only completed receipts. For silent events without receipts it reports the
missing evidence and exits 2, without converting a heartbeat or log mtime into
an invented finish time. This corrects the cost-undercount risk in `779022f`.
The five additional unknown **research** prefix events are separately retained;
they are not the four deployment-budget blockers. An open event for a currently
running worker is also not evidence of a lost finish record.

### The 1074/537 traceback is historical

Switch report line 16524 contains the exact tensor-size exception. Its caller
path is `/user-volume/offpolicy-misranking/src/...`, not the `bdd727e` pinned
runtime. The exported old failure matches the cache-recomputation issue fixed
and reproduced in the previous patch, but does not establish a new failure of
the guarded helper. Old failed logs also remain under some now-completed
branches. Do not overwrite those logs or interpret every retained tail as a
current failure.

## Repairs from this analysis

- Recognize CUDA 802 from either the supervisor exception or individual rank
  errors. Record `failure_kind=cuda_system_not_ready` and an actionable diagnosis.
  Do not retry the same failure under the unrelated host-allocation setting.
  Exit 78 before any training claim, retaining the completed probe cost.
- Stop no-argument MoPPS bulk retry on admission failure or worker stop status
  78/130/137/143. Do not probe all eight branches on the same failed node.
- Make bulk retry use explicit zero idle timeout. The controller now honors
  zero even when a busy branch has a fresh heartbeat, so nodes can take other
  independent branches. Locks remain authoritative; active work is not stolen.
  Normal `run` and explicitly positive wait timeouts retain their behavior.
- Do not bury the current admission failure under unrelated historical branch
  tracebacks. An ordinary MoPPS controller failure also prints the explicit
  retry command after its diagnostics.
- Include admission, per-rank and all runtime-receipt JSON in Switch `why` and
  `export`, matching the evidence already provided by MoPPS `why`.
- Retain stale-event inspection and receipt recovery without treating a silent
  owner or a last log write as proof of a completed deployment allocation.
- Preserve the exact existing MoPPS manifests, mixed-version receipts, parent
  Switch data and costs; append `nonblocking-retry-runtime.json` for the queue
  change. No sampler, trainer, reward, optimizer or budget changes.

Switch source-map fingerprint remains
`b7803071821dd7aa68035370e37c77fdbaecec13d9e96ea6574d91b10758f5a5`.
MoPPS changes from
`46f08b629491f95ec180ead37440dd92c1161d19d34a81dfd14ffebfcdeb9756`
to `b49a7417bf1f00c8c164c6bd9d0aa480dd4d7438e2de4d6f8d505d53a3e2dab2`.
The exact exported mixed receipt chain is covered by regression tests.

## Resume sequence

Leave healthy current work alone. On a free node, with the same roots and
existing environment, update the checkout:

```bash
git pull --ff-only
```

For the diagnosed MoPPS historical failures, on a node that passes admission:

```bash
bash scripts/run_mopps_comparison.sh retry
```

This is one sweep of the recorded failed branches. Busy branches are skipped;
it is not a request to create fresh tasks or wait for missing prefixes. After
the missing parent prefixes are published, use the regular MoPPS `run` for
the remaining fresh continuations. Each worker needs one four-GPU node;
separate nodes claim independent branches, not ranks of one multi-node job.

For unfinished Switch prefixes/development work, including recovered costs:

```bash
bash scripts/run_selection_switch.sh run
```

Admission will stop a still-unhealthy node explicitly; this patch does not
repair host infrastructure. Stop an old detached launcher only when that
specific launcher must be replaced, using its `stop` mode; Ctrl+C only closes
the console view. Do not delete locks, run roots, manifests or cost records.

Verification results are recorded in [the bug-fix log](SELECTION_SWITCH_BUGFIX_LOG.md).
