# MBPP uploaded why analysis

## Evidence actually inspected

Read all three uploaded parts `/home/kms/w1.txt`, `w2.txt`, `w3.txt` from set
`mbpp-why-qttwx9g0`, concatenating their payloads after removing part headers.
The payload contains 9,041 lines, 2,745 complete JSON records and 887 inventory
entries. Its last JSON record is incomplete because the exporter reached its
explicit three-file limit. The earlier five MBPP failure summaries and the Pair
lock diagnostic were also inspected, but are not substituted for this newer set.

SHA-256 of uploaded files:

- w1: `0c5f305a9802d42d72f0c9de00cfffb32ff10377197a5e6edcd1574fac132e86`
- w2: `6ec47d23d217a5766704bc373e22cec5c675e9eb69ef1a6b183f5247d368a826`
- w3: `99c1c5de8705e82300ca7ba3767ddf1503349426a420599acce74bc25db33703`

This export omits full manifests, completion seals and live ownership probes.
Counts below describe saved records, not a newly certified scientific result or
proof that a recorded PID is alive on the remote server.

## Quality root

All 48 branch directories are represented:

| Saved condition | Count |
| --- | ---: |
| Canonical result says complete, with a curve record | 37 |
| Separate budget-recovery evaluation says complete, canonical_complete=false | 3 |
| Interrupted training, open cost event, no restartable checkpoint in inventory | 2 |
| Gated branches awaiting development results | 6 |

The three reporting-only recoveries are:

- DEV s2/t50 selection_reduced: used 28387.864953, cap 28376.947149 GPU-s.
- TEST s4/t25 random_full: used 28389.435711, cap 28380 GPU-s.
- TEST s4/t50 selection_reduced: used 28383.561583, cap 28376.929755 GPU-s.

They are measured saved-policy evaluations, not canonical budget-compliant
results. In particular, DEV s2/t50 cannot silently become the missing gate-fit
label. The gate model is absent in the uploaded export.

The two interrupted branches are s3/t25 selection_full and s4/t100
random_reduced. Their inventories contain only 2685 and 2724 bytes of training
statistics at policy root respectively, with no adapter, optimizer, policy
manifest, budget stop or rolling checkpoint. Recorded progress says train/running
at 345.217 and 375.242 seconds, but the corresponding failures report unclosed
cost events. These are not proof of live work. The exported evidence cannot
authorize a cost refund, invent missing weights, or establish a resume point.

## Other roots

The budget-accounted on-policy root contains 42 recorded branch directories,
21 canonical result records and 21 selection failures caused by exhausted branch
allocations. The missing gated results must not be reported as completion.

The difficulty root contains six recorded branch directories, three result/curve
pairs and two training progress records with saved rolling checkpoints. The
export does not prove those workers are still alive. These are separate
experimental conditions; they are not replacements for the missing quality
results. The default all queue currently schedules quality only by design.

## Operational defects and fixes

1. The export placed every historical admission before worker/controller logs.
   It included 1,514 complete admission records and then truncated: no controller
   log section survived. Quality alone contributed 1,418 records with 72,914.739
   recorded GPU-s. This is observed admission cost, not a claim that all of it was
   unnecessary, and the truncated export is not a complete admission inventory.
   Current console/cleanup exits now come first; worker errors precede large
   checkpoint inventories. Each root retains only eight recent admission records,
   each bounded to 16 KiB, with explicit total/sample labels. Nested curve progress
   and read-only lease probes are included. The three-part size limit is unchanged.

2. MBPP admitted GPUs before the worker could explain that only review-dependent
   work remained. A CPU readiness check now recognizes a complete registered
   task inventory containing only DONE, REVIEW and gate-dependent WAIT entries,
   including a reviewed development branch. It reports each blocker and returns
   80 before launching the inner GPU worker. Live leases/progress, incomplete
   inventories, unfinished prefixes, ordinary failures, pending evaluations,
   resumable work and unrecovered budget stops do not trigger this shortcut.
   This does not change any canonical result or experimental condition.

The current server's controller exit was omitted by the old export, so these
files cannot establish the exact latest launch error. Local regression tests
cover the exported blocker pattern and the real controller's no-admission exit.
Remote execution, current leases and missing-file recovery remain unverified.
