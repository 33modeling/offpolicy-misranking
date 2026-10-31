# Worker startup registration and plan-lock contention — 2026-10-31

The user reported all registered nodes showing LAUNCH and asked for actual worker registration and execution. This environment has no current remote filesystem or SSH endpoint; both local users have no SSH config. The actual remote wait cause and rollout remain unverified. The added probe is strictly local/read-only and does not imply that a remote PID is healthy.

A genuine startup process now writes an external, heartbeat-backed PREPARING receipt before model discovery, pinned package activation and plan preparation. It lives beside the experiment root, outside immutable plans, inputs, runs and queue protocols. Status shows the recorded preparation step, with no optimizer progress or GPU cost inferred. A real queue-worker preflight receipt ends this startup registration; task leases alone continue to define RUN. Failed or stale startup records never fabricate active training.

The existing prepare_missing function acquires a shared root-wide .start-prepare.lock even when all plans already exist, before model and input validation. The operational wrapper now verifies existing plans with exactly the original snapshot identity, package and bundle checks outside that writer lock. Missing-plan creation still uses the original function and its atomic locks. In a model chain, peer-owned creation yields75 before GPU work instead of blocking a child indefinitely in that plan lock; the coordinator retries automatically. Other locks and immutable model/adapter/learner files are unchanged.

A real held-lock test proves complete plans validate without entering serialized preparation. Separate tests retain wrong-weight refusal, stop-before-model-work, peer-owned missing-plan yield, early startup visibility and registered-worker handoff. The first focused run had137passes/1fixturefailure: the test itself created the empty lock file after its before snapshot; the baseline was corrected inside that test-held lock. Original evidence is retained.

Final registration/status checks: **70 passed in78.86seconds**. Four separate neighboring modules: **199passed,1official-tokenizer skip,12warnings in132.47seconds**. New/final files pass Ruff. No original model/adapter/learner file was changed. The initial137passes include68MBPP/bootstrap/handoff checks; these overlap previous audits and are not a new whole-suite run. An isolated CPU reproduction of the original unmodified prepare_missing function confirms a process with an existing plan waits in the kernel on a peer-held .start-prepare.lock; the new wrapper verifies complete valid plans while that lock is held. This reproduces a code contention path, not the actual remote root cause.

A same-checkout pull applies to the next automatically spawned child. An already blocked old Python child does not reload in memory. Avoid assuming manual whole-coordinator restart is required or that rollout is already complete. To inspect an existing local child without signals/GPU/queue changes:

```sh
python3 scripts/srgc_launch_probe.py
```

The probe prints only recognized current-user startup processes, kernel wait channels and selected startup lock paths. It distinguishes kernel-recorded held/waiting locks from unconfirmed open file descriptors and never prints process arguments or credentials. Output from one affected remote node is needed to identify its actual blocker.
