# V8 sequential experiment dispatch

User request (2026-11-10): "일단 1개를 완전히 끝내고 다른거를 시작하게 해저".
This replaces the earlier request to fill one cohort and spill spare nodes into another.

The default launch command remains:

```sh
sh scripts/run_v8_nodes.sh
```

Order: OLMo MATH → Qwen MBPP → OLMo MBPP → Qwen MATH. Nodes run ready
tasks in parallel within the current cohort. Training, budget extensions, endpoint
evaluation, collection and analysis must finish before the next cohort starts.
Only the shared queue's `complete` phase admits the next cohort; missing inputs,
failed tasks, an empty offer list and 100% training progress do not prove completion.
A live task lease is checked again at admission. An unavailable earlier runtime
remains in the order rather than being silently dropped from startup selections.
Historical primary-result barriers are scoped to the current cohort so they cannot
wait for results from a later cohort that is deliberately not running yet.

The controller records the full order with `spill_to_next=false` in its policy
receipt. Existing queue bindings, task identities, frozen numerical runtimes,
model/data definitions, seeds, updates, reward/evaluation protocols and checkpoint
files are preserved. The previous coordinator is included in the historical runtime
archive for exact-source resume. Existing bounded retry/native-crash controls remain.
The retained low-level fair dispatcher is tested separately for archival callers;
the production CLI explicitly enables the sequential policy.

Workers already running use frozen controller snapshots. Apply the new order by
safely stopping old workers on each node, pulling master and running the same command.
Do not reset the queue or remove checkpoints. No remote node was contacted or restarted
while implementing this change.

Validation is CPU control testing, not a new learning experiment or evidence of
remote H100 completion. Model performance, throughput, wall time and remote progress
are unmeasured. No training, GPU allocation, paid API, deployment or local server
was started. The new tests exercise four-cohort completion, six concurrent workers,
no spill when the current cohort is full, final evaluation/finalize ordering,
missing-runtime blocking, finite retries and checkpoint preservation.
