# Existing code consulted for V7 execution

Read-only source: `/home/kms/dev/offpolicy-misranking`, commit
`09c95b9b954d0aa47a1ee60b11f2037ce6a6f615`.
The source checkout was not modified. V7 retains its independent implementation
of the author-specified online procedure; the old diagnostic protocol is not
imported as an online switching algorithm.

| Existing source | Applied in V7 |
| --- | --- |
| scripts/selector_pair_cost_measure.py | Confirmed an existing selection_interval=25 block runner; it ranks the full pool, so it is not the author's 40-candidate/top-four algorithm. V7 now implements that explicit 25-update algorithm independently. |
| `scripts/run_selector_pair.sh` | Four-GPU admission check, bounded CPU/thread pools, input checks before GPU work |
| `scripts/_selection_worker.sh` | Track the real worker process, forward termination, wait before releasing work leases |
| `scripts/selector_pair_parallel.py` | Shared task leases, completion receipts, ready-task scheduling, skip completed work |
| `scripts/selector_pair_deploy.py` | Bind execution to code hashes and use an isolated fixed checkout |
| `src/train_policy_grpo.py::_response_logps_batch` | Pad variable-length responses with attention masks and batch log-probability forwards |
| `src/grads.py::grad_params` and `prompt_gradient` | Enable only the required scoring derivatives; accumulate response gradients before projection |
| `scripts/apply_valgrads_memory_patch.py` | Chunk the LM head and recompute its activation blocks in backward to avoid retaining full vocabulary tensors |
| `configs/olmo3_rlzero.json` | Pin the model and tokenizer to the recorded OLMo-3 revision `a81bae42db3975be1671e27b9c9a56da1a9f980f` |

The new queue separates each seed's prefix from its four continuations, so
different nodes can train arms from exactly the same saved model and optimizer.
V7 uses its original fixed CountSketch mapping; it does not replace the
projection hash with the existing runner's different mapping. Response batches
change only computation grouping. The eight scoring responses, full 40-vs-40
contrast, four selected training prompts, fresh training responses and temporal
rule remain unchanged. The 2026-09-27 author clarification changes selection
refresh to every 25 updates; top-four IDs persist between refreshes.

Tests compare gradients and optimizer updates before and after batching,
chunked-head values and derivatives against full forwards, and distributed
updates against one-rank updates. These are correctness checks on small CPU
models. Multi-node scheduling and reduced redundant work are implemented;
actual 7B acceleration remains to be timed on the allocated GPU hardware.
