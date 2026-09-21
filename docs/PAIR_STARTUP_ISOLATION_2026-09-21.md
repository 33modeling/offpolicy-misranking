# Pair Startup Isolation

## Report and Verified Paths

The user reported that starting another selector worker affected existing work.
The exact command and remote symptom have not yet been supplied, so these local
reproductions do not establish which path caused that incident.

Two interference paths were identified in the launch code:

- Pair leftover cleanup ran before node ownership was acquired. A duplicate
  invocation could inspect and signal old event workers before discovering
  that the node was already occupied. A free phase lock alone did not protect
  every live controller or distinguish another allocation visible in /proc.
- The restart wrapper requested `restart_shared=True` unconditionally. Even
  when the root already supported parallel workers, the default command could
  intentionally send TERM to an existing local controller. That is unsuitable
  for simply adding a worker.

## Changes

- Acquire the local or shared-filesystem physical-node lock before any Pair
  GPU recovery. A duplicate launch returns busy without invoking recovery.
- Hold node ownership through recovery, close inherited lock descriptors in
  the helper, and release ownership if recovery fails.
- Preserve a visible live Pair controller even when its phase lock is free.
  Recheck for a controller immediately before cleanup.
- Require matching PID/mount namespaces and cgroup before selecting leftover
  processes. Preserve an event if any matched descendant has a different or
  unverifiable allocation. Hostname and GPU usage are not ownership proof.
- Default `restart_selector_pair.sh` now joins the queue without signalling
  any existing controller. An exclusive legacy controller is preserved and
  reported as blocking admission. Only `--restart-current` opts into the
  existing verified local handoff; all previous ownership checks remain.
- Refresh the pinned runtime and generated standalone Bash wrapper so both
  direct and staged launches receive the isolation changes.

No scientific source, target, allocation cap, checkpoint, result, or recorded
cost is changed. Existing workers are not restarted by this patch.

## Bash Usage

From the experiment checkout, after updating it, add a worker on a free node:

```bash
bash scripts/restart_selector_pair.sh
```

This uses a separate pinned runtime, preserves current workers, and retains
normal GPU/node admission. On an occupied node the new invocation exits busy.

The direct launcher also preserves the current owner:

```bash
bash scripts/run_selector_pair.sh run
```

Only when intentionally replacing the controller on this allocation:

```bash
bash scripts/restart_selector_pair.sh --restart-current
```

That explicit command may interrupt in-progress work. It is not the command
for adding another node.

## Regression Coverage

Real process and flock tests cover duplicate local/shared-node admission,
recovery failure releasing locks, a live controller with a free phase lock,
different or unreadable namespaces/cgroups, foreign descendants, genuine
leftover recovery, and preservation of other experiment roots. Handoff tests
cover default non-interruption and explicit-only local restart. Deployment
tests compare the staged safety helpers against the current maintained files.
No remote GPU task is launched or signalled by this validation.
