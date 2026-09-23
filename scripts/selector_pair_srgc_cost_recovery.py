"""Extend the existing interrupted-cost policy to unpublished SR-GC measurements."""
from __future__ import annotations

import contextlib
from pathlib import Path

import selector_pair_cost_recovery as pair_recovery
import selector_pair_gpu as worker
import selector_pair_srgc_score as score


def recover(root, protocol, *, now=None):
    import selector_pair_srgc as srgc
    root = pair_recovery.checked(Path(root))
    srgc.validate(root, protocol)
    scope = pair_recovery.checked(root / 'sr-gc')
    policy = pair_recovery.recovery_module()
    original_read, original_recover = policy.read_events, policy.recover
    names = {f's{seed}-t{step}' for seed in worker.pair.TEST_SEEDS for step in worker.pair.STEPS}
    phases = {'sr-gc-' + stage for stage in (*score.STAGES, 'r-validation', 'r-candidate', 'aggregate')}

    def checked(directory):
        pair_recovery.checked(directory)
        if directory.parent != scope or directory.name not in names:
            raise ValueError('not a scheduled SR-GC measurement directory; costs preserved')
        pair_recovery.checked_cost_files(directory)
        pair_recovery.checked(directory / '.decision.lock')
        for phase in phases:
            for shard in range(4):
                pair_recovery.checked(directory / f'{phase}-{shard}.log')
        return directory

    def read(directory, *, repair=False):
        checked(directory)
        raw, events = original_read(directory, repair=repair)
        for event in events:
            if (event.get('phase') not in phases or event.get('ledger') != 'deployment'
                    or event.get('gpus') != 4 or event.get('gpu_type') != protocol['gpu_type']):
                raise ValueError('SR-GC cost allocation differs from the frozen measurement')
        pending = worker.core.cost_summary(events)['incomplete_events']
        if pending:
            if len(pending) != 1:
                raise ValueError('overlapping SR-GC cost events; original costs preserved')
            index = next(i for i, row in enumerate(events) if row['event_id'] == pending[0])
            if any(row['event_id'] != pending[0] or row['state'] != 'started' for row in events[index:]):
                raise ValueError('SR-GC ledger is not a closed serial prefix plus one open phase')
            worker.pair.finished_events(events[:index])
            for path in (directory / 'decision.json', root / 'test-decisions.json'):
                pair_recovery.checked(path)
                if path.exists():
                    raise ValueError('SR-GC costs are sealed by a published decision; preserved')
        else:
            worker.pair.finished_events(events)
        return raw, events

    def close(scope_root, relative, event_id, **kwargs):
        directory = checked(scope_root / relative)
        if kwargs.get('evidence_kind') in {'stale_owner_last_evidence', 'confirmed_local_stop_last_evidence'}:
            kwargs['reason'] = ('Estimated interrupted-attempt duration from last observed evidence plus the '
                                'existing margin; not directly measured and not a guaranteed upper bound')
            kwargs['evidence_extra'] = {**kwargs.get('evidence_extra', {}), 'duration_is_estimate': True,
                                       'directly_measured': False}
        try:
            with contextlib.ExitStack() as locks:
                locks.enter_context(worker.pair_lease(pair_recovery.checked(root / '.pair-barrier.lock'), shared=True))
                guarded = False

                def locked_read(target, *, repair=False):
                    nonlocal guarded
                    # The reviewed closer already holds decision and meter
                    # leases here. Also exclude detached scoring workers.
                    if not guarded:
                        paths = [directory / f'.{stage}-{shard}.lock'
                                 for stage in score.STAGES for shard in range(4)]
                        paths += [directory / 'ranking' / f'.{stage}-{shard}.lock'
                                  for stage in ('validation', 'candidate') for shard in range(4)]
                        for path in paths:
                            locks.enter_context(worker.pair_lease(pair_recovery.checked(path), shared=True))
                        guarded = True
                    return read(target, repair=repair)

                policy.read_events = locked_read
                return original_recover(scope_root, relative, event_id, **kwargs)
        except (worker.PairLockBusy, BlockingIOError):
            return {'status': 'active', 'reason': 'SR-GC decision, meter, or scoring lease held; costs preserved'}
        finally:
            policy.read_events = read

    policy.read_events = read
    policy.owner_lock_path = lambda _, directory: checked(directory) / '.decision.lock'
    policy.recover = close
    rows = policy.close_stale(scope, now=now)
    for row in rows:
        detail = (f"{row['seconds']:g}s charged; evidence={row['evidence']['kind']}; prior costs preserved"
                  if row['status'] == 'recovered' else row.get('reason', ''))
        print(f"[SR-GC recover-cost] {row.get('directory', '')}: {row['status']} {detail}", flush=True)
    return rows
