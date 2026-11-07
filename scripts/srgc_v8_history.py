#!/usr/bin/env python3
"""Export live V8 training curves, separate evaluations and checkpoint coverage.

Usage: python scripts/srgc_v8_history.py --root /path/to/v8 --output /path/to/exports
Reads JSON/JSONL only; never loads pickle/torch checkpoints or invokes training,
evaluation, a model API or a server. Immutable export generations are published
through latest.json only after every output file has been written.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

TERMINAL = {"completed_horizon", "completed_budget"}
IDENTITY = ("study_id", "cohort_id", "run_id", "method_id", "training_seed", "role")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value, low=0, high=1):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high and math.isfinite(value)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.sources, self.issues = {}, []

    def issue(self, code, path, detail, severity='error'):
        self.issues.append(dict(code=code, path=str(path), detail=detail, severity=severity))

    def safe(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f'input path escapes root: {path}')
        return path

    def read(self, path):
        path = self.safe(path)
        body = path.read_bytes()
        self.sources[str(path.relative_to(self.root))] = sha(body)
        return body

    def json(self, path):
        value = json.loads(self.read(path), parse_constant=self.reject_constant)
        if not isinstance(value, dict):
            raise ValueError('expected a JSON object')
        return value

    @staticmethod
    def reject_constant(value):
        raise ValueError(f'nonfinite JSON value: {value}')

    def lines(self, path):
        if not Path(path).exists():
            return []
        raw = self.read(path)
        lines = raw.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b'\n'):
            self.issue('unfinished_line', path, 'Last unterminated line omitted; source left unchanged.', 'warning')
            lines.pop()
        rows = []
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=self.reject_constant)
                if not isinstance(row, dict):
                    raise ValueError('expected a JSON object')
                rows.append((row, f'{Path(path).relative_to(self.root)}:{number}'))
            except (ValueError, UnicodeError) as exc:
                self.issue('invalid_jsonl', path, f'line {number}: {exc}')
        return rows


def unique(rows, key, reader, path):
    """Conflicting duplicates are retained as issues and excluded from the curve."""
    out, conflicts = {}, set()
    for row, source in rows:
        value = row.get(key)
        if not integer(value):
            reader.issue('invalid_update', path, f'{source}: invalid {key}')
            continue
        if value in out and out[value][0] != row:
            reader.issue('conflicting_record', path, f'conflicting {key}={value}')
            conflicts.add(value)
        else:
            out[value] = (row, source)
    return {k: v for k, v in out.items() if k not in conflicts}, conflicts


def attempt_updates(reader, rdir, manifest):
    """Completed update u belongs to the last attempt with start_update < u.

    In particular, a resume from checkpoint60 keeps the previous attempt's
    completed update60. Its cost event at boundary60 instead belongs to the
    new attempt (update61); those two coordinate systems must not be mixed.
    """
    attempts = manifest.get('attempts', [])
    if not isinstance(attempts,list) or any(not isinstance(a,dict) for a in attempts):
        reader.issue('unknown_attempt_lineage',rdir/'manifest.json','attempts must be a list of objects')
        return {},set()
    order = []
    for attempt in attempts:
        aid, start = attempt.get('attempt_id'), attempt.get('start_update')
        if not isinstance(aid, str) or not aid or '/' in aid or aid in {'.', '..'} or not integer(start):
            reader.issue('unknown_attempt_lineage', rdir/'manifest.json', 'Attempt ID/start_update is unavailable or invalid.')
            return {}, set()
        if aid in [a for a, _ in order]:
            reader.issue('unknown_attempt_lineage', rdir/'manifest.json', f'duplicate attempt ID {aid}')
            return {}, set()
        order.append((aid, start))
    rows = []
    for aid, start in order:
        for row, source in reader.lines(rdir/'attempts'/aid/'updates.jsonl'):
            u = row.get('completed_updates')
            if not integer(u, 1) or u <= start:
                reader.issue('invalid_attempt_update', rdir, source)
                continue
            owner = next((a for a, s in reversed(order) if s < u), None)
            if owner == aid:
                rows.append((row, source))
    declared = {aid for aid, _ in order}
    for file in sorted((rdir/'attempts').glob('*/updates.jsonl')):
        if file.parent.name not in declared:
            reader.read(file)
            reader.issue('orphan_attempt', file, 'Not in manifest; excluded from surviving training lineage.', 'warning')
    return unique(rows, 'completed_updates', reader, rdir)


def checkpoint(reader, rdir, row, source, verify):
    result = {**row, 'source': source, 'status': 'invalid_index', 'payload_loaded': False}
    if not isinstance(row.get('path'), str) or not digest(row.get('sha256')):
        reader.issue('invalid_checkpoint_index', rdir, source)
        return result
    try:
        path = reader.safe(rdir/row['path'])
        if not path.is_relative_to(rdir.resolve()):
            raise ValueError('checkpoint path escapes its run directory')
        if not path.is_file():
            result['status'] = 'missing_file'
            return result
        result.update(status='present_unhashed', bytes=path.stat().st_size)
        if verify:
            before = path.stat()
            h = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1 << 20), b''):
                    h.update(block)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                result['status'] = 'changed_during_hash'
            else:
                result.update(status='verified_sha256' if h.hexdigest()==row['sha256'] else 'hash_mismatch',
                              actual_sha256=h.hexdigest())
    except (OSError, ValueError) as exc:
        reader.issue('checkpoint_unavailable', rdir, str(exc))
        result['status'] = 'unavailable'
    if result['status'] in {'hash_mismatch', 'changed_during_hash'}:
        reader.issue(result['status'], rdir, source)
    return result


def training_rows(reader, rdir, manifest):
    updates, conflicts = attempt_updates(reader, rdir, manifest)
    stats, stat_conflicts = unique(reader.lines(rdir/'training_stats.jsonl'), 'completed_update', reader, rdir)
    # Completion exports can contain inherited prefix rewards. Never use a
    # stale finalized tail during an active restart in place of current logs.
    parent = manifest.get('task', {}).get('parent', {}).get('completed_update', 0)
    if not integer(parent):
        raise ValueError('invalid parent completed_update')
    canonical = manifest.get('status') in TERMINAL
    rows = []
    for u in sorted(set(updates) | set(stats) | conflicts | stat_conflicts):
        raw, source = updates.get(u, ({}, None))
        total, count = None, None
        metrics = raw.get('metrics')
        if metrics is None:metrics = {}
        if not isinstance(metrics, dict):
            reader.issue('invalid_training_metrics',rdir,f'update {u}: metrics must be an object')
            metrics = {}
        value = metrics.get('sample_reward')
        if source and value is not None and not finite(value):
            reader.issue('invalid_training_reward', rdir, f'update {u}: {value!r}')
            value = None
        sources = [source] if source else []
        if u in stats and (canonical or u <= parent or u in updates):
            rec, stat_source = stats[u]
            total, count = rec.get('reward_sum'), rec.get('valid_response_count')
            if integer(count, 1) and finite(total, 0, count):
                candidate = total/count
                if value is not None and abs(candidate-value)>1e-9:
                    conflicts.add(u)
                    reader.issue('training_summary_conflict', rdir, f'update {u}: summary and raw batch reward differ')
                value = candidate
                sources.append(stat_source)
            elif total is not None or count not in (None, 0):
                reader.issue('invalid_training_counts', rdir, f'update {u}: invalid reward sum/response count')
                conflicts.add(u)
        if u in conflicts | stat_conflicts:
            value = None
        if not sources and u not in conflicts | stat_conflicts:
            continue
        rows.append(dict(completed_update=u, reward_fraction=value, reward_sum=total,
                         valid_response_count=count, status='conflicting' if u in conflicts | stat_conflicts else
                         ('recorded' if value is not None else 'unmeasured'),
                         reward_scope='training_batch', inherited_prefix=u<=parent, sources=sources,
                         lineage_observed=(u in updates or (u in stats and (canonical or u<=parent)))
                         and u not in conflicts | stat_conflicts))
    return rows


def evaluation_records(reader, cohort):
    records = []
    for path in sorted((cohort/'evaluations').glob('*.json')):
        try:
            records.append((path,reader.json(path)))
        except (OSError,ValueError) as exc:
            reader.issue('invalid_evaluation_file',path,str(exc))
    return records


def evaluations(reader, cohort, manifest, checkpoints, records):
    rows = []
    for path,rec in records:
        if rec.get('run_id') != manifest['run_id']:
            continue
        row = {key: rec.get(key) for key in ('completed_update', 'split', 'evaluation_spec_hash',
                                           'checkpoint_hash', 'measurement_status')}
        row.update(source=str(path.relative_to(reader.root)), reported_reward_fraction=rec.get('reward_fraction'),
                   reward_fraction=None, status='invalid_record')
        u, k, n = rec.get('completed_update'), rec.get('responses_per_question'), rec.get('n_questions')
        counts = rec.get('per_question_correct')
        snap = checkpoints.get(u) if integer(u) else None
        reason = None
        if not integer(u) or rec.get('split') not in ('dev', 'final') or not digest(rec.get('evaluation_spec_hash')):
            reason = 'invalid update/split/evaluation identity'
        elif not snap or snap.get('sha256') != rec.get('checkpoint_hash') or snap['status'] not in {'verified_sha256','present_unhashed'}:
            reason = 'checkpoint not linked to the retained index and present file'
        elif not snap.get('training_lineage_observed'):
            reason = 'checkpoint is not linked to the surviving training lineage'
        elif not integer(k, 1) or not integer(n, 1) or not isinstance(counts, dict) or len(counts)!=n:
            reason = 'missing question-level integer counts or denominator'
        elif any(not integer(v) or v>k for v in counts.values()):
            reason = 'correct count outside [0, responses_per_question]'
        elif not finite(rec.get('reward_fraction')) or abs(sum(counts.values())/(k*n)-rec['reward_fraction'])>1e-9:
            reason = 'aggregate reward disagrees with question-level counts'
        if reason:
            reader.issue('invalid_evaluation', path, reason)
            row['reason'] = reason
        else:
            row.update(status='validated_record', reward_fraction=rec['reward_fraction'], n_questions=n,
                       responses_per_question=k, measurement_status=rec.get('measurement_status','not_recorded'),
                       checkpoint_verification=snap['status'],
                       per_question_counts_sha256=sha(json.dumps(counts,sort_keys=True).encode()))
        rows.append(row)
    # Different evaluation protocols stay separate. No best-result selection.
    keys, conflicts = {}, set()
    for row in rows:
        key = json.dumps([row.get('completed_update'),row.get('split'),row.get('evaluation_spec_hash')],sort_keys=True)
        if key in keys and {k:v for k,v in keys[key].items() if k!='source'} != {k:v for k,v in row.items() if k!='source'}:
            conflicts.add(key)
        keys[key] = row
    for key in conflicts:
        keys[key].update(status='conflicting', reward_fraction=None)
        reader.issue('conflicting_evaluation', cohort, str(key))
    return list(keys.values())


def export(root, *, every=25, verify_checkpoints=False):
    if not integer(every, 1):
        raise ValueError('every must be a positive integer')
    reader = Reader(root)
    if not reader.root.is_dir():
        raise ValueError(f'input directory not found: {reader.root}')
    manifests = sorted((reader.root/'runs').glob('*/manifest.json'))
    if not manifests:
        manifests = sorted(reader.root.glob('*/runs/*/manifest.json'))
    if not manifests:
        raise ValueError('no V8 run manifests found (expected <cohort>/runs/<run_id>/manifest.json)')
    manifests = sorted({path.resolve() for path in manifests})
    runs, evaluation_cache = [], {}
    for path in manifests:
        try:
            manifest = reader.json(path)
            if not isinstance(manifest.get('task',{}),dict):
                raise ValueError('task must be an object')
            rdir, cohort = path.parent, path.parent.parent.parent
            if any(key not in manifest for key in IDENTITY) or manifest['run_id']!=rdir.name or not integer(manifest['training_seed']):
                raise ValueError('manifest identity is incomplete or disagrees with run directory')
            training = training_rows(reader, rdir, manifest)
            indexed, collisions = unique(reader.lines(rdir/'snapshots.jsonl'), 'completed_update', reader, rdir)
            checkpoints = {u:checkpoint(reader,rdir,row,source,verify_checkpoints) for u,(row,source) in indexed.items()}
            lineage = {row['completed_update'] for row in training if row['lineage_observed']}
            for u,item in checkpoints.items():
                item['training_lineage_observed'] = u in lineage
            if cohort not in evaluation_cache:
                evaluation_cache[cohort] = evaluation_records(reader,cohort)
            ev = evaluations(reader,cohort,manifest,checkpoints,evaluation_cache[cohort])
            task = manifest.get('task', {})
            parent = task.get('parent', {}).get('completed_update',0)
            last_logged = max((row['completed_update'] for row in training),default=None)
            reported = manifest.get('completed_updates',manifest.get('last_completed_update'))
            observed = max(last_logged if last_logged is not None else parent, max(checkpoints,default=parent), reported if integer(reported) else parent)
            required = set(range(parent+every,observed+1,every))
            required.update(u for u in task.get('retain_updates',[]) if integer(u) and parent<u<=observed)
            if integer(task.get('horizon')) and parent<task['horizon']<=observed:
                required.add(task['horizon'])
            by_update = {row['completed_update']:row for row in training}
            coverage = []
            for u in sorted(required):
                valid = [row for row in ev if row.get('completed_update')==u and row['status']=='validated_record']
                def eval_status(split):
                    if any(row['split']==split for row in valid):return 'recorded'
                    if any(row.get('completed_update')==u and row.get('split')==split for row in ev):return 'invalid_record'
                    return 'unmeasured'
                coverage.append(dict(completed_update=u, training_status=by_update.get(u,{}).get('status','unmeasured'),
                                     checkpoint_status=checkpoints.get(u,{}).get('status','missing_index'),
                                     final_evaluation_status=eval_status('final'),
                                     development_evaluation_status=eval_status('dev'),
                                     final_evaluation_records=sum(row['split']=='final' for row in valid),
                                     development_evaluation_records=sum(row['split']=='dev' for row in valid)))
            missing = [c['completed_update'] for c in coverage if c['training_status']!='recorded' or
                       c['checkpoint_status'] not in {'present_unhashed','verified_sha256'}]
            # An atomic manifest replacement while reading means this export
            # cannot assert a single stable run view. Preserve it as incomplete.
            stable = sha(reader.safe(path).read_bytes())==reader.sources[str(path.relative_to(reader.root))]
            if not stable:
                reader.issue('manifest_changed',path,'Run manifest changed during export; rerun to obtain a stable view.')
            runs.append({**{k:manifest[k] for k in IDENTITY}, 'run_status':manifest.get('status','unknown'),
                         'source_manifest':str(path.relative_to(reader.root)), 'last_logged_update':last_logged,
                         'reported_completed_update':reported, 'observed_record_boundary':observed,
                         'parent_completed_update':parent, 'coverage_complete':not missing and stable,
                         'missing_recording_updates':missing, 'training':training, 'checkpoints':list(checkpoints.values()),
                         'evaluations':ev, 'coverage':coverage})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reader.issue('run_unavailable',path,str(exc))
    errors = sum(i['severity']=='error' for i in reader.issues)
    statuses = {}
    for run in runs:
        statuses[run['run_status']] = statuses.get(run['run_status'],0)+1
    return dict(schema='srgc-v8-live-history/v1', root=str(reader.root), exported_at=datetime.now(timezone.utc).isoformat(),
                every_updates=every, checkpoints_hashed=verify_checkpoints, runs=runs, sources_sha256=reader.sources,
                issues=reader.issues, run_status_counts=statuses, integrity_ok=errors==0, coverage_complete=bool(runs) and all(r['coverage_complete'] for r in runs),
                scope='Recorded training batches and separately recorded dev/final evaluation counts; no interpolation, no evaluation execution, no training launch. Snapshot payloads are never loaded; optimizer/RNG contents and remote model runtime are not verified. Missing held-out points remain unmeasured and do not require a synchronous evaluation. Coverage describes discovered runs through their recorded boundary, not completion of all scheduled or future runs; per-file hashes do not certify a globally atomic snapshot.')


def csv_text(rows, columns):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer,fieldnames=columns,extrasaction='ignore',lineterminator='\n')
    writer.writeheader()
    writer.writerows({key:json.dumps(value,sort_keys=True) if isinstance(value,(list,dict)) else value
                      for key,value in row.items()} for row in rows)
    return buffer.getvalue()


def write_outputs(report, output):
    output, root = Path(output).resolve(), Path(report['root']).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('output must be separate from the input tree (including ancestors and symlinks)')
    output.mkdir(parents=True,exist_ok=True)
    directory = output/('generation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:12])
    directory.mkdir()
    rows = {name:[] for name in ('training','checkpoints','coverage','final','dev')}
    for run in report['runs']:
        identity = {key:run[key] for key in IDENTITY}
        for name in ('training','checkpoints','coverage'):
            rows[name].extend(dict(identity,**row) for row in run[name])
        for row in run['evaluations']:
            if row.get('split') in ('final','dev'):
                rows[row['split']].append(dict(identity,**row))
    fields = list(IDENTITY)+['completed_update']
    text = {'history.json':json.dumps(report,indent=2,allow_nan=False)+'\n',
            'training-curve.csv':csv_text(rows['training'],fields+['reward_fraction','reward_sum','valid_response_count','status','reward_scope','inherited_prefix','lineage_observed','sources']),
            'heldout-final.csv':csv_text(rows['final'],fields+['reward_fraction','reported_reward_fraction','status','evaluation_spec_hash','checkpoint_hash','checkpoint_verification','measurement_status','n_questions','responses_per_question','per_question_counts_sha256','source']),
            'heldout-development.csv':csv_text(rows['dev'],fields+['reward_fraction','reported_reward_fraction','status','evaluation_spec_hash','checkpoint_hash','checkpoint_verification','measurement_status','n_questions','responses_per_question','per_question_counts_sha256','source']),
            'checkpoint-index.csv':csv_text(rows['checkpoints'],fields+['path','sha256','actual_sha256','status','bytes','payload_loaded','training_lineage_observed','source']),
            'recording-coverage.csv':csv_text(rows['coverage'],fields+['training_status','checkpoint_status','final_evaluation_status','development_evaluation_status','final_evaluation_records','development_evaluation_records'])}
    for name, body in text.items():
        with (directory/name).open('x') as handle:
            handle.write(body);handle.flush();os.fsync(handle.fileno())
    manifest = {'schema':report['schema'], 'generation':directory.name, 'files_sha256':{name:sha(body.encode()) for name,body in text.items()},
                'integrity_ok':report['integrity_ok'], 'coverage_complete':report['coverage_complete']}
    body = json.dumps(manifest,indent=2)+'\n'
    with (directory/'manifest.json').open('x') as handle:
        handle.write(body);handle.flush();os.fsync(handle.fileno())
    def sync_directory(path):
        descriptor = os.open(path,os.O_RDONLY | os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    sync_directory(directory)
    sync_directory(output)
    pointer = json.dumps({'generation':directory.name,'manifest_sha256':sha(body.encode())},indent=2)+'\n'
    fd, temporary = tempfile.mkstemp(prefix='.latest-',dir=output)
    try:
        with os.fdopen(fd,'w') as handle:
            handle.write(pointer);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,output/'latest.json')
        sync_directory(output)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,help='One cohort directory or the parent V8_ROOT')
    parser.add_argument('--output',type=Path,required=True,help='Separate export directory; prior generations are preserved')
    parser.add_argument('--every',type=int,default=25)
    parser.add_argument('--verify-checkpoints',action='store_true',help='Stream checkpoint SHA256; default checks paths/files only')
    parser.add_argument('--strict',action='store_true',help='Exit3 for invalid records or incomplete training/checkpoint coverage; skipped held-out evaluations are allowed')
    args = parser.parse_args(argv)
    try:
        report = export(args.root,every=args.every,verify_checkpoints=args.verify_checkpoints)
        directory = write_outputs(report,args.output)
    except (OSError,ValueError) as exc:
        parser.exit(2,f'{exc}\n')
    print(json.dumps({'generation':str(directory),'runs':len(report['runs']),'integrity_ok':report['integrity_ok'],
                      'coverage_complete':report['coverage_complete'],'run_status_counts':report['run_status_counts'],'issues':len(report['issues'])},indent=2))
    return 3 if args.strict and not (report['integrity_ok'] and report['coverage_complete']) else 0


if __name__=='__main__':
    raise SystemExit(main())
