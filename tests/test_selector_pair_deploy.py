"""Pinned runtime deployment never updates or repairs a live working tree."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/selector_pair_deploy.py'


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def commit(repo, message):
    git(repo, 'add', '.')
    git(repo, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-qm', message)
    return git(repo, 'rev-parse', 'HEAD')


@pytest.fixture
def deploy():
    spec = importlib.util.spec_from_file_location('pair_deploy_tests', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(tmp_path, deploy, monkeypatch):
    upstream, checkout = tmp_path / 'upstream', tmp_path / 'checkout'
    upstream.mkdir()
    git(upstream, 'init', '-q')
    for directory in ('src', 'scripts', 'vendor'):
        (upstream / directory).mkdir()
    (upstream / '.gitignore').write_text('.work/\n')
    (upstream / 'src/selector_pair_gpu.py').write_text('LEGACY_EX = True\n')
    (upstream / 'scripts/run_selector_pair.sh').write_text('#!/bin/bash\necho old\n')
    (upstream / 'vendor/fixture.whl').write_bytes(b'fixture vendor bytes')
    (upstream / 'requirements.txt').write_text('fixture-package==1\n')
    old = commit(upstream, 'legacy')
    git(tmp_path, 'clone', '-q', str(upstream), str(checkout))
    (upstream / 'src/selector_pair_gpu.py').write_text('DISTRIBUTED_QUEUE = True\n')
    (upstream / 'scripts/run_selector_pair.sh').write_text('#!/bin/bash\necho queue\n')
    pinned = commit(upstream, 'reviewed queue')
    (upstream / 'src/selector_pair_gpu.py').write_text('UNREVIEWED = True\n')
    newer = commit(upstream, 'not approved')
    monkeypatch.setattr(deploy, 'PINNED_COMMIT', pinned)
    return checkout, upstream, old, pinned, newer


def evidence(repo):
    return {
        'head': git(repo, 'rev-parse', 'HEAD'),
        'refs': git(repo, 'for-each-ref', '--format=%(refname) %(objectname)'),
        'status': git(repo, 'status', '--porcelain'),
        'files': {str(path.relative_to(repo)): (path.read_bytes(), path.stat().st_mtime_ns,
                                               path.stat().st_ino)
                  for directory in ('src', 'scripts', 'vendor')
                  for path in (repo / directory).rglob('*') if path.is_file()},
    }


def test_old_dirty_checkout_preserved_but_reviewed_queue_staged(prepared, deploy):
    repo, _, old, pinned, newer = prepared
    (repo / 'src/selector_pair_gpu.py').write_text('USER_UNCOMMITTED = True\n')
    (repo / 'src/user-note.txt').write_text('preserve this untracked file\n')
    before = evidence(repo)
    target = deploy.stage_runtime(repo)
    assert target == repo / '.work/pair-runtimes' / pinned
    assert target.name != newer
    assert (target / 'src/selector_pair_gpu.py').read_text() == 'DISTRIBUTED_QUEUE = True\n'
    assert not (target / 'src/user-note.txt').exists()
    assert (target / 'vendor/fixture.whl').read_bytes() == b'fixture vendor bytes'
    assert evidence(repo) == before
    assert git(repo, 'rev-parse', 'HEAD') == old
    assert not (repo / '.git/FETCH_HEAD').exists()


def test_clean_cache_reused_without_file_rewrites(prepared, deploy):
    repo, *_ = prepared
    target = deploy.stage_runtime(repo)
    before = {str(path): (path.stat().st_ino, path.stat().st_mtime_ns)
              for path in target.rglob('*')}
    assert deploy.stage_runtime(repo) == target
    assert before == {str(path): (path.stat().st_ino, path.stat().st_mtime_ns)
                      for path in target.rglob('*')}


def test_corrupt_cache_refused_without_repair(prepared, deploy):
    repo, *_ = prepared
    target = deploy.stage_runtime(repo)
    source = target / 'src/selector_pair_gpu.py'
    source.write_text('corrupted active runtime\n')
    before = source.read_bytes(), source.stat().st_ino, source.stat().st_mtime_ns
    with pytest.raises(RuntimeError, match='changed'):
        deploy.stage_runtime(repo)
    assert before == (source.read_bytes(), source.stat().st_ino, source.stat().st_mtime_ns)


def test_forged_manifest_cannot_bless_changed_code(prepared, deploy):
    repo, *_ = prepared
    target = deploy.stage_runtime(repo)
    source = target / 'src/selector_pair_gpu.py'
    source.write_text('corrupted active runtime\n')
    manifest = target / deploy.MANIFEST
    content = json.loads(manifest.read_text())
    content['files']['src/selector_pair_gpu.py']['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    content['files']['src/selector_pair_gpu.py']['size'] = source.stat().st_size
    manifest.write_text(json.dumps(content))
    with pytest.raises(RuntimeError, match='changed'):
        deploy.stage_runtime(repo)


def test_fetch_failure_never_stages_or_changes_working_tree(prepared, deploy):
    repo, *_ = prepared
    git(repo, 'remote', 'set-url', 'origin', str(repo / 'missing-remote'))
    before = evidence(repo)
    with pytest.raises(RuntimeError, match='fetch failed'):
        deploy.stage_runtime(repo)
    assert evidence(repo) == before
    assert not (repo / '.work').exists()


def test_pinned_symlink_refused_before_staging(prepared, deploy, monkeypatch):
    repo, upstream, *_ = prepared
    (upstream / 'src/symlink.py').symlink_to('/etc/passwd')
    monkeypatch.setattr(deploy, 'PINNED_COMMIT', commit(upstream, 'unsafe fixture'))
    with pytest.raises(RuntimeError, match='symlink'):
        deploy.stage_runtime(repo)
    assert not (repo / '.work').exists()


def test_symlinked_cache_preserved_and_refused(prepared, deploy, tmp_path):
    repo, *_ = prepared
    external = tmp_path / 'unrelated'
    external.mkdir()
    (external / 'keep').write_text('preserve')
    (repo / '.work').symlink_to(external, target_is_directory=True)
    with pytest.raises(RuntimeError, match='symlinked'):
        deploy.stage_runtime(repo)
    assert sorted(path.name for path in external.iterdir()) == ['keep']
    assert (external / 'keep').read_text() == 'preserve'


def test_existing_empty_runtime_is_not_overwritten(prepared, deploy):
    repo, _, _, pinned, _ = prepared
    target = repo / '.work/pair-runtimes' / pinned
    target.mkdir(parents=True)
    inode = target.stat().st_ino
    with pytest.raises(RuntimeError, match='missing'):
        deploy.stage_runtime(repo)
    assert target.stat().st_ino == inode
    assert list(target.iterdir()) == []


def test_unexpected_cache_file_refused(prepared, deploy):
    repo, *_ = prepared
    target = deploy.stage_runtime(repo)
    (target / 'src/extra.py').write_text('unapproved')
    with pytest.raises(RuntimeError, match='changed'):
        deploy.stage_runtime(repo)
    assert (target / 'src/extra.py').read_text() == 'unapproved'


def test_archive_substitution_cannot_replace_pinned_blob(prepared, deploy, monkeypatch):
    repo, upstream, *_ = prepared
    (upstream / '.gitattributes').write_text('src/selector_pair_gpu.py export-subst\n')
    (upstream / 'src/selector_pair_gpu.py').write_text('"$Format:%H$"\n')
    monkeypatch.setattr(deploy, 'PINNED_COMMIT', commit(upstream, 'archive substitution fixture'))
    with pytest.raises(RuntimeError, match='differs from'):
        deploy.stage_runtime(repo)


def test_runtime_size_limit_precedes_cache_creation(prepared, deploy, monkeypatch):
    repo, *_ = prepared
    monkeypatch.setattr(deploy, 'MAX_ARCHIVE_BYTES', 1)
    with pytest.raises(RuntimeError, match='size limit'):
        deploy.stage_runtime(repo)
    assert not (repo / '.work').exists()


def test_git_output_limit_is_enforced(prepared, deploy):
    repo, *_ = prepared
    with pytest.raises(RuntimeError, match='output limit'):
        deploy.git(repo, 'show', 'HEAD:src/selector_pair_gpu.py', limit=1)


@pytest.mark.parametrize('name', ['/absolute', '../escape', 'src/../escape', 'src//alias', './alias', 'src\\escape'])
def test_unsafe_member_names_rejected(deploy, name):
    with pytest.raises(RuntimeError, match='unsafe path'):
        deploy.safe_name(name)


def test_atomic_publication_does_not_replace_even_empty_directory(deploy, tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir()
    target.mkdir()
    inode = target.stat().st_ino
    with pytest.raises(RuntimeError, match='appeared during staging'):
        deploy.publish(source, target)
    assert target.stat().st_ino == inode
    assert source.is_dir()
