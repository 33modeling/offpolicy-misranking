"""Package isolation, dependency-free status and failed-start diagnostics."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from srgc_research.dispatch.gemma4 import cli, entry, runtime
from srgc_research.tests.test_gemma4 import prepared


def fake_install(command, **kwargs):
    target = Path(command[command.index('--target') + 1])
    for name, version in runtime.PACKAGES.items():
        folder = target / f"{name.replace('-', '_')}-{version}.dist-info"
        folder.mkdir()
        (folder / 'METADATA').write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")


def test_packages_are_installed_once_outside_existing_venvs(tmp_path):
    env = {'GROUP_VOLUME': str(tmp_path), 'OM_WORK': str(tmp_path/'work')}
    preserved = tmp_path/'work/.venv-cu126/untouched'
    preserved.parent.mkdir(parents=True)
    preserved.write_bytes(b'existing Llama runtime')
    with patch.object(runtime.subprocess, 'run', side_effect=fake_install) as install:
        target = runtime.package_overlay(env)
        assert runtime.package_overlay(env) == target
        install.assert_called_once()
        command = install.call_args.args[0]
        assert '--no-deps' in command and '--only-binary=:all:' in command
        assert not any(argument.startswith(('torch==', 'torchvision', 'torchaudio')) for argument in command)
    assert target.is_relative_to(tmp_path/'work/gemma-runtime-cache/packages')
    assert preserved.read_bytes() == b'existing Llama runtime'


def test_failed_install_does_not_publish_partial_overlay(tmp_path):
    env = {'GROUP_VOLUME': str(tmp_path), 'OM_WORK': str(tmp_path/'work')}
    with patch.object(runtime.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['pip'])), pytest.raises(subprocess.CalledProcessError):
        runtime.package_overlay(env)
    assert not list((tmp_path/'work/gemma-runtime-cache/packages').rglob('.runtime.json'))
    assert not list((tmp_path/'work/gemma-runtime-cache/packages').glob('.*.tmp-*'))


def test_changed_package_receipt_is_rejected(tmp_path):
    env = {'GROUP_VOLUME': str(tmp_path), 'OM_WORK': str(tmp_path/'work')}
    with patch.object(runtime.subprocess, 'run', side_effect=fake_install):
        target = runtime.package_overlay(env)
    metadata = next(target.glob('transformers-*.dist-info/METADATA'))
    metadata.write_text('Metadata-Version: 2.1\nName: transformers\nVersion: 0.0.0\n')
    with pytest.raises(ValueError, match='packages changed'):
        runtime.package_overlay(env)


def test_status_shortcut_does_not_install_import_gpu_or_touch_storage(tmp_path, capsys):
    root = tmp_path/'work/srgc-rebuttal/gemma4-12b-pt-v1'
    for name in ('math', 'mbpp'):
        prepared(root, name)
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    with patch.dict(os.environ, {'GROUP_VOLUME': str(tmp_path), 'OM_WORK': str(tmp_path/'work')}), patch.dict(sys.modules, {'torch': None, 'transformers': None, 'peft': None}), patch.object(runtime, 'activate') as install:
        assert entry.main(['status', '--root', str(root)]) == 0
        install.assert_not_called()
    assert 'Gemma-4-12B-PT' in capsys.readouterr().out
    assert {path: path.read_bytes() for path in before} == before


def test_invalid_arguments_and_missing_model_never_install(tmp_path):
    with patch.object(runtime, 'activate') as install, pytest.raises(SystemExit):
        entry.main(['all', 'invalid-action'])
    install.assert_not_called()
    with patch.dict(os.environ, {'GROUP_VOLUME': str(tmp_path), 'OM_WORK': str(tmp_path/'work')}), patch.object(runtime, 'activate') as install, pytest.raises(ValueError, match='No model download'):
        entry.main(['all'])
    install.assert_not_called()


def test_missing_source_plan_child_error_is_printed_inline(tmp_path, capsys):
    def child(command, log, *args, **kwargs):
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('FileNotFoundError: actual missing cohort.json\n')
        return 1
    with patch.object(cli, 'resolve_snapshot', return_value=(tmp_path, 'a'*64)), patch.object(cli.adapter, 'runtime_packages'), patch('srgc_rebuttal.cluster.run_child', side_effect=child), pytest.raises(subprocess.CalledProcessError):
        cli.prepare_missing(('math',), tmp_path, {})
    text = capsys.readouterr().out
    assert 'GEMMA FAILURE DETAILS' in text and 'actual missing cohort.json' in text


def test_broken_optional_media_is_not_imported_in_text_process(tmp_path):
    fake = tmp_path/'torchvision'
    fake.mkdir()
    (fake/'__init__.py').write_text("raise RuntimeError('operator torchvision::nms does not exist')\n")
    info = tmp_path/'torchvision-0.21.0.dist-info'
    info.mkdir()
    (info/'METADATA').write_text('Metadata-Version: 2.1\nName: torchvision\nVersion: 0.21.0\n')
    script = '''
from srgc_research.dispatch.gemma4.runtime import disable_optional_media
disable_optional_media()
from transformers import Gemma4UnifiedForCausalLM
import sys
assert 'torchvision' not in sys.modules
print('PASS: text Gemma imports without torchvision')
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[2], env={**os.environ, 'PYTHONPATH': str(tmp_path)+os.pathsep+os.environ.get('PYTHONPATH', '')}, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert 'PASS:' in result.stdout
