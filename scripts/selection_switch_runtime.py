#!/usr/bin/env python3
"""Run a switch launcher from an isolated, verified local Git checkout."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


CODE_PATHS = ("src", "scripts", "config", "configs")
PATH_OPTIONS = {"--matrix", "--pool", "--pool-manifest", "--eval-prompts"}
LAUNCHERS = {"switch": "run_selection_switch.sh", "mopps": "run_mopps_comparison.sh"}


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repo), *args],
        text=True, stderr=subprocess.PIPE).strip()


def clean_revision(repo):
    revision = git(repo, "rev-parse", "HEAD")
    dirty = git(repo, "status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS)
    if dirty:
        raise ValueError(f"runtime source is dirty; nothing was discarded: {repo}\n{dirty}")
    if git(repo, "rev-parse", "HEAD") != revision:
        raise ValueError("checkout changed during runtime preflight; retry after the update finishes")
    return revision


def snapshot(repo, cache):
    repo, cache = repo.resolve(), cache.resolve()
    if cache == repo or cache.is_relative_to(repo):
        raise ValueError("runtime cache must be outside the live repository")
    revision = clean_revision(repo)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / revision
    with (cache / ".snapshot.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            if target.is_symlink() or clean_revision(target) != revision:
                raise ValueError(f"cached runtime changed; existing workers and cache were not modified: {target}")
        else:
            temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=cache))
            try:
                subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "clone", "--quiet",
                                "--no-hardlinks", "--no-checkout", str(repo), str(temporary)],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                git(temporary, "checkout", "--quiet", "--detach", revision)
                git(temporary, "remote", "remove", "origin")
                if clean_revision(temporary) != revision:
                    raise ValueError("new runtime clone does not match the selected commit")
                temporary.rename(target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    return target, revision


def runtime_arguments(arguments, repo):
    result = []
    path_next = False
    for value in arguments:
        if path_next:
            result.append(str((repo / value).absolute()))
            path_next = False
        elif value in PATH_OPTIONS:
            result.append(value)
            path_next = True
        elif "=" in value and value.split("=", 1)[0] in PATH_OPTIONS:
            key, path = value.split("=", 1)
            result.append(f"{key}={(repo / path).absolute()}")
        else:
            result.append(value)
    return result


def launch(repo, cache, kind, arguments):
    repo = repo.resolve()
    target, revision = snapshot(repo, cache)
    launcher = target / "scripts" / LAUNCHERS[kind]
    if not launcher.is_file():
        raise ValueError(f"launcher is absent from the selected commit: {launcher}")
    env = dict(os.environ)
    for key in ("OM_WORK", "VENV_DIR", "MODELS_DIR", "DATASETS_DIR", "OM_OLMO3_ROOT", "SWITCH_ROOT", "MOPPS_ROOT"):
        if env.get(key):
            env[key] = str((repo / env[key]).absolute())
    old_sources = {str(repo / "src")}
    if env.get("OM_REPO"):
        old_sources.add(str(Path(env["OM_REPO"]).absolute() / "src"))
    paths = [path for path in env.get("PYTHONPATH", "").split(os.pathsep)
             if path and str(Path(path).absolute()) not in old_sources]
    env.update(OM_REPO=str(target), SWITCH_RUNTIME_REPO=str(target), SWITCH_RUNTIME_COMMIT=revision,
               SWITCH_PYTHON=sys.executable, MOPPS_PYTHON=sys.executable,
               PYTHONPATH=os.pathsep.join([str(target / "src"), *paths]), PYTHONDONTWRITEBYTECODE="1")
    # Shell entrypoints resolve these before re-entry; keep output storage unchanged.
    if env.get("OUT_ROOT"):
        env["SWITCH_ROOT" if kind == "switch" else "MOPPS_ROOT"] = env["OUT_ROOT"]
    print(f"[runtime] commit={revision} source={repo} pinned={target}", flush=True)
    os.execvpe("bash", ["bash", str(launcher), *runtime_arguments(arguments, repo)], env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--kind", choices=LAUNCHERS, default="switch")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    try:
        launch(args.repo, args.cache, args.kind, arguments)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None)
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        parser.exit(2, f"[runtime preflight failed] {exc}\n{detail or ''}\n")


if __name__ == "__main__":
    main()
