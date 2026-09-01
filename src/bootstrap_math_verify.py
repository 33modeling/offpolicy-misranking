#!/usr/bin/env python3
"""Install the pinned pure-Python math verifier bundle without network access."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHEEL_DIR = ROOT / "vendor" / "wheels"
WHEELS = {
    "antlr4_python3_runtime-4.13.2-py3-none-any.whl": (
        "fe3835eb8d33daece0e799090eda89719dbccee7aa39ef94eed3818cafa5a7e8"
    ),
    "latex2sympy2_extended-1.11.0-py3-none-any.whl": (
        "aebb77d52ce269e25028e4bea89ddb14d242ba36bcf7b636496fb5fd9728d234"
    ),
    "math_verify-0.9.0-py3-none-any.whl": (
        "3703e7c4885354027fa84409d762a596a2906d1fd4deb78361876bd905a76194"
    ),
    "mpmath-1.3.0-py3-none-any.whl": (
        "a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c"
    ),
    "sympy-1.14.0-py3-none-any.whl": (
        "e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5"
    ),
}
BUNDLE_ID = hashlib.sha256(
    json.dumps(WHEELS, sort_keys=True).encode("utf-8")
).hexdigest()[:16]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_wheels() -> None:
    for name, expected in WHEELS.items():
        wheel = WHEEL_DIR / name
        if not wheel.is_file():
            raise ValueError(f"bundled dependency missing: {wheel}")
        if _sha256(wheel) != expected:
            raise ValueError(f"bundled dependency hash mismatch: {wheel}")


def _extract(target: Path) -> None:
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        for name in WHEELS:
            with zipfile.ZipFile(WHEEL_DIR / name) as archive:
                for member in archive.infolist():
                    relative = Path(member.filename)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(
                            f"unsafe path in bundled wheel: {member.filename}"
                        )
                archive.extractall(temporary)
        (temporary / ".bundle.json").write_text(
            json.dumps({"bundle_id": BUNDLE_ID, "wheels": WHEELS}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _smoke(target: Path) -> None:
    sys.path.insert(0, str(target))
    import antlr4
    import latex2sympy2_extended
    import mpmath
    import sympy
    from math_verify import parse, verify

    for module in (
        sys.modules["math_verify"],
        latex2sympy2_extended,
        antlr4,
        sympy,
        mpmath,
    ):
        module_path = Path(module.__file__).resolve()
        if target.resolve() not in module_path.parents:
            raise RuntimeError(f"dependency loaded outside bundled runtime: {module_path}")
    if not verify(parse(r"\frac{1}{2}"), parse("0.5")):
        raise RuntimeError("bundled math_verify equivalence smoke test failed")


def bootstrap(cache_root: Path) -> Path:
    _verify_wheels()
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / f"math-verify-{BUNDLE_ID}"
    lock_path = cache_root / ".math-verify-bootstrap.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stamp = target / ".bundle.json"
        valid_stamp = False
        try:
            valid_stamp = json.loads(stamp.read_text(encoding="utf-8")) == {
                "bundle_id": BUNDLE_ID,
                "wheels": WHEELS,
            }
        except (OSError, json.JSONDecodeError):
            pass
        if valid_stamp:
            try:
                _smoke(target)
                return target
            except (ImportError, OSError, RuntimeError):
                quarantine = cache_root / f".{target.name}.invalid.{time.time_ns()}"
                target.replace(quarantine)
        else:
            if target.exists():
                quarantine = cache_root / f".{target.name}.invalid.{time.time_ns()}"
                target.replace(quarantine)
        _extract(target)
        _smoke(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    print(bootstrap(args.cache_root.resolve()))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"[math-verify-bootstrap-abort] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
