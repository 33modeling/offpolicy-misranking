"""Read-only resolution of the state point used by the GPU worker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SUITE_SCHEMA = "offpolicy-selection-gate-gpu/one-shot-v1"


def resolve_state_point(child, step):
    """Return ``(point_path, error)``; an error must never be shown as READY.

    Published states bind their point in suite.json, just as the worker's
    selection_gate_gpu.entries does. Directory ordering is not authority.
    Unpublished/legacy single-point states retain the previous fallback.
    The returned fallback path on error is for display only, not permission to
    start work or evidence that existing results are absent.
    """
    child = Path(child)
    expected = child / "points" / f"view-{step}"
    suite_path = child / "suite.json"
    try:
        if suite_path.exists() or suite_path.is_symlink():
            suite = json.loads(suite_path.read_text())
            if not isinstance(suite, dict) or suite.get("schema") != SUITE_SCHEMA:
                raise ValueError("unsupported GPU gate suite")
            items = suite.get("points")
            if not isinstance(items, list) or len(items) != 1:
                raise ValueError("switch state suite must bind exactly one point")
            item = items[0]
            name = item.get("name") if isinstance(item, dict) else None
            if (not isinstance(name, str) or not name or name in {".", ".."}
                    or Path(name).name != name):
                raise ValueError("invalid point path in suite.json")
            out = child / "points" / name
            if hashlib.sha256((out / "contract.json").read_bytes()).hexdigest() != item.get("sha256"):
                raise ValueError("point contract changed")
            return out, ""

        points_root = child / "points"
        if not points_root.exists():
            if points_root.is_symlink():
                raise ValueError("state points link is unavailable; saved work cannot be inspected")
            return expected, ""
        points = sorted(path for path in points_root.iterdir() if path.is_dir() or path.is_symlink())
        if any(not point.is_dir() for point in points):
            raise ValueError("state point link is unavailable; saved work cannot be inspected")
        if len(points) > 1:
            names = ", ".join(point.name for point in points)
            raise ValueError(f"multiple state points without suite authority: {names}")
        return (points[0] if points else expected), ""
    except (OSError, ValueError, TypeError) as exc:
        return expected, str(exc)
