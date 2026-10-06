"""Save validated report snapshots without rereading or modifying live endpoints."""

import json
from pathlib import Path
import re
import sys
import time
import uuid

from srgc_rebuttal.runtime import atomic_json


def publish(report, scope):
    if scope not in {"mechanism", "support", "switch_validation"}:
        raise ValueError("unknown result collection scope")
    roots = set()
    for row in report["rows"]:
        folder = Path(row["output"]).resolve()
        if not re.fullmatch(r"seed-[0-9]+", folder.name):
            raise ValueError(f"invalid seed output directory: {folder}")
        if folder.parent.is_dir():
            roots.add(folder.parent)
    sources = report.get("source_results", {})
    resolved_sources = {}
    for name, value in sources.items():
        path = Path(name).resolve()
        if not any(path.is_relative_to(root) for root in roots):
            raise ValueError(f"result source outside verified run directories: {path}")
        resolved_sources[path] = value
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:12]
    report["collections"] = []
    for root in sorted(roots):
        destination = root / "results" / scope
        bundle = destination / "exports" / stamp
        report["collections"].append(dict(
            output_root=str(root), report_path=str(destination / "results.json"),
            bundle_report_path=str(bundle / "results.json"), directory=str(bundle),
            file_count=sum(path.is_relative_to(root) for path in resolved_sources)))
    # Validate all JSON before publishing any latest pointer. Each report embeds
    # the full multi-root snapshot; individual copies stay under their own root.
    json.dumps(report, allow_nan=False)
    for collection in report["collections"]:
        root = Path(collection["output_root"])
        bundle = Path(collection["directory"])
        for source, value in resolved_sources.items():
            if source.is_relative_to(root):
                atomic_json(bundle / "raw" / source.relative_to(root), value)
        atomic_json(Path(collection["bundle_report_path"]), report)
    for collection in report["collections"]:
        atomic_json(Path(collection["report_path"]), report)
    return report


def print_paths(report, *, json_output=False):
    stream = sys.stderr if json_output else sys.stdout
    for collection in report.get("collections", []):
        print(f"COLLECTED JSON: {collection['report_path']}", file=stream)
        print(f"COLLECTED FILES: {collection['directory']} ({collection['file_count']} originals)", file=stream)
