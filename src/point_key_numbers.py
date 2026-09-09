"""KEY NUMBERS lines for scored experiment points, shared by every status tool.

    python src/point_key_numbers.py --root <runs root>      # every scored point under the root
    python src/point_key_numbers.py <point dir> [<point dir> ...]

One line per scored point and scoring, with the numbers the paper's gates
read: split-half floor against 2*chance (point estimate; the paper gate uses
the 95% lower bound), the fresh selector's precision, the four stale
precisions, token KL and trajectory ESS. 'current' rows come from the point
root, 'pinned' rows from the scoring parked by scripts/rescore_math500.sh.
The OLMo status (src/rlzero_status.py) and the Qwen status
(scripts/status_qwen35.sh) both print these lines, so the routine status
upload carries the numbers off a cluster that cannot push.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

POINT_NAME = re.compile(r"-s(\d+)-([a-z0-9]+)-d(\d+)$")


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _fmt(value, digits: int = 3) -> str:
    if value is None or isinstance(value, bool):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def scorings_of(run: Path) -> list[tuple[str, Path]]:
    """('current', run) when the root holds a report, then ('pinned', newest parked scoring)."""
    scorings = []
    if _nonempty(run / "report.json"):
        scorings.append(("current", run))
    parking = run / "pinned-scoring"
    parked = sorted(p for p in parking.glob("*") if p.is_dir()) if parking.is_dir() else []
    if parked and _nonempty(parked[-1] / "report.json"):
        scorings.append(("pinned", parked[-1]))
    return scorings


def point_lines(run: Path, dataset: str, seed: int, drift: int) -> list[str]:
    """KEY NUMBERS lines for one point (empty when it has no scoring yet)."""
    scorings = scorings_of(run)
    if not scorings:
        return []
    config = _load(run / "run_config.json") or {}
    lines = []
    for kind, where in scorings:
        report = _load(where / "report.json") or {}
        div = _load(where / "divergence_stats.json") or {}
        n = config.get("n_train")
        k = report.get("k")
        chance = k / n if isinstance(n, (int, float)) and n and isinstance(k, (int, float)) else None
        floor = report.get("noise_floor")
        gate = "-       "
        if isinstance(floor, (int, float)) and not isinstance(floor, bool) and chance is not None:
            gate = "GATE-OK " if floor >= 2 * chance else "gate-LOW"
        fresh = (report.get("certagrad") or {}).get("precision_vs_oracle")
        stale = " ".join(
            f"{e}={_fmt((report.get(e) or {}).get('precision'))}" for e in ("g00", "g01", "g10", "g11")
        )
        lines.append(
            f" {dataset} s{seed} d{drift} {kind:<7} floor={_fmt(floor)} {gate} "
            f"fresh={_fmt(fresh)} {stale} KL={_fmt(div.get('token_kl_beta_pi'), 6)} "
            f"ESS={_fmt(div.get('traj_ess_frac_g11'))}"
        )
    return lines


def parse_point_name(name: str) -> tuple[str, int, int] | None:
    match = POINT_NAME.search(name)
    if not match:
        return None
    return match.group(2), int(match.group(1)), int(match.group(3))


def discover_points(root: Path, max_depth: int = 3) -> list[tuple[str, int, int, Path]]:
    """Scored points under `root`, at most `max_depth` levels down, ordered by dataset, seed, drift."""
    found = []
    if not root.is_dir():
        return found
    stack = [(root, 0)]
    while stack:
        folder, depth = stack.pop()
        try:
            children = [p for p in folder.iterdir() if p.is_dir()]
        except OSError:
            continue
        for child in children:
            parsed = parse_point_name(child.name)
            if parsed and scorings_of(child):
                found.append((parsed[0], parsed[1], parsed[2], child))
            elif depth + 1 < max_depth:
                stack.append((child, depth + 1))
    return sorted(found, key=lambda item: (item[0], item[1], item[2]))


HEADER = (" KEY NUMBERS per scored point: floor vs 2*chance is the point-estimate check "
          "(the paper gate uses its 95% lower bound); 'pinned' = scoring parked by rescoring")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("points", nargs="*", type=Path, help="point directories")
    parser.add_argument("--root", type=Path, default=None, help="scan this runs root for scored points")
    parser.add_argument("--no-header", action="store_true")
    args = parser.parse_args(argv)
    items = []
    if args.root is not None:
        items.extend(discover_points(args.root))
    for point in args.points:
        parsed = parse_point_name(point.name)
        if parsed is None:
            print(f"[skip] {point}: name does not end in -s<seed>-<dataset>-d<drift>", file=sys.stderr)
            continue
        items.append((parsed[0], parsed[1], parsed[2], point))
    lines = []
    for dataset, seed, drift, run in items:
        lines.extend(point_lines(run, dataset, seed, drift))
    if not args.no_header:
        print(HEADER)
    if lines:
        print("\n".join(lines))
    else:
        print(" (no scored point yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
