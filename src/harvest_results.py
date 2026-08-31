#!/usr/bin/env python3
"""Validate and package the canonical 27B/7B RLVR result matrices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import re
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ANALYSIS_FILES = (
    "REGIME.json",
    "REGIME.csv",
    "REGIME_SUMMARY.csv",
    "FINAL_REPORT.md",
)
MARKER = ".regime_analysis.key"
REGIME_SCHEMA = "offpolicy-regime-map/v2"
HARVEST_SCHEMA = "offpolicy-rlvr-harvest/v2"
EXPECTED_DATASETS = ("gsm8k", "math500")
EXPECTED_DRIFTS = (0, 25, 100, 400)
EXPECTED_POLICIES = {
    "stale_g00",
    "stale_g10",
    "stale_g01",
    "stale_g11",
    "passrate_beta",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class HarvestError(ValueError):
    """A source matrix or published bundle is not safe to harvest."""


@dataclass(frozen=True)
class AnalysisSnapshot:
    root: Path
    files: dict[str, bytes]
    document: dict
    csv_rows: list[dict[str, str]]
    model: str


def _read_regular(path: Path) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise HarvestError(f"missing analysis artifact: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        raise HarvestError(f"analysis artifact must be a nonempty regular file: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise HarvestError(f"cannot read analysis artifact: {path}") from exc


def _parse_marker(root: Path, payload: bytes) -> dict[str, str]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise HarvestError(f"{root / MARKER}: marker is not UTF-8") from exc
    if len(lines) != 1 + len(ANALYSIS_FILES) or not SHA256_RE.fullmatch(lines[0]):
        raise HarvestError(f"{root / MARKER}: invalid report-cache marker structure")

    recorded: dict[str, str] = {}
    expected_paths = {(root / name).resolve(): name for name in ANALYSIS_FILES}
    for line in lines[1:]:
        match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
        if match is None:
            raise HarvestError(f"{root / MARKER}: malformed sha256 entry")
        raw_path = Path(match.group(3))
        resolved = (
            raw_path.resolve()
            if raw_path.is_absolute()
            else (Path.cwd() / raw_path).resolve()
        )
        name = expected_paths.get(resolved)
        if name is None or name in recorded:
            raise HarvestError(
                f"{root / MARKER}: unexpected or duplicate artifact path"
            )
        recorded[name] = match.group(1)
    if set(recorded) != set(ANALYSIS_FILES):
        raise HarvestError(
            f"{root / MARKER}: marker does not bind every analysis output"
        )
    return recorded


def _read_coherent_files(root: Path) -> dict[str, bytes]:
    last_error: HarvestError | None = None
    for attempt in range(3):
        marker_before = _read_regular(root / MARKER)
        files = {name: _read_regular(root / name) for name in ANALYSIS_FILES}
        marker_after = _read_regular(root / MARKER)
        try:
            if marker_before != marker_after:
                raise HarvestError(f"{root}: analysis marker changed while harvesting")
            recorded = _parse_marker(root, marker_after)
            for name, payload in files.items():
                actual = hashlib.sha256(payload).hexdigest()
                if actual != recorded[name]:
                    raise HarvestError(
                        f"{root / name}: content differs from analysis marker"
                    )
            return {**files, MARKER: marker_after}
        except HarvestError as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.05)
    assert last_error is not None
    raise last_error


def _reject_nonfinite(value: object, location: str = "REGIME.json") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise HarvestError(f"{location}: non-finite numeric value")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{location}[{index}]")


def _csv_value(value: object) -> str:
    return "" if value is None else str(value)


def _validate_csv(payload: bytes, rows: list[dict], name: str) -> list[dict[str, str]]:
    try:
        stream = io.StringIO(payload.decode("utf-8"), newline="")
    except UnicodeDecodeError as exc:
        raise HarvestError(f"{name}: CSV is not UTF-8") from exc
    reader = csv.DictReader(stream)
    if not rows:
        raise HarvestError(f"{name}: corresponding JSON rows are empty")
    expected_fields = list(rows[0])
    if reader.fieldnames != expected_fields:
        raise HarvestError(f"{name}: header differs from REGIME.json")
    parsed = list(reader)
    if len(parsed) != len(rows):
        raise HarvestError(f"{name}: row count differs from REGIME.json")
    for index, (actual, expected) in enumerate(zip(parsed, rows, strict=True), start=2):
        normalized = {
            field: _csv_value(expected.get(field)) for field in expected_fields
        }
        if actual != normalized:
            raise HarvestError(f"{name}:{index}: row differs from REGIME.json")
    return parsed


def _validate_document(
    root: Path,
    files: dict[str, bytes],
    expected_seeds: tuple[int, ...],
) -> tuple[dict, list[dict[str, str]], str]:
    try:
        document = json.loads(files["REGIME.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HarvestError(f"{root / 'REGIME.json'}: invalid JSON") from exc
    if not isinstance(document, dict) or document.get("schema") != REGIME_SCHEMA:
        raise HarvestError(f"{root / 'REGIME.json'}: expected schema {REGIME_SCHEMA}")
    _reject_nonfinite(document, str(root / "REGIME.json"))
    expected_protocol = {
        "topk_frac": 0.10,
        "retention_threshold": 0.50,
        "replication_fraction": 0.80,
        "minimum_final_bootstrap": 10_000,
    }
    for key, expected in expected_protocol.items():
        if document.get(key) != expected:
            raise HarvestError(f"{root / 'REGIME.json'}: unexpected {key}")
    first_bootstrap = document.get("first_bootstrap")
    if (
        not isinstance(first_bootstrap, int)
        or isinstance(first_bootstrap, bool)
        or first_bootstrap < 10_000
    ):
        raise HarvestError(
            f"{root / 'REGIME.json'}: final bootstrap count is below 10000"
        )

    rows = document.get("rows")
    summary = document.get("summary")
    if (
        not isinstance(rows, list)
        or not rows
        or not all(isinstance(row, dict) for row in rows)
    ):
        raise HarvestError(f"{root / 'REGIME.json'}: rows must be nonempty objects")
    if (
        not isinstance(summary, list)
        or not summary
        or not all(isinstance(row, dict) for row in summary)
    ):
        raise HarvestError(f"{root / 'REGIME.json'}: summary must be nonempty objects")

    expected_cells = set(
        itertools.product(EXPECTED_DATASETS, expected_seeds, EXPECTED_DRIFTS)
    )
    actual_cells: set[tuple[str, int, int]] = set()
    selector_sets: dict[tuple[str, int, int, str], set[str]] = {}
    cell_runs: dict[tuple[str, int, int], str] = {}
    identities: set[tuple[object, ...]] = set()
    models: set[str] = set()
    for row in rows:
        try:
            dataset = row["dataset"]
            seed = row["seed"]
            drift = row["drift"]
            model = row["model"]
            stratum = row["stratum"]
            policy = row["policy"]
            run = row["run"]
        except KeyError as exc:
            raise HarvestError(
                f"{root / 'REGIME.json'}: row lacks {exc.args[0]}"
            ) from exc
        if (
            not isinstance(dataset, str)
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or not isinstance(drift, int)
            or isinstance(drift, bool)
            or not isinstance(model, str)
            or not model
            or not isinstance(stratum, str)
            or stratum not in {"all", "mixed_reward", "identical_reward"}
            or not isinstance(policy, str)
            or not isinstance(run, str)
            or not run
        ):
            raise HarvestError(f"{root / 'REGIME.json'}: invalid row identity fields")
        cell = (dataset, seed, drift)
        if cell not in expected_cells:
            raise HarvestError(
                f"{root / 'REGIME.json'}: row is outside the registered matrix: {cell}"
            )
        identity = (dataset, seed, drift, stratum, policy)
        if identity in identities:
            raise HarvestError(
                f"{root / 'REGIME.json'}: duplicate analysis row: {identity}"
            )
        identities.add(identity)
        actual_cells.add(cell)
        models.add(model)
        previous_run = cell_runs.setdefault(cell, run)
        if previous_run != run:
            raise HarvestError(
                f"{root / 'REGIME.json'}: one matrix cell names multiple runs: {cell}"
            )
        if row.get("final_resampling") is not True:
            raise HarvestError(
                f"{root / 'REGIME.json'}: provisional row cannot be harvested"
            )
        selector_sets.setdefault((*cell, stratum), set()).add(policy)
    if actual_cells != expected_cells:
        missing = sorted(expected_cells - actual_cells)
        raise HarvestError(
            f"{root / 'REGIME.json'}: incomplete registered matrix; missing={missing}"
        )
    missing_all = sorted(cell for cell in expected_cells if (*cell, "all") not in selector_sets)
    bad_policies = sorted(
        key for key, policies in selector_sets.items() if policies != EXPECTED_POLICIES
    )
    if missing_all:
        raise HarvestError(
            f"{root / 'REGIME.json'}: missing all-prompt rows at {missing_all}"
        )
    if bad_policies:
        raise HarvestError(
            f"{root / 'REGIME.json'}: incomplete selector set at {bad_policies}"
        )
    if len(models) != 1:
        raise HarvestError(
            f"{root / 'REGIME.json'}: expected exactly one model, got {sorted(models)}"
        )

    grouped_seeds: dict[tuple[str, str, int, str, str], set[int]] = {}
    for row in rows:
        key = (row["model"], row["dataset"], row["drift"], row["stratum"], row["policy"])
        grouped_seeds.setdefault(key, set()).add(row["seed"])
    summary_keys: set[tuple[str, str, int, str, str]] = set()
    for row in summary:
        try:
            key = (row["model"], row["dataset"], row["drift"], row["stratum"], row["policy"])
            seeds = row["seeds"]
            status = row["status"]
        except KeyError as exc:
            raise HarvestError(
                f"{root / 'REGIME.json'}: summary row lacks {exc.args[0]}"
            ) from exc
        if key in summary_keys:
            raise HarvestError(f"{root / 'REGIME.json'}: duplicate summary row: {key}")
        summary_keys.add(key)
        if key not in grouped_seeds or seeds != len(grouped_seeds[key]):
            raise HarvestError(
                f"{root / 'REGIME.json'}: summary seed count differs from analysis rows: {key}"
            )
        if status not in {"effective", "ineffective", "inconclusive"}:
            raise HarvestError(
                f"{root / 'REGIME.json'}: provisional or unknown summary status: {status}"
            )
    if summary_keys != set(grouped_seeds):
        raise HarvestError(
            f"{root / 'REGIME.json'}: summary groups differ from analysis rows"
        )

    csv_rows = _validate_csv(files["REGIME.csv"], rows, str(root / "REGIME.csv"))
    _validate_csv(
        files["REGIME_SUMMARY.csv"], summary, str(root / "REGIME_SUMMARY.csv")
    )
    return document, csv_rows, next(iter(models))


def load_snapshot(root: Path, expected_seeds: tuple[int, ...]) -> AnalysisSnapshot:
    root = root.resolve()
    files = _read_coherent_files(root)
    document, csv_rows, model = _validate_document(root, files, expected_seeds)
    return AnalysisSnapshot(root, files, document, csv_rows, model)


def _input_digest(
    primary: AnalysisSnapshot,
    replication: AnalysisSnapshot,
    git: str,
    code_paths: list[Path],
) -> str:
    digest = hashlib.sha256(b"offpolicy-rlvr-harvest-input/v2\0")

    def add(label: str, payload: bytes) -> None:
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    add("git", git.encode())
    for path in code_paths:
        add(f"code:{path.as_posix()}", _read_regular(path))
    for label, snapshot in (("primary_27b", primary), ("replication_7b", replication)):
        for name in (*ANALYSIS_FILES, MARKER):
            add(f"{label}:{name}", snapshot.files[name])
    return digest.hexdigest()


def build_bundle(
    primary_root: Path,
    replication_root: Path,
    output: Path,
    git: str,
    code_paths: list[Path],
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40,64}", git):
        raise HarvestError("git revision is not a full hexadecimal object ID")
    primary = load_snapshot(primary_root, (0, 1, 2, 3, 4))
    replication = load_snapshot(replication_root, (0, 1, 2))
    if primary.model == replication.model:
        raise HarvestError("primary and replication results resolve to the same model")
    digest = _input_digest(primary, replication, git, code_paths)

    output.mkdir(parents=True, exist_ok=True)
    (output / "REPORT.md").write_text(
        "# RLVR Experiment Results\n\n"
        "## Primary: Qwen3.8-27B\n\n"
        + primary.files["FINAL_REPORT.md"].decode("utf-8").strip()
        + "\n\n## Scale replication: Qwen2.5-7B\n\n"
        + replication.files["FINAL_REPORT.md"].decode("utf-8").strip()
        + "\n",
        encoding="utf-8",
    )
    result = {
        "schema": HARVEST_SCHEMA,
        "git": git,
        "input_digest": digest,
        "primary_27b": primary.document,
        "replication_7b": replication.document,
    }
    (output / "RESULTS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, str]] = []
    fieldnames = ["experiment"]
    for label, snapshot in (("primary_27b", primary), ("replication_7b", replication)):
        for row in snapshot.csv_rows:
            rows.append({"experiment": label, **row})
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with (output / "RESULTS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git", required=True)
    parser.add_argument("--code", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    try:
        digest = build_bundle(
            args.primary,
            args.replication,
            args.output,
            args.git,
            args.code,
        )
    except (HarvestError, OSError, UnicodeDecodeError) as exc:
        print(f"[harvest-abort] {exc}", file=sys.stderr)
        return 1
    print(f"[harvest] validated input digest {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
