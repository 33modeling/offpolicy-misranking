#!/usr/bin/env python3
"""Export paper-code incidents and the complete available Git history without erasing records."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "research-bug-board"
REPO = "https://github.com/33modeling/offpolicy-misranking"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def preserve(previous, current):
    """Existing incidents and events are permanent; corrections must append events."""
    rows = {row["id"]: row for row in current}
    if len(rows) != len(current):
        raise ValueError("duplicate incident id")
    for old in previous:
        new = rows.get(old["id"])
        if new is None:
            raise ValueError(f"cannot delete incident {old['id']}")
        if {k: v for k, v in old.items() if k != "events"} != {k: v for k, v in new.items() if k != "events"}:
            raise ValueError(f"cannot overwrite incident {old['id']}; append a correction event")
        if new.get("events", [])[:len(old.get("events", []))] != old.get("events", []):
            raise ValueError(f"cannot erase or rewrite events for {old['id']}")


def history():
    raw = git("log", "--all", "--reverse", "--topo-order",
              "--format=%H%x1f%aI%x1f%an%x1f%s%x1f%B%x1e")
    rows = []
    for record in raw.split("\x1e"):
        if not record.strip():
            continue
        sha, at, author, title, body = record.strip().split("\x1f", 4)
        attribution = [line for line in body.splitlines()
                       if re.match(r"(?:Co-Authored-By|Generated-By|Agent):", line, re.I)]
        rows.append({"hash": sha, "at": at, "author": author, "title": title,
                     "url": f"{REPO}/commit/{sha}", "attribution": attribution,
                     "is_fix": bool(re.search(r"\bfix\b|\brepair\b|\bbug\b|버그|오류|누락|복구|고침|회귀", title, re.I)),
                     "gemma": "gemma" in title.lower()})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", default="HEAD", help="record the audited source revision")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = git("rev-parse", args.through).strip()
    incidents = json.loads((BOARD / "incidents.json").read_text())
    data_path, commits_path = BOARD / "data.json", BOARD / "commits.json"
    previous = json.loads(data_path.read_text()) if data_path.exists() else None
    published = subprocess.run(["git", "show", "HEAD:research-bug-board/data.json"], cwd=ROOT,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if published.returncode == 0:
        preserve(json.loads(published.stdout)["incidents"], incidents)
    if previous:
        preserve(previous["incidents"], incidents)
    for name in git("ls-files", "research-bug-board/archive").splitlines():
        expected = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT)
        path = ROOT / name
        if not path.exists() or path.read_bytes() != expected:
            raise ValueError(f"cannot erase or overwrite published archive {name}")
    rows = history()
    if (previous and previous["coverage"]["source_commit"] == source and commits_path.exists()):
        # Keep this audit reproducible after committing the board itself.
        rows = json.loads(commits_path.read_text())["commits"]
    if commits_path.exists():
        old = json.loads(commits_path.read_text())["commits"]
        known = {row["hash"]: row for row in rows}
        if any(known.get(row["hash"]) != row for row in old):
            raise ValueError("available history lost or rewrote a previously published commit")
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    gemma = [row for row in rows if row["gemma"]]
    coverage = {"source_commit": source, "source_url": f"{REPO}/commit/{source}",
                "all_available_commits": len(rows), "fix_commit_records": sum(r["is_fix"] for r in rows),
                "gemma_named_commits": len(gemma), "first_commit": rows[0]["hash"],
                "first_commit_at": rows[0]["at"], "first_gemma_commit": gemma[0]["hash"],
                "first_gemma_at": gemma[0]["at"], "command": "git log --all --reverse --topo-order",
                "branch_snapshot": (previous["coverage"]["branch_snapshot"] if previous and previous["coverage"]["source_commit"] == source else git("for-each-ref", "--format=%(refname):%(objectname)").splitlines())}
    current = {"schema": 1, "title": "논문 코드 버그 현황판", "recorded_at": now,
               "repository": REPO, "coverage": coverage, "incidents": incidents,
               "retention": "항목·수정·실패·기존 이벤트를 삭제하지 않는다. 정정은 새 이벤트로 추가한다.",
               "reported_scores": [{"label": "1%대", "source": "사용자 보고 및 2026-10-21 감사 문서", "verified": False},
                                   {"label": "10%대", "source": "사용자 대화 보고", "verified": False},
                                   {"label": "약 14%", "source": "2026-10-22 문서의 사용자 보고", "verified": False},
                                   {"label": "83%", "source": "IT 전환 후 사용자 대화 보고", "verified": False}],
               "measurement_note": "위 비율은 서로 같은 조건으로 검증된 학습 곡선이 아니다. 원격 원시 기록이 없는 값은 미검증으로 남긴다."}
    if previous:
        same = {k: v for k, v in previous.items() if k != "recorded_at"} == {k: v for k, v in current.items() if k != "recorded_at"}
        if same:
            current["recorded_at"] = previous["recorded_at"]
    exports = {data_path: current, commits_path: {"schema": 1, "commits": rows}}
    for path, value in exports.items():
        content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"stale ledger: {path}")
        else:
            path.write_text(content)
    print(json.dumps({"incidents": len(incidents), **{k: coverage[k] for k in
                     ("all_available_commits", "fix_commit_records", "gemma_named_commits", "source_commit")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
