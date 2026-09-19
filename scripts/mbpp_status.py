"""One read-only dashboard for the MBPP suites, without unrelated experiments."""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selection_switch_status as switch_status


ARM_NAMES = {"selection_reduced": "Selection", "random_reduced": "Random",
             "selection_full": "Full selection", "random_full": "Full random", "gated": "Gate policy"}
REMARKS = {"EVAL": "평가·결과 저장 남음", "RESUME": "체크포인트 검증·재개 대기",
           "REVIEW": "저장 파일 확인 필요", "BUDGET": "예산 소진으로 중단",
           "FAILED": "실패 원인 확인 필요", "STALE": "실행 신호 끊김; 확인 필요",
           "INVALID": "저장 기록 검증 필요", "SAVING": "결과 저장 중", "BLOCKED": "진행 차단"}


def arm_name(arm):
    return ARM_NAMES.get(arm.split("/", 1)[0],
                         {"prefix": "Shared training", "diagnostic": "Shared diagnostic",
                          "curve-parent": "Shared evaluation"}.get(arm.split("/", 1)[0], arm))


def display_state(task, running_directories=()):
    if not task:
        return "-"
    if task["status"] == "DONE":
        return "DONE"
    directory = task.get("directory", "")
    if active(task) or directory and any(path.startswith(directory + "/") for path in running_directories):
        return "RUN"
    return "READY" if task["status"] == "READY" else "WAIT"


def remark(task):
    parts = []
    if task.get("status") in REMARKS:
        parts.append(REMARKS[task["status"]])
    if active(task) and task.get("phase"):
        parts.append("단계: " + task["phase"].replace("fresh-r", "on-policy").replace("fresh_r", "on-policy"))
    if task.get("status") == "WAIT" and task.get("reason"):
        parts.append(task["reason"])
    return "; ".join(parts)


def completion(suite):
    branches = [task for task in suite.get("tasks", []) if task.get("kind") == "branch"]
    done = sum(task["status"] == "DONE" for task in branches)
    return (f"{100 * done / len(branches):.1f}%" if branches else "-", done, len(branches))


def columns(text):
    return sum(0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in "WF" else 1
               for char in str(text))


def wrap(text, width):
    """Wrap full labels without ellipses, respecting Korean terminal width."""
    rows, line = [], ""
    for word in str(text).split():
        if line and columns(line + " " + word) <= width:
            line += " " + word
            continue
        if line:
            rows.append(line)
        line = ""
        for char in word:
            if line and columns(line + char) > width:
                rows.append(line)
                line = ""
            line += char
    if line:
        rows.append(line)
    return rows or [""]


def table(headers, rows, widths):
    lines = []
    for row in (headers, *rows):
        cells = [wrap(value, width) for value, width in zip(row, widths)]
        for index in range(max(map(len, cells))):
            values = [cell[index] if index < len(cell) else "" for cell in cells]
            lines.append("  ".join(value + " " * (width - columns(value))
                                   for value, width in zip(values, widths)).rstrip())
    return lines


def label(root):
    name = Path(root).name
    if name == "selection-switch-mbpp-v1":
        return "on-policy"
    return name.removeprefix("selection-switch-mbpp-").removesuffix("-v1")


def snapshot(roots, *, now=None):
    now = time.time() if now is None else now
    suites = []
    for root in roots:
        try:
            suites.append(switch_status.snapshot(Path(root), now=now, local_gpus=False,
                                                node_namespace="mbpp"))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            suite = {"prepared": False, "root": str(root), "updated": now, "error": str(exc)}
            try:
                suite["nodes"] = switch_status.node_view.launcher_nodes(
                    Path(root), [], now=now, node_namespace="mbpp")
            except OSError:
                suite["nodes"] = []
            suites.append(suite)
    return {"updated": now, "suites": suites}


def active(task):
    # Publication status and process activity are independent: an EVAL branch
    # can still have a worker. Never downgrade its saved result to RUN/READY.
    return task.get("heartbeat_fresh", task.get("status") == "RUNNING")


def node_assignments(data):
    """Merge shared launcher evidence once, retaining *every* live assignment."""
    hosts = {}
    for suite in data["suites"]:
        for node in suite.get("nodes", []):
            host = str(node["host"]).rstrip("_")
            previous = hosts.get(host)
            log_age, pid_age = node.get("last_age"), node.get("pid_age")
            ages = [age for age in (log_age, pid_age) if age is not None]
            age = min(ages) if ages else None
            if (pid_age is not None and -5 <= pid_age < switch_status.node_view.HEARTBEAT_GRACE
                    and (log_age is None or log_age >= switch_status.node_view.HEARTBEAT_GRACE)):
                node = {**node, "state": "UNKNOWN", "source_root": None, "reason": "",
                        "detail": "Recent launcher PID record; waiting for current task or controller log."}
            prior_age = previous.get("evidence_age") if previous else None
            if previous is None or (age is not None and (prior_age is None or age < prior_age)):
                hosts[host] = {**node, "host": host, "evidence_age": age, "assignments": []}
    for suite in data["suites"]:
        for task in suite.get("tasks", []):
            if task.get("status") == "STALE" and task.get("host"):
                host = str(task["host"]).rstrip("_")
                node = hosts.setdefault(host, {"host": host, "state": "STALE", "evidence_age": None,
                                               "assignments": []})
                age = task.get("heartbeat_age")
                prior_age = node.get("evidence_age")
                if age is not None and (prior_age is None or age < prior_age):
                    node.update(evidence_age=age, state="STALE", source_root=str(Path(suite["root"]).resolve()),
                                detail=f"Last task s{task['seed']}/t{task['step']} {task['arm']}; heartbeat stale, ownership unconfirmed.")
            if not active(task):
                continue
            host = str(task.get("host") or "unknown-owner").rstrip("_")
            node = hosts.setdefault(host, {"host": host, "state": "RUN", "evidence_age": None,
                                           "assignments": []})
            node["state"] = "RUN"
            node["assignments"].append((suite["root"], task))
    for node in hosts.values():
        if node["assignments"]:
            node["state"] = "RUN"
        node["assignments"].sort(key=lambda pair: (pair[0], pair[1]["seed"], pair[1]["step"], pair[1]["arm"]))
        age = node.get("evidence_age")
        node["current"] = bool(node["assignments"] or node.get("launcher_alive") is True
                               or age is not None and -5 <= age < switch_status.node_view.HEARTBEAT_GRACE)
        if node["state"] == "-":
            node["state"] = "UNKNOWN"
    return sorted(hosts.values(), key=lambda node: (not node["current"], not bool(node["assignments"]),
                                                   switch_status.node_view.host_sort_key(node["host"])))


def render_nodes(data, *, width, all_nodes=False):
    """One full node name -> experiment mapping, with no interleaved columns."""
    nodes = node_assignments(data)
    current = [node for node in nodes if node["current"]]
    lines = ["NODE ASSIGNMENTS", f"NODES {len(current)} current",
             "# Node -> Experiment | Status | Progress | Remarks"]
    progress = {suite["root"]: completion(suite)[0] for suite in data["suites"]}
    visible = nodes if all_nodes else current
    for index, node in enumerate(visible, 1):
        if node["assignments"]:
            grouped = {}
            for root, task in node["assignments"]:
                key = (root, task["seed"], task["step"], arm_name(task["arm"]))
                detail = remark(task)
                details = grouped.setdefault(key, [])
                if detail and detail not in details:
                    details.append(detail)
            for (root, seed, step, arm), details in grouped.items():
                lines.append(f"{index}. {node['host']} -> {label(root)} / seed {seed} / step {step} / {arm}"
                             f" | RUN | {progress.get(root, '-')} | {'; '.join(details) or '-'}")
        else:
            detail = {"WAIT": "작업 배정 대기", "HOLD": "작업 배정 대기", "ADMIT": "장치 점검 중",
                      "COOL": "장치 오류 후 대기", "LIVE": "작업 배정 확인 중",
                      "STALE": "실행 신호 끊김", "UNKNOWN": "배정 확인 안 됨",
                      "EXITED": "실행 종료", "GONE": "오래된 실행 기록"}.get(node["state"], "배정 확인 안 됨")
            lines.append(f"{index}. {node['host']} -> 배정 없음 | WAIT | - | {detail}")
    if not nodes or not all_nodes and not current:
        lines.append("No current MBPP node evidence.")
    hidden = len(nodes) - len(current)
    if hidden and not all_nodes:
        lines.append(f"{hidden} old node(s) hidden; --all shows history.")
    return lines


def render(data, *, width=120, all_tasks=False):
    width = max(80, width)
    stamp = datetime.fromtimestamp(data["updated"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [f"MBPP EXPERIMENTS  {stamp}",
             "READY: 실행 가능 | DONE: 결과 저장 완료 | WAIT: 대기·중단 | RUN: 실행 중",
             "Progress: 실험 묶음의 완료 분기 수 / 전체 분기 수. 시간 경과로 추정하지 않습니다."]
    rows, running, notices = [], [], []
    for suite in data["suites"]:
        name = label(suite["root"])
        if not suite.get("prepared"):
            rows.append([name, "-", "-", "-", "-", "-", "-", "설정 읽기 실패" if suite.get("error") else "준비 전"])
            notices.append(f"{name}: 설정 읽기 실패: {suite['error']}" if suite.get("error") else f"{name}: 준비 전")
            continue
        tasks = suite.get("tasks", [])
        branches = [task for task in tasks if task.get("kind") == "branch"]
        active_tasks = [task for task in tasks if active(task)]
        running += [(name, task) for task in active_tasks]
        directories = [task.get("directory", "") for task in active_tasks]
        counts = Counter(display_state(task, directories) for task in branches)
        prefixes = [task for task in tasks if task.get("kind") == "prefix"]
        prefix_done = sum(task["status"] == "DONE" for task in prefixes)
        note = f"공통 학습 {prefix_done}/{len(prefixes)}"
        budgets = sum(task['status'] == 'BUDGET' for task in branches)
        evaluations = sum(task['status'] == 'EVAL' for task in branches)
        if budgets:
            note += f"; 예산 소진 {budgets}개"
        if evaluations:
            note += f"; 평가·결과 저장 남음 {evaluations}개"
        rows.append([name, completion(suite)[0], f"{counts['DONE']}/{len(branches)}", counts['READY'],
                     counts['DONE'], counts['WAIT'], counts['RUN'], note])
        trained = suite.get("training_published", 0)
        if trained > counts['DONE']:
            notices.append(f"{name}: 학습 결과 {trained}개 저장됨; 평가·결과 확정 대기 {evaluations}개.")
    lines += table(["Experiment", "Progress", "Completed", "READY", "DONE", "WAIT", "RUN", "Remarks"],
                   rows, [12, 8, 9, 5, 4, 4, 3, width - 59])
    for suite in data["suites"]:
        lines += ["", f"FULL STATUS — {label(suite['root'])}"]
        if not suite.get("prepared"):
            lines.append("WAIT: 설정 읽기 실패" if suite.get("error") else "WAIT: 준비 전 (저장된 실험 설정 없음)")
            continue
        tasks = suite.get("tasks", [])
        prefixes = {(task["seed"], task["step"]): task for task in tasks if task.get("kind") == "prefix"}
        branches = {(task["seed"], task["step"], task["arm"]): task for task in tasks if task.get("kind") == "branch"}

        directories = [task.get("directory", "") for task in tasks if active(task)]

        matrix = []
        for seed in (*switch_status.rule.DEV_SEEDS, *switch_status.rule.TEST_SEEDS):
            for step in switch_status.rule.STEPS:
                notes = [f"{arm_name(task['arm'])}: {remark(task)}" for task in tasks
                         if task["seed"] == seed and task["step"] == step and remark(task)]
                matrix.append([f"{seed} / {step}", "개발" if seed in switch_status.rule.DEV_SEEDS else "검증",
                               display_state(prefixes.get((seed, step)), directories),
                               *[display_state(branches.get((seed, step, arm)), directories) for arm in ARM_NAMES],
                               "; ".join(dict.fromkeys(notes)) or "-"])
        widths = ([11, 5, 6, 9, 6, 14, 11, 11, width - 89] if width >= 110
                  else [11, 4, 5, 9, 6, 9, 6, 6, width - 72])
        lines += table(["Seed / Step", "Role", "Prefix", *ARM_NAMES.values(), "Remarks"], matrix, widths)
    running_experiments = {(name, task["seed"], task["step"], arm_name(task["arm"])) for name, task in running}
    lines += ["Selection / Random: 공통 진단 비용 차감 후 비교. Full selection / Full random: 전체 예산 대조군.",
              "Gate policy: 전환 규칙 적용. '-': 해당 상태에서 실행 대상 아님.",
              "", f"CURRENT RUN {len(running_experiments)}"]
    if not running:
        lines.append("No fresh RUN heartbeat in these MBPP suites; saved completions above are retained.")
    lines += render_nodes(data, width=width, all_nodes=all_tasks)
    lines += notices
    if all_tasks:
        for suite in data["suites"]:
            lines += ["", f"ROOT {suite['root']}"]
            for task in suite.get("tasks", []):
                lines.append(f"{display_state(task)} {task['directory']}" + (f" — {remark(task)}" if remark(task) else ""))
    return "\n".join(part for line in lines for part in
                     (wrap(line, width) if columns(line) > width else [line]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--all", action="store_true", dest="all_tasks")
    args = parser.parse_args()
    data = snapshot(args.root)
    print(render(data, width=max(80, shutil.get_terminal_size((120, 40)).columns),
                 all_tasks=args.all_tasks))
    return int(any(suite.get("error") for suite in data["suites"]))


if __name__ == "__main__":
    raise SystemExit(main())
