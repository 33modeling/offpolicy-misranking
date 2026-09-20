"""One read-only dashboard for the MBPP suites, without unrelated experiments."""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selection_switch_status as switch_status
from _status_summary import gate_label, mbpp_suite_label


ARM_NAMES = {"selection_reduced": "Selection", "random_reduced": "Random",
             "selection_full": "Full selection", "random_full": "Full random", "gated": "Gate policy"}
REMARKS = {"EVAL": "평가·결과 저장 남음", "RESUME": "체크포인트 검증·재개 대기",
           "REVIEW": "저장 파일 확인 필요", "BUDGET": "GPU 시간 한도 도달; 결과 미완료",
           "FAILED": "실패 원인 확인 필요", "STALE": "실행 신호 끊김; 확인 필요",
           "INVALID": "저장 기록 검증 필요", "SAVING": "결과 저장 중", "BLOCKED": "진행 차단"}


def arm_name(arm, names=None):
    return (ARM_NAMES if names is None else names).get(arm.split("/", 1)[0],
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
    if task.get("owner_active") and not task.get("heartbeat_fresh"):
        parts.append("평가 작업 잠금 유지; 시간 차이·진행 신호 확인 필요")
    if task.get("posthoc_evaluation_saved"):
        parts.append("복구 평가 저장됨; 동일예산 완료 아님")
    elif task.get("status") == "EVAL" and task.get("training_published"):
        # A sealed result already contains the final evaluation. Only the
        # convergence curve remains; do not describe this as unfinished training.
        parts.append("최종 평가 저장됨; 곡선 평가 남음 (재학습 없음)")
    elif task.get("status") in REMARKS:
        parts.append(REMARKS[task["status"]])
    if active(task) and task.get("phase"):
        parts.append("단계: " + task["phase"].replace("fresh-r", "on-policy").replace("fresh_r", "on-policy"))
    if task.get("status") == "WAIT" and task.get("reason"):
        parts.append(task["reason"])
    return "; ".join(parts)


def completion(suite):
    value = counts(suite)
    return value["progress"], value["done"], value["planned"]


def counts(suite):
    """Count planned continuation branches, never phases or shared prefixes."""
    rule = switch_status.rule
    registered = {(seed, step, arm) for seed in (*rule.DEV_SEEDS, *rule.TEST_SEEDS)
                  for step in rule.STEPS for arm in (rule.DEV_ARMS if seed in rule.DEV_SEEDS else rule.TEST_ARMS)}
    if "registered_tasks" in suite:
        registered = {tuple(key) for key in suite["registered_tasks"]}
    planned = len(registered)
    tasks = suite.get("tasks", [])
    branches, conflicting = {}, set()
    for task in tasks:
        key = (task.get("seed"), task.get("step"), task.get("arm"))
        if task.get("kind") != "branch" or key not in registered or task.get("unverified"):
            continue
        if key in branches and branches[key] != task:
            conflicting.add(key)
        branches[key] = task
    for key in conflicting:
        branches.pop(key)
    directories = [task.get("directory", "") for task in tasks if active(task)]
    states = Counter(display_state(task, directories) for task in branches.values())
    # A missing/unreadable root is not proof that its old results disappeared.
    # Its planned slots remain visible, but are explicitly unverified.
    unknown = max(0, planned - len(branches))
    states["WAIT"] += unknown
    return {"planned": planned, "done": states["DONE"], "remaining": planned - states["DONE"],
            "recovered": sum(bool(task.get("posthoc_evaluation_saved")) and task["status"] != "DONE"
                             for task in branches.values()),
            "unknown": unknown, "states": states, "progress": f"{100 * states['DONE'] / planned:.1f}%"}


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


def label(root, protocol=None):
    return mbpp_suite_label(root, protocol)


def suite_label(suite):
    return suite.get("display_label") or label(suite["root"], suite.get("protocol"))


def snapshot(roots, *, now=None, retained_roots=()):
    now = time.time() if now is None else now
    suites = []
    for root in dict.fromkeys(Path(root).resolve() for root in roots):
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
    data = {"updated": now, "suites": suites}
    observed = {suite["root"] for suite in suites}
    retained = [root for root in dict.fromkeys(Path(root).resolve() for root in retained_roots)
                if str(root) not in observed and root.is_dir()]
    if retained:
        data["retained_suites"] = snapshot(retained, now=now)["suites"]
    return data


def observed_suites(data):
    return [*data["suites"], *data.get("retained_suites", [])]


def active(task):
    # Publication status and process activity are independent: an EVAL branch
    # can still have a worker. Never downgrade its saved result to RUN/READY.
    return bool(task.get("owner_active") or task.get("heartbeat_fresh", task.get("status") == "RUNNING"))


def task_progress(root, task):
    """Current phase evidence only: bounded log tails, never model/rollout files."""
    phase = str(task.get("phase") or "")
    if re.fullmatch(r"[\w-]+", phase) and "train" not in phase:
        directory = Path(root) / task.get("directory", "")
        if directory.resolve().is_relative_to(Path(root).resolve()):
            batches = []
            for rank in range(4):
                path = directory / f"{phase}-{rank}.log"
                if not path.resolve().is_relative_to(Path(root).resolve()):
                    break
                lines = switch_status.node_view._tail_lines(path, size=16384)
                # Reports from completed previous attempts must not look like
                # current progress while a new worker is still loading.
                try:
                    progress = directory / "progress.json"
                    if path.stat().st_mtime < progress.stat().st_mtime - switch_status.number(task.get("seconds")) - 2:
                        break
                except OSError:
                    break
                gradients = re.findall(r"\[(?:fresh_r|on.policy)\].*?\((\d+)/(\d+)\)", "\n".join(lines))
                rollouts = re.findall(r"\brollout\s+(\d+)/(\d+)", "\n".join(lines))
                matches, basis = (gradients, "Gradient 처리") if gradients else (rollouts, "응답 생성")
                if not matches:
                    break
                done, total = map(int, matches[-1])
                if not 0 <= done <= total or total <= 0:
                    break
                batches.append((basis, done, total))
            if len(batches) == 4 and len({row[0] for row in batches}) == 1:
                done, total = sum(row[1] for row in batches), sum(row[2] for row in batches)
                return f"{100 * done / total:.1f}%", f"{batches[0][0]} {done}/{total}개"
    elapsed = switch_status.number(task.get("seconds"), -1)
    limit = switch_status.number(task.get("timeout"), 0)
    if elapsed >= 0 and limit > 0:
        note = "학습 시간 한도 사용률" if "train" in phase else "현재 단계 시간 한도 사용률"
        step, start = task.get("training_step"), task.get("step")
        if "train" in phase and isinstance(step, int) and isinstance(start, int) and step >= start:
            note += f"; 업데이트 {step - start}회 완료"
        return f"{min(100., 100 * elapsed / limit):.1f}%", note + " (결과 완료율 아님)"
    return "확인 중", "처리 건수·시간 한도 기록 없음"


def node_assignments(data):
    """Merge shared launcher evidence once, retaining *every* live assignment."""
    hosts = {}
    for suite in observed_suites(data):
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
    for suite in observed_suites(data):
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
    # Keep a server in the same name-sorted position as it changes RUN/WAIT.
    # Historical nodes remain opt-in and below current nodes, never deleted.
    return sorted(hosts.values(), key=lambda node: (not node["current"],
                                                   switch_status.node_view.host_sort_key(node["host"])))


def render_nodes(data, *, width, all_nodes=False):
    """Aligned, display-width-aware assignments; never truncate a node name."""
    nodes = node_assignments(data)
    current = [node for node in nodes if node["current"]]
    lines = ["NODE ASSIGNMENTS", f"NODES {len(current)} current"]
    rows = []
    retained = {suite["root"] for suite in data.get("retained_suites", [])}
    names = data.get("arm_names", ARM_NAMES)
    labels = {suite["root"]: suite_label(suite)
              + (" (기본 실행 제외)" if suite["root"] in retained else "") for suite in observed_suites(data)}
    visible = nodes if all_nodes else current
    for index, node in enumerate(visible, 1):
        if node["assignments"]:
            grouped = {}
            for root, task in node["assignments"]:
                key = (root, task["seed"], task["step"], arm_name(task["arm"], names))
                detail = remark(task)
                entry = grouped.setdefault(key, {"details": [], "tasks": []})
                entry["tasks"].append(task)
                details = entry["details"]
                if detail and detail not in details:
                    details.append(detail)
            for (root, seed, step, arm), entry in grouped.items():
                details = entry["details"]
                # A nested curve phase is more specific than its parent branch.
                task = max(entry["tasks"], key=lambda row: row.get("directory", "").count("/"))
                percent, basis = task_progress(root, task)
                details.append(basis)
                rows.append([f"{index}.", node['host'],
                             f"{labels[root]} / seed {seed} / step {step} / {arm}",
                             "RUN", percent, '; '.join(details) or '-'])
        else:
            detail = {"WAIT": "작업 배정 대기", "HOLD": "작업 배정 대기", "ADMIT": "장치 점검 중",
                      "COOL": "장치 오류 후 대기", "LIVE": "작업 시작 전 검증 중",
                      "BLOCKED": "작업 차단; 실험 미완료", "FAILED": "실행 실패; 실험 미완료",
                      "STOPPING": "실행 종료 처리 중",
                      "STALE": "실행 신호 끊김", "UNKNOWN": "배정 확인 안 됨",
                      "EXITED": "실행 종료", "GONE": "오래된 실행 기록"}.get(node["state"], "배정 확인 안 됨")
            state = {"HOLD": "WAIT", "LIVE": "CHECK", "UNKNOWN": "CHECK", "-": "CHECK",
                     "BLOCKED": "BLOCK", "FAILED": "FAIL", "EXITED": "EXIT", "STOPPING": "STOP"}.get(node["state"], node["state"])
            if node["state"] in {"BLOCKED", "FAILED", "EXITED"} and node.get("reason"):
                detail += "; " + node["reason"]
            elif node["state"] == "LIVE":
                last = node.get("detail", "")
                if last.startswith(("[recover-cost]", "[sweep ")):
                    detail = "비용·저장 기록 확인 중; 아직 작업 미배정"
                elif last.startswith(("[dispatch]", "[dispatch-task]", "[pass ", "[queue]")):
                    detail = "실행 가능한 작업 검색 중; 아직 작업 미배정"
            rows.append([f"{index}.", node['host'], "배정 없음", state, "-", detail])
    headers = ["#", "Node", "Experiment", "Status", "Progress", "Remarks"]
    number_width = max([columns(headers[0]), *(columns(row[0]) for row in rows)])
    node_width = max([columns(headers[1]), *(columns(row[1]) for row in rows)])
    # Keep full host names on one line when six useful columns fit. Padding is
    # based on terminal cells, not Python len(): Korean labels occupy two cells.
    remaining = width - number_width - node_width - 6 - 8 - 10
    if remaining >= 40:
        experiment_width = min(max([24, *(columns(row[2]) for row in rows)]),
                               max(24, remaining * 3 // 5))
        widths = [number_width, node_width, experiment_width, 6, 8,
                  remaining - experiment_width]
        lines += table(headers, rows, widths)
    else:
        # A very narrow terminal or exceptionally long host cannot hold all six
        # columns. Put each complete host above its aligned work columns, so a
        # wrapped hostname is never interleaved with another field's content.
        lines.append("# Node")
        detail_width = max(40, width - 2)
        experiment_width = max(18, (detail_width - 20) * 3 // 5)
        widths = [experiment_width, 6, 8, detail_width - experiment_width - 20]
        for row in rows:
            host_lines = wrap(row[1], width - number_width - 1)
            lines.append(row[0].ljust(number_width) + " " + host_lines[0])
            lines.extend(" " * (number_width + 1) + part for part in host_lines[1:])
            lines.extend("  " + line for line in table(headers[2:], [row[2:]], widths))
    lines.append("노드 Progress는 현재 단계 기준입니다. 시간 한도 사용률과 실제 처리 건수는 비고에서 구분합니다.")
    if not nodes or not all_nodes and not current:
        lines.append(f"No current {data.get('subject', 'MBPP')} node evidence.")
    hidden = len(nodes) - len(current)
    if hidden and not all_nodes:
        lines.append(f"{hidden} old node(s) hidden; --all shows history.")
    return lines


def idle_nodes(data):
    """Only fresh, explicitly waiting controllers without observed GPU work.

    Missing/stale task evidence alone is not an idle-node certificate. In
    particular, admission, recovery, and a task in a retained suite must not
    be listed as an unused allocation.
    """
    grace = switch_status.node_view.HEARTBEAT_GRACE
    return [node for node in node_assignments(data)
            if node["current"] and not node["assignments"] and node["state"] in {"WAIT", "HOLD"}
            and node.get("launcher_alive") is not False
            and node.get("evidence_age") is not None and -5 <= node["evidence_age"] < grace]


def render_idle_nodes(data):
    nodes = idle_nodes(data)
    lines = [f"작업 없는 노드: {len(nodes)}개 (배정 대기 확인)"]
    for index, node in enumerate(nodes, 1):
        age = max(0, int(node["evidence_age"]))
        lines.append(f"{index}. {node['host']} | WAIT | 작업 배정 대기; 확인 {age}초 전")
    lines.append("최근 대기 신호가 있는 노드만 표시합니다. 장치 점검·복구 중·신호 끊김은 제외합니다.")
    return lines


def render(data, *, width=120, all_tasks=False):
    width = max(80, width)
    names = data.get("arm_names", ARM_NAMES)
    subject = data.get("subject", "MBPP")
    stamp = datetime.fromtimestamp(data["updated"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    totals = [counts(suite) for suite in data["suites"]]
    planned = sum(item["planned"] for item in totals)
    done = sum(item["done"] for item in totals)
    unknown = sum(item["unknown"] for item in totals)
    remaining = planned - done
    aggregate = Counter()
    for item in totals:
        aggregate.update(item["states"])
    lines = [f"{subject} EXPERIMENTS  {stamp}",
             f"현재 조회 범위 {len(totals)}개 조건 | 총 계획 {planned}개 | 완료 확인 {done}개 | 남음 {remaining}개",
             f"남음 {remaining}개 = RUN {aggregate['RUN']}개 + READY {aggregate['READY']}개 + WAIT {aggregate['WAIT']}개"
             + (f" (기록 미확인 {unknown}개 포함)" if unknown else ""),
             "READY: 실행 가능 | DONE: 결과 저장 완료 | WAIT: 대기·중단·확인 필요 | RUN: 실행 중",
             "Progress: 완료 확인 / 계획. 남음에는 미확인 분기가 포함되며, 기록 없음은 삭제·미실행의 증거가 아닙니다.",
             "학습 분기 수 기준입니다. 공통 학습·선택·평가 단계를 별도 실험으로 더하지 않습니다."]
    recovered = sum(item["recovered"] for item in totals)
    if recovered:
        lines.insert(3, f"복구 평가 완료 {recovered}개 (동일예산 DONE 제외; 위 남음에 포함)")
    if data.get("retained_suites"):
        preserved_done = sum(counts(suite)["done"] for suite in data["retained_suites"])
        lines.append(f"다른 조건의 완료 결과 {preserved_done}개 보존 — 현재 조건과 합산하지 않음; 아래 기존 기록에 표시")
    rows, running, notices = [], [], []
    for suite in data["suites"]:
        name = suite_label(suite)
        count = counts(suite)
        states = count["states"]
        condition = ("학습 한도 공통; 선택 비용 별도 기록"
                     if suite.get("protocol", {}).get("accounting") == "matched" else
                     "선택·학습 한도 공통" if suite.get("protocol", {}).get("accounting") == "budget" else "")
        if not suite.get("prepared"):
            note = ("설정 읽기 실패" if suite.get("error") else "실험 설정 확인 불가") + f"; 기록 미확인 {count['unknown']}개"
            rows.append([name, count["planned"], count["done"], count["remaining"], count["progress"],
                         states["READY"], states["WAIT"], states["RUN"], f"{condition}; {note}".lstrip("; ")])
            if suite.get("error"):
                notices.append(f"{name}: 설정 읽기 실패: {suite['error']}")
            continue
        tasks = suite.get("tasks", [])
        branches = [task for task in tasks if task.get("kind") == "branch"]
        active_tasks = [task for task in tasks if active(task)]
        running += [(name, task) for task in active_tasks]
        prefixes = [task for task in tasks if task.get("kind") == "prefix"]
        prefix_done = sum(task["status"] == "DONE" for task in prefixes)
        shared_label = suite.get("shared_label", "공통 학습")
        note = f"{condition}; {shared_label} {prefix_done}/{len(prefixes)}".lstrip("; ")
        if count["unknown"]:
            note += f"; 기록 미확인 {count['unknown']}개"
        budgets = sum(task['status'] == 'BUDGET' for task in branches)
        evaluations = sum(task['status'] == 'EVAL' for task in branches)
        curves = sum(task['status'] == 'EVAL' and task.get('training_published', False) for task in branches)
        if budgets:
            note += f"; GPU 시간 한도 도달 {budgets}개 (결과 미완료)"
        if evaluations > curves:
            note += f"; 평가·결과 저장 남음 {evaluations - curves}개"
        if curves:
            note += f"; 최종 평가 저장됨·곡선 남음 {curves}개"
        if count["recovered"]:
            note += f"; 복구 평가 완료 {count['recovered']}개 (동일예산 DONE 제외)"
        rows.append([name, count["planned"], count["done"], count["remaining"], count["progress"],
                     states["READY"], states["WAIT"], states["RUN"], note])
        trained = suite.get("training_published", 0)
        if trained > count['done']:
            notices.append(f"{name}: 최종 평가 결과 {trained}개 저장됨; 곡선 평가 남음 {curves}개"
                           f"; 곡선 기록 확인 필요 {max(0, trained - count['done'] - curves)}개.")
    lines += table(["Experiment", "계획", "DONE", "남음", "Progress", "READY", "WAIT", "RUN", "Remarks"],
                   rows, [26, 4, 4, 4, 8, 5, 4, 3, width - 74])
    for suite in data["suites"]:
        lines += ["", f"FULL STATUS — {suite_label(suite)}"]
        count = counts(suite)
        lines.append(f"계획 {count['planned']}개 | 완료 확인 {count['done']}/{count['planned']}"
                     f" | 남음 {count['remaining']}개 | {count['progress']}")
        if count["recovered"]:
            lines.append(f"복구 평가 완료 {count['recovered']}개 (동일예산 DONE 제외)")
        if not suite.get("prepared"):
            lines.append(f"WAIT {count['unknown']}개: " + ("설정 읽기 실패" if suite.get("error") else "실험 설정 확인 불가")
                         + "; 완료 여부 미확인")
            continue
        if suite.get("protocol", {}).get("gate"):
            lines.append("Gate policy 판단: " + gate_label(suite["protocol"]["gate"]))
        if suite.get("protocol", {}).get("budget_gpu_seconds") is not None:
            budget = suite['protocol']['budget_gpu_seconds']
            kind = "진단·학습 시간 한도" if suite['protocol'].get('accounting') == 'matched' else "선택·진단·학습 시간 한도"
            lines.append(f"{kind}: {budget} GPU-seconds")
        if suite.get("protocol", {}).get("accounting") == "matched":
            lines.append("선택 비용은 위 학습 한도에서 차감하지 않으며, 총 GPU 시간에 포함합니다. 평가·곡선 저장까지 끝나야 DONE입니다.")
        lines += suite.get("details", [])
        tasks = suite.get("tasks", [])
        prefixes = {(task["seed"], task["step"]): task for task in tasks if task.get("kind") == "prefix"}
        branches = {(task["seed"], task["step"], task["arm"]): task for task in tasks if task.get("kind") == "branch"}

        directories = [task.get("directory", "") for task in tasks if active(task)]

        matrix = []
        state_points = suite.get("state_points", [
            (seed, step, "개발" if seed in switch_status.rule.DEV_SEEDS else "검증")
            for seed in (*switch_status.rule.DEV_SEEDS, *switch_status.rule.TEST_SEEDS)
            for step in switch_status.rule.STEPS])
        for seed, step, role in state_points:
            notes = [f"{arm_name(task['arm'], names)}: {remark(task)}" for task in tasks
                     if task["seed"] == seed and task["step"] == step and remark(task)]
            matrix.append([f"{seed} / {step}", role,
                           display_state(prefixes.get((seed, step)), directories),
                           *[display_state(branches.get((seed, step, arm)), directories) for arm in names],
                           "; ".join(dict.fromkeys(notes)) or "-"])
        widths = ([11, 5, 6, 9, 6, 14, 11, 11, width - 89] if width >= 110
                  else [11, 4, 5, 9, 6, 9, 6, 6, width - 72])
        if names != ARM_NAMES:
            fixed = [11, 5, 6, *[max(6, columns(name)) for name in names.values()]]
            widths = [*fixed, width - sum(fixed) - 2 * len(fixed)]
        lines += table(["Seed / Step", "Role", suite.get("prefix_heading", "Prefix"), *names.values(), "Remarks"], matrix, widths)
    if data.get("retained_suites"):
        lines += ["", "기본 실행 제외 — 기존 기록 보존 (위 계획·완료·남음 합계에서 제외)"]
        for suite in data["retained_suites"]:
            name = suite_label(suite)
            count = counts(suite)
            states = count["states"]
            lines.append(f"{name}: 기존 계획 {count['planned']}개 | 완료 확인 {count['done']}개 | 남음 {count['remaining']}개"
                         f" | RUN {states['RUN']}개 | READY {states['READY']}개 | WAIT {states['WAIT']}개")
            if count["unknown"]:
                lines.append(f"기록 미확인 {count['unknown']}개; 기존 결과가 삭제됐다는 뜻이 아닙니다.")
            if suite.get("error"):
                notices.append(f"{name}: 설정 읽기 실패: {suite['error']}")
            running += [(name, task) for task in suite.get("tasks", []) if active(task)]
        lines.append("이 표시는 기존 작업을 중단하지 않습니다. 기본 실행 제외 작업의 노드도 아래에 표시합니다.")
    running_experiments = {(name, task["seed"], task["step"], arm_name(task["arm"], names)) for name, task in running}
    lines += data.get("legend", ["On-policy: 현재 정책으로 계산한 gradient 기반 선택. Difficulty: 저장된 정답률 기반 선택.",
              "선택비용 포함: 선택·진단·학습에 같은 예산 적용. 선택비용 별도: 선택 비용을 예산 밖에 기록.",
              "선택비용 별도도 총 GPU 비용에는 포함합니다. 평가 비용은 모든 조건에서 별도로 기록합니다.",
              "GPU 시간 한도에 도달해도 평가 결과가 없으면 미완료이며, 보상 0점이 아닙니다.",
              "Selection / Random: 공통 진단 비용 차감 후 비교. Full selection / Full random: 전체 예산 대조군.",
              "Gate policy: 전환 규칙 적용. '-': 해당 상태에서 실행 대상 아님."])
    lines += ["", f"CURRENT RUN {len(running_experiments)}"]
    if not running:
        lines.append(f"No fresh RUN heartbeat in these {subject} suites; saved completions above are retained.")
    lines += render_nodes(data, width=width, all_nodes=all_tasks)
    lines += notices
    if all_tasks:
        for suite in observed_suites(data):
            lines += ["", f"ROOT {suite['root']}"]
            directories = [task.get("directory", "") for task in suite.get("tasks", []) if active(task)]
            for task in suite.get("tasks", []):
                lines.append(f"{display_state(task, directories)} {task['directory']}"
                             + (f" — {remark(task)}" if remark(task) else ""))
    lines += ["", *render_idle_nodes(data)]
    return "\n".join(part for line in lines for part in
                     (wrap(line, width) if columns(line) > width else [line]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--retained-root", type=Path, action="append", default=[],
                        help="observe legacy work without adding it to the current experiment plan")
    parser.add_argument("--all", action="store_true", dest="all_tasks")
    args = parser.parse_args()
    data = snapshot(args.root, retained_roots=args.retained_root) if args.retained_root else snapshot(args.root)
    print(render(data, width=max(80, shutil.get_terminal_size((120, 40)).columns),
                 all_tasks=args.all_tasks))
    return int(any(suite.get("error") for suite in observed_suites(data)))


if __name__ == "__main__":
    raise SystemExit(main())
