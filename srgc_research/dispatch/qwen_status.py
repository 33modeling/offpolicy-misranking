"""Read-only terminal status for SRGC queues and their recorded progress (Qwen entry; layout shared by all models)."""

import argparse
import json
import math
import os
import socket
import sys
import time
from collections import Counter
from pathlib import Path

from scripts import srgc_live_status as live
from scripts.srgc_qwen35_storage import default_root

from .rank_failures import last_error_line

REPO = Path(__file__).resolve().parents[2]
ARMS = ("cache", "prefix", "on_policy", "switch", "sr", "random")
HEADERS = ("Cache", "Prefix", "On-policy", "Switch", "SR", "Random")
LIVE_WORKERS = {"running", "preflight", "idle", "serving_other_dataset",
                "launching", "waiting", "switching", "preparing"}
RESUMABLE = {"recoverable", "interrupted"}
ERRORS = {"failed", "invalid", "attempts_exhausted"}


def load_reports(dataset, root, environment):
    """Use recorded identities without importing model packages or binding queues."""
    from .model_status import load_reports as load

    return load("qwen35", dataset, root, environment)


def age(seconds):
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        return "-"
    seconds = max(0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def bar(done, total, width=16):
    filled = int(width * done / total) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _cell(task):
    if task is None:
        return "-"
    state = task["status"]
    if state == "complete":
        reward = task.get("reward_percent")
        return "DONE" if reward is None else f"DONE {reward:.1f}%"
    if state == "running":
        if task["arm"] == "cache" and "cache_saved_prompts" in task:
            count = max(task["cache_saved_prompts"], task["cache_exported_prompts"])
            return f"RUN {count}/{task['cache_total_prompts']}"
        done, total = task.get("completed_steps"), task.get("total_steps")
        return f"RUN {done}/{total}" if done is not None and total else "RUN"
    if state.startswith("waiting_for_"):
        return "WAIT " + state.removeprefix("waiting_for_")
    if state == "failed":
        return "ERROR"
    return {"ready": "READY", "recoverable": "RESUME", "interrupted": "RESUME",
            "invalid": "INVALID", "attempts_exhausted": "RETRY LIMIT"}.get(state, state.upper())


def cell(task):
    return "-" if task is None else live.switch_cell(task, _cell(task))


def phase_cell(task):
    if task.get("phase_stage") == "evaluation":
        return "CHECKPOINT EVAL" if task.get("evaluation_step") is not None else "FINAL EVAL"
    return {"training": "TRAIN", "selection": "SELECT", "checkpoint_save": "SAVE",
            "checkpoint_load": "LOAD", "finalizing": "FINALIZING",
            "startup": "STARTUP", "preparation": "PREP"}.get(task.get("phase_stage"), "-")


def selection_text(task):
    """Saved scoring prompts of a running selection block, or None when not observed."""
    progress = task.get("selection_progress")
    if not progress or task.get("phase_stage") != "selection":
        return None
    count = lambda value: "?" if value is None else value
    bound = "<=" if progress["check"] else ""
    return (f"cand {count(progress['candidate_saved'])}/{bound}{progress['candidate_max']}"
            f" val {count(progress['validation_saved'])}/{progress['validation_total']}")


def worker_error(node):
    message = " ".join(str(node.get("error") or "").split())
    expected = {"KeyboardInterrupt", "KeyboardInterrupt:",
                *(f"SystemExit: {code}" for code in (129, 130, 137, 143))}
    return "" if node["status"] == "stopped" and message in expected else message


def table(headers, rows):
    rows = [[str(value) for value in row] for row in rows]
    widths = [max(len(header), *(len(row[i]) for row in rows)) if rows else len(header)
              for i, header in enumerate(headers)]
    line = lambda values: "  ".join(f"{value:<{width}}" for value, width in zip(values, widths)).rstrip()
    return [line(headers), line(["-" * width for width in widths]), *(line(row) for row in rows)]


def heartbeat_age(row):
    value = row.get("heartbeat_age_seconds")
    return value if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else float("inf")


def scoped_coordinators(rows, *, model, root):
    family = "gemma4" if model and model.startswith("gemma4") else model
    return [row for row in rows
            if family is not None and row.get("model") == family
            and row.get("status_model") in {None, model}
            and (not row.get("root") or root is not None
                 and Path(row["root"]).resolve() == Path(root).resolve())]


def leased_task_text(node):
    """Explain the recorded work without treating ownership as forward progress."""
    task = node.get("task_row") or {}
    arm = task.get("arm") or str(node.get("task") or "").rsplit(".", 1)[-1]
    work = {"cache": "초기 응답 캐시", "prefix": "공통 Prefix", "switch": "Switch",
            "on_policy": "On-policy", "sr": "SR", "random": "Random"}.get(arm, str(arm or "실험"))
    stage = {"training": "Prefix 업데이트" if arm == "prefix" else "모델 업데이트",
             "selection": "후보·검증 데이터 선별", "checkpoint_save": "체크포인트 저장",
             "checkpoint_load": "체크포인트 복원", "preparation": "학습 준비",
             "startup": "자식 초기화", "finalizing": "결과 마무리",
             "evaluation": "중간 평가" if task.get("evaluation_step") is not None else "최종 평가"}.get(task.get("phase_stage"))
    if task.get("endpoint_ready") is True:
        text = f"{work} 결과 저장 완료 · 작업 종료/잠금 반환 대기"
    else:
        text = f"{work} 작업 확보 · " + (f"마지막 기록: {stage}" if stage else "실행 단계 미기록")
    if not node.get("heartbeat_stale") and not math.isfinite(heartbeat_age(node)):
        text += " · worker 응답 미기록, 진행 미확인"
    return text


def launcher_text(node):
    if node["status"] == "waiting":
        code = node.get("previous_exit_code", node.get("exit_code"))
        if code == 76:
            return "배정 가능한 작업을 잡지 못함 · 같은 모델 재시도 대기 (종료 코드76)"
        if code == 75:
            return "자원 점검/잠금 확보 보류 · 같은 모델 재시도 대기 (종료 코드75)"
        if code in (129, 137):
            return f"자식 작업 중단 · 저장 작업 재시도 대기 (종료 코드{code})"
        reason = node.get("idle_reason", "")
        labels = {"worker failed; retrying saved work before fresh work": "직전 작업 실패 · 저장 작업 재시도 대기",
                  "worker interrupted; retrying saved work": "자식 작업 중단 · 저장 작업 재시도 대기",
                  "GPU allocation owned": "GPU 자원 확보 대기 · 상세 원인 미확인",
                  "current model waiting on peers or retry delay; later models blocked": "같은 모델 재시도 대기 · 상세 차단 조건 미기록"}
        return labels.get(reason, reason or "실행 대기 · 원인 미기록")
    return node["idle_reason"]


def node_rows(snapshots, *, coordinators=(), model=None, root=None, hide_stale=False):
    """Merge worker, held-task and coordinator evidence once per host."""
    sessions = {}
    for report in snapshots:
        for worker in report["workers"]:
            key = (worker.get("host", "-"), worker.get("worker_id", worker.get("pid")))
            row = {**worker, "label": report["label"], "task_row": next(
                (task for task in report["tasks"] if task["task"] == worker.get("task")), None)}
            # A shared worker writes an idle receipt to the inactive dataset.
            # In a dataset-only view this still describes a busy allocation.
            if row["status"] == "idle" and row.get("active_dataset"):
                row["status"] = "serving_other_dataset"
            previous = sessions.get(key)
            if previous is not None:
                current_age = heartbeat_age(row)
                previous_age = heartbeat_age(previous)
                if abs(current_age - previous_age) <= 5:
                    priority = {"running": 4, "serving_other_dataset": 3, "preflight": 2, "idle": 1}
                    if priority.get(previous["status"], 0) > priority.get(row["status"], 0):
                        continue
                elif current_age > previous_age:
                    continue
            sessions[key] = row
    for report in snapshots:
        for task in report["tasks"]:
            # reports.snapshot marks RUN only while an execution/task lease is
            # held. A missing/stale worker receipt must not hide that owner.
            host = task.get("host")
            if task["status"] != "running" or not host or host == "-":
                continue
            owner = task.get("worker_id")
            matches = [(key, row) for key, row in sessions.items()
                       if key[0] == host and row.get("task") == task["task"]
                       and row["label"] == report["label"]]
            key = ((host, owner) if owner else min(matches, key=lambda item: heartbeat_age(item[1]))[0]
                   if matches else (host, f"task:{task['task']}"))
            worker = sessions.get(key, {})
            sessions[key] = {**worker, "host": host, "worker_id": owner or worker.get("worker_id"),
                             "status": "running", "source": "task_lease",
                             "heartbeat_stale": bool(worker) and heartbeat_age(worker) > 60,
                             "task": task["task"], "task_row": task, "label": report["label"]}
    for coordinator in scoped_coordinators(coordinators, model=model, root=root):
        state = {"running": "launching", "waiting": "waiting", "switching": "switching",
                 "stale": "heartbeat_stale", "finished": "complete",
                 "stopped": "stopped", "failed": "failed"}.get(coordinator["status"])
        if state is None or (state not in LIVE_WORKERS and heartbeat_age(coordinator) > 900):
            continue
        reason = coordinator.get("idle_reason") or ("worker 시작 미확인 · 실행기 응답만 기록됨 · 원인 미기록"
                 if state in LIVE_WORKERS else "model launcher " + state)
        if not coordinator.get("root"):
            reason += "; experiment root not recorded"
        row = {**coordinator, "status": state, "source": "model_chain", "task": None,
               "task_row": None, "label": str(coordinator.get("active_dataset") or "model").upper(),
               "idle_reason": reason}
        sessions[(row["host"], "model-chain:" + str(row.get("worker_id", row.get("pid"))))] = row
    if model == "gemma4_it" and root is not None:
        from .startup import snapshot as startup_snapshot

        for row in startup_snapshot(root):
            sessions[(row["host"], "startup:" + row["worker_id"])] = {
                **row, "source": "startup", "task": None, "task_row": None, "label": "STARTUP"}
    nodes = {}

    def activity(row):
        # A newer idle launcher must not hide a different live session whose
        # task lease proves it is actually running on this host.
        if row.get("source") == "task_lease":
            return 4
        if row["status"] in {"running", "heartbeat_stale"} and row["task_row"] and row["task_row"]["status"] == "running":
            return 3
        if row["status"] == "serving_other_dataset":
            return 2
        # The launcher's child heartbeat is newer than its own worker's
        # admission receipt; show that admission, not "worker not registered".
        if row["status"] == "preflight" and row.get("source") != "model_chain" and heartbeat_age(row) <= 120:
            return 1.5
        if row.get("source") == "startup":
            return 1.25
        return int(row["status"] in LIVE_WORKERS)

    for row in sessions.values():
        if hide_stale and row["status"] == "heartbeat_stale":
            continue
        host = row.get("host", "-")
        previous = nodes.get(host)
        if previous is None or (activity(row), -heartbeat_age(row)) > (
                activity(previous), -heartbeat_age(previous)):
            nodes[host] = row
    return sorted(nodes.values(), key=lambda row: (row["status"] not in LIVE_WORKERS, row.get("host", "-")))


def progress(task):
    parts = []
    def fraction(done, total):
        if (type(done) is int and type(total) is int and total > 0
                and 0 <= done <= total):
            return f"{bar(done, total)} {100 * done / total:.1f}%"
        return "progress not recorded"

    if task["arm"] == "cache":
        if "cache_saved_prompts" in task:
            count = max(task["cache_saved_prompts"], task["cache_exported_prompts"])
            total = task["cache_total_prompts"]
            parts.append(f"{count}/{total} prompts {fraction(count, total)}")
            write_age = task.get("cache_last_write_age_seconds")
            if write_age is not None:
                parts.append(f"last write {age(write_age)} ago")
    else:
        if task.get("phase_stage") == "finalizing":
            reward = task.get("saved_reward_percent")
            parts.append("result saved" + (f" {reward:.1f}%" if reward is not None else ""))
            parts.append("waiting for worker/lease release")
        evaluating = task.get("phase_stage") == "evaluation"
        evaluation = task.get("final_evaluation")
        if evaluating:
            if task.get("evaluation_step") is not None:
                step = task['evaluation_step']
                parts.append(f"checkpoint eval at update {step}/{task['total_steps']}")
                if step < task['total_steps']:
                    parts.append(f"then resume training at update {step + 1}")
                elif task['arm'] == 'prefix':
                    parts.append("then publish prefix completion")
            else:
                parts.append(f"final eval at update {task.get('completed_steps', '?')}/{task['total_steps']}")
            if evaluation:
                saved, total = evaluation["saved_prompts"], evaluation["total_prompts"]
                if saved is None:
                    parts.append(f"EVAL ?/{total} prompts | awaiting saved progress")
                else:
                    parts.append(f"EVAL {saved}/{total} prompts {fraction(saved, total)}")
                    if saved == total:
                        parts.append("awaiting evaluation record" if task.get("evaluation_step") is not None
                                     else "awaiting endpoint")
            else:
                parts.append("EVAL progress not recorded")
        if task.get("current_step") is not None:
            parts.append(f"update {task['current_step']}/{task['total_steps']} in progress")
        if task.get("completed_steps") is not None:
            done, total = task["completed_steps"], task["total_steps"]
            parts.append(f"TRAIN {done}/{total} completed" + ("" if evaluating else f" {fraction(done, total)}"))
        elif not evaluating and task.get("total_steps"):
            parts.append(f"TRAIN ?/{task['total_steps']} completed | progress not recorded")
        if task.get("phase_stage"):
            parts.append(task["phase_stage"].replace("_", " "))
            parts.append(f"last progress {age(task.get('progress_age_seconds'))} ago")
        selection = selection_text(task)
        if selection is not None:
            check = task["selection_progress"]["check"]
            parts.append(f"SELECT {selection} prompts saved"
                         + (" (switch check: candidates + SR contrast)" if check else ""))
            if task["selection_progress"].get("read_error"):
                parts.append(f"selection cache read error: {task['selection_progress']['read_error']}")
        if evaluation:
            parts.append(f"{evaluation['responses_per_prompt']} responses/prompt")
            if evaluation.get("read_error"):
                parts.append(f"evaluation cache read error: {evaluation['read_error']}")
    if not parts:
        parts.append("starting; progress not recorded yet")
    if type(task.get("attempt")) is int and task["attempt"] > 0:
        parts.append(f"attempt {task['attempt']}")
        if task["attempt"] > 1 and task.get("log"):
            from .rank_failures import previous_attempt_line
            ended = previous_attempt_line(task["log"])
            if ended:
                parts.append(f"previous attempt ended: {ended}")
    ranks = []
    for row in task.get("progress", []):
        updated = row.get("updated")
        if isinstance(updated, (int, float)) and math.isfinite(updated) and row.get("stage"):
            rank = f"r{row['rank']} " if type(row.get("rank")) is int else ""
            ranks.append(f"{rank}{row['stage']} ({age(time.time() - updated)} ago)")
    if ranks:
        parts.append("rank progress: " + "; ".join(ranks))
    decision = live.switch_detail(task)
    if decision:
        parts.append(decision.split(" | transitions ", 1)[0])
    return " | ".join(parts)


def resume_backlog(snapshots, root):
    """Inspect the dispatch marker without claiming jobs or changing receipts."""
    path = Path(root) / ".dispatch/resume-first.json"
    if not path.exists():
        return [], None
    try:
        saved = live.read_object(path)
        entries = saved.get("tasks")
        if saved.get("schema") != "qwen-resume-first-v1" or not isinstance(entries, list):
            raise ValueError("invalid resume-first marker")
        lookup = {(report["label"], task["task"]): task for report in snapshots for task in report["tasks"]}
        visible = {report["label"] for report in snapshots}
        pending = {}
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2 or any(not isinstance(value, str) for value in entry):
                raise ValueError("invalid resume-first task")
            plan, key = entry
            if Path(plan).parent.resolve() != (Path(root) / "experiments").resolve():
                raise ValueError("resume-first plan is outside this experiment")
            label = "MATH" if Path(plan).name.endswith("-math.json") else "MBPP" if Path(plan).name.endswith("-mbpp.json") else None
            if label is None:
                raise ValueError("unrecognized resume-first dataset")
            task = lookup.get((label, key))
            if task is None and label in visible:
                raise ValueError("resume-first task is absent from this dataset report")
            if task is None or task["status"] != "complete":
                pending[(label, key)] = task or {"status": "outside_view"}
        return list(pending.values()), None
    except (OSError, ValueError, TypeError) as error:
        return [], f"resume-first status unavailable: {error}"


def idle_reason(node, snapshots, backlog):
    if node.get("idle_reason"):
        return " ".join(str(node["idle_reason"]).split())
    if backlog:
        counts = Counter(row["status"] for row in backlog)
        return (f"resume priority: {len(backlog)} unfinished, {counts['running']} running; "
                "independent READY work may run on free nodes")
    tasks = [row for report in snapshots for row in report["tasks"]]
    if any(report.get("stop_requested") for report in snapshots):
        return "stop requested"
    if tasks and all(row["status"] == "complete" for row in tasks):
        return "tasks complete; worker finishing"
    if any(row["status"] in {"ready", "recoverable", "interrupted"} for row in tasks):
        return "ready/resume work recorded; waiting for claim"
    if any(row["status"] in live.ATTENTION for row in tasks):
        return "failed/unavailable tasks; see ATTENTION"
    return "waiting for cache/prefix or another node"


def render(snapshots, root, *, host=None, now=None, model="Qwen3.5-9B", include_initial=False,
           status_model="qwen35", coordinators=None):
    host = socket.gethostname() if host is None else host
    now = time.time() if now is None else now
    tasks = [task for report in snapshots for task in report["tasks"]]
    counts = Counter(task["status"] for task in tasks)
    done, total = counts["complete"], len(tasks)
    waiting = sum(count for state, count in counts.items() if state.startswith("waiting_for_"))
    issues = sum(counts[state] for state in ERRORS)
    resumable = sum(counts[state] for state in RESUMABLE)
    from .model_chain import snapshot as chain_snapshot

    coordinators = chain_snapshot(now=now) if coordinators is None else coordinators
    coordinators = scoped_coordinators(coordinators, model=status_model, root=root)
    # Expired worker/coordinator records are not shown (2026-10-31 user request);
    # their heartbeat age stays in the JSON snapshot.
    nodes = node_rows(snapshots, coordinators=coordinators, model=status_model, root=root, hide_stale=True)
    local = next((row for row in nodes if row["host"] == host), None)
    local_state = (f"ACTIVE ({local['status'].upper()})" if local["status"] in LIVE_WORKERS
                   else local["status"].upper()) if local else "NOT REGISTERED"
    lines = [f"{model} | SRGC | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}",
        (f"Tasks {bar(done, total)} {done}/{total} completed"
         f" | RUN {counts['running']} | READY {counts['ready']} | WAIT {waiting}"
         f" | RESUME {resumable} | ERROR {issues}"),
        f"Local node: {host} | {local_state}"]
    error_count = sum(len(report["errors"]) for report in snapshots)
    if error_count:
        lines.insert(2, f"READ ERRORS: {error_count}; inspect ATTENTION below")
    if any(report.get("stop_requested") for report in snapshots):
        labels = ", ".join(report["label"] for report in snapshots if report.get("stop_requested"))
        lines.append(f"STOP REQUESTED: {labels}")
    backlog, backlog_error = resume_backlog(snapshots, root)
    if backlog_error:
        lines.append(backlog_error)
    if nodes:
        live_nodes = sum(row["status"] in LIVE_WORKERS for row in nodes)
        node_counts = Counter(row["status"] for row in nodes)
        lines += ["", (f"NODES  {live_nodes} active | {len(nodes) - live_nodes} stopped"
                       f" | RUN {node_counts['running'] + node_counts['serving_other_dataset']} | IDLE {node_counts['idle']}"
                       f" | ADMISSION {node_counts['preflight']}"
                       f" | PREPARING {node_counts['preparing']} | LAUNCH {node_counts['launching']} | WAIT {node_counts['waiting']}"
                       f" | HANDOFF {node_counts['switching']}")]
        rows = []
        for node in nodes:
            state = {"running": "RUN", "serving_other_dataset": "RUN OTHER", "preflight": "ADMISSION"}.get(
                node["status"], node["status"].upper())
            current = f"{node['label']} {node['task']}" if node.get("task") else "-"
            reason = idle_reason(node, snapshots, backlog) if node["status"] == "idle" else "-"
            heartbeat = age(node.get("heartbeat_age_seconds"))
            if node.get("source") == "task_lease":
                reason = leased_task_text(node)
            if node.get("source") == "model_chain":
                current = f"{node['model']} {node['label']} launcher"
                reason = launcher_text(node)
            if node.get("source") == "startup":
                current = f"worker pid={node['pid']}"
                reason = node["idle_reason"]
            if node["status"] == "preflight" and node.get("source") != "model_chain":
                reason = "worker 등록됨 · 실행 전 GPU·모델 점검"
            if node["status"] == "serving_other_dataset":
                current = str(node["active_dataset"])
                reason = "shared worker is running the other dataset"
            rows.append([("* " if node.get("host") == host else "") + node.get("host", "-"), state,
                         phase_cell(node["task_row"]) if node["task_row"] else "-",
                         current, cell(node["task_row"]).removeprefix("RUN ") if node["task_row"] else "-",
                         heartbeat, reason])
        lines += table(("Node", "State", "Phase", "Task", "Progress", "Heartbeat", "상태·대기 사유"), rows)
        from .model_chain import launch_details

        for node in nodes:
            if node.get("source") == "model_chain":
                details = launch_details(node)
                if details:
                    lines.append(f"  {node.get('host', '-')} launcher records: {details}")
    else:
        lines += ["", "NODES  no worker records"]
    for report in snapshots:
        rows = report["tasks"]
        if report.get("preparing"):
            lines += ["", f"{report['label']} PREPARING | initial model/input preparation lease is active"]
        panel = report.get("preflight")
        if panel:
            detail = (f"{panel['saved_prompts']}/{panel['total_prompts']} prompts recorded"
                      if "saved_prompts" in panel else "see ATTENTION")
            if "success_percent" in panel:
                detail += (f" | {panel['correct']:g}/{panel['responses']} correct ({panel['success_percent']:.2f}%)"
                           f" | empty {panel['empty_responses']} | truncated {panel['truncated_responses']}")
            lines += ["", f"{report['label']} PREFLIGHT {panel['state']} | {detail}"]
            if panel.get("path"):
                lines.append(f"  Fixed small panel; full initial-cache accuracy is separate. Records: {panel['path']}")
        if not report.get("prepared", True):
            if report.get("initialization_state") == "not_initialized":
                state = "PREPARING" if report.get("preparing") else "NOT INITIALIZED"
                lines += ["", f"{report['label']}  {state} | IT experiment plan has not been created",
                          f"  Plan: {report['missing_plan']}"]
                if report.get("stop_requested"):
                    lines.append("  Automatic preparation is stopped by the recorded stop request.")
                elif report.get("preparing"):
                    lines.append("  Automatic preparation is already in progress.")
                else:
                    lines.append("  Status/results automatically initialize missing plans on inspection.")
            else:
                lines += ["", f"{report['label']}  NOT PREPARED / UNREADABLE"]
            continue
        finished = Counter(row["arm"] for row in rows if row["status"] == "complete")
        train_total = sum(row["arm"] not in {"cache", "prefix"} for row in rows)
        train_done = sum(finished[arm] for arm in ARMS[2:])
        seeds = sorted({row["seed"] for row in rows})
        lines += ["", (f"{report['label']}  {bar(train_done, train_total)} train {train_done}/{train_total}"
                       f" | cache {finished['cache']}/{len(seeds)} | prefix {finished['prefix']}/{len(seeds)}")]
        lookup = {(row["seed"], row["arm"]): row for row in rows}
        def current_cell(row):
            if not row or row["status"] != "complete":
                return cell(row)
            reward = row.get("reward_percent")
            measured = type(reward) in (int, float) and math.isfinite(reward) and 0 <= reward <= 100
            return live.switch_cell(row, f"{reward:.1f}%" if measured else "DONE")

        lines += table(("Seed", *HEADERS), [[seed, *(current_cell(lookup.get((seed, arm))) for arm in ARMS)]
                                            for seed in seeds])
        initial = {item["seed"]: item["success_percent"]
                   for item in report.get("initial_candidate_accuracy", [])}
        if initial:
            lines.append("Init  " + " | ".join(
                f"seed {seed} {initial[seed]:.1f}%" if seed in initial else f"seed {seed} -"
                for seed in seeds))
        switching = [row for row in rows if live.switch_info(row) is not None]
        if switching:
            lines.append(f"{report['label']} SWITCH DECISIONS")
            decisions = []
            for row in switching:
                info = live.switch_info(row)
                decisions.append([row["seed"], info["state"],
                    info["step"] if info["step"] is not None else "-",
                    info["first_sr_update"] if info["first_sr_update"] is not None else "-",
                    info["selector"]])
            lines += table(("Seed", "Decision", "Switch after update", "First SR update", "Selector"), decisions)
        from .status_current import render_results

        lines.extend(render_results(report))
    running = [(report["label"], row) for report in snapshots for row in report["tasks"]
               if row["status"] == "running"]
    if running:
        lines += ["", "RUNNING"]
        details = []
        for label, row in running:
            done, total = row.get("completed_steps"), row.get("total_steps")
            train = f"{done if done is not None else '?'}/{total}" if total else "-"
            evaluation = row.get("final_evaluation")
            eval_count = (f"{evaluation['saved_prompts'] if evaluation['saved_prompts'] is not None else '?'}/{evaluation['total_prompts']}"
                          if evaluation and row.get("phase_stage") == "evaluation" else "-")
            details.append([label, row["task"], "RUN", phase_cell(row), train, eval_count,
                            selection_text(row) or "-", row.get("attempt", "-")])
        lines += table(("Dataset", "Task", "State", "Phase", "Train", "Eval", "Select", "Attempt"), details)
        for label, row in running:
            lines.append(f"  {label} {row['task']} | node {row.get('host') or '-'} | {progress(row)}")
    pending = [(report["label"], row) for report in snapshots for row in report["tasks"]
               if row["status"] in RESUMABLE]
    if pending:
        lines += ["", "RESUME PENDING"]
        for label, row in pending:
            reason = "worker lease released" if row["status"] == "recoverable" else "execution interrupted"
            lines.append(f"  {label} {row['task']} | RESUME | node {row.get('host') or '-'} | {reason}")
    attention = [(report["label"], row) for report in snapshots for row in report["tasks"]
                 if row["status"] in ERRORS]
    errors = [(report["label"], error) for report in snapshots for error in dict.fromkeys(report["errors"])]
    worker_errors = [(node, message) for node in nodes if (message := worker_error(node))]
    if attention or errors or worker_errors:
        lines += ["", "ATTENTION"]
        for label, row in attention:
            reason = row.get("validation_error") or last_error_line(row.get("log", ""))
            lines.append(f"  {label} {row['task']} | {cell(row)}" + (f" | {reason}" if reason else ""))
            if row.get("log"):
                lines.append(f"    log: {row['log']}")
        for node, message in worker_errors:
            lines.append(f"  node {node.get('host', '-')} | {message}")
        lines += [f"  {label} | {error}" for label, error in errors]
    warnings = [(report["label"], warning) for report in snapshots for warning in dict.fromkeys(report["warnings"])]
    if warnings:
        lines += ["", "WARNINGS", *(f"  {label} | {warning}" for label, warning in warnings)]
    lines += ["", "State RUN = live execution lease. Phase TRAIN/EVAL can change within the same attempt.",
              "Train = completed updates; Eval = saved prompts. They use separate totals. ? = progress not recorded.",
              "FINALIZING = validated result saved; waiting for worker/lease release.",
              "Seed table % = held-out final eval reward. Init = initial policy on training candidates (not held-out).",
              "RUN = active lease; LAUNCH/WAIT/HANDOFF = live dispatcher. SW@N = switch after update N.",
              f"Root: {root}"]
    from .model_chain import render_status
    lines.extend(render_status(now=now, rows=[row for row in coordinators
                                            if row["status"] in {"running", "switching", "waiting"}]))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("math", "mbpp", "all"))
    parser.add_argument("action", choices=("status",))
    parser.add_argument("--root", type=Path, default=default_root(os.environ))
    parser.add_argument("--json", action="store_true", help="print the same recorded status as JSON")
    args = parser.parse_args(argv)
    try:
        snapshots = load_reports(args.dataset, args.root, os.environ)
        from .status_current import print_current
        if args.json:
            from .model_chain import snapshot as chain_snapshot

            coordinators = chain_snapshot()
            print(json.dumps({"model": "Qwen/Qwen3.5-9B", "root": str(args.root.resolve()),
                              "datasets": snapshots, "model_chain": coordinators,
                              "nodes": node_rows(snapshots, coordinators=coordinators,
                                                 model="qwen35", root=args.root)}, indent=2))
        else:
            text = render(snapshots, args.root.resolve())
            print_current(text)
        return int(any(report["errors"] for report in snapshots))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"STATUS ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
