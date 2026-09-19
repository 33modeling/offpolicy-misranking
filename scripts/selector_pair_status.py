"""Read-only selector-pair dashboard, using the existing status table format."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mbpp_status as display
from _switch_state_point import resolve_state_point
import selection_gate as core
import selector_pair as pair
import selector_pair_gpu as gpu

LABELS = {"on_policy": "On-policy", "cached": "Cached", "adaptive": "Adaptive", "random": "Random"}


def read(path):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except FileNotFoundError:
        return {"_error": "saved link unavailable"} if path.is_symlink() else {}
    except (OSError, ValueError) as exc:
        return {"_error": str(exc)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observe_branch(root, seed, step, name, branch, *, ready, observations):
    role = "development" if seed in pair.DEV_SEEDS else "test"
    task = dict(seed=seed, step=step, name=name, role=role, status="READY" if ready else "WAIT",
                reason="" if ready else "준비·테스트 결정 고정 대기", directory="")
    if branch is None:
        return task
    point, error = resolve_state_point(root / "branches" / branch / "states" / f"s{seed}-t{step}", step)
    arm = "selection_reduced" if role == "development" else "random_full" if name == "random" else "selection_full"
    directory = point / arm
    task["directory"] = str(directory.relative_to(root))
    result, receipt, curve = (read(directory / path) for path in ("result.json", "result.sha256.json", "curve.json"))
    if error:
        task.update(status="WAIT", reason=error)
    elif result or receipt or curve:
        try:
            if (result.get("complete") is not True or result.get("schema") != display.switch_status.rule.SCHEMA
                    or receipt.get("sha256") != digest(directory / "result.json")):
                raise ValueError("결과·발행 영수증 검증 필요")
            task["training_published"] = True
            if not curve:
                raise ValueError("최종 평가 저장됨; 곡선 평가 남음 (재학습 없음)")
            if (curve.get("schema") != display.switch_status.rule.SCHEMA
                    or curve.get("result_sha256") != receipt["sha256"]
                    or not isinstance(curve.get("points"), dict) or not curve["points"]):
                raise ValueError("곡선·결과 연결 검증 필요")
            task.update(status="DONE", reason="결과·곡선 저장 완료")
        except (OSError, ValueError) as exc:
            task.update(status="EVAL" if task.get("training_published") and not curve else "WAIT", reason=str(exc))
    else:
        attempt = read(directory / "pair-attempt.json")
        saved = display.switch_status.saved_policy_state(directory / "policy")
        if attempt.get("error"):
            detail = display.switch_status.short_error(attempt["error"])
            task.update(status="BUDGET" if "budget exhausted" in detail or "allocation exhausted" in detail else "WAIT",
                        reason="실패 원인 확인 필요: " + detail)
        elif saved:
            task.update(status=saved[0], reason=saved[1])
        elif attempt:
            task.update(status="WAIT", reason="이전 실행 기록; 완료 여부 미확인")
    relevant = [(updated, path, value) for updated, path, value in observations
                if path.parent == directory or directory in path.parents]
    fresh = [(updated, path, value) for updated, path, value in relevant if value.get("_fresh")]
    if fresh and task["status"] != "DONE":
        task.update(status="RUN", reason="", **{key: fresh[0][2].get(key) for key in
                    ("host", "phase", "seconds", "timeout")})
    elif relevant and task["status"] == "READY":
        task.update(status="WAIT", reason="실행 신호 끊김; 확인 필요")
    return task


def snapshot(root, *, now=None):
    root = Path(root).resolve()
    now = time.time() if now is None else now
    p = read(root / "pair.json")
    prepared, error, choices = False, "", {}
    try:
        if p.get("_error"):
            raise ValueError(p["_error"])
        if p.get("schema") == pair.SCHEMA:
            if p.get("protocol_id") != core.fingerprint({k: v for k, v in p.items() if k != "protocol_id"}):
                raise ValueError("Pair manifest changed")
            if (not isinstance(p.get("branch_manifests"), dict)
                    or set(p["branch_manifests"]) != set(gpu.BRANCHES)):
                raise ValueError("Branch manifests missing")
            for name, value in p["branch_manifests"].items():
                if digest(root / "branches" / name / "switch.json") != value:
                    raise ValueError("Frozen branch manifest changed: " + name)
            prepared = True
            if (root / "test-decisions.json").exists():
                choices = gpu.decisions(root, p)
        elif p and p.get("schema") != gpu.BOOTSTRAP_SCHEMA:
            raise ValueError("Unknown pair manifest")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        error = str(exc)
    observations = gpu.pair_progress(root)
    nodes, activity = {}, []
    for updated, path, value in observations:
        fresh = value.get("state") == "running" and -5 <= now - updated < 60
        value["_fresh"] = fresh
        host = str(value.get("host") or "unknown")
        node = nodes.setdefault(host, {"host": host, "current": False})
        if not fresh:
            continue
        node["current"] = True
        node["progress_age"] = now - updated
        relative = str(path.parent.relative_to(root))
        match = re.search(r"states/s(\d+)-t(\d+)/", relative)
        if not match:
            node["state"] = "ADMIT"
            continue
        branch_name = relative.split("/")[1]
        arm = ("random" if "random_full" in path.parts else "adaptive" if branch_name.startswith("adaptive-")
               else branch_name)
        if "/curve-parent" in relative:
            arm = "curve-parent"
        elif "/measurement" in relative or "/gate_measurement" in relative:
            arm = "diagnostic"
        task = {**value, "directory": relative, "seed": int(match[1]), "step": int(match[2]),
                "kind": "phase", "arm": arm, "status": "RUNNING", "heartbeat_fresh": True,
                "training_step": display.switch_status.last_training_step(path.parent / "policy/grpo_stats.jsonl")}
        activity.append(task)
    for path in sorted((root / "queue-workers").glob("*.json")):
        worker = read(path)
        if not worker or worker.get("protocol_id") != p.get("protocol_id"):
            continue
        host = str(worker.get("host") or "unknown")
        node = nodes.setdefault(host, {"host": host, "current": False})
        if display.switch_status.number(worker.get("updated")) > node.get("worker_updated", 0):
            node.update(worker=worker, worker_updated=display.switch_status.number(worker.get("updated")))
    for node in nodes.values():
        worker = node.get("worker", {})
        fresh = -5 <= now - node.get("worker_updated", 0) < 60
        node["current"] |= fresh and worker.get("state") in {"RUN", "WAIT"}
        node.setdefault("state", "WAIT" if fresh and worker.get("state") == "WAIT" else
                        "LIVE" if fresh and worker.get("state") == "RUN" else "STALE")
        match = re.fullmatch(r"(?:development|test)/s(\d+)-t(\d+)", str(worker.get("task", "")))
        if (fresh and worker.get("state") == "RUN" and match
                and not any(task["host"] == node["host"] for task in activity)):
            activity.append(dict(host=node["host"], kind="phase", arm="상태 작업", seed=int(match[1]),
                                 step=int(match[2]), directory=worker["task"], status="RUNNING",
                                 heartbeat_fresh=True, phase="분기 단계 확인 중"))
    tasks = []
    for seed in (*pair.DEV_SEEDS, *pair.TEST_SEEDS):
        for step in pair.STEPS:
            development = seed in pair.DEV_SEEDS
            for name in (pair.SELECTORS if development else LABELS):
                branch = "on_policy" if name == "random" else name
                if name == "adaptive":
                    choice = choices.get(f"s{seed}-t{step}", {}).get("selector")
                    branch = f"adaptive-{choice}" if choice in pair.SELECTORS else None
                task = observe_branch(root, seed, step, name, branch,
                                      ready=prepared and (development or bool(choices)),
                                      observations=observations)
                if not prepared or (not development and error):
                    task.update(status="RUN" if task["status"] == "RUN" else "WAIT",
                                reason=error or "실험 설정 확인 불가; 완료 여부 미확인")
                tasks.append(task)
    return dict(root=str(root), updated=now, prepared=prepared, error=error, tasks=tasks,
                nodes=sorted(nodes.values(), key=lambda node: display.switch_status.node_view.host_sort_key(node["host"])),
                activity=activity, target_reward=p.get("target_reward"),
                budget_gpu_seconds=p.get("training_cap_gpu_seconds"),
                test_decisions_frozen=bool(choices))


def dashboard_data(data):
    """Adapt Pair evidence to the same tables, counters and node view as MBPP."""
    tasks = [{**task, "kind": "branch", "arm": task["name"],
              "status": "RUNNING" if task["status"] == "RUN" else task["status"]}
             for task in data["tasks"] if data["prepared"]]
    tasks += data["activity"]
    root = Path(data["root"])
    branch_root = root / "branches/on_policy"
    protocol = read(branch_root / "switch.json")
    for seed in (*pair.DEV_SEEDS, *pair.TEST_SEEDS):
        for step in pair.STEPS:
            path = branch_root / "prefixes" / f"seed-{seed}" / f"prefix-{step}.json"
            cert = read(path)
            valid = False
            try:
                expected = protocol["prefix_source"]["seeds"][str(seed)][f"prefix-{step}"]
                valid = (data["prepared"] and cert.get("schema") == display.switch_status.rule.SCHEMA
                         and cert.get("seed") == seed and cert.get("step") == step and digest(path) == expected)
            except (OSError, KeyError, TypeError):
                pass
            tasks.append(dict(kind="prefix", seed=seed, step=step, arm="prefix",
                              directory=str(path.parent.relative_to(root)),
                              status="DONE" if valid else "WAIT", reason="" if valid else "인증된 공통 학습 기록 확인 필요"))
    nodes = []
    for node in data["nodes"]:
        age = data["updated"] - node.get("worker_updated", 0)
        if "progress_age" in node:
            age = min(age, node["progress_age"])
        nodes.append(dict(host=node["host"], state=node["state"], last_age=age))
    suite = dict(root=data["root"], prepared=data["prepared"], error=data["error"],
                 display_label="Selector pair", tasks=tasks, nodes=nodes,
                 registered_tasks=[(s, t, arm) for s in (*pair.DEV_SEEDS, *pair.TEST_SEEDS)
                                   for t in pair.STEPS for arm in (pair.SELECTORS if s in pair.DEV_SEEDS else LABELS)],
                 protocol={"accounting": "matched", "budget_gpu_seconds": data["budget_gpu_seconds"]},
                 training_published=sum(task.get("training_published", False) for task in tasks),
                 details=[f"개발 18개 / 검증 24개 | 목표 보상: {data['target_reward']}",
                          "테스트 결정 고정: " + ("DONE" if data["test_decisions_frozen"] else "WAIT"),
                          *(["WAIT: " + data["error"]] if data["error"] else [])])
    return dict(updated=data["updated"], suites=[suite], subject="SELECTOR PAIR", arm_names=LABELS,
                legend=["On-policy: 현재 정책 gradient 기반 선택. Cached: 저장된 정답률 기반 선택.",
                        "Adaptive: 개발 데이터로 고정한 전환 규칙. Random: 무작위 선택 대조군.",
                        "선택비용 별도도 총 GPU 비용에는 포함합니다. 평가 비용은 모든 조건에서 별도로 기록합니다.",
                        "GPU 시간 한도에 도달해도 평가 결과가 없으면 미완료이며, 보상 0점이 아닙니다.",
                        "'-': 해당 상태에서 실행 대상 아님. 완료는 결과 영수증·곡선 기록 기준이며 전체 검증은 report에서 수행합니다."])


def render(data, *, width=120, all_tasks=False):
    return display.render(dashboard_data(data), width=width, all_tasks=all_tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    data = snapshot(args.root)
    print(json.dumps(data, indent=2) if args.json else render(data,
          width=shutil.get_terminal_size((120, 40)).columns, all_tasks=args.all), flush=True)
    return int(bool(data["error"]))


if __name__ == "__main__":
    raise SystemExit(main())
