"""Read-only execution views; publication receipts remain a separate fact."""


def active(task):
    return bool(task.get("owner_active") or task.get("heartbeat_fresh", task.get("status") == "RUNNING"))


def execution_state(task, running_directories=()):
    directory = str(task.get("directory") or "").rstrip("/")
    if (active(task) or task.get("task_lease_held") or
            directory and any(path.startswith(directory + "/") for path in running_directories)):
        return "RUNNING"
    return task.get("status", "UNKNOWN")


def execution_tasks(tasks):
    """Project one root at a time so relative paths cannot join different suites."""
    directories = [str(task["directory"]).rstrip("/") for task in tasks
                   if task.get("directory") and (active(task) or task.get("task_lease_held"))]
    return [{**task, "publication_status": task.get("publication_status", task.get("status")),
             "execution_inferred": not (active(task) or task.get("task_lease_held"))
                                   and execution_state(task, directories) == "RUNNING",
             "status": execution_state(task, directories)} for task in tasks]


def current_tasks(tasks):
    """Prefer the actual nested phase over its parent receipt in CURRENT views."""
    running = [task for task in tasks if task.get("status") == "RUNNING"
               and not task.get("execution_inferred")]
    return [task for task in running if not any(
        other is not task and task.get("directory") and
        str(other.get("directory", "")).startswith(task["directory"].rstrip("/") + "/") and
        (other.get("host"), other.get("worker_id")) == (task.get("host"), task.get("worker_id"))
        for other in running)]
