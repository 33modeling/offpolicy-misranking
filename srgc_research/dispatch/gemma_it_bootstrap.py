"""Prepare absent IT plans through the normal launcher before status/results."""

from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import uuid

from srgc_rebuttal import cluster
from srgc_rebuttal.reports import locked
from srgc_rebuttal.runtime import Busy, lease

REPO = Path(__file__).resolve().parents[2]


def preparation_lock(root):
    root = Path(root)
    return root.parent / f".{root.name}.auto-prepare.lock"


def has_saved_work(root, dataset):
    root = Path(root)
    run = root / "runs" / dataset
    return ((run / ".queue/protocol.json").exists()
            or any((run / ".queue/tasks").glob("*.json"))
            or any(run.glob("seed-*"))
            or any((root / "inputs").glob(f"{dataset}-seed-*.cache"))
            or any((root / "inputs").glob(f"{dataset}-seed-*.cache-responses.jsonl")))


def discard_previous_pt_path(environment):
    """Select IT automatically when an older generic override still names PT."""
    old_path = environment.get("SRGC_GEMMA_MODEL_PATH")
    if old_path and not environment.get("SRGC_GEMMA_IT_MODEL_PATH"):
        native = Path(old_path) / "generation_config.json"
        if native.is_file() and json.loads(native.read_text()).get("eos_token_id") == 1:
            environment.pop("SRGC_GEMMA_MODEL_PATH")
            environment.pop("SRGC_GEMMA_MODEL_SHA256", None)
            print("GEMMA selecting pinned IT weights instead of the previous PT path", file=sys.stderr, flush=True)


def ensure_plans(root, dataset, environment):
    """Create real cohort inputs once; never start admission/cache/training."""
    from .gemma4_it.storage import group_work, inside

    group, _ = group_work(environment)
    root = inside(root, group)
    inside(root / "experiments", root)
    names = ("math", "mbpp") if dataset == "all" else (dataset,)

    def missing():
        return [name for name in names
                if not (root / "experiments" / f"gemma4-12b-it-{name}.json").is_file()]

    def stopped():
        return any((root / f".startup-control/{name}/stop.json").exists()
                   or (root / f"runs/{name}/.queue/stop.json").exists() for name in names)

    absent = missing()
    if not absent or stopped() or any(has_saved_work(root, name) for name in absent):
        return
    for path in (root / "experiments").glob("*.json"):
        plan = json.loads(path.read_text())
        if not isinstance(plan, dict) or plan.get("model") != "google/gemma-4-12B-it":
            raise ValueError(f"IT root contains another model's experiment plan: {path}; existing files preserved")
    if locked(root / ".start-prepare.lock"):
        return  # The running launcher owns source selection and preparation.
    try:
        with lease(preparation_lock(root)), lease(root / ".start-prepare.lock") as guard:
            absent = missing()
            if not absent or stopped() or any(has_saved_work(root, name) for name in absent):
                return
            selected = "all" if len(absent) == 2 else absent[0]
            command = ["sh", str(REPO / "scripts/run_srgc_gemma4.sh"), selected, "prepare",
                       "--root", str(root)]
            child_env = {**environment, "SRGC_GEMMA_IT_PROTOCOL": "1",
                         "SRGC_GEMMA_LEGACY_PROTOCOL": "0", "SRGC_MODEL_CHAIN": "0",
                         "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}
            # This launcher starts a new verification handoff. An old PT
            # parent's digest/path must not block or overwrite the IT snapshot.
            child_env.pop("SRGC_GEMMA_MODEL_SHA256", None)
            discard_previous_pt_path(child_env)
            log = root / "startup-logs" / f"auto-prepare-{uuid.uuid4().hex}.log"
            print(f"GEMMA automatically preparing missing IT plans ({', '.join(absent)}); "
                  f"no training launched; log={log}", file=sys.stderr, flush=True)
            # The existing child runner streams output, preserves the raw log,
            # and tears down the entire process group on cancellation.
            with redirect_stdout(sys.stderr):
                code = cluster.run_child(command, log, child_env, pass_fds=(guard.fileno(),),
                                         should_stop=stopped, interval=1)
            if code:
                raise RuntimeError(f"Gemma automatic preparation failed (exit {code}); log: {log}")
            if missing():
                raise RuntimeError(f"Gemma preparation did not create required IT plans: {', '.join(missing())}; "
                                   f"log: {log}")
    except Busy:
        return  # Another status request is already creating these plans.
