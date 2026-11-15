"""Print the last completed update of every Llama run, two lines per dataset."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from srgc_research.dispatch.result_export import work_root  # noqa: E402


def main():
    path = work_root() / "results" / "llama31-results.json"
    if not path.is_file():
        print(f"NO RESULT FILE: {path}")
        return 1
    data = json.loads(path.read_text())
    for name, snapshot in data.get("datasets", {}).items():
        plan = snapshot.get("plan", {})
        total = plan.get("total_updates")
        tasks = snapshot.get("tasks", [])
        short = [t for t in tasks if t.get("step") != total]
        questions = sorted({len(e["endpoint"]["per_question_reward"])
                            for e in snapshot.get("endpoints", [])
                            if isinstance(e.get("endpoint", {}).get("per_question_reward"), list)})
        print(f"{name.upper()}: planned {total} updates, {len(tasks)} runs, "
              f"{len(tasks) - len(short)} reached {total}, eval questions {questions or 'not in file'}, "
              f"responses {plan.get('responses')}")
        for t in short:
            print(f"  SHORT seed {t.get('seed')} {t.get('arm')}: step {t.get('step')} ({t.get('status')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
