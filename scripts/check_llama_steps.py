"""Save the last completed update of every Llama run to a text file under exports/."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from srgc_research.dispatch.result_export import work_root  # noqa: E402


def main():
    work = work_root()
    path = work / "results" / "llama31-results.json"
    lines = []
    out = work / "exports" / time.strftime("llama-steps-%Y%m%dT%H%M%SZ.txt", time.gmtime())
    out.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        out.write_text(f"NO RESULT FILE: {path}\n")
        print(out)
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
        lines.append(f"{name.upper()}: planned {total} updates, {len(tasks)} runs, "
              f"{len(tasks) - len(short)} reached {total}, eval questions {questions or 'not in file'}, "
              f"responses {plan.get('responses')}")
        for t in short:
            lines.append(f"  SHORT seed {t.get('seed')} {t.get('arm')}: step {t.get('step')} ({t.get('status')})")
    out.write_text("\n".join(lines) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
