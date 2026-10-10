"""Run the unchanged measurement CLI with cleanup scoped to its own ranks."""

from pathlib import Path
from unittest.mock import patch

from scripts import srgc_process_guard as guard
from srgc_research import information_cli

TARGETS = ("srgc_research/information_rank.py", "srgc_research/dispatch/information_rank_run.py")
OWNERS = ("srgc_research.information_cli", "srgc_research.dispatch.information_run",
          "srgc_research.dispatch.information_queue")
RANK_WRAPPER = Path(__file__).with_name("information_rank_run.py")


def rank_dispatch(runner):
    def run_child(command, *args, **kwargs):
        command = list(command)
        if "torch.distributed.run" in command:
            for index, argument in enumerate(command):
                if Path(argument).as_posix().endswith("/srgc_research/information_rank.py"):
                    command[index:index + 1] = [str(RANK_WRAPPER), "--rank-script", argument]
                    break
        return runner(command, *args, **kwargs)
    return run_child


def main(argv=None):
    from srgc_rebuttal import cluster
    with patch.object(guard, "TARGET_MARKERS", TARGETS), \
            patch.object(guard, "OWNER_MARKERS", OWNERS), \
            patch.object(cluster, "run_child", rank_dispatch(cluster.run_child)):
        return information_cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
