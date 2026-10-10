"""Run the unchanged measurement CLI with cleanup scoped to its own ranks."""

from unittest.mock import patch

from scripts import srgc_process_guard as guard
from srgc_research import information_cli

TARGETS = ("srgc_research/information_rank.py",)
OWNERS = ("srgc_research.information_cli", "srgc_research.dispatch.information_run",
          "srgc_research.dispatch.information_queue")


def main(argv=None):
    with patch.object(guard, "TARGET_MARKERS", TARGETS), \
            patch.object(guard, "OWNER_MARKERS", OWNERS):
        return information_cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
