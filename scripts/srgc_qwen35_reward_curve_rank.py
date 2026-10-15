#!/usr/bin/env python3
"""Keep Qwen rank isolation while adding durable rewards and rollout records."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from srgc_research.dispatch.reward_curve_rank import main
    sys.argv[1:1] = ["--model", "qwen35"]
    main()
