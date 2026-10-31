#!/bin/sh
# Run on a node stuck at LAUNCH: ends only launch children waiting on a startup
# lock so the model chain relaunches them from the pulled code.
cd "$(dirname "$0")/.."
exec python3 scripts/srgc_launch_unstick.py "$@"
