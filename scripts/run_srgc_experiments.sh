#!/bin/sh
# Common entry point; training environments remain owned by the existing runners.
set -eu
cd "$(dirname "$0")/.."
exec python3 scripts/srgc_experiments.py "$@"
