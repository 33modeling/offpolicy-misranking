#!/bin/sh
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
exec python3 "$HERE/v8_stop.py"
