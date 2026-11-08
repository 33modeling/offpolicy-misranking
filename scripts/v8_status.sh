#!/bin/sh
# 읽기 전용 V8 현황: sh scripts/v8_status.sh  (계속 보기: sh scripts/v8_status.sh --watch)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
exec python3 "$HERE/v8_status.py" "$@"
