#!/bin/sh
# V8 결과 해석 (한글): sh scripts/v8_explain.sh -> 화면 + ~/v8-explain.txt (숫자 전체는 ~/v8-recount.txt)
# 학습·평가는 건드리지 않는다. 기록된 결과를 현재 코드로 다시 집계하고 cohort마다 판정과 해석을 쓴다.
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
V8_EXPLAIN_ONLY=1 exec sh "$HERE/v8_recount.sh" "$@"
