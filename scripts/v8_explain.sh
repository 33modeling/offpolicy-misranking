#!/bin/sh
# V8 결과 보고서 (한글): sh scripts/v8_explain.sh -> 화면 + ~/v8-explain.txt (재집계 영문 원본은 ~/v8-recount.txt)
# cohort마다 원본 값(실험 설정, D 신호, 정책별 시드 행, 비교, 비용)을 전부 적고 항목마다 아래에 무슨 실험인지와 결과를 단다. 학습·평가는 건드리지 않는다.
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
V8_EXPLAIN_ONLY=1 exec sh "$HERE/v8_recount.sh" "$@"
