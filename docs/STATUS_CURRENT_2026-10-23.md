# status 화면의 현재 결과·GPU 시간

사용자 정정: JSON 파일을 저장하라는 뜻이 아니라 status에 결과와 GPU 시간도
표시하라는 지시다. 앞선 자동 JSON 저장 구현을 제거했다.

Gemma IT/PT, Qwen, Llama의 기존 `status` 명령은 현재 작업·노드·오류와 함께
seed/arm별 `Result (%)`, `GPU-h`를 간단한 표로 표시한다. 현재 실험의 측정된
결과를 읽으며, 조회할 때 이전 화면이나 파일의 값을 더하지 않는다.
전체 시도별·단계별 GPU 통계, 재시작 분석과 과거 워커 히스토리를 덧붙이지 않는다.

status는 결과 JSON 파일을 새로 만들거나 덮어쓰지 않는다. 이전에 생성된 파일과
실험의 원시 timer, reward curve, 체크포인트, 기존 `results` 파일도 수정하지 않는다.
명시적인 `--json`은 화면 대신 stdout에 JSON을 출력하는 옵션으로 유지한다.
준비된 실험의 status는 읽기 전용이며, 기존 빈 IT root의 초기 준비 동작은 그대로다.
터미널에서는 현재 화면을 교체하고 파이프·JSON에는 화면 제어 문자를 넣지 않는다.

화면의 continuation GPU 시간은 기존 retained-update 기준이다. cache/prefix는
공통 비용으로 구분한다. 기록된 부분 시간은 `>=`, 미측정·잘못된 기록은 `-`로
표시하며 측정된 0과 구분한다. 검증된 완료 결과만 Result 열에 표시한다.

학습은 완료 update/전체 update와 퍼센트를 표시한다. 진행 중인 update를 완료 수에
포함하지 않는다. 최종 및 중간 checkpoint eval은 저장된 prompt/전체 prompt와
퍼센트를 표시한다. 중간 평가는 해당 update의 sampling stream과 cache를 읽고,
새 학습 기록이 나오면 과거 eval을 다시 표시하지 않는다. 저장 수가 불명확하면
`?`로 표시한다. 모든 prompt를 저장해도 결과 기록 전에는 task를 DONE으로 바꾸지 않는다.

[회귀 검사](../srgc_research/tests/test_status_current.py)는 결과·GPU 시간의 화면 표시,
반복 조회 시 비누적·무수정, JSON 파일 쓰기 호출 금지, dataset 범위, 측정된 0과
미측정·부분·잘못된 값의 구분, 학습 및 최종/중간 eval 진행률을 검사한다.
원시 검증 로그는 `/home/kms/reports/status-display-20261023/`에 보존한다.
최종 **192개 검사가 68.25초에 통과**했다.
[검증 요약](evidence/status-display-20261023.json)에 범위와 원시 로그 경로를 기록했다.
검사 수치는 CPU fixture이며 실제 GPU 실험 결과가 아니다. 학습 프로세스나 원격
노드를 시작·재시작하지 않고, 학습 adapter/engine digest도 보존한다.
