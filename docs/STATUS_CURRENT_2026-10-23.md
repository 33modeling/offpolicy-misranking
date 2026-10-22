# 현재 상태와 학습·eval 진행률

사용자 지시: status는 누적 결과를 늘어놓지 않고 현재 현황을 보여준다.
GPU 시간·결과는 최신 값만 저장하고, 학습과 eval 진행률은 화면에 유지한다.

Gemma IT/PT, Qwen, Llama의 기존 `status` 명령을 그대로 사용한다.
화면에는 현재 작업·노드·오류·진행률을 표시한다. GPU 시간의 전체 시도별 표,
단계별 누적 표, 재시작 분석, 숨긴 워커 히스토리 및 과거 PT 안내는 출력하지 않는다.
터미널에서는 화면을 교체하고, 파이프·JSON 출력에는 화면 제어 문자를 넣지 않는다.

각 조회는 `$OM_WORK/status/<model>-latest.json` 하나를 원자적으로 교체한다.
이전 파일을 읽거나 시간·결과를 더하지 않는다. 데이터셋을 한정해 조회하면
파일도 해당 조회 범위로 교체하며, 과거 조회의 다른 데이터셋을 합치지 않는다.
`--json`의 `saved_results`와 텍스트 마지막 줄에서 저장 위치를 확인할 수 있다.
저장 오류가 발생해도 현재 화면을 표시하고, stderr와 종료 코드로 실패를 알린다.

최신 파일에는 모델·revision·실험 root·plan 경로, seed·arm별 현재 결과와 GPU
시간을 저장한다. continuation은 기존 retained-update 기준을 사용하며,
cache/prefix 시간은 공통 비용으로 구분한다. 미측정 값은 MISSING/null로 남긴다.
실험의 원시 timer, reward curve, 체크포인트와 기존 `results` 파일은 변경하지 않는다.

학습은 완료 update 수/전체 update 수와 퍼센트를 표시한다. 진행 중인 update를
완료 수에 포함하지 않는다. eval은 저장된 prompt 수/전체 prompt 수와 퍼센트를
표시한다. 최종 eval과 별도 checkpoint eval을 구분하고, 중간 평가는 해당 update의
sampling stream과 cache를 읽는다. 새 학습 기록이 나오면 이전 eval을 다시 표시하지 않는다.
저장 수가 불명확하면 `?`로 표시하며 0%로 추정하지 않는다. 모든 prompt를 저장해도
eval 결과 기록이 끝날 때까지 task를 DONE으로 바꾸지 않는다.

검증은 [신규 회귀 검사](../srgc_research/tests/test_status_current.py)와 기존
상태·노드·히스토리·switch·GPU 시간·submitted-time 검사로 수행한다.
반복 조회 시 파일이 하나만 남고 시간 값이 더해지지 않는지, 새 값과 조회 범위로
교체되는지, 학습·최종/중간 eval 진행률, JSON/TTY 출력, 저장 실패 및 손상된
checkpoint 진행 기록의 격리를 검사한다. 학습 adapter/engine digest도 이전 값과 비교한다.

최종 **191개 검사가 69.04초에 통과**했다. adapter/engine digest도 일치했다.
[검증 요약](evidence/status-current-20261023.json)에 검사 범위와 원시 로그 경로를 기록했다.

원시 검증 로그: `/home/kms/reports/status-current-20261023/`.
모든 수치 예시는 CPU 회귀 fixture이며 실제 GPU 실험 결과가 아니다.
이 작업으로 학습이나 원격 노드를 시작·재시작하지 않았다.
