# SRGC status의 전환 시점 표시

Qwen3.5, Llama3.1, Gemma4의 공개 status 명령과 공통 SRGC status,
추가 switch 실험의 status에 저장된 전환 시점을 표시한다.

```sh
sh scripts/run_srgc_qwen35.sh status
sh scripts/run_srgc_llama31.sh status
sh scripts/run_srgc_gemma4.sh status
sh scripts/run_srgc_qwen35.sh math status --json
```

`Switch` 셀의 `SW@75`는 학습 update 75가 완료된 시점에서 전환했다는 뜻이다.
SR로 학습하는 첫 update는 76이다. 데이터셋별 `SWITCH DECISIONS` 표에는
각 seed의 전환 여부, 전환 경계, 첫 SR update, 저장된 selector를 함께 표시한다.
진행 중인 항목의 상세 줄에도 같은 정보를 표시한다.

| 표시 | 근거 |
| --- | --- |
| `SWITCHED`, `SW@N` | 저장된 progress 또는 검증된 endpoint에 전환 경계 N이 있다. |
| `NOT YET` | 저장된 progress에는 전환이 없다고 기록되어 있다. 이후 전환 여부는 아직 모른다. |
| `NEVER SWITCHED` | 검증된 최종 endpoint에 전환이 없다고 기록되어 있다. |
| `PENDING` | 대기 중이며 전환 기록이 아직 없다. |
| `NOT RECORDED` | 실행 중이거나 중단됐지만 확인할 수 있는 전환 기록이 없다. |
| `INVALID RECORD` | 전환 step의 자료형·범위·진행 step과 맞지 않거나 결과가 무효다. |

고정 step 실험도 이름의 `switch_fixed200`을 실제 전환 기록으로 간주하지 않는다.
추가 실험의 `switch_repeat`은 첫 전환과 저장된 전체 전환 경계를 표시하며,
마지막 전환에 따라 현재 저장된 selector가 SR인지 on-policy인지 구분한다.
완료된 추가 실험은 검증된 endpoint에서 최종 전환 이력을 읽고, 읽는 도중
결과 파일이 바뀌면 확인 불가 오류로 표시한다.

이 표시의 시점은 학습 step이다. 파일 수정 시각이나 조회 시각을 실제 전환
날짜로 대신 표시하지 않는다. 저장 주기에 따라 실제 전환과 표시 사이에
지연이 있을 수 있다. 상태 조회는 plan, 응답, checkpoint, 결과, task receipt를
수정하지 않으며, 기존 frozen runtime 및 adapter/engine identity도 보존한다.

## 검증

2026-10-12 로컬 검증: **87 passed, 9 subtests passed**. 변경 Python 파일의
Ruff와 `git diff --check`도 통과했다. Qwen/Llama/Gemma adapter와 공통 engine의
SHA-256이 변경 전과 같은지 별도로 대조했다.

```sh
python3 -m pytest -q \
  srgc_research/tests/test_switch_status.py \
  srgc_research/tests/test_qwen_status.py \
  srgc_research/tests/test_model_status.py \
  srgc_rebuttal/tests/test_live_reporting.py \
  srgc_rebuttal/tests/test_worker_status.py
```

세 모델의 실제 공개 shell entry에서 text/JSON과 두 데이터셋을 확인한다.
완료·실행 중·미전환·기록 없음·재개 step 불일치·잘못된 전환 값·반복 전환·
결과 교체 감지를 검사하고, 조회 전후 실험 파일의 바이트가 같은지 검증한다.
실제 원격 H100 노드와 별도 `qwen35.sh` 바로가기의 설치 상태는 이 로컬
검증으로 확인하지 않는다.
