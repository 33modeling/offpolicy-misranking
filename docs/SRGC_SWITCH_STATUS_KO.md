# SRGC status의 전환 시점 표시

## GPU 시간 표시

2026-10-14부터 Qwen·Llama·Gemma의 MATH/MBPP, 공통 OLMo SRGC,
추가 arm·메커니즘·replicate·고정 전환·규칙·support, 정보 측정,
현재 등록된 N01–N07 연구 실험 및 기존 new-seed status에 GPU 시간 표를 추가했다.
학습 코드·모델 adapter·기존 frozen runtime은 바꾸지 않고 저장된 비용 receipt를 읽는다.

`GPU TIMES (GPU-hours)`는 seed·arm별 Prep/Select/Train/Eval/Check/Start,
`S+T+P`와 `Phases`를 표시한다. S+T+P는 준비+선별+학습이며 평가·checkpoint·startup을
제외한다. Phases는 scope의 기록된 모든 phase 비용이다. cache와 prefix, 공통 startup은
한 seed에서 공유하는 별도 비용으로 표시하며 네 continuation에 중복 합산하지 않는다.
GPU-hour는 GPU-seconds/3600으로 계산하며 wall-hour와 다르다. invocation과 그 안에
포함된 phase 비용을 더하지 않는다.

| 표시 | 의미 |
| --- | --- |
| `COMPLETE` | 완료 scope와 비용 측정이 확인됨 |
| `PARTIAL`, `>=` | 완료된 receipt까지의 누적 비용 하한; 실행 중 timer는 추정하지 않음 |
| `MISSING`, `-` | 시간 미측정·미수신; 0으로 대체하지 않음 |
| `INVALID` | 비용 자료를 읽거나 검증할 수 없음; 보상·진행 상태와 구분 |

같은 데이터셋·arm의 모든 seed가 COMPLETE일 때만 S+T+P 평균과 표본 SD를 표시한다.
JSON에는 `timing.gpu_seconds`와 `timing.gpu_hours`의 정확한 값, 측정 상태와
회계 범위를 함께 넣는다. 보상 평가와 학습-batch 보상 기록은 변경하지 않는다.

```sh
sh scripts/run_srgc_qwen35.sh all status
sh scripts/run_srgc_llama31.sh all status
sh scripts/run_srgc_gemma4.sh all status
sh scripts/run_srgc_qwen35.sh math status --json
sh scripts/run_srgc.sh all status
sh scripts/run_srgc_sr_refresh.sh all status all
sh scripts/run_srgc_research.sh all n01 status --json
sh scripts/run_srgc_information.sh all status --json
```

동결된 package 내부의 과거 `reports`를 편집하지 않고 공개 status entry에서 표시한다.
진행 중인 학습 checkout이나 plan을 새 코드로 교체하지 않는다. GPU 시간 조회는
학습·캐시 생성·GPU 모델 로드·결과 재작성·새 작업 배정을 실행하지 않는다.

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
