# Code check coordination log (2026-08-20)

이 문서는 동시 Codex 작업자가 현재 코드 점검의 기준, 발견 사항, 수정 범위를 확인할 수
있도록 남기는 공유 로그다. 숫자 판독은 `HARVEST-0820.md`, 전체 무결성 감사는
`FULL_AUDIT_2026-08-20.md`를 함께 본다.

## 작업 격리

- 공유 checkout `/home/kms/dev/offpolicy-misranking`은 수정하지 않는다.
- 격리 worktree: `/home/kms/dev/offpolicy-misranking-codecheck-0820`
- 작업 브랜치: `audit/codecheck-20260820`
- 기준 master: `b5a0853`
- 기존 무결성 수정 `a50540d`를 기준 master 위에 결합한 커밋: `78edf02`
- 완료 전과 push 직전에 `origin/master` 이동 여부를 다시 확인한다.
- 점검 중 `origin/master`가 `56a4edc`로 이동했다. 이 커밋은 세대 일반화를 추가했지만
  실패 산출물을 최종 경로에 먼저 쓰고 harvest가 실패 시에도 0으로 끝날 수 있어 그대로
  채택하지 않는다. `run_select`와 관련 스크립트의 세대 일반화만 통합 대상으로 삼는다.

## READOUT 0-byte 원인 판정

실제 Python traceback은 기존 `harvest.sh`가 stderr를 저장하지 않아 사후 복원할 수 없다.
하지만 0-byte 파일을 만드는 코드 경로는 확정됐다.

1. `python readout_summary.py | tee READOUT.md`에서 `tee`가 Python 실행 완료 전에
   `READOUT.md`를 생성하거나 truncate한다.
2. `readout_summary.py`는 모든 run 분석이 끝난 뒤에 첫 Markdown을 출력한다.
3. 중간 run에서 빈 oracle, score ID coverage 불일치, JSON 오류, judge timeout 등 처리되지
   않은 예외가 나면 stdout은 한 줄도 나오지 않는다.
4. 기존 `harvest.sh`는 `set -e`가 없고 해당 stderr도 파일로 보존하지 않아 다음 산출물로
   진행한다. 그 결과 실패 원인 없이 0-byte `READOUT.md`가 전달 폴더에 남는다.

`READOUT.md` 0-byte는 실험 결과 부재가 아니라 harvest 실행 실패다.

## 추가로 확인한 harvest 결함

- 분 단위 `STAMP_DIR`를 재사용하므로 같은 분의 재실행이나 동시 실행이 서로 덮어쓰거나
  `STATS.md`에 중복 append할 수 있다.
- 기존 보강안도 readout을 `tee`로 최종 경로에 바로 써서 실패 시 빈/부분 최종 파일을
  남긴다.
- readout과 kcurve의 stderr가 보존되지 않는다.
- stats는 run header를 먼저 기록한 뒤 계산 실패를 허용해 빈 섹션을 만든다.
- `|| true`와 stderr 폐기로 reversal/stats 실패가 정상 수확처럼 보일 수 있다.
- 산출물 nonempty 검증과 원자적 publish 단계가 없다.

## 수정 결정

1. 수확 디렉터리는 초 단위와 `mktemp`를 사용해 동시 실행 충돌을 막는다.
2. 각 생성기는 임시 stdout/stderr에 기록한다. 허용된 exit code와 nonempty 검사를 통과한
   뒤에만 최종 `.md`로 원자적으로 rename한다.
3. 실패한 stdout은 `.partial`, stderr는 `.err`로 보존하고 harvest 자체는 nonzero로 끝낸다.
4. stats는 run별 임시 파일이 성공한 경우에만 합친다. 하나라도 실패하면 최종
   `STATS.md`를 publish하지 않는다.
5. `readout_summary.py`는 run별 artifact 오류를 문맥이 포함된 오류로 바꾸고, 유효한
   corrected run이 하나도 없으면 nonzero로 끝낸다.
6. 성공, Python 실패, 빈 stdout, stats 부분 실패, 동시 디렉터리 생성을 회귀 테스트한다.

## 추가 코드 판정

- 기존 `canonical_gate_report`는 protocol marker가 있어도 raw artifact가 없으면 저장된
  `report.json`으로 fallback했고, score ID가 다르면 교집합만 사용했다. confirmatory
  판정에서 누락 prompt를 숨길 수 있어 fallback을 제거했다.
- `score_artifacts.py`를 공통 validator로 추가해 oracle, g00/g10/g01/g11, split-half가
  모두 존재하고 finite이며 prompt ID 집합이 정확히 같아야 judge/readout/stats/reversal이
  진행되게 했다.
- `readout_summary.py`는 `v2-*`만 보지 않고 완료 또는 corrected protocol이 있는 모든
  직접 run을 검사한다. 한 corrected run이라도 malformed이면 partial 보고서를 출력하고
  nonzero로 끝낸다.
- 다른 Codex의 `56a4edc`가 주장한 “v2 하드코딩이 0-byte의 직접 원인”은 단독으로는
  성립하지 않는다. 대상이 0건이어도 구 코드는 Markdown 헤더를 출력한다. 하드코딩은
  누락 원인이고, 0-byte 자체는 첫 출력 전 예외와 `tee` 선생성·오류 은폐가 결합한 결과다.

## 검증 현황

- 통과: artifact contract, data sandbox, frontier, hard pool, pool qualification,
  reversal, judge, readout, harvest 회귀 테스트
- 통과: theory verifier, 전체 Python `py_compile`, 전체 shell `bash -n`, Ruff `F/E9/B`,
  `git diff --check`
- 로컬 미실행: `test_core.py`, `test_protocol.py`는 `torch` 없음;
  `test_contract.py`는 `transformers` 없음. GPU/실험 환경에서 재실행해야 한다.

## 진행 상태

| 항목 | 상태 |
|---|---|
| 동시 작업 격리 | 완료 |
| 원인 분석 | 완료 |
| harvest 원자적 출력 수정 | 완료 |
| readout run 단위 검증 | 완료 |
| 회귀 테스트 | 완료 |
| 전체 테스트·정적 검사 | CPU 범위 완료 |
| 최신 master 세대 일반화 통합 | 진행 중 |
| push | 대기 |
