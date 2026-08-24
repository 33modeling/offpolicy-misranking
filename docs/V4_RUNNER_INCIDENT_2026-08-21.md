# v4 runner 장애 및 수정 기록 (2026-08-21)

상태: **해결** (`abe9dbb`; 중단 재개 경로는 2026-08-22, false-success는
2026-08-24 보강)

이 문서는 3개 독립 H100 4-GPU 클러스터에서 v4 confirmatory matrix를 실행하는
과정에서 발생한 재시작, config 계보, GPU 잔류 문제를 기록한다. 실험 계산식이나
`src/experiment.py`의 과학적 로직 변경 기록이 아니라 launcher 운영 장애 기록이다.

## 1. 최종 실행 계약

각 클러스터 checkout에서 다음 중 하나만 실행한다.

```bash
git pull
bash scripts/go_v4.sh 1  # 27B seeds 0,1 / 7B seed 0
bash scripts/go_v4.sh 2  # 27B seeds 2,3 / 7B seed 1
bash scripts/go_v4.sh 3  # 27B seed 4 / 7B seeds 2,3,4
```

- 세 명령은 **서로 독립된 클러스터**에서 하나씩 실행한다.
- 추가 finalize 플래그나 `OM_V4_FINALIZE_ONLY` 입력은 필요 없다.
- 각 모델/seed는 GSM8K와 MATH500을 실행한다. 포화된 27B DAPO는 제외한다.
- 시작 전에 이전 v4 작업 정리와 GPU 메모리 해제 검사를 자동 수행한다.

## 2. 관찰된 증상

1. 스모크가 실패했으나 콘솔에는 마지막 몇 줄만 보여 정확한 stage/예외를 찾기 어려웠다.
2. `git pull` 뒤 같은 경로를 재사용하면 다음 오류로 스모크 또는 본실행이 즉시 중단됐다.

   ```text
   [config-abort] existing artifacts use a different run config: ['git']
   ```

3. 실패한 실행의 Python/CUDA 자식이 남아 GPU를 점유했고, 새 실행이 GPU 건강검사나
   `run_14b.sh`의 2GB 점유 검사에서 다시 중단됐다.
4. 무출력 stage를 자동 재시작하도록 한 중간 수정 이후에도 종료와 재시도 사이의
   프로세스/GPU 해제가 확정되지 않아 같은 장애가 반복됐다.

## 3. 근본 원인

### 3.1 config lock과 run 경로의 계약 불일치

`scripts/run_14b.sh`는 run 최초 생성 시 `run_config.json`에 Git HEAD와 설정 digest를
기록하고, 이후 하나라도 달라지면 fail-closed로 중단한다. 이 검사는 서로 다른 코드의
산출물 혼합을 막으므로 정상이고 유지해야 한다.

문제는 launcher가 smoke 경로만 commit hash로 분리하고, 본실행 경로
`v4-27b-s*`와 `v4-7b-s*`는 Git 변경 뒤에도 그대로 재사용한 것이다. 따라서 이전
commit이 만든 부분 산출물이 있으면 config lock이 의도대로 `['git']`을 검출했지만,
launcher에는 이를 보존 격리하고 새 run으로 전환하는 단계가 없었다.

### 3.2 프로세스 수명주기와 GPU 검사 순서

무출력 watchdog은 `setsid` process group을 종료하도록 확장됐지만, 정리 함수가 TERM
전송 뒤 모든 자식의 종료를 확인하지 않은 채 다음 재시도로 넘어갈 수 있었다. 또한
`go_v2.sh`의 GPU 건강검사가 기존 run 정리보다 먼저 실행되므로, 이전 CUDA 자식이
남은 상태에서는 정리 코드까지 도달하기 전에 실패했다.

최종 수정은 상위 진입점 `go_v4.sh`에서 이전 v4 wrapper/자식을 먼저 종료하고 GPU
메모리 해제를 확인한 뒤에만 `go_v2.sh`를 호출한다. 재시도 내부 정리도 TERM 대기 후
KILL을 보장한다.

## 4. 수정 이력과 불완전했던 지점

| commit | 변경 | 판정 |
|---|---|---|
| `cfe9025` | 27B와 7B confirmatory matrix 추가 | 모델 축 구성 |
| `6296319` | 3개 4-GPU worker 분배 초안 | 독립 클러스터 조건 반영이 불충분 |
| `6fef6d1` | 독립 클러스터별 seed 배정 확정 | 현재 배정 계약 |
| `99f61be` | 로그 무변화 stage 자동 재시작 | stage 밖 침묵과 전체 자식 수명주기 미포함 |
| `f7ff6a8` | smoke 전체를 `setsid` process group으로 감시 | config 경로 충돌과 시작 전 GPU 잔류 미해결 |
| `ee9a6b9` | commit별 smoke 경로, 자동 failure diagnostic | smoke만 격리해 canonical 본실행 경로 충돌이 남음 |
| `abe9dbb` | stale v4 process/GPU 정리, 전체 run 자동 격리, 종료 대기 | 현재 최종 수정 |

중간 수정의 핵심 실수는 오류 메시지를 더 잘 출력하는 것과 오류의 원인을 제거하는
것을 혼동한 점이다. 진단 TXT는 보조 기록일 뿐이며 사용자가 파일을 직접 열어야만
재실행할 수 있는 흐름은 허용하지 않는다.

## 5. 현재 시작 및 재시도 흐름

1. `src/`와 `scripts/`가 clean committed snapshot인지 확인한다.
2. worker 번호, GPU 4장, 두 모델 snapshot을 확인한다.
3. `src/cleanup_run_processes.py`가 현재 launcher의 조상 PID는 보호하면서 같은 Unix
   사용자의 이전 최상위 `go_v4.sh`, v4 worker와 자식을 찾고 TERM 후 KILL한다.
4. 추가 race를 막기 위해 다음 run namespace에 명시적으로 `pkill`한다.

   ```bash
   pkill -TERM -f -- "--run $OM_WORK/runs/v4-"
   pkill -KILL -f -- "--run $OM_WORK/runs/v4-"
   ```

5. 모든 GPU의 사용 메모리가 2GB 이하가 될 때까지 최대 60초 기다린다. 남은 점유가
   있으면 PID, process name, memory를 콘솔에 출력하고 중단한다.
6. 최신 `go_v4.sh`는 `resume_v4.sh`를 거쳐 각 미완료 v4 `run_config.json`의
   generation commit을 선택한다. 현재 HEAD와 다르면 `$OM_WORK/code-snapshots/`의
   detached worktree에서 해당 run의 원래 코드를 실행한다.
7. 해당 snapshot의 `prepare_run_path.py`가 같은 Git을 확인하므로 canonical run은
   격리되지 않는다. `go_v2.sh`는 DONE run을 건너뛰고 미완료 run의 기존 stage/shard를
   재사용한다. 서로 다른 generation commit은 run별 worktree로 분리해 재개한다.
8. 스모크를 통과한 뒤 본실행으로 진입한다. 실패 시 오류 서명과 로그 tail은 콘솔에
   자동 출력되고, 재시도 전 process group 종료를 기다린다.

## 6. 데이터 안전 불변조건

- `run_config.json`의 immutable digest 검사는 제거하거나 우회하지 않는다.
- 다른 Git commit의 산출물을 같은 run 디렉터리에서 이어 쓰지 않는다.
- 충돌한 산출물은 삭제하지 않고 quarantine으로 이동한다.
- quarantine 폴더는 `runs/` 밖에 두어 run selection/harvest가 자동 선택하지 못하게 한다.
- 자동 격리는 **Git 불일치**만 처리한다. 같은 commit에서 사용자가 임의 환경변수로
  설정을 바꿔 digest가 달라지면 기존 fail-closed 오류가 그대로 발생한다.
- v4 시작 정리는 `v4-` run namespace와 `RUN_LABEL=v4-*`로 제한한다. 정리 후에도
  무관한 GPU 작업이 남아 있으면 그것을 강제 종료하지 않고 PID를 출력하고 중단한다.
- 하나의 물리 노드에서 v4 worker 둘을 동시에 실행하면 서로 stale worker로 간주한다.
  현재 계약은 독립 클러스터당 worker 하나이므로 의도된 동작이다.

## 7. 검증 범위

최종 수정에서 확인한 항목:

- `bash -n`: `go_v4.sh`, `go_v2.sh`, `diagnose_run_failure.sh`
- `py_compile`: `cleanup_run_processes.py`, `prepare_run_path.py`
- `test_cleanup_run_processes.py`: 가짜 이전 최상위 `scripts/go_v4.sh`와
  `RUN_LABEL=v4-*` worker를 생성한 뒤 실제 종료 확인
- 같은 Git run no-op, 다른 Git run 보존 이동, malformed config 보존 이동
- failure diagnostic에서 console/nested stage 오류 노출
- `test_run_select.py`, `test_harvest.py` 회귀 통과

로컬 머신에서는 모델 snapshot/H100이 없으므로 실제 GPU 통합 스모크는 실행하지 않았다.
최종 GPU 통합 검증은 각 클러스터에서 worker 번호 `1`, `2`, `3`으로 실행하는
`go_v4.sh`가 담당한다.

## 8. 잔여 운영 경계

- 프로세스가 Linux D-state에 빠지면 SIGKILL도 즉시 회수하지 못한다. 60초 뒤 GPU
  점유가 남는 경우 출력된 PID 상태와 NFS/Xid를 확인하고 노드를 교체한다.
- 현재 commit별 smoke 경로는 감사 추적에는 안전하지만 오래된 smoke가 누적된다.
  자동 삭제하지 않으며 실험 종료 후 별도 보존 정책으로 정리한다.
- watchdog 임계값은 7B 5분, 27B 20분이다. 모델 로딩 시간이 이보다 긴 새 하드웨어나
  파일시스템에서는 로그 진행 신호를 추가한 뒤 임계값을 조정해야 한다.

관련 문서: `docs/BACKLOG.md`, `docs/TROUBLESHOOTING.md`,
`docs/FULL_AUDIT_2026-08-20.md`.

## 9. 2026-08-22 운영자 중단 뒤 재개 보강

약 이틀 실행 뒤 모든 GPU worker가 외부 요인으로 종료된 경우, 완료된 결과를 새 Git
HEAD로 덮거나 전체 matrix를 다시 시작하면 안 된다. 최신 checkout에서 각 클러스터는
기존과 동일하게 `bash scripts/go_v4.sh <1|2|3>`만 실행한다. launcher가 원래
run마다 기록된 generation commit으로 자동 진입하며, 호출 checkout의 `OM_REPO`와 `PYTHONPATH`를
제거한 뒤 snapshot의 `setup_env.sh`로 다시 설정해 두 revision의 Python 코드가 섞이지
않게 한다. 모든 worker 종료 후 `bash scripts/collect_v4.sh` 한 번으로 20-run의
missing/incomplete/artifact 상태를 검사하고 최종 결과를 생성한다.

## 10. 2026-08-24 주말 실행의 false-success 종료

### 증상

클러스터 `1`, `2`, `3`을 주말 동안 실행했지만 GPU process가 모두 종료된 뒤에도
배정 matrix가 완성되지 않았다. Launcher 마지막 메시지는 run을 재개했다고 표시할 수
있어 정상 종료와 미완료 종료를 구분할 수 없었다.

### 직접 원인

`go_v2.sh`는 내부 retry를 모두 소진해 `RESULT[$KEY]=0`인 run이 남아도
`OM_SKIP_POSTPROCESS=1`이면 무조건 `exit 0`을 반환했다. V4 worker는 후처리를 한 번만
수행하기 위해 이 변수를 항상 설정한다. `resume_v4.sh`도 child exit만 신뢰하고
`DONE`이나 protocol/report를 다시 검사하지 않은 채 "재개 완료"를 출력했다. 따라서
실제 계산이 실패한 상태가 shell success로 바뀌고, 클러스터 launcher가 스스로 끝나는
경로가 있었다.

### 수정

1. `go_v2.sh`는 선택된 run 중 하나라도 미완료면 후처리 생략 여부와 무관하게
   non-zero로 종료한다.
2. `resume_v4.sh`는 한 run 실패로 slot 전체를 즉시 중단하지 않는다. 같은 pass의
   나머지 run을 계속 실행해 주말 GPU 시간을 보존한다.
3. 각 run은 `DONE`, `run_config.json`, `manifest.json`, `score_protocol.json`,
   `oracle_protocol.json`, `report.json`이 모두 nonempty일 때만 완료로 인정한다.
4. Pass 종료마다 plan을 새로 만들고 완료된 run은 제외한다. 남은 run만 기본 3개의
   supervisor pass 동안 다시 실행한다.
5. 마지막 pass에도 미완료가 있으면 run 이름을 출력하고 launcher 자체가 non-zero로
   종료한다. 성공 메시지는 완전한 배정 계약을 만족한 경우에만 나온다.

기존 partial artifact는 삭제하거나 새 commit과 섞지 않는다. 각 run의
`run_config.json.git`이 가리키는 detached worktree에서 기존 shard와 `.partial`부터
재개하며, 최신 launcher가 바깥에서 완료 계약을 검증한다.

### 검증과 운영 경계

- `test_go_v2_exit_status.py`: 실패한 단일 worker가 `OM_SKIP_POSTPROCESS=1`에서도
  exit 1을 유지하는지 검증한다.
- `test_v4_resume_shell.py`: 한 run을 의도적으로 계속 실패시켜도 나머지 다섯 run이
  실행되고, 다음 pass에서는 실패 run만 재시도되며 최종 exit가 1인지 검증한다.
- `test_v4_resume_commit.py`: 여섯 필수 artifact를 모두 갖춰야 complete인지 검증한다.
- Artifact contract, failure diagnostic, cleanup, run-path preservation, run selection,
  readout, harvest 회귀 테스트를 함께 통과했다.
- 로컬 system Python에는 PyTorch/Transformers가 없어 GPU-dependent 전체 suite는
  collection할 수 없었다. 실제 H100 integration은 클러스터 재개 시 확인한다.
- Supervisor는 살아 있는 노드 안의 child failure를 복구한다. 운영자가 VM·pod·최상위
  batch job을 종료하면 supervisor도 사라지므로, 노드가 다시 준비된 뒤 같은
  `bash scripts/go_v4.sh <1|2|3>` 명령을 한 번 다시 실행해야 한다.
