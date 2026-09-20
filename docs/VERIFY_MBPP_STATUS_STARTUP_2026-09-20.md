# MBPP 시작 즉시 종료·status 표시 문제 검증 (2026-09-20)

요청: "status가 제대로 안 나오는 문제, mbpp 시작시 그냥 종료. 검증해봐". 코드 수정 없음(다른 세션이
`scripts/run_mbpp_experiments.sh`·`mbpp_queue_readiness.py` 등을 편집 중이고 pytest 실행 중). 로컬(GPU 없음,
스크래치 OM_WORK) 재현 + 사용자가 클러스터에서 가져온 TXT 3개 + 레포 테스트로 확인했다.

## 근거 파일

| 파일 | 시각 | 내용 |
|---|---|---|
| `~kms/mbpp-storage-5pj74_28.txt` | 09-19 07:19 | `STORAGE_UNAVAILABLE /group-volume/minsoo3.kim/offpolicy-misranking: work/runs unavailable` → START BLOCKED |
| `~kms/mbpp-why-el8fwo9_.txt` | 09-19 15:40 | quality 루트 s1/t50: `fresh-r-candidate exceeded 7093s`; 노드 run284373-wts-16 NCCL preflight 실패(`Cuda failure 802 'system not yet initialized'`); fresh 루트 실패 21건 |
| `~kms/mbpp_why.txt` | 09-20 11:23 | s4/t50 selection_reduced: train 중 `CUDA error: unspecified launch failure` 뒤 `used=28383.6 > budget=28376.9` 소진; s4/t25 random_full: verify-inputs만 끝났는데 `used=28389 > 28380` 소진 |

## 1. "시작시 그냥 종료" — 설계상 조기 종료 경로 4개, 그중 2개는 실제로 일어났다

1. **저장소 감사 차단 (실제 발생, exit 2).** `run_mbpp_experiments.sh run`은 GPU·컨트롤러보다 먼저
   `check_mbpp_storage.sh`를 돌리고, `$OM_WORK/runs`가 안 보이면 `STORAGE_UNAVAILABLE`로 즉시 끝난다
   (`run_mbpp_experiments.sh:155-161`, `mbpp_storage_audit.py:140-143`). 09-19 07:19 TXT가 정확히 이 경우다.
   원인은 코드가 아니라 그 노드의 group-volume 마운트/경로(OM_USER·OM_WORK). 로컬 재현: 빈 OM_WORK → 같은 메시지, exit 2.
2. **신규 볼륨에서 자동 초기화 거부 (로컬 재현, exit 2).** `runs/`가 있어도 요청 루트 어디에도 `switch.json`이
   없으면 `NO_EXISTING_RUN`으로 차단한다(`mbpp_storage_audit.py:144-146`, "cannot distinguish fresh setup from
   lost/wrong storage; no automatic initialization"). 준비된 적 없는 루트에 대고 기본 명령을 치면 항상 즉시 종료다.
3. **노드 admission 실패 → 홀딩 없이 종료 (실제 발생, exit 75/78).** MBPP는 NCCL preflight 실패나 GPU 점유를
   재시도하지 않고 `[blocked] MBPP node unavailable ...; no holding/retry loop`로 나간다
   (`run_experiments.sh:684-691`). 09-19 why TXT의 `Cuda failure 802` 노드가 이 경로다.
4. **터미널에서는 분리 실행이 기본.** `[ -t 1 ]`이면 `setsid nohup ... &`로 컨트롤러를 떼어 놓고 콘솔 로그만
   tail한다(`run_experiments.sh:505-521`). tail이 끝나거나 컨트롤러가 죽으면 `[detached-exit] rc=N`을 찍고
   프롬프트로 돌아온다. 사용자 관점에서는 "명령이 그냥 끝남"으로 보인다.

추가로 **예산 소진 분기는 다시 잡히지 않는다.** 실패한 시도의 GPU초도 분기 예산에 그대로 계상되므로
(`selection_switch_gpu.py:1653-1659 remaining_allocation`), CUDA 오류나 SIGTERM으로 죽은 시도 한 번이 분기를
영구 소진시킨다(09-20 TXT의 s4/t50, s4/t25). 소진·격리 분기만 남으면 `[done] ... releasing the node`(exit 0) 또는
`[WAIT] only checkpoint-review branches remain`(exit 80)으로 즉시 나간다 — 이것도 "그냥 종료"로 보인다.
이 사고는 `origin/docs/mbpp-budget-incident-20260919`(INCIDENT_MBPP_BUDGET_2026-09-19.md, AGENTS.md)에 기록돼 있고,
소진 분기를 학습 없이 평가만 하는 `scripts/mbpp_budget_recovery.py` 경로가 있다.

## 2. "status가 제대로 안 나옴"

- 로컬에서는 `run_experiments.sh status`, `run_mbpp_experiments.sh status` 둘 다 정상 출력(스크래치 루트, 각 exit 0).
- 오늘 커밋 657fdd5·8ec0c0e·71fdddb와 `docs/MBPP_STATUS_REFRESH_2026-09-20.md`가 표시 버그 3개를 고쳤다:
  (a) finish receipt가 `progress.json`보다 먼저 써지면 마지막 heartbeat가 늙을 때까지 RUN으로 남던 문제,
  (b) `--all` 목록이 중첩 평가를 반영 안 하던 문제, (c) posthoc 예산 복구 평가가 요약에 안 나오던 문제.
  문서 스스로 "원격 관찰 없이는 어느 문제가 사용자 화면이었는지 단정 못 함"이라고 적었다.
- **명령 혼동 가능성.** `bash scripts/run_experiments.sh status`는 MATH(selection-switch-v1) 화면이라 MBPP 루트를
  넘기지 않으면 `NOT PREPARED` / `no launcher evidence yet`만 나온다. MBPP는
  `bash scripts/run_mbpp_experiments.sh status --watch`가 맞다(런처 도움말: "never the math view").

## 3. 레포 테스트 (kms python, torch 없음)

`tests/test_mbpp_storage_audit.py test_mbpp_experiments_launcher.py test_mbpp_status_refresh.py
test_mbpp_budget_dispatch.py test_mbpp_budget_recovery.py test_mbpp_queue_readiness.py test_experiments_status.py
test_mbpp_status_dashboard.py test_mbpp_node_guard.py` → **231 passed, 1 skipped, 3 failed**. 실패 3건은 모두
`test_mbpp_budget_dispatch.py`가 `src/additive_experiment.py`를 import하며 `No module named 'torch'` — 환경 문제이지
코드 실패가 아니다.

## 4. 클러스터에서 원인을 하나로 좁히는 명령 (읽기 전용)

```bash
bash scripts/run_mbpp_experiments.sh why        # ~/mbpp-why-*.txt 저장
bash scripts/check_mbpp_storage.sh all          # ~/mbpp-storage-*.txt 저장
bash scripts/run_mbpp_experiments.sh logs       # console.mbpp.<node>.log: [abort]/[blocked]/[done]/[detached-exit] rc= 줄 확인
bash scripts/run_mbpp_experiments.sh status --watch
```

`[detached-exit] rc=` 값으로 구분: 2=저장소 감사 차단, 75=노드 busy, 78=NCCL/CUDA admission 실패,
79=GPU fault 쿨다운, 80=review 분기만 남음, 0=`[done]`(잡을 분기 없음).

## 5. 코드 관점 제안 (미적용 — 동시 편집 중, 예산·재개 정책은 AGENTS.md상 승인 필요)

- 포그라운드 경로: `[detached-exit]` 앞에 `rc_reason`을 같이 찍어 종료 이유를 한 줄로(사용자는 nohup/분리 실행을
  싫어함 — tmux 포그라운드 옵션 `EXPERIMENTS_DETACHED=1 bash scripts/run_experiments.sh run`을 도움말에 노출).
- 예산: SIGTERM/CUDA fault로 학습 step 0에서 죽은 시도의 계상 규칙은 정책 결정 사항. 지금은 한 번의 노드 장애가
  분기를 영구 소진시킨다.
