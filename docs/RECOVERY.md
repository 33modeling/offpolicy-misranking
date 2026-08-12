# 복구·백업 런북 — 실험이 안 됐을 경우의 경로들

정본 상태는 전부 group-volume(`$OM_WORK`)에 있고 모든 스크립트가 DONE/산출물
스킵으로 재개되므로, **어떤 상황에서도 "처음부터 다시"는 없다.**

## 상황 1 — run이 ✘로 남거나 CUDA 에러가 반복

1. `DISABLE_ADDMM_CUDA_LT=1 bash scripts/go_full.sh ...` — cublasLt 경로 폴백.
2. 그래도 반복 → **노드 교체** (이 클러스터 커널 계열 문제의 검증된 해법):
   새 인스턴스에서 `git pull`(또는 코드 동기화) → `bash scripts/provision.sh`
   → `bash scripts/preflight.sh` → `bash scripts/go_full.sh`.
   group-volume이 상태를 들고 있으므로 완주분은 전부 스킵되고 즉시 재개된다.
3. `dmesg | grep -i xid` 에 이벤트가 있으면 하드웨어 — 관리자 보고.

## 상황 2 — 부분 완주 상태로 시간이 없음

완주(DONE)분만으로 진행한다. `tables.sh`·`frontier.sh`는 DONE 필터가 있어
부분 상태에서 안전하게 돈다. 논문 표는 있는 seed로 mean±sd(n 명시).
seed 3개 미만이어도 v1 수치가 바닥을 받친다 (아래 상황 3).

## 상황 3 — v2가 끝내 불가/결과 불발

concept.md의 **"v2 결과 사전 판정 규칙"**(2026-08-12 등록)을 따른다 —
결과를 본 뒤 경로를 정하면 체리피킹이므로 세 분기가 미리 고정돼 있다.
전량 실패 시 v1 수치 경로: 원고는 v1만으로 완결(9p, 게이트 PASS), 단일 seed
한계는 limitations 유지.

## 상황 4 — 산출물 유실 대비 (백업)

주기적으로(최소 각 go_* 완주 직후):

```bash
bash scripts/backup_results.sh    # 정본 소형 파일 → 레포 results/backup + 로컬 커밋
# 온라인 셸에서: git push          # GitHub = 오프사이트 백업
```

rollout 원본(대용량)은 백업하지 않는다 — 재현 가능(seed 고정)하고, 판정에
필요한 것은 report/scores/manifest 전부 백업에 포함된다.

## 마감 역산 (ICLR 2027, 공식 확인 2026-08-12)

- abstract 마감 **9/18 AOE**, full paper **9/25 AOE**
- 권장 내부 마감: **수치 동결 9/20** (표·frontier 절 반영 여유 5일)
- **경로 결정점 9/10**: 이날까지 v2+보강이 불안하면 상황 3 분기 발동
- 실행 순서: go_full → go_boost → go_35(Qwen3.5 스케일 스윕) → go_27b —
  뒤쪽 블록은 시간 없으면 자른다 (27b가 첫 번째 삭제 후보, full이 최후 보루)

## 재실험 사다리 (R-plan, 2026-08-13 — 원인 불명이어도 작동)

전제: 어느 단계에 서 있든 완주분(DONE)은 보존되고, 각 단은 논문 티어와 대응한다.

| 단 | 실험 범위 | 논문 티어 |
|---|---|---|
| 완전 | 5-seed × {gsm8k, dapo, math500} + downstream | 주 경로 (오차대 완비) |
| 표준 | 3-seed × {gsm8k, math500} | 충분 (dapo 없이 성립) |
| 최소 | 3-seed × gsm8k | 주표 오차대만 확보 |
| 바닥 | 0 (전량 실패) | v1 수치 경로 — 원고 이미 완결 |

절차:

```bash
bash scripts/go_retry.sh     # R1: 같은 노드+cublasLt 폴백 검증런(gsm8k s0)
                             #  통과 → 같은 env로 전체 자동 재개
                             #  실패 → DIAGNOSIS.txt 자동 생성·중단 → 노드 교체
bash scripts/diagnose.sh     # 언제든 원샷 진단 리포트 (사진 한 장으로 전달)
```

노드 교체 시(R2): 새 인스턴스에서 `git pull → provision.sh → preflight.sh →
go_full.sh` — group-volume이 상태를 들고 있어 완주분 전부 스킵. R2도 실패하면
세 번째 인스턴스 유형(B300 등)으로 — 이 계열 커널 문제는 노드 의존이 정설.
시간이 없으면 표에서 한 단씩 내려간다: `SEEDS_ALL="0 1 2"`, `DATASETS` 축소.
