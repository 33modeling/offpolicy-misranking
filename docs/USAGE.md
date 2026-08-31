# USAGE.md — 사용 방법

> 실행·재개·복구·수확·진단 절차서. 설계 구조는
> [`ARCHITECTURE.md`](ARCHITECTURE.md), 장애 사례 정본은
> [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md), 모듈 목록은 [`CODE.md`](CODE.md),
> 전체 문서 우선순위는 [`README.md`](README.md)에 있다.
> 겹치는 내용은 여기서 요약만 하고 링크한다.
>
> 작성 기준 커밋: `2244e89` (2026-08-24). 아래 기본값은 전부 코드에서 읽은
> 실제 값이다. 문서 서술과 구현이 어긋나는 부분은
> [`ARCHITECTURE.md` 12절](ARCHITECTURE.md#12-실측--기존-문서-서술과-구현이-어긋나는-곳)에
> 정리했고, 이 문서의 절차는 **구현 기준**으로 적었다.

---

## 0. 30초 요약

| 하려는 일 | 명령 |
|---|---|
| 셸 준비 | `source scripts/setup_env.sh` |
| 클러스터 1회 셋업 | `bash scripts/provision.sh` |
| 전체 실험 실행: v4 → 7B regime → 수확 (3-cluster) | `bash scripts/go_v4.sh 1` / `2` / `3` |
| 27B만 재실행 (공유 큐) | `bash scripts/go_v4_27b.sh` |
| 중단 후 재개 | 같은 명령을 그대로 다시 실행 |
| 진행 상황만 보기 (실행 중 안전) | `bash scripts/progress_snapshot.sh <라벨>` |
| 계약 판정 | `bash scripts/check_contract.sh` |
| 20-run 집계만 복구 | `bash scripts/collect_v4.sh` |
| 수확만 복구 | `bash scripts/harvest.sh` |
| 실패 진단 | `bash scripts/diagnose_run_failure.sh <RUN_DIR>` |

---

## 1. 준비

### 1.1 셸 환경

**모든 셸에서 한 번 source한다.** 실행 스크립트는 내부에서 다시 source하므로
직접 명령을 조합할 때만 필요하지만, 경로 확인용으로도 먼저 돌려 보는 편이 좋다.

```bash
cd /home/kms/dev/offpolicy-misranking
source scripts/setup_env.sh
```

출력 두 줄로 `OM_WORK`·`VENV_DIR`·모델 경로·오프라인 여부를 확인한다.
경로가 다르면 **source 전에** override 한다.

```bash
export GROUP_VOLUME=/mnt/group-volume
export MODELS_DIR=/group-volume/nait-models
source scripts/setup_env.sh
```

group-volume이 없으면 `OM_WORK`가 `$OM_REPO/.work`로 자동 폴백하고 경고를 찍는다.
`setup_env.sh`는 없는 경로에 대해 **경고만 하고 셸을 죽이지 않는다**.

부수 효과 하나: 레포 루트의 `*.log`·`READOUT.md`·`DIAGNOSIS.txt`를
`$OM_WORK/console-logs/`로 옮긴다. 로그는 체크아웃에 남기지 않는 규약이다.

### 1.2 1회 프로비저닝 (온라인 머신)

```bash
source scripts/setup_env.sh
bash scripts/provision.sh
```

멱등이다. venv(torch 2.7.1+cu126, `constraints/h100-cu126.txt` 고정),
모델 고정 revision 스냅샷, GSM8K 로컬 jsonl, 로직 테스트까지 한 번에 한다.
hf-mirror.com 폴백이 내장돼 있고 pip/curl/HF 전부 타임아웃이 걸려 있어
폐쇄망에서 무한 대기 대신 에러로 끝난다. 사내 미러가 있으면
`PIP_INDEX_URL=<미러>`로 우회한다.

데이터셋을 따로 받을 때:

```bash
bash scripts/fetch_datasets.sh              # 전부
bash scripts/fetch_datasets.sh mbpp         # 골라서
bash scripts/prepare_domain_datasets.sh     # 비수학 주행렬 3종 고정·검증
bash scripts/fetch_transfer_models.sh       # 비Qwen 7B 두 모델 고정·검증
OM_ONLINE=1 source scripts/setup_env.sh     # 다운로드 머신에서만 오프라인 해제
```

### 1.3 실행 전 점검

```bash
bash scripts/preflight.sh          # 코드 버전·venv·GPU·디스크·모델·데이터
bash scripts/gpu_check.sh          # GPU별 matmul / SDPA 분리 판정
bash scripts/check_data.sh gsm8k 512 100   # 로더가 찾는 위치와 실제 로드 확인
```

`gpu_check.sh`의 판정은 두 갈래다. **matmul까지 실패하면 코드가 아니라 노드를
바꾼다.** SDPA만 실패하면 `OM_ATTN=eager`로 우회한다(v4는 이미 eager가 기본).

---

## 2. 실행 — 클러스터별 절차

### 2.1 전체 행렬 (v4 20 run → 7B regime 24 point → 수확)

독립된 H100 4장 클러스터 **세 곳**에서 번호만 달리해 실행한다.

```bash
# 클러스터 A
git pull && bash scripts/go_v4.sh 1     # 27B seed 0,1 / 7B seed 0
# 클러스터 B
git pull && bash scripts/go_v4.sh 2     # 27B seed 2,3 / 7B seed 1
# 클러스터 C
git pull && bash scripts/go_v4.sh 3     # 27B seed 4   / 7B seed 2,3,4
```

배정표는 `src/v4_resume_commit.py`의 `SLOTS`에 있다. 세 워커가 같은
`$OM_WORK/runs`를 쓰므로 결과가 seed별 canonical 경로에 바로 모인다.
수동 복사가 필요 없다.

`go_v4.sh`는 이제 상위 runner로 들어가 아래 전체 흐름을 수행한다.

```
go_v4.sh <slot>
  └─ go_offpolicy.sh <slot>           # node + global slot singleton
       ├─ resume_v4.sh <slot>
       │    ├─ cleanup_run_processes.py --run-prefix $OM_WORK/runs/v4-
       │    ├─ pkill TERM → 3초 → KILL
       │    └─ pass 1..3 (OM_V4_SUPERVISOR_PASSES)
       │         └─ v4_resume_commit.py plan → ensure_snapshot → go_v2.sh
       ├─ resume_regime.sh → go_additional.sh     # 완료 point 생략
       └─ collect_v4.sh → harvest.sh              # 입력 동일 시 재사용
```

같은 slot을 두 노드에서 실행하거나 같은 노드에서 다른 slot을 겹쳐 실행하면 시작 전에
거부한다. regime family/run 잠금은 클러스터 간 생성 중복을 막고, 최종 regime 분석,
모델별 TABLES/FRONTIER, harvest는 입력 키가 같으면 기존 결과를 그대로 재사용한다.
27B fixed-drift v4는 이 흐름에 포함되지만 27B regime confirmation은 7B boundary 동결
후의 별도 후속 게이트다.

각 워커가 실제로 쓰는 설정(`resume_v4.sh` 기준):

| 항목 | 27B | 7B |
|---|---|---|
| 모델 | `$MODELS_DIR/Qwen3.8-27B-BF16` | `$MODELS_DIR/Qwen2.5-7B-Instruct` |
| `N_TRAIN` | 512 (gsm8k) / 400 (math500) | 같음 |
| `OM_LORA_TARGETS` | `all-linear` | 미설정(= `q_proj,v_proj`) |
| `OM_GEN_BATCH` | `8` | 미설정(= K 전체 한 배치) |
| `OM_SKIP_HYBRID` | `1` | `0` |
| `OM_STALL_MINUTES` | `20` | `5` |
| 공통 | `BEHAVIOR_K=8 FRESH_K=32 VAL_K=8 MICRO_GROUP=4 HYBRID_PROMPTS=64 K_CELL=8 DRIFT=100 MAX_NEW_TOKENS=512 PROJ_DIM=4096 GRAD_LAYERS=4 CLIP_CAP=10.0 TEMPERATURE=1.0 TOPK_FRAC=0.10 RADIUS_MODE=gaussian OM_TOP_P=1.0 OM_THINKING=off OM_ATTN=eager N_VAL=100 OM_MAX_RETRIES=5 OM_GPUS=0,1,2,3 OM_SKIP_POSTPROCESS=1` | |

`run_config.json`이 이미 있는 run은 그 파일에 기록된 값이 위 기본값을 덮어쓴다
(`v4_resume_commit.py env <config>`가 `export`/`unset` 문을 만들어 `eval` 된다).

### 2.2 27B 전용 재실행 (공유 flock 큐)

7B가 이미 끝났고 27B만 현재 코드로 다시 돌릴 때 쓴다. **사용할 모든 4-H100
클러스터에서 인자 없이 같은 명령**을 실행한다.

```bash
bash scripts/go_v4_27b.sh
```

각 워커가 `$OM_WORK/locks/v4-27b-s<seed>-<dataset>.lock`을 `flock -n`으로
선점해 10개 작업 중 하나씩 가져간다. 완료하면 다음 미완료 작업을 잡는다.
클러스터가 죽으면 lock이 풀려 다른 워커가 그 부분 산출물부터 이어받는다.
클러스터 수를 지정할 필요가 없다.

수동 분할이 필요하면:

```bash
bash scripts/go_v4_27b.sh --plan 2 3    # 배정만 출력 (실행 안 함)
bash scripts/go_v4_27b.sh 2 3           # 3개 워커 중 2번째
```

시작 전 이 스크립트만 하는 일:

- `src/scripts` worktree dirty면 abort.
- FLA 0.5.2 준비 확인. 없으면 `flock`으로 직렬화해 공유 venv에
  `flash-linear-attention[cuda]==0.5.2`를 설치하고 fused recurrent/chunk
  kernel import를 재확인한다. 실패하면 **GPU 실험을 시작하지 않는다** —
  Qwen3.8 Gated DeltaNet의 PyTorch recurrent fallback이 ULF의 원인이었다.
- GPU 0~3에서 `scripts/check_27b_fla.py` 스모크(각 300초 타임아웃).
- 이전 v4 프로세스 정리 후 2GB 초과 GPU가 남으면 abort.
- `run_config.json`이 있는데 `validate_v4_27b.py` 기대값과 다르면
  **`--force-quarantine`으로 격리**하고 새로 시작.

`OM_GEN_BATCH=4`, `OM_MAX_RETRIES=10`을 쓴다. 27B 10 run이 다 끝나고 7B 10 run도
있으면 `flock`으로 단일화해 `validate_v4_27b.py` 전수 검증 후
`collect_v4.sh`를 자동 실행하고 `$OM_WORK/results/v4/V4_COMPLETE`에
`completed=<시각>` / `git=<commit>`을 남긴다.

> **주의.** `go_v4.sh` 경로로 만든 27B 산출물은 `gen_batch=8`,
> `linear_attention_backend=torch`라 `validate_v4_27b.py`를 통과하지 못한다.
> `go_v4_27b.sh`를 나중에 돌리면 그 run들을 전부 격리하고 처음부터 다시
> 시작한다. 두 27B 진입점을 섞지 말 것
> ([ARCHITECTURE 12.2](ARCHITECTURE.md#122-go_v4sh-경로의-27b-run은-validate_v4_27bpy를-통과할-수-없다)).

### 2.3 단일 run 직접 실행 (복구·진단용)

```bash
source scripts/setup_env.sh
MODEL_14B=$MODELS_DIR/Qwen2.5-7B-Instruct \
OUT_ROOT=$OM_WORK/runs/repair-s0 DATASET=gsm8k SEED=0 \
N_TRAIN=512 N_VAL=100 FRESH_K=32 HYBRID_PROMPTS=64 \
bash scripts/run_14b.sh
```

`run_14b.sh`는 한 (seed, dataset)만 처리한다. GPU 자동 감지, config digest lock,
샤딩·병합 검증, `DONE` 생성까지 전부 여기서 일어난다. 감시·재시도는 없다 —
그건 `go_v2.sh`의 몫이다.

여러 seed/dataset을 감시와 함께 돌리려면 `go_v2.sh`를 쓴다.

```bash
SEEDS="0 1 2" DATASETS="gsm8k math500" \
RUN_BASE=$OM_WORK/runs/v4-7b RESULTS_BASE=$OM_WORK/results/v4-7b \
bash scripts/go_v2.sh
```

### 2.4 tmux 포그라운드로 돌린다

장시간 실행은 `nohup`이 아니라 tmux 포그라운드에 두고 로그를 실시간으로 본다.
`go_v2.sh`의 워처가 15초마다 진행 줄을 콘솔로 뱉으므로, 화면이 조용한지
움직이는지가 곧 1차 진단이다.

```bash
tmux new -s v4
git pull && bash scripts/go_v4.sh 1
```

---

## 3. 중단·재개

### 3.1 재개 방법: 같은 명령을 다시 실행한다

모든 스테이지가 산출물 스킵으로 재개되고, rollout은 `.partial`에서 프롬프트
단위로 이어진다. **"처음부터 다시"는 어떤 경우에도 필요 없다.**

```bash
git pull && bash scripts/go_v4.sh 1     # 완료 run 스킵, 미완료만 이어서
```

`git pull` 후에도 안전하다. 각 run은 `run_config.json`에 기록된 **generation
commit**의 격리 worktree에서 계산을 이어가고, 감시·재시도만 최신 코드를 쓴다.
없는 commit은 `git fetch --no-tags origin <sha>`로 자동으로 받아 온다.

재개 계획만 먼저 확인하려면:

```bash
source scripts/setup_env.sh
python3 src/v4_resume_commit.py plan "$OM_WORK/runs" 1 "$(git rev-parse HEAD)"
```

stdout에 `name model seed dataset commit config` 탭 구분 행이,
stderr에 `[resume-v4-plan] <run>: <commit12> (<출처>)`와
`DONE skip=N, resume/start=M`이 나온다. 출처는
`recorded run_config` / 반대 데이터셋 run 이름 / `existing <모델> majority` /
`existing v4 majority` / `current checkout (no existing v4 config)` 중 하나다.

특정 run의 재개 환경을 확인하려면:

```bash
python3 src/v4_resume_commit.py env "$OM_WORK/runs/v4-27b-s1/run_config.json"
```

### 3.2 클라우드 운영자가 노드·잡을 죽인 경우

노드 안의 supervisor도 함께 사라지므로 **같은 `go_v4.sh <slot>` 명령을 다시
실행**해야 한다. 기존 shard와 `.partial`부터 재개되고 완료 run은 다시 계산하지
않는다.

### 3.3 supervisor가 3 pass 뒤에도 미완료를 남기면

`[resume-v4-abort] cluster N: supervisor 3회 뒤에도 미완료 run 존재`와 함께
exit 1이다. 미완료 run 이름은 그 위 `== cluster N pass P: 미완료 K개: ...`
줄에 있다. pass 수를 늘리려면:

```bash
OM_V4_SUPERVISOR_PASSES=5 bash scripts/go_v4.sh 1
```

늘리기 전에 7절의 진단 순서를 먼저 밟는다. 같은 자리에서 3번 실패하는 것은
보통 재시도로 풀리지 않는다.

### 3.4 `existing artifacts use a different run config: [...]`

immutable `run_config.json`과 현재 설정의 digest가 다르다는 뜻이다. 대괄호
안에 달라진 키가 나온다.

- `['git']`만 다르면 코드 revision 차이다. `resume_v4.sh` 경로는 기록된 commit
  worktree로 들어가므로 정상 재개에서는 이 오류가 나오지 않는다. 직접
  `run_14b.sh`를 돌리다 만난 것이라면 같은 commit으로 checkout하거나
  `prepare_run_path.py`로 격리한다.
- K·n_train 같은 실험 설정이 다르면 **새 `OUT_ROOT`를 쓴다.** 기존 폴더를
  덮어쓰면 안 된다.

수동 격리:

```bash
python3 src/prepare_run_path.py "$OM_WORK/runs/v4-27b-s1" \
  --expected-git "$(git rev-parse HEAD)" \
  --quarantine-root "$OM_WORK/quarantine/manual"
```

`--quarantine-unconfigured`는 `run_config.json`이 없는 비어 있지 않은 디렉터리도,
`--force-quarantine`은 기록된 git과 무관하게 이동시킨다. **삭제하지 않고
rename만 한다.**

### 3.5 중단하고 정리해야 할 때

```bash
bash scripts/reset_run.sh --dry-run          # 지울 것만 확인
bash scripts/reset_run.sh                    # soft: 프로세스 종료 + 투영 기반 산출물만 삭제
bash scripts/reset_run.sh --hard             # run 디렉터리 전체 삭제
```

soft는 rollout·drift adapter·`prompts.json`을 보존하므로 재개가 빠르다.
대상은 `OUT_ROOT`(기본 `$OM_WORK/runs/gate`)이므로 **반드시 `OUT_ROOT`를 명시**한다.

프로세스만 정리하려면(run 범위 한정, 다른 작업 안 건드림):

```bash
python3 src/cleanup_run_processes.py --run-prefix "$OM_WORK/runs/v4-"
```

---

## 4. 실행 중 상태 확인

### 4.1 진행 스냅샷 — `progress_snapshot.sh` (읽기 전용, 실행 중 안전)

돌고 있는 잡을 전혀 건드리지 않는다. GPU를 쓰지 않고 run 디렉터리에도 쓰지 않는다.

```bash
bash scripts/progress_snapshot.sh clusterA
```

산출: `$OM_WORK/progress/<MMDD-HHMM>-cluster<라벨>/`

| 파일 | 내용 |
|---|---|
| `PROGRESS.md` | 코드 commit, GPU 표, run별 아티팩트 존재표, config 요약, `rollouts_*.jsonl`·`*.partial` 행 수와 크기, 에러 있는 로그 파일 수, 최신 로그 tail 3줄, 콘솔 로그 tail 5줄 |
| `contract.txt` | `check_contract.sh` 출력 |
| `<run>/` | 완료 run의 소형 아티팩트 사본 (`run_config`·`manifest`·두 protocol·`report`·`DONE`·rollout manifest) |
| `judge-<run>.txt` | `report.json`+`score_protocol.json`이 있는 run의 judge 판정 (180초 타임아웃) |

rollout 원본(대용량 jsonl)은 복사하지 않으므로 폴더 전체를 그대로 전달할 수 있다.
라벨을 생략하면 `hostname -s`를 쓴다.

`.partial` 행 수가 곧 진행률이다. 예를 들어 27B fresh rollout shard가
`rollouts_fresh_train.shard0.partial: 1024행`이면 `1024 / FRESH_K(32) = 32`개
프롬프트를 완주한 것이다.

### 4.2 계약 판정 — `check_contract.sh`

run이 P0-1·P0-2 계약 **수정 이후** 산출물인지 판정한다.

```bash
bash scripts/check_contract.sh                       # $OM_WORK/runs 전체
bash scripts/check_contract.sh "$OM_WORK/runs/v4-27b-s1"   # 하나만
```

판정 근거는 두 가지다.

1. `rollouts_*.manifest.json`에 `"top_k": 0`이 있는가 (P0-1)
2. `rollouts_behavior_train.jsonl` 첫 행에 `"resp_end"`가 있는가 (P0-2)

| 출력 | 뜻 |
|---|---|
| `✅ 수정 후` | 둘 다 있음 — 논문 수치로 사용 가능 |
| `❌ 수정 전 — 논문 수치 불가` | 둘 다 없음 |
| `⚠ 혼재 — 폴더 재사용 의심` | 하나만 있음. 같은 폴더에 두 세대가 섞였을 가능성 |

이름에 `smoke`가 든 디렉터리는 건너뛴다. 종료 코드는 항상 0이므로
스크립트에서 자동 게이트로 쓰지 말고 사람이 읽는다.

이것은 **1차 선별**이다. 정밀 검증은 채점 단계에 내장된
`artifact_contract.validate_generation_contract()`가 하고, 그 결과가
`score_protocol.json`·`oracle_protocol.json`의 `generation_validation`에 남는다.

### 4.3 그 밖의 상태 명령 (v1 경로 기본값 주의)

```bash
bash scripts/check.sh "$OM_WORK/runs/v4-7b-s0"     # 3줄 진단 + 결론
bash scripts/status.sh "$OM_WORK/runs/v4-7b-s0"    # 진행 위치·ETA·산출물 체크리스트
bash scripts/result.sh "$OM_WORK/runs/v4-7b-s0"    # report 원문 + judge 판정
```

세 스크립트는 `_find_root.sh`를 통해 대상을 정하는데 **인자가 없으면 v1 경로
(`$OM_WORK/runs/gate`)를 기본으로 잡는다.** v4 run을 보려면 경로를 넘겨야 한다.
`check.sh`가 마지막에 안내하는 `nohup bash scripts/babysit.sh ...`는 현재
비활성 스크립트이므로 따르지 말 것(3.1절의 재개 방법을 쓴다).

`result.sh`는 `score_protocol.json`·`oracle_protocol.json`이 없는 report를
`[거부] 교정 protocol 없는 역사적 report`로 걸러낸다.

---

## 5. 수확과 병합

### 5.1 순서

```
(20 run 완주)
  │
  ├─ bash scripts/collect_v4.sh      # 완결성 검사 → 모델별 TABLES/FRONTIER → harvest
  │     └─ 내부에서 bash scripts/harvest.sh
  │
  └─ (또는) bash scripts/harvest.sh  # 표 없이 판독 산출물만
```

### 5.2 `collect_v4.sh` — 20-run 자동 취합

```bash
bash scripts/collect_v4.sh
```

GPU를 쓰지 않고 run 디렉터리도 수정하지 않는다. 하는 일:

1. `v4-{27b,7b}-s{0..4}[-math500]` 20개에 대해 11종 필수 산출물 +
   `divergence_stats*.json` 존재를 확인한다. 없는 것은
   `[missing-run]` / `[incomplete-run]` / `[missing-artifact]`로 **분리 출력**한다.
   하나라도 없으면 `[collect-v4-abort]`와 함께 exit 1이고 GPU는 재실행하지 않는다.
2. 입력 키가 기존 완료 마커와 같으면 27B·7B 보고서를 재사용한다. 변경됐을 때만
   `$OM_WORK/results/.v4-collect.XXXXXX` staging에서 **각각** 표를 만든다
   (`tables.sh` → `TABLES.md`, `frontier.sh` → `FRONTIER.md` + `frontier.json`).
   모델을 섞은 표를 만들지 않는 것이 요점이다.
3. 셋 다 비어 있지 않으면 `$OM_WORK/results/v4-27b/`·`v4-7b/`로 원자적 게시.
4. `harvest.sh` 실행. 전역 잠금 안에서 입력 키가 같은 기존 수확 폴더가 있으면 새 폴더나
   bootstrap을 만들지 않고 그 폴더를 반환한다. regime 누락처럼 수확 후반이 실패해도
   정상 완료된 `STATS.md`는 별도 입력 키로 체크포인트되어 다음 수확에서 재사용된다.

> **한계.** `collect_v4.sh`는 **provenance를 검사하지 않는다.** commit·모델
> 해시·고정 설정이 20 run에서 같은지 보지 않는다. 27B는
> `go_v4_27b.sh`가 `validate_v4_27b.py`로 따로 검증하지만, 7B와 전체 행렬에
> 대해 그에 해당하는 자동 검사는 현재 실행 경로에 없다
> ([ARCHITECTURE 12.1·12.3](ARCHITECTURE.md#121-go_v4sh-본문-대부분이-실행되지-않는다-영향-큼)).
> 수치를 원고에 넣기 전에 아래 수동 검증을 한 번 돌린다.

```bash
# 27B 10 run 전수 검증
python3 src/validate_v4_27b.py "$OM_WORK/runs" \
  --expected-git "$(git rev-parse HEAD)" \
  --expected-model-hash "$(sha256sum "$MODELS_DIR/Qwen3.8-27B-BF16/config.json" | cut -d' ' -f1)"

# 20 run의 commit·설정 동일성 눈으로 확인
for d in "$OM_WORK"/runs/v4-*-s*/; do
  case "$d" in *smoke*) continue;; esac
  python3 -c "
import json,sys
c=json.load(open(sys.argv[1]))
print(f\"{sys.argv[2]:28s} git={str(c.get('git'))[:12]} model={str(c.get('model_config_sha256'))[:12]} \"
      f\"n_train={c.get('n_train')} fresh_k={c.get('fresh_k')} gen_batch={c.get('gen_batch')} \"
      f\"fla={c.get('fla_core_version')} dirty={bool(c.get('git_status'))}\")
" "$d/run_config.json" "$(basename "$d")"
done
```

### 5.3 `harvest.sh` — 폴더 하나로 수확

```bash
bash scripts/harvest.sh
```

`$OM_WORK/readouts/<YYYY-MM-DD_HHMMSS>-harvest.XXXXXX/` 폴더 하나를 만들고
마지막 줄에 그 경로를 찍는다. **그 폴더만 전달하면 된다.**

성공 폴더의 Markdown은 아래 두 개뿐이다.

| 파일 | 내용 |
|---|---|
| `RESULTS.md` | regime final report + 모델별 TABLES + cross-run READOUT |
| `APPENDIX.md` | KCURVE/KCURVE_ALL + REVERSAL + STATS + 모델별 FRONTIER |

기계 판독 파일은 `REGIME-<이름>.json`/CSV,
`REGIME_SUMMARY-<이름>.csv`, 선택적 `REGIME_COLLECTION-<이름>.json`,
`PROVENANCE.json`, `HARVEST_MANIFEST.sha256`만 별도로 둔다. 개별 Markdown은 검증 중
임시 파일이며 성공 시 `RESULTS.md`/`APPENDIX.md`로 합친 뒤 삭제한다.

실패 처리 규약이 2026-08-20 수확 사고의 결과다. **실패를 숨기지 않는다.**
종료 코드가 허용 목록 밖이거나 출력이 비어 있으면 최종 이름을 주지 않고
`<이름>.partial.md`와 `<이름>.err`로 남긴 뒤 `HARVEST_FAILURES.md`에 기록하고
exit 1한다. `READOUT.md`가 0바이트로 전달되던 사고의 재발 방지다.

v4 run이 하나라도 있으면 시작 시 20-run × 6종 완결성을 먼저 검사한다.
누락이 있으면 `v4-matrix:incomplete-<N>-artifacts`를 실패 목록에 넣고
`V4_MATRIX.err`에 run별 누락 목록을 남긴다. **보고서 생성 자체는 계속 진행하되
최종 exit는 1이다.** `docs/results/2026-08-24/`의 번들이 그 상태로 도착한 예다.

`$OM_WORK/results/regime-*`가 하나라도 있으면 regime의 JSON/CSV/최종 보고서 4종을
모두 필수 산출물로 취급한다. 하나라도 없거나 비어 있으면 수확은 실패하고
`PROVENANCE.json`을 만들지 않는다.

### 5.4 개별 분석만 돌리기

`harvest.sh`가 전부 포함하므로 조기 확인용이다.

```bash
bash scripts/read_now.sh          # READOUT + REVERSAL만 원자적 publish
bash scripts/tables.sh            # TABLES.md
bash scripts/frontier.sh          # FRONTIER.md + frontier.json
bash scripts/kcurve.sh            # 사전등록 K-curve
bash scripts/kcurve_all.sh        # 전 세대 K-curve
bash scripts/reversal_freq.sh     # 반전율 + 닻
bash scripts/selection.sh         # 방법별 top-k 선택 내역·겹침 행렬
```

`tables.sh`·`frontier.sh`는 인자가 없으면 corrected protocol이 있는 완주 run을
자동 수집하고, 세대가 하나면 `$OM_WORK/results/v<N>/`, 여러 세대면
`results/all/`에 쓴다. `OM_RESULTS`로 출력 경로를 직접 지정할 수 있다.

```bash
OM_RESULTS=$OM_WORK/results/v4-7b bash scripts/tables.sh "$OM_WORK"/runs/v4-7b-s*
```

### 5.5 백업

```bash
bash scripts/backup_results.sh
git push            # 온라인 셸에서 (클러스터는 GitHub egress 없음)
```

소형 정본만(`report`·`manifest`·`scores_*`·`protocol`·`divergence_stats`·
`downstream_*`) 레포 안 `results/backup/`으로 모아 로컬 커밋한다. rollout
jsonl과 `.pt`는 제외한다 — seed 고정으로 재현 가능하고, 판정에 필요한 것은
전부 백업에 들어간다. **레포에 커밋을 만드는 유일한 스크립트**이므로
실행 시점을 의식할 것.

### 5.6 교정 전 산출물 되살리기

```bash
python3 src/rescore_completed_run.py RUN_DIR       # GPU 필요 — score 재계산 포함
python3 src/recompute_oracle_scores.py RUN_DIR     # GPU 불필요 — oracle/report만
```

`rescore_completed_run.py`는 dirty tree를 거부하고, 생성 계약을 먼저 검증한 뒤
통과할 때만 재점수화한다. **계약 검증에 실패하면 우회하지 말고 새
`OUT_ROOT`에서 generation부터 다시 한다.** `recompute_oracle_scores.py`는
이미 새 score 프로토콜이 있는 run에만 쓴다.

두 경로 모두 `postprocess_manifest.json`에 입력·출력 해시와
`source_run_git`/`postprocess_git`을 남긴다. `score_protocol.json`과
`oracle_protocol.json`을 **수동으로 만들지 말 것** — 판정기가 마커의 존재를
계약 통과의 증거로 취급한다.

---

## 6. 환경변수 전체 목록

### 6.1 경로·환경 (`scripts/setup_env.sh`)

| 변수 | 기본값 | 의미 |
|---|---|---|
| `GROUP_VOLUME` | `/group-volume` | 공유 볼륨 마운트 |
| `OM_USER` | `minsoo3.kim` | group-volume 안 사용자 폴더명 |
| `OM_REPO` | 레포 루트 (자동) | 체크아웃 경로 |
| `OM_WORK` | `$GROUP_VOLUME/$OM_USER/offpolicy-misranking`, 없으면 `$OM_REPO/.work` | 작업 루트 (모든 산출물) |
| `STORAGE_ROOT` | `$OM_WORK` | 별칭 |
| `MODELS_DIR` | `$GROUP_VOLUME/models`, 없으면 `$OM_WORK/models` | 모델 스냅샷 |
| `MODEL_QWEN25_05B` | `$MODELS_DIR/Qwen2.5-0.5B-Instruct` | 스모크용 |
| `MODEL_QWEN25_7B` | `$MODELS_DIR/Qwen2.5-7B-Instruct` | 재현 축 |
| `DATASETS_DIR` | `$GROUP_VOLUME/$OM_USER/datasets` → `$GROUP_VOLUME/datasets` → `$OM_WORK/data` | 데이터셋 배치본 |
| `OM_DATA` | `$OM_WORK/data` | provision이 받은 로컬 jsonl |
| `VENV_DIR` | `$OM_WORK/.venv-cu126` | venv |
| `HF_HOME` | `$OM_WORK/cache/huggingface` | HF 캐시 |
| `PIP_CACHE_DIR` | `$OM_WORK/cache/pip` | pip 캐시 |
| `TMPDIR` | `$OM_WORK/tmp` | 임시 |
| `PYTHONPYCACHEPREFIX` | `$OM_WORK/cache/pycache` | pyc 격리 |
| `PYTHONPATH` | `$OM_REPO/src` 선두 추가 | import 경로 |
| `OM_ONLINE` | `0` | `1`이면 HF 오프라인 해제 |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / `HF_DATASETS_OFFLINE` | `1` (오프라인) | 컴퓨트 노드 기본 |
| `HF_ENDPOINT` | 미설정 | HF 미러 (클러스터 egress 없음) |
| `PIP_INDEX_URL` | 미설정 | 사내 pip 미러 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 단편화 완화 |
| `PRESCREEN_SEED` | `104729` | hard pool 이름에 들어가는 seed |
| `OM_FETCH_TIMEOUT` | `3600` | provision 다운로드 타임아웃(초) |

### 6.2 실험 설정 (`run_14b.sh`가 `run_config.json`에 고정)

| 변수 | run_14b 기본값 | v4 실제값 | 의미 |
|---|---|---|---|
| `MODEL_14B` | `$MODELS_DIR/Qwen2.5-14B-Instruct` | 27B 또는 7B 경로 | 모델 스냅샷 (없으면 abort) |
| `OUT_ROOT` | `$OM_WORK/runs/gate-14b` | `$OM_WORK/runs/v4-<모델>-s<seed>[-<ds>]` | run 디렉터리 |
| `DATASET` | `gsm8k` | `gsm8k` \| `math500` | 데이터셋 (`dapo-math`·`mbpp`·`kk`·`arc-challenge`·`apps` 가능) |
| `SEED` | `0` | `0..4` | 생성 샘플링·LoRA init·tie-break (프롬프트 분할은 고정) |
| `N_TRAIN` | `256` | `512` (gsm8k) / `400` (math500) | 후보 프롬프트 수 |
| `N_VAL` | `50` | `100` | validation 프롬프트 수 |
| `BEHAVIOR_K` | `8` | `8` | β rollout 수/프롬프트 |
| `FRESH_K` | `16` | `32` | π fresh rollout 수/프롬프트 |
| `VAL_K` | `8` | `8` | validation rollout 수/프롬프트 |
| `MICRO_GROUP` | `4` | `4` | oracle micro-group 크기 (`FRESH_K`의 약수, 그룹 수가 짝수여야 함) |
| `HYBRID_PROMPTS` | `24` | `64` | hybrid 대상 프롬프트 수 |
| `K_CELL` | `8` | `8` | hybrid 셀별 K (≥2) |
| `DRIFT` | `100` | `100` | drift SFT step 수 |
| `MAX_NEW_TOKENS` | `512` | `512` | 생성 상한 |
| `PROJ_DIM` | `4096` | `4096` | CountSketch 투영 차원 |
| `GRAD_LAYERS` | `4` | `4` | gradient 대상 마지막 decoder block 수 |
| `CLIP_CAP` | `10.0` | `10.0` | IS 가중치 양측 clip (≥1) |
| `TEMPERATURE` | `1.0` | `1.0` | **1.0이 아니면 실행 거부** |
| `TOPK_FRAC` | `0.10` | `0.10` | top-k 비율 |
| `RADIUS_MODE` | `gaussian` | `gaussian` | CertaGrad 반경 (`hoeffding` 가능) |
| `OM_POOL_FILE` | 미설정 | 미설정 | 사전 구성 풀 jsonl로 데이터셋 내용 대체 |

### 6.3 런타임 스위치

| 변수 | 기본값 | 의미 |
|---|---|---|
| `OM_TOP_P` | `1.0` | 생성 top-p. **1.0이 아니면 실행 거부** (raw softmax 계약) |
| `OM_THINKING` | 미설정 (= off) | `on`이면 chat template의 thinking 모드 유지 |
| `OM_ATTN` | `run_14b.sh`가 `eager` 강제 | `eager`\|`sdpa`\|`flash_attention_2`. fused SDPA 커널 병든 노드 우회 |
| `OM_GEN_BATCH` | 미설정 (= K 전체 한 배치) | generate 배치 상한. 27B는 4 또는 8 |
| `OM_LORA_TARGETS` | 미설정 (= `q_proj,v_proj`) | 콤마 목록 또는 `all-linear`. DeltaNet류는 `all-linear` |
| `OM_SKIP_HYBRID` | `0` | `1`이면 hybrid 전체 생략 (27B는 π+β 동시 상주 불가) |
| `OM_GPUS` | 미설정 (= 전 GPU) | `"0,1"`처럼 사용할 GPU 제한 |
| `OM_SKIP_GPU_CHECK` | `0` | `1`이면 2GB 초과 점유 검사 무시 (비권장) |
| `OM_RETRY_INDEX` | `1` | shard의 물리 GPU 배정 회전 인덱스 |
| `OM_ALLOW_DIRTY` | `0` | `1`이면 dirty `src/scripts`에서도 run 초기화 허용 |
| `OM_EOS_IDS` | 미설정 | 구버전 rollout 절단용 EOS id 목록 (예: `151645,151643`) |

### 6.4 오케스트레이션

| 변수 | 기본값 | 의미 |
|---|---|---|
| `SEEDS` | `0 1 2` | `go_v2.sh` seed 목록 |
| `DATASETS` | `gsm8k dapo-math` | `go_v2.sh` 데이터셋 목록 |
| `SEEDS_ALL` | `0 1 2 3 4` | `go_retry.sh` seed 목록 |
| `EXPECTED_V4_SEEDS` | `0 1 2 3 4` | `go_v4.sh` 집계 대상 seed |
| `RUN_BASE` | `$OM_WORK/runs/v2` | run 경로 접두사 |
| `RUN_BASE_SMOKE` | `$RUN_BASE-smoke` | 스모크 run 경로 (v4는 commit·클러스터별) |
| `RUN_LABEL` | `basename($RUN_BASE)` | 콘솔 로그 파일명 접두사 |
| `RESULTS_BASE` | `$OM_WORK/results/v2` | 결과 수집 경로 |
| `OM_RESULTS` | 세대 자동 판별 | `make_tables`/`frontier` 출력 경로 |
| `OM_MAX_RETRIES` | `2` | run당 재시도 (v4 5, 27B rerun 10) |
| `OM_STALL_MINUTES` | `5` | watchdog 로그 무변화 임계 (27B 20) |
| `OM_SKIP_POSTPROCESS` | `0` | `1`이면 워커가 표를 만들지 않음 (병렬 워커용) |
| `OM_PIPELINE_REPO` | `$PWD` | 계산 코드 worktree |
| `OM_PIPELINE_SCRIPT` | `$OM_PIPELINE_REPO/scripts/run_14b.sh` | 실제 실행할 파이프라인 |
| `OM_V4_SUPERVISOR_PASSES` | `3` | `resume_v4.sh` 재계획 pass 수 |
| `OM_V4_RESUME_WRAPPED` | `0` | 내부의 도달 불가 legacy gate. 사용자가 설정하지 않는다. `go_v4.sh`는 `resume_v4.sh`에 위임한다 |
| `OM_ENABLE_LEGACY_RUNNER` | `0` | `1`이어야 `run_h100_all.sh`·`babysit.sh` 실행 |
| `MODEL_7B` / `MODEL_27B` | `$MODELS_DIR/Qwen2.5-7B-Instruct` / `$MODELS_DIR/Qwen3.8-27B-BF16` | `go_v4.sh` 모델 |
| `REPO27B` | `Qwen/Qwen3.6-27B` | `fetch_27b.sh`·`go_new.sh` 다운로드 대상 |
| `OM_TARGET` | 미설정 | `_find_root.sh` 대상 (`7b`\|`14b`\|`7bm`\|`14bm`\|`fast`\|경로) |
| `FORCE_HARD` | `0` | `go_hard.sh` precheck 우회 (폐기됨) |
| `OM_SKIP_EXTRA` | `0` | `go_v3.sh`에서 mbpp·kk 생략 |

### 6.5 데이터셋 위치 override

`MATH500_DIR`, `MBPP_DIR`, `KK_DIR`, `DAPO_DIR`, `APPS_DIR`. 지정하면
`_candidate_roots()`가 **그 경로만** 본다. 자동 탐색이 엉뚱한 폴더를 잡을 때
쓴다.

---

## 7. 실패 시 진단 순서

**순서대로 밟는다.** 아래로 갈수록 비싸다.

### 1단계 — 정말 멈춘 것인가 (30초)

```bash
timeout 20 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

- **util > 0** → 계산 중이다. 로그가 조용해도 정상일 수 있다. score의 β-pass,
  val-grads, oracle-grads는 수십 분 무출력 구간이다. `go_v2.sh`의 워처도
  GPU·CPU 활동이 있으면 죽이지 않는다.
- **util 0%가 지속** → 진짜 hang이다. 2단계로.

콘솔에 `[워처] 로그 N분 무변화지만 계산 활동 확인 (GPU x%, CPU +ys) — 계속 실행`이
찍히면 1단계는 통과한 것이다.

### 2단계 — 자동 진단 리포트 (1분)

```bash
bash scripts/diagnose_run_failure.sh "$OM_WORK/runs/v4-27b-s1" \
     "$OM_WORK/console-logs/v4-27b-resume-cluster1-gsm8k-s1.log" 1
```

`<RUN_DIR>/FAILURE_DIAGNOSTIC.txt`를 만들고 화면에도 찍는다. 내용:
에러 시그니처 집계(traceback/OOM/killed/abort/CUDA/watchdog 등 최근 80줄),
최근 로그 8개의 tail 30줄, 필수 산출물 9종 존재표, 남은 run 프로세스,
GPU 상태. `go_v2.sh`는 실패할 때마다 이걸 자동으로 부른다.

노드 전체 상황이 필요하면:

```bash
bash scripts/diagnose.sh     # $OM_WORK/console-logs/DIAGNOSIS.txt 하나로
```

코드 commit, GPU, 디스크, run별 DONE 현황, 전 로그의 에러 시그니처 빈도표,
`dmesg | grep -i xid`, 최근 로그 tail을 한 파일에 모은다.

### 3단계 — 증상별 분기

| 증상 | 판정·조치 |
|---|---|
| `CUDA error: unspecified launch failure` | `bash scripts/gpu_check.sh` → matmul까지 실패면 **노드 교체**, SDPA만 실패면 `OM_ATTN=eager`(이미 기본). 27B Gated DeltaNet 경로는 `OM_ATTN` 관할 밖이므로 FLA kernel 확인 (TROUBLESHOOTING C5·C6·C7) |
| util 0% 지속인데 에러 없음 | fused 커널 동결 또는 group-volume 스톨. `ps -eo pid,stat,wchan \| awk '$2~/D/'`와 `dmesg`의 `nfs not responding` / `Xid` 확인 → RECOVERY 상황 1 (노드 교체) |
| OOM | 좀비 프로세스 확인 → `python3 src/cleanup_run_processes.py --run-prefix "$OM_WORK/runs/v4-"` 후 `nvidia-smi`로 해제 확인. 27B는 `OM_GEN_BATCH`를 줄인다 |
| `existing artifacts use a different run config: [...]` | 3.4절 |
| `[merge-abort] ... 누락 N개 ... exact-K 실패 M개` | GPU 수가 바뀐 재시작. 메시지에 나오는 `rm` 명령대로 정리 후 재실행 |
| `rollout 산출물 행 수 불일치: X != n×K` | `.partial`이 보존돼 있다. 재실행하면 `salvage_partial`이 완주분만 승계한다 |
| `[abort] <ds> 데이터 로드 실패` | `bash scripts/check_data.sh <ds>` — 로더가 찾아본 위치 전체가 출력된다. `fetch_datasets.sh`로 확보하거나 `<DS>_DIR` 지정 |
| `[abort] <ds> 데이터 로드 600초 초과` | HF datasets stale lock. `find "$HF_HOME" -name '*.lock' -mmin +30 -delete` (run_14b가 자동으로도 한다) |
| `corrected off-policy score protocol is missing` | 그 run은 교정 전 산출물이다. `check_contract.sh`로 확인 후 5.6절 재점수화 또는 새 `OUT_ROOT` |
| `generation provenance changed after scoring: artifact_sha256` | 채점 후 rollout 파일이 바뀌었다. 그 run은 폐기 대상 |
| `[27b-runtime] FLA 0.5.2 not ready` | `go_v4_27b.sh`가 자동 설치를 시도한다. 실패하면 fallback으로 27B를 돌리지 말 것 |
| `[abort] src/scripts worktree is dirty` | 코드를 먼저 커밋한다. **문서만 바꿨어도 `src`·`scripts`가 dirty면 실행이 막힌다** |
| `==== [key] ✘ child는 성공했지만 필수 artifact가 불완전` | 2026-08-24에 막은 false-success 경로. 진단 출력의 누락 artifact를 본다 |
| `[harvest-abort] ... v4-matrix:incomplete-N-artifacts` | 20 run이 안 찼다. `V4_MATRIX.err`에 run별 누락 목록 |
| `[collect-v4-abort]` | 5.2절의 세 카테고리 출력을 본다. GPU는 재실행되지 않았다 |

### 4단계 — 복구 분기

1. `gpu_check.sh`에서 matmul 또는 여러 커널이 실패하거나 `dmesg`에 Xid가 있으면
   코드 옵션을 바꾸지 말고 **노드를 교체**한다. 새 노드에서 `git pull`,
   `provision.sh`, `preflight.sh` 뒤 사용하던 동일 진입점을 다시 실행한다.
2. SDPA만 실패하면 `OM_ATTN=eager`를 쓴다(v4 기본값). 27B Gated DeltaNet 실패는
   `go_v4_27b.sh`의 FLA 0.5.2 스모크를 통과한 노드에서만 재개한다.
3. 설정·commit 불일치는 파일을 직접 지우지 않는다. `go_v4_27b.sh`와 regime runner가
   기존 run을 quarantine한 뒤 canonical 경로를 다시 만든다. 고정 v4의 부분 run은
   `go_v4.sh <slot>`이 기록된 generation commit에서 재개한다.
4. 클라우드 운영자가 top-level job을 종료하면 노드 안 supervisor도 사라진다. 공유
   볼륨의 완주 artifact와 `.partial`은 남으므로 같은 명령을 다시 실행한다.
5. 부분 matrix는 진행 확인용 표만 만들 수 있다. 제출용 수치는 필수 artifact와
   lineage 검사를 모두 통과한 완결 matrix에서만 동결한다.

---

## 8. CPU에서 안전하게 돌릴 수 있는 것

실험 중에도 안전한 읽기 전용·CPU 전용 명령이다.

```bash
bash scripts/progress_snapshot.sh <라벨>     # 진행 스냅샷 (읽기 전용)
bash scripts/check_contract.sh               # 계약 판정
bash scripts/harvest.sh                      # 수확 (기존 산출물 재집계)
bash scripts/read_now.sh                     # READOUT + REVERSAL
bash scripts/tables.sh / frontier.sh / kcurve.sh / reversal_freq.sh
python3 src/judge.py <RUN_DIR>               # 게이트 판정
python3 src/stats_extra.py <RUN_DIR>         # 정확 p·bootstrap CI
python3 src/stats_extra.py --sign 12 3       # 부호검정만
python3 scripts/verify_theory.py             # 이론 열거 검산 (표준 라이브러리만)
python3 src/c2_sweep.py / src/c2_diagnose.py # C2 재판정
```

### 8.1 테스트

`tests/`의 CPU 회귀 테스트는 전부 모델·GPU 없이 돈다(GPU를 쓰는 셸 스크립트는 fake
`nvidia-smi`로 대체한다). `test_rollout_resume.py`의 최상위 `sys.exit`도
`91025ca`에서 `__main__` 아래로 이동해 전체 pytest 수집이 가능하다.

```bash
export OM_WORK=/tmp/om-test GROUP_VOLUME=/nonexistent CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=$PWD/src
PY=.work/.venv-cu126/bin/python      # 또는 $VENV_DIR/bin/python

# 전체 회귀 테스트 (2026-08-25: 51 passed)
$PY -m pytest tests/ -q

# 개별 스크립트의 상세 PASS 로그가 필요할 때만 직접 실행
for f in tests/test_cleanup_run_processes.py tests/test_data_sandbox.py \
         tests/test_failure_diagnostic.py tests/test_frontier.py \
         tests/test_pool_qualification.py tests/test_prepare_run_path.py \
         tests/test_rollout_resume.py; do
  echo "== $f"; $PY "$f" || echo "FAIL $f"
done
```

`OM_WORK`를 임시 경로로 돌려두는 것이 중요하다. 일부 셸 회귀 테스트가
`setup_env.sh`를 거치므로 실험 볼륨을 건드리지 않게 격리한다.

정적 검사:

```bash
$PY -m ruff check --no-cache src scripts tests
bash -n scripts/*.sh
git diff --check
```

---

## 9. 하지 말아야 할 것

| 금지 | 이유 |
|---|---|
| 실행 중 공유 checkout에서 `git pull` | 셸이 다음 스테이지에서 새 Python 프로세스를 시작하면 한 run에 코드 버전이 섞인다. `verify_code_snapshot()`이 잡아서 중단시키기는 한다 |
| `src/`·`scripts/`를 dirty로 둔 채 실행 | `go_v4.sh`·`go_v4_27b.sh`·`run_14b.sh`가 전부 abort한다 |
| `score_protocol.json`·`oracle_protocol.json` 수동 생성 | 마커의 존재가 계약 통과의 증거로 쓰인다. 5.6절의 교정 코드로만 만든다 |
| 같은 폴더에 다른 K·n으로 재실행 | immutable digest가 막지만, 막히지 않는 조합이 있으면 조용히 오염된다. 새 `OUT_ROOT`를 쓴다 |
| `go_v4.sh`와 `go_v4_27b.sh`를 27B에 섞어 쓰기 | 2.2절 주의 — 전자의 산출물이 후자에서 전부 격리된다 |
| in-place로 `VAL_K` 늘리기 | `--stage val-deepen`과 `deepen_val.sh`는 의도적으로 비활성이다. 더 깊은 validation이 필요하면 새 run |
| `OM_ENABLE_LEGACY_RUNNER=1`로 confirmatory 실행 | `run_h100_all.sh`·`babysit.sh`는 manifest·code lock·exact merge 계약 이전 runner다 |
| `TEMPERATURE`·`OM_TOP_P`를 1.0 외의 값으로 | raw-softmax IS 계약이 깨진다. `experiment.py`가 실행을 거부한다 |
| 진행 로그 tail을 집계 통계로 쓰기 | BACKLOG에 명시된 금지. 전체 reward 분포는 산출물에서 다시 계산한다 |
| 부분 matrix 수치를 원고 표에 넣기 | 20/20이 contract·lineage 검사를 통과한 단일 matrix에서만 확정한다 |

---

## 10. 자주 쓰는 경로

```bash
source scripts/setup_env.sh

$OM_WORK/runs/v4-27b-s0            # run 디렉터리
$OM_WORK/runs/v4-7b-s2-math500
$OM_WORK/runs/v4-27b-s0/logs/main.log
$OM_WORK/console-logs/v4-*.log     # 워커 콘솔 로그
$OM_WORK/results/v4-27b/TABLES.md
$OM_WORK/results/v4/V4_COMPLETE
$OM_WORK/readouts/                 # harvest 산출
$OM_WORK/progress/                 # progress_snapshot 산출
$OM_WORK/quarantine/v4/            # 격리된 run
$OM_WORK/code-snapshots/           # 재개용 worktree
$OM_WORK/locks/                    # 27B 공유 큐 lock
```

한 run이 어디까지 갔는지 한눈에 보기:

```bash
RUN=$OM_WORK/runs/v4-27b-s1
for a in prompts.json run_config.json manifest.json \
         rollouts_behavior_train.jsonl rollouts_fresh_train.jsonl \
         rollouts_fresh_val.jsonl val_groups.pt oracle_micro_groups.pt \
         scores_offpolicy.json scores_oracle.json scores_splithalf.json \
         score_protocol.json oracle_protocol.json report.json DONE; do
  [ -s "$RUN/$a" ] && echo "ok      $a" || echo "missing $a"
done
ls "$RUN"/*.partial 2>/dev/null && wc -l "$RUN"/*.partial
```
