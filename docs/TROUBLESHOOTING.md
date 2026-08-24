# 트러블슈팅 기록 — 2026-08-07 ~ 08-10 게이트 구축에서 잡은 주요 에러

증상 → 원인 → 수정 순으로 기록한다. 같은 함정을 다시 밟지 않기 위한 정본.
(커밋 해시는 이 레포 기준)

## A. 치명 — 실행을 죽이던 것들

### A1. 투영 OOM: 밀집 JL 행렬이 최대 131GB 할당
- **증상**: oracle/score 단계 진입 직후 CUDA OOM으로 파이프라인 사망.
- **원인**: `project_grads`가 청크(8M 원소)×dim(4096)짜리 fp32 가우시안 행렬을
  통째로 생성 — 8M×4096×4B ≈ 131GB.
- **수정**: 전역 원소 위치의 정수 해시(splitmix형) 기반 **CountSketch**로 교체.
  추가 메모리 수 MB. 같은 grad→같은 투영, 청크 크기 불변, cosine(0.9191→0.9198)·
  norm(비율 1.008) 보존을 테스트로 고정. RNG 스트림 방식은 청크 경계가 결과를
  바꿔서 기각하고 위치 해시로 확정. (`1685bf4`)

### A2. 7B LoRA 학습 OOM
- **증상**: drift SFT/GRPO-lite 학습 중 OOM.
- **수정**: gradient checkpointing(`use_reentrant=False` + `enable_input_require_grads`)
  + 학습 시퀀스 1280 토큰 상한 + score의 fp32 logits 동시 보유 절반(micro_batch 4→2). (`1685bf4`)

### A3. 클러스터 "GPU 유휴 3시간 → 잡 킬"
- **증상**: 장시간 CPU 구간(다운로드·피팅·판정)에서 잡이 통째로 사라짐.
- **수정 1차(실패)**: 5분 간격 2초 버스트 — 사용률 평균 0%로 잡혀 무효.
- **수정 2차**: **소형 커널(256×256) 연속 발사** — 사용률 지표는 커널 실행 시간
  비율이라 상시 36~57%로 찍힘(로컬 실측, 0% 샘플 없음). 실연산 미미, 본 작업에
  자연 양보. (`dadb060`) GAUGE-CPT 레포에도 이식(`4755b59`).

### A4. 다중 babysit 재시작 폭풍 ← "진행하다 자꾸 멈춤"의 핵심
- **증상**: 감시자(babysit)를 띄울 때마다 누적 → 여러 감시자가 서로의 실행을
  "죽었다"고 오판, 정리(reset pkill)로 **서로의 run을 죽이며** 무한 재시작.
- **수정**: 싱글턴 잠금(`$OM_WORK/.babysit.lock`) — 두 번째부터는 "이미 실행 중"
  출력 후 종료. 재시작 직후 60초 생존 확인·사망 사유 자동 기록.
- **교훈**: 자동 재시작 데몬은 반드시 싱글턴으로 설계할 것.

### A5. 좀비 프로세스 위 재시작 → OOM
- **증상**: 죽은 실행의 keepalive/학습 프로세스가 GPU를 점유한 채 새 실행 시작 → OOM.
- **수정**: 시작 전 GPU 점유 검사(2GB+ 발견 시 PID 출력·중단), reset이
  experiment/run/keepalive 고아까지 pkill. (`03556d1`, babysit 계열 커밋)

## B. 정합성 — 조용히 결과를 오염시키던 것들

### B1. 부분 파일이 완성본으로 오인 (재개 오염)
- **증상**: 중단된 rollout jsonl이 남아 다음 실행이 "존재=완료"로 스킵 →
  일부 프롬프트가 통째로 빠진 채 oracle/score 계산.
- **수정**: **원자적 쓰기**(.tmp에 쓰고 완료 시 rename) + reset의 완결성 검증
  (파일의 distinct prompt 수를 prompts.json과 대조, 미달 시 삭제). (`aba4ea3`, `8603f92`)

### B2. 산출물 경로 이력 불일치 → "전부 미판정"
- **증상**: result/status가 빈 출력 — 실행 이력마다 기본 경로가 달랐음
  (`outputs/h100` → `$OM_WORK/runs/gate` → fast 분리 `gate-fast`).
- **수정**: `_find_root.sh` 공용 자동 탐색(모든 이력 경로에서 최신 산출물 채택,
  위치를 첫 줄에 명시) + fast/full **산출물 경로 분리**로 혼입 원천 차단. (`5847468` 계열)

### B3. fast/full 설정 혼입
- **증상**: full 재시작이 이전 fast 실행의 fresh(K=16)를 완성본으로 재사용 —
  "왜 자꾸 fast냐"의 원인.
- **수정**: 모드별 OUT_ROOT 분리 + 시작 로그에 유효 설정(DRIFTS/FRESH_K/...) 명시.

### B5. 재개 시 val fresh 수집 영구 누락 ← "진행하다가 터짐"의 정체
- **증상**: 14B run이 중간까지 진행 후 `rollouts_fresh_val.jsonl` 없음(line 37)으로 사망.
  최초 실행은 멀쩡하고 **크래시 후 재시작한 run만** 죽어서 원인이 오래 숨음.
- **원인**: rollout-fresh의 val 수집이 "train 샤드 파일 없음" else 분기 안에 있었다
  — 재시작하면 train 샤드가 '이미 존재—스킵' 되면서 val 수집도 함께 증발.
- **수정**: val 검사를 train 스킵 여부와 무관한 독립 블록으로 분리 (`936b267` 후속).
  교훈: **재개 스킵은 산출물 단위로 독립적이어야 한다** — 한 스테이지가 산출물
  두 개를 만들면 스킵 판단도 두 번 해야 한다.
- **전수 점검(`d51b479`)**: 같은 부류를 코드 전체에서 수색해 4건 추가 수정 —
  ① stage_oracle val 가드가 산출물 2개 중 1개만 검사 ② merge-grads oracle 도출
  가드 동일 문제 ③ analyze 스킵 가드가 oracle 산출물 3개 중 1개만 검사
  ④ json/pt 저장 전면 원자화(.tmp→rename — 깨진 파일이 exists() 통과하는 것 차단)
  ⑤ run_14b 샤드 병합을 `cat`에서 **커버리지 검증 병합**으로 교체(전 프롬프트
  존재·무중복 확인, GPU 수 변경 재시작의 조용한 누락/중복 차단; 위반 시 정리
  명령까지 출력하고 중단).

### B7. 이종-n 샤드·옛-π 산출물 잔재 — 실행 전 자동 격리 (사전 추적으로 발견)
- **증상(예정돼 있던)**: GPU 분할 수(n)가 바뀐 재시작에서 옛 n의 샤드가 이름이
  같아 스킵→병합 abort로 사망하거나, π 재학습 이전의 score 샤드가 병합 때 새
  값을 **덮어써 조용히 오염**. 실제 터지기 전에 실행 경로 전체 정적 추적으로 발견.
- **수정**: run_14b가 prep 직후 ① adapter보다 오래된 π-의존 산출물 전부 격리
  ② 현재 n과 커버리지가 안 맞는 rollout 샤드 + 인덱스 초과 score/micro 샤드
  격리 (stale-*/ 폴더로 이동, 삭제 아님). 시나리오 5종 + 나이 규칙 셸 테스트로 검증.
- 같은 날 함께 잡은 것: go7_14의 14B 실행줄에 FRESH_K=32 누락(재실험 목적 자체를
  무효화할 뻔), 건강검사 실패 시 사인 은폐(2>/dev/null), venv 부재 시 오진.
- 교훈: **"실행 전에 실행 경로를 처음부터 끝까지 종이 위에서 밟아보는 것"이
  사후 디버깅 왕복 열 번보다 싸다.**

### B6. 재개 시 drift 재학습 → 정책 불일치 오염 (잠복하다 발견)
- **증상**: 없음(조용함) — 재개할 때마다 drift LoRA가 다시 학습되는데 초기화가
  랜덤이라 π가 매번 조금 다른 정책이 된다. 이전에 계산된 점수와 새로 계산되는
  oracle이 **다른 π**를 기준으로 하게 될 수 있는 잠복 오염.
- **완화 요인**: run_14b는 2×2 점수(score-shard)도 무조건 재계산하는 구조라
  "한 번의 재개 안에서는" 내부 일관성이 유지된다 (val fresh 샘플만 옛 π 출신 —
  기준 방향으로만 쓰이므로 영향 미미). hybrid 점수는 스킵 가드가 있어 stale로
  남을 수 있음 — 14B C1′ 판정에 hybrid를 쓸 때는 재생성 여부 확인 필요.
- **수정**: drift 스테이지에 adapter 존재 시 스킵 추가 (재학습하려면 폴더 삭제).
  교훈: **랜덤성이 있는 스테이지는 반드시 산출물 스킵 가드가 있어야 한다** —
  결정적 스테이지의 재실행은 낭비지만, 랜덤 스테이지의 재실행은 오염이다.

### B4. phase1 하나 실패 → 전체 exit
- **증상**: 병렬 파이프라인 중 하나만 넘어져도 성한 결과까지 버리고 종료.
- **수정**: 부분 실패 비치명화 — 살아남은 파이프라인으로 report·판정까지 진행.

## C. 라이브러리/환경 함정

### C1. transformers 5.x `apply_chat_template` 반환형
- **증상**: `'tokenizers.Encoding' object has no attribute 'to'`.
- **수정**: 템플릿을 `tokenize=False`로 텍스트만 뽑고 별도 토크나이즈
  (`add_special_tokens=False`). 4/5 양쪽 호환. (`65b159c`)

### C2. LoRA `merge_and_unload()` 후 requires_grad 전체 소실
- **증상**: `element 0 of tensors does not require grad` — backward 사망.
- **수정**: `grad_params()`가 전체 동결 후 대상 레이어만 활성화(메모리 절약 겸용).

### C3. 생성 길이 절단 → 보상 전멸
- **증상**: 정답 0/전체 — 모든 응답이 정확히 max_new_tokens에서 잘림.
- **원인**: 128 토큰은 GSM8K가 `####` 정답에 도달하기 전.
- **수정**: 384+ 토큰, temp 0.7. **rollout 로그에 정답 수를 찍어 즉시 보이게** 한 것이
  재발 방지의 핵심.

### C4. uv 부트스트랩 차단(사내망 403) / pypi·HF 무한 대기
- **수정**: venv+pip 경로(constraints 고정), pip `--timeout 60`·curl 타임아웃·
  `HF_HUB_ETAG_TIMEOUT=15`+미러 폴백 — "멈춘 것처럼 보임"을 "명확한 에러"로 전환.

### C5. cuDNN SDPA 'unspecified launch failure' — 두 번 잡은 버그
- **증상**: Hopper(H100)에서 forward 중 `CUDA error: unspecified launch failure`.
  비동기 에러라 보고 지점이 매번 다름(modeling_qwen2.py:261 → 재발 때는 :47).
- **1차 수정**: `load_model()` 안에서 `torch.backends.cuda.enable_cudnn_sdp(False)`.
  **그러나 재발** — drift 학습(rollout.py의 직접 `from_pretrained`)과
  downstream(train_downstream.py)은 load_model을 안 거쳐 미적용이었다.
- **최종 수정**: rollout.py **모듈 import 시점** 전역 비활성(모든 스테이지가
  rollout을 import). 교훈: 프로세스 전역이어야 하는 설정을 특정 로더 함수 안에
  두지 말 것 — 보고 라인이 달라도 같은 병일 수 있다.

### C6. ULF 3차 재발 — cuDNN이 아니라 그 노드의 fused SDPA 커널 전체
- **증상**: cuDNN SDPA 전역 비활성(C5) 후에도 특정 14B 인스턴스에서 **생성 시작
  직후** `this_peer_finished: CUDA error: unspecified launch failure` (transformers
  generate 루프 내부에서 보고).
- **진단 도구**: `scripts/gpu_check.sh` — GPU마다 ① 순수 matmul ② SDPA를 분리
  실행해 "하드웨어/드라이버 병"과 "attention 커널 병"을 즉석 판정.
- **수정**: `OM_ATTN=eager` 환경변수 (신규 스위치 — 생성·drift 학습·downstream의
  모든 모델 로드에 `attn_implementation` 강제). eager는 fused 커널을 아예 안 쓰므로
  느리지만 확실. **해당 노드에서 eager로 에러 소멸 확인(2026-08-10)** → cuDNN만이
  아니라 그 노드의 fused SDPA 커널 계열(flash/efficient 포함) 전체가 불안정했던 것.
- 교훈: 같은 증상이 수정 후 재발하면 "같은 원인의 잔재"보다 **한 층 아래의 더
  넓은 원인**(개별 커널 → 커널 계열 → 드라이버/하드웨어)을 의심하고, 층별로
  분리 판정하는 진단부터 만들 것. matmul까지 실패하는 날은 코드가 아니라 노드를
  바꿔야 한다.

### C7. ULF 4차 — 27B(Qwen3.8) linear-attention 경로 + shard 전량 재시작 (2026-08-21)
- **증상**: v4-27b-s4 fresh rollout이 46/128 지점에서
  `CUDA error: unspecified launch failure`. 트레이스는 `modeling_qwen3_5.py`의
  `torch_recurrent_gated_delta_rule`(linear attention torch 폴백)을 가리키나
  비동기 보고라 발생 지점 불확정. 모델 재로드 후 재시도에서도 다른 지점에서 재발.
- **C6과 다른 점**: `OM_ATTN=eager`는 full-attention 레이어의
  `attn_implementation`만 바꾼다. Qwen3.8의 Gated DeltaNet 재귀는 그 스위치
  관할 밖이고 이미 torch 폴백에서 죽고 있어 "fused 커널 우회" 카드가 없다.
  노드 병 판정은 C6의 층별 진단(`gpu_check.sh`) 그대로 — matmul까지 죽으면 노드 교체.
- **우리 코드의 실제 버그**: `collect_rollouts`가 shard 전체를 `.tmp`에 쓰고
  마지막에 rename했기 때문에 크래시 시 수 시간 진행분이 통째로 버려지고
  재시도가 프롬프트 0부터 다시 돌았다. 간헐 ULF 노드에서는 2.5시간짜리 shard가
  영원히 완주하지 못하는 구조.
- **수정**(`bbf884e`): 프롬프트 단위 `.partial` 내구 저장(매 프롬프트 flush) +
  재시작 시 `salvage_partial()`이 K개 완주 프롬프트만 남기고(찢긴 꼬리 줄·미완
  프롬프트 제거) 그 지점부터 재개. 구버전 `.tmp` 진행분 승계. 발행은 행수
  = n_prompts×K 검증 통과 시에만(fail-closed). `.partial` 확장자는 merge·stale
  청소기의 `.jsonl` 글롭에 안 걸리게 선택.
- 교훈: "원자적 발행"과 "내구 진행"은 상충하지 않는다 — 발행 원자성은 rename
  1회로 유지하고 내구성은 부분 파일+구제 규칙으로 얻는다. 크래시가 일상인
  환경에서 all-or-nothing 쓰기는 그 자체가 가용성 버그다.

## E. 데이터셋 확장(math500/mbpp/kk)에서의 시행착오 — 2026-08-10 하루치

### E1. math500 허브 직행 → 오프라인 노드 즉사
- **증상**: `DATASET=math500` 첫 실행이 dataset 로드에서 사망.
- **원인**: provision이 GSM8K만 로컬로 받아뒀고 math500은 `load_dataset()` 허브
  직행 코드였다. **실행 명령을 주기 전에 데이터 확보를 확인 안 한 프로세스 실수.**
- **수정**: provision·fetch_datasets에 로컬 jsonl 수급 추가 + 로더 로컬 폴백.

### E2. DATASET 변수 누락 → gsm8k로 오실행
- **증상**: math500을 돌린다고 생각했는데 이미 끝난 gsm8k가 재실행되며 노드 점유.
- **수정(절차)**: 실행 직후 `head -3 log`에서 OUT_ROOT 접미사(`-math500`) 확인을
  표준 절차화. 환경변수 기반 분기는 "실행 후 즉시 확인"이 유일한 안전장치.

### E3. 로컬 미발견 시 조용한 허브 폴백 → "이상한 곳에서 찾는" 에러
- **증상**: 사용자가 데이터를 받아뒀는데 오프라인 노드가 HF URL 에러를 뱉음.
- **원인**: 로컬 탐색 실패 시 아무 말 없이 허브로 넘어가는 폴백 — 에러 지점이
  원인(경로 불일치)과 무관한 곳(네트워크)에 찍힘.
- **수정**: 허브까지 실패하면 **찾아본 위치 전체를 나열**하는 에러로 교체.
  침묵 폴백은 원격 디버깅의 적.

### E4. 가드와 로더의 경로 판단 분리 → 데이터가 있는데 abort
- **증상**: `/group-volume/datasets/math500`이 실존하는데 run_14b 사전 검사가 사망.
- **원인**: 로더는 베이스 3곳을 탐색하는데 가드는 `$DATASETS_DIR` 한 곳만 검사
  — setup_env가 사용자 폴더를 우선하도록 바뀌자 즉시 어긋남. **같은 판단을 두
  곳에서 따로 구현하면 반드시 어긋난다.**
- **수정**: 가드 삭제, **로더 자신을 preflight로 실행** (`load_prompts(ds,1,1)`).
  검사 통과 = 로드 성공이 구조적으로 보장.

### E5. 원본 MATH 형식(문제당 개별 .json 트리) 미지원
- **수정**: `_load_rows_any()`가 jsonl → parquet → json 트리(HF 메타 제외) →
  `load_from_disk` 순으로 전부 수용. 정답도 `answer` 필드 또는 solution의
  `\boxed{}`(중첩 중괄호 카운팅) 양쪽 처리.

## D. 관측성 — 버그는 아니지만 버그처럼 보이게 하던 것들

- 첫 진행 로그가 5건 처리 후에야 출력 → 7B에서 수 분 침묵 = 멈춤 오인.
  → 시작 배너 + 매건 로그(소요초·정답수·진행%·ETA).
- status의 ETA 정규식이 신형 로그 형식 미매치 → 조용한 기능 상실. (재점검에서 발견)
- pgrep -f 유령 매치(진단 명령 자신의 cmdline) → 패턴을 실행 형태로 한정.
- 최종형: `check.sh` 3줄 진단(🟢/🟡/🔴/✅ 결론까지), 하트비트(10분), babysit 생존 로그.

## 총평

죽인 원인 1위는 인프라(유휴 킬·다중 감시자·경로 이력)였고, 코드 버그 1위는
투영 OOM이었다. 공통 교훈: **(1) 장시간 파이프라인은 원자적 산출물 + 완결성 검증 +
싱글턴 감시자가 기본기, (2) 침묵은 버그와 같다 — 모든 장기 루프에 진행·ETA를 박을 것,
(3) 원격에서 디버깅할 때는 "결론까지 내주는 진단 명령" 하나가 왕복 열 번보다 낫다.**

E절(데이터셋 확장 하루치)의 추가 교훈: **(4) 같은 판단(경로·설정)을 두 곳에서
구현하지 말 것 — 가드가 필요하면 실제 코드를 그대로 실행하라, (5) 폴백은 반드시
"무엇을 시도했는지"를 남겨라 — 조용한 폴백은 에러를 원인에서 먼 곳에 찍는다,
(6) 실행 명령을 건네기 전에 그 명령의 전제(데이터·pull 상태)를 먼저 검사하라.**

### A6. "자꾸 멈춰" — 재시작 무한루프 (무출력 스테이지 × 스킵 부재, 2026-08-14)
- **증상**: 콘솔·로그 완전 정지가 반복. 재시작해도 매번 같은 자리에서 "멈춤".
- **원인(복합)**: ① score(β 2-pass)·val-grads·oracle-grads 샤드에 재시작 스킵이
  없어 저장분을 두고도 처음부터 재계산, ② 하필 그 구간들이 수십 분 무출력
  (진행 print 부재 + 워처는 변화 시만 출력)이라 정상 진행이 hang과 구별 불가 →
  사용자가 재시작 → 다시 같은 무출력 구간 → 무한루프.
- **수정**: 세 스테이지 exists-스킵(모델 로드 전 종료), β-pass 25개·val 20개
  단위 진행 print, nvidia-smi 전 호출 timeout 20(드라이버 wedge 즉시 판정).
  **v4 운영 정책(8/24 변경)**은 로그가 임계 시간 동안 조용하더라도 GPU utilization 또는
  해당 process group의 CPU 누적 시간이 증가하면 정상 계산으로 보고 유지한다. 로그·GPU·
  CPU가 함께 정지한 경우에만 process group을 종료하고 저장분부터 자동 재시작한다.
  임계값은 7B 5분, 27B 20분이며 스모크와 본 run 모두 최대 5회 재시도한다.
- **v4 전체 run 계보(8/21 최종 수정)**: `git pull` 뒤 이전 run을 같은 경로에서
  재사용하면 immutable `run_config.json`의 `git`이 달라
  `existing artifacts use a different run config: ['git']`으로 중단된다. smoke는 현재
  12자리 commit hash를 경로에 포함하고, canonical 본실행 경로의 이전 commit 산출물은
  시작 전에 `$OM_WORK/quarantine/v4/`로 자동 보존 이동한다. 시작 시 stale v4 worker를
  종료하고 GPU 메모리 해제까지 확인한다.
- **8/24 false-success 원인**: `OM_SKIP_POSTPROCESS=1` 경로가 실패 run을 남기고도 exit 0을
  반환했고 상위 launcher가 child exit만 믿어 전체 완료로 오인했다. 현재는 필수 artifact
  6종을 직접 검사하고, 한 run이 실패해도 나머지를 실행한 뒤 미완료 run만 3 pass 재시도한다.
- **기존 run에 최신 복구 로직 적용**: 계산 파일은 `run_config.json`에 기록된 commit의
  격리 worktree에서 실행하지만 감시·재시도 supervisor는 현재 checkout 버전을 사용한다.
  이 분리가 없으면 `git pull` 뒤에도 옛 watchdog 버그가 그대로 재현된다.
- **27B `rc=1` 뒤 다른 shard `rc=143`**: 첫 shard의 CUDA ULF로 `wait`가 실패하자
  `run_14b.sh`가 즉시 exit하고 EXIT cleanup이 정상 계산 중인 형제 shard에 SIGTERM을
  보내던 증상이다. `143=128+15`이므로 형제 shard의 독립 CUDA 실패가 아니다. 현재는
  모든 shard가 끝날 때까지 기다려 정상 결과를 보존하고, 실패 shard만 `.partial`에서
  재시도한다. 재시도마다 shard의 물리 GPU 배정을 회전해 특정 GPU 반복 실패도 피한다.
- **27B clean rerun**: 7B가 이미 완료된 경우 `go_v4_27b.sh WORKER TOTAL`을 사용한다.
  10개 27B run을 최대 10개 4-H100 클러스터로 분배하며, 과거 불완전 run은 삭제하지
  않고 quarantine한 뒤 최신 commit으로 실행한다.
- **27B linear-attention backend**: 과거 ULF trace는 Qwen3.8 Gated DeltaNet의
  `torch_recurrent_gated_delta_rule` fallback을 가리켰다. 전용 rerun은 FLA 0.5.2의
  fused recurrent/chunk kernel을 자동 설치하고 import를 확인한다. fallback이면 시작하지
  않으며 backend와 FLA version을 run config에 기록해 run 간 혼입을 차단한다.
- **과거 완료 표식 재사용**: 27B clean rerun 전의 `V4_COMPLETE`와 결과 표가 남아 있어도
  현재 generation commit이 표식과 다르면 다시 수집한다. 수집 전 27B 10개 run의 commit,
  모델 hash, FLA backend, 고정 설정 및 run-config digest를 전수 검증한다.
- **잔여 경계**: util 0%가 지속되면 진짜 hang — 이 노드군의 fused 커널병(C5·C6)
  이 에러 대신 동결로 나타나는 케이스 또는 group-volume 스톨(D-state).
  그때는 RECOVERY 상황 1(노드 교체)이 정답. 진단: ps -eo pid,stat,wchan | awk
  '$2~/D/', dmesg의 nfs not responding / Xid.
