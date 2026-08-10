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
