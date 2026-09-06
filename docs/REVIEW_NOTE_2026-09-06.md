# 검수 메모 (2026-09-06) — Claude 작성, 검증되지 않은 부분 명시

## 경위 (사실)

- 2026-09-05: "offpolicy misrank 코드 버그 검수" 요청에 대해 **잘못된 체크아웃**
  `~/dev/offpolicy-misranking-audit`(브랜치 `audit/p1-integrity`, master보다 116커밋 뒤,
  SFT-drift 시절 코드)을 리뷰하고 패치·커밋(`5dfda89`, 해당 브랜치에만)했다.
  논문 실험은 master의 RLVR(GRPO)이므로 **그 리뷰의 결론은 이 실험에 적용되지 않는다.**
  특히 "drift 학습이 채점 표본을 공유해 IS 항등식이 깨진다"는 지적은 SFT 경로 이야기이며,
  GRPO는 자기 fresh 표본으로 학습하고 β는 d0에서 한 번 뽑아 재사용하므로 해당 없음.
- 2026-09-06: master `c78e24c` 기준으로 RLVR 경로를 다시 읽었다.

## 읽은 범위 (사실)

`scripts/run_olmo3_rlzero.sh`, `scripts/run_matrix.sh`, `scripts/run_point.sh`,
`scripts/recover_rollout_stage.sh`, `scripts/gpu_keepalive.py`, `src/train_policy_grpo.py`,
`src/experiment.py`(score/oracle/report/main), `src/grads.py`, `src/rollout.py`,
`src/rollout_contract.py`, `src/reuse_behavior.py`, `src/recovery_policy.py`,
`src/cleanup_run_processes.py`, `src/compact_artifacts.py`, `src/data.py`(검증기 부분),
`configs/olmo3_rlzero*.json`, `docs/OLMO3_RLZERO_RUNBOOK.md`, `docs/EXPERIMENT.md`.
읽지 않은 것: `regime_map.py`, `regime_contract.py` 본문, `rlzero_status.py`,
`qualify_rlzero_signal.py`, `qualify_domain_data.py`, `code_sandbox.py`, 테스트 전부.
**실행 로그·산출물은 하나도 보지 않았다**(클러스터 접근 불가).

## 판단 (추정 — 검증 안 됨)

1. 읽은 범위에서 로직 버그를 찾지 못했다. "없다"가 아니라 "못 찾았다"이며, 위 미열람
   파일과 실제 산출물에는 적용되지 않는다.
2. "3일째 첫 family 미완"은 다음 어림 계산과 일치한다:
   family당 생성 ≈ fresh 4점×512×32 + GRPO 400step×32 + β 512×8 ≈ 16만 시퀀스 × ≤2048토큰,
   HF `generate` batch 8·eager attention에서 GPU당 250~300 tok/s 가정 → point당 ~12h,
   GRPO ~10h, family ~55~60h. 10 family/3노드 → 9~10일.
   **가정한 수치(tok/s, cap 비율, verify 시간)는 로그로 확인되지 않았다.** 확인 명령:
   ```bash
   R=$OM_WORK/runs/olmo3-1025-7b-base-rlzero-grpo-h100-v2
   grep -h "tok/s=" $R/family-*/*/logs/fresh-shard*.log | tail -4
   grep -c "family-retry\|cuda-recovery\|regime-hard-stall" $R/logs/*.log
   ```
   재시도/복구 카운트가 0이 아니면 "순수 연산 시간" 판단은 틀린 것이다.
3. 속도 레버(미구현·미검증): `OM_GEN_BATCH` 8→32(OLMo-3 7B KV 토큰당 524KB, 2.3k토큰×32
   ≈ 38GB + 가중치 15GB로 80GB 내 — 실측 아님), val rollout을 shard 0에만 두는 불균형 해소,
   keepalive duty 15%→5%, mbpp 검증기 순차 8초 타임아웃. 전부 계약 해시 밖의 실행 파라미터이나
   **진행 중 point에는 적용 불가**(run_config에 gen_batch 기록됨).

## 사용자 지시

"기록해놔. 니가 한게 사실이 아닐 수도 있다는 것도 기록해." — 이 문서의 2·3절은 그 취지로
추정임을 명시한다. 이 파일은 커밋하지 않았다(docs/는 dirty 검사 범위 밖).

## 2026-09-06 후반 — 실제 실행 경로 점검 결과 (master)

원인이 확인되어 고친 것(전부 push, 최신 커밋은 `git log`):
1. 추가실험 런처가 math-verify 번들을 PYTHONPATH에 안 올림 → `data.py:802` (`004f5f8`)
2. Qwen shard 파일명(`model.safetensors-0000N-of-0000M`)이 다운로드 패턴에 안 맞아 가중치 0개로 prepare 완료 처리 (`dbee3d1`)
3. 런처가 찾은 모델 폴더가 아니라 `$MODELS_DIR/<pinned>` 빈 껍데기를 검사 (`c3c39e0`) — 반복된 "가중치 없음"의 실제 원인
4. transformers 5: `apply_chat_template(tokenize=True)`가 dict 반환 → 토크나이저 검사 오판 (`83467b0`)
5. 추가실험이 OLMo primary 노드에서 며칠 조용히 대기 → 명시적 거부 (`bb679e2`)
6. `regime_contract.build_matrix`가 pinned 폴더명·manifest를 강제 → 임의 폴더/무 manifest 허용, provenance 기록 (`75bafbb`)
7. 클러스터 터미널이 한글을 떨어뜨림 → 터미널 메시지 전부 영어 (`88bb0c6`)

추가한 운영 장치: 실패 시 `DIAGNOSIS/ACTION/EVIDENCE` + 폴더/venv/데이터 상태 자동 출력, `[progress] stage=k/8` 라인, CUDA OOM 시 배치 반감 재시도(생성·logprob·gradient), 인자 없는 한 줄 실행(`run_qwen35_9b.sh`), 후속실험 프로파일(`run_followup.sh`).

**검증되지 않은 것**: 위 수정 후 9B `run`이 학습 단계까지 진입했는지는 2026-09-06 03:10 KST 기준 미확인(마지막 확인 지점: 토크나이저 검사 통과 전). FLA 커널·스모크·GRPO 첫 step은 클러스터에서만 확인 가능.

## 2026-09-06 저녁 — 전체 코드 재점검 (4영역 병렬 리뷰, master 21750eb 기준)

범위: 셸 런치 경로 / 모델·스냅샷·데이터 / GRPO 학습·롤아웃 코어 / 점수·regime·리포트.
방법: 코드 읽기 + CPU 재현(작은 랜덤 Qwen3.5 모델, 실제 토크나이저·config, 합성 산출물).
확정·수정 목록은 `BACKLOG.md`의 `OM-2026-09-06-*`. 여기에는 판단만 적는다.

- 가장 큰 것: **완료 판정 불가**(`OM-…-01`). 09-03 `r_high_budget` 추가와 08-31의 정확 일치
  검사가 서로를 몰랐다. OLMo 워커의 generation 코드가 09-03 이후라면 본실험 "첫 family
  미완"의 원인일 수 있다. 클러스터에서 `lacks exact R/A/B` 문구를 로그에서 찾으면 확정된다.
- **EOS**(`OM-…-04`): 9B는 `<|im_end|>`에서 멈추지 않아 모든 응답이 2048 토큰까지 갈 수
  있었다. "느린 게 정상"이라던 앞선 추정은 이것 때문일 가능성이 크다. 보상은 맞았다
  (`resp_end_index`가 잘라냄), 시간만 낭비.
- **MBPP 게이트·수학 콤마**(`OM-…-02/03`): 정답을 0점 처리하는 경로가 있었다. GRPO 보상
  신호 자체를 왜곡하므로 이 수정 전 산출물은 재사용하지 않는다(9B는 산출물 없음).
- weight_decay: 토치 기본 0.01이 계약에 없이 적용되던 것을 0으로 명시했다. 보상 분산 0인
  스텝에서도 옵티마이저는 계속 밟는다(모멘텀만 적용). 표준 GRPO 구현과 같고, "스텝 수가
  정책 이동을 색인한다"는 계약 문구를 유지하기 위해 스킵으로 바꾸지 않았다.
- 맞는 것으로 확인된 것: 손실 식·토큰 정규화·logprob 정렬·5e-3 온폴리시 불변식(bf16에서
  비트 동일)·OOM 재시도 트랜잭션성·체크포인트 왕복·FIRST 부트스트랩 분할·가우시안 천장
  fail-closed·`PINNED_OFFICIAL_FILES` 해시·LoRA 타깃이 텍스트 디코더에만 존재.
- 미해결(결정 필요): MBPP 5개 불가능 프롬프트(데이터 해시 변경), `make_tables` T3/T6.
- 검증되지 않은 것: 전부 CPU·로컬. FLA 커널, 4랭크 학습, 실제 9B 속도는 클러스터에서만.
