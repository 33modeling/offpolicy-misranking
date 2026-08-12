# 백로그 — 컨셉 #68 (2026-08-13 기준, 마감 역산 포함)

담당: 🖥 = 클러스터(사용자 실행), 🤖 = 로컬(Claude 작업). 순서는 위→아래.

## 상태 (2026-08-12 1차 판독): 잠정 분기 3

완주 5 판독 — gsm8k C1 미재현(포화 이동 의심, base 0.80) / dapo C1 재현 /
hybrid 회복 실패(dapo) / C2 인증불가 안정 재현. 상세는 concept.md "v2 1차 판독".
신규 최우선: **A0. equal-K 코드 대조(21/21 아티팩트 판별)** → floor 판독 →
분기 3 원고 전략. 트랙 B 후속 블록은 분기 확정 후 재판단.

## 트랙 0 — 지금 돌고 있는 것

- [ ] 🖥 v2 재실행 수확 (go_retry.sh / go_full.sh 반복 — DONE 누적, s1 완주 확인됨)
- [ ] 🖥 완주분 나올 때마다 `bash scripts/backup_results.sh`
- [ ] 🖥→🤖 **결과 파일 전달**: 종료 요약 + results/v2의 TABLES.md·FRONTIER.md·judge-*.txt
      (부분 완주 상태라도 s1 judge 먼저 보내면 조기 판독 가능)

## 트랙 A — 원고 (결과 수령 시 즉시, 🤖)

- [ ] A1. judge 검증 → **사전 판정 규칙 3분기** 확정 (concept.md 등록분 — 결과 보기 전 고정됨)
- [ ] A2. 본문 수치 일괄 교체: 메인 표(seed mean±sd)·인라인 수치·"21 of 21"류 집계·부록 표
- [ ] A3. **D5 문구 반영**: 규칙(1) 재서술 + 편향 귀속 문장 + below-chance 서술 조정
      (단독 과판매 금지 — 부호검정 p=1.9e-6·다조건 결합이 주 방어, D6 발견)
- [ ] A4. **D6 수치 삽입**: stats_extra.py로 초기하 p·부트스트랩 CI 산출 → 표 각주/부록
- [ ] A5. frontier 절 병합: sec_frontier.tex의 \todo를 FRONTIER.md(F1~F4)로 채워
      main.tex sec:limits/sec:discuss 사이에 병합, main_frontier.* 삭제
- [ ] A6. 재컴파일·검수 → 발표자료 메인 덱 RESULTS/OUTLOOK 수치 갱신
- [ ] A7. (선택) 축 앵커 인용 추가: per-decision IS(Precup)·DICE 계열 → refs.bib
- [ ] A8. (선택) intro에 CTPO/MinPRO "정리 전제 위반" 승격 문장

## 트랙 B — 실험 블록 (앞 블록 완주 후 순서대로, 🖥)

- [ ] B1. go_full 완주 (A: gsm8k+dapo 5-seed / B: math500 5-seed / C: downstream 4소스)
- [ ] B2. go_boost (D: drift 50/200/400×3seed — β 재사용 / E: 14B math500×3seed / F: mbpp·kk×3seed — 코드·논리 도메인)
- [ ] B3. go_35 (Qwen3.5 0.8/2/4B — 능력 스윕 + 신세대 게이트)
- [ ] B4. go_27b (스모크 관문 → DAPO hard-slice 프리스크린 → 3-seed; 코드는 DATASETS_27B에 apps)
- [ ] B5. real_drift_check.sh (D1 — B1의 downstream_random 어댑터 필요)
- 시간 부족 시 삭제 순서: B4 → B3 → B2의 E → ... (B1이 최후 보루, RECOVERY.md 티어표)

## 트랙 C — 조건부 (트리거 발생 시에만)

- [ ] C1. 실패 반복 → go_retry.sh, 그래도 실패 → DIAGNOSIS.txt 전달 + 노드 교체 (RECOVERY 상황 1)
- [ ] C2. 클러스터 전면 불능 → **D7**: Qwen3.5-0.8B/2B를 아무 안정 GPU에서 (go_35 그대로, n 축소 가능)
- [ ] C3. v2 결과가 v1과 모순 → 사전 판정 분기 3 (병기 + 교정 민감성 보고, hybrid 축 유지)
- [ ] C4. 2D-REFRESH(순차 audit)가 replay에서 audit_* 격파 → 방법 기여 승격 + GPU 본실험 검토

## 마감 게이트 (ICLR 2027 공식)

| 날짜 | 게이트 |
|---|---|
| **9/10** | 경로 결정점 — v2+보강 불안하면 축소 티어/v1 폴백 발동 |
| **9/18 AOE** | abstract 마감 |
| **9/20** | 내부 수치 동결 (표·frontier 반영 여유 5일) |
| **9/25 AOE** | full paper 마감 |

## 완료 (참고)

- ✅ 원고: 제목 확정·용어 정규화·서지 16편 전수 검증·9p/11p 클린 빌드
- ✅ 인프라: go_* 스택 전부, preflight·diagnose·retry·backup, 버그 픽스 3건
- ✅ 방어: 리뷰어 공격 포인트·D1~D7 설계(D6 구현), 사전 판정 규칙, 이론 전수 검증 통과
- ✅ 발표자료: 메인 14섹션 상세판 + 레퍼런스 18덱 + 허브 (렌더 검증)
