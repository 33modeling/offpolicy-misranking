# 백로그 — 컨셉 #68 (2026-08-13 오후 기준, 마감 역산 포함)

담당: 🖥 = 클러스터(사용자 실행), 🤖 = 로컬(Claude 작업). 순서는 위→아래.

## 상태 (2026-08-13 오후): 재편 분기 발동 — 원고 v0.1 push 완료

P3-0 precheck **NO-GO** + P4-0 kcurve **구조적 부재** → 사전 등록 재편 분기
(concept.md "P4 설계서" 절) 발동. go_hard 폐기. 원고 v0.1(`paper/main.tex`,
`505c83f`)은 3단 주장 계층(possibility/realization map/impossibility)으로
전면 재작성 — 신규 보강 3종 반영: **측정 상한 명제**(floor→ceiling, gsm8k
g00 0.294가 ceiling≈0.30 도달), Fisher 결합 p=1.0e-6(gsm8k 추적),
**함정 4호**(포화-공유 인공 겹침, 합성 실증 — kcurve 하강 곡선의 정체).
v1 drift400 below-chance는 단독 p=0.27 확인, 방향성 관찰로 강등됨.

그록 리뷰 수신(8/12 작성, new-paper-ideas `658ecec`,
`68-.../grok/`) — 구조 지적(시제 모순·처방↔한계 포지션·CertaGrad)은 v0.3
재편이 선반영했음을 대조 확인. 잔여 반영분은 A6·A7(원고)·B4·B5(실험),
안전 단락(B12)은 즉시 반영 완료.

## 트랙 0 — 지금 돌고 있는 것

- [ ] 🖥 go_retry 수확 (s3 gsm8k/dapo — 사용자 실행 중)
- [ ] 🖥 s3 완주 시 `git pull && bash scripts/harvest.sh` **한 번** →
      마지막 줄에 찍히는 폴더 **하나만** 전달 (KCURVE·READOUT·REVERSAL 동봉;
      kcurve 표결 2→3개: 뒤집히면 트랙 C1, 유지되면 원고 표만 갱신)

## 트랙 A — 원고 (🤖)

- [ ] A1. 원고 v0.1 사용자 피드백 반영 (구조·주장 수위·분량)
- [ ] A2. 잔여 seed 수확 시 수치 일괄 갱신: 표 2(v2)·표 3(kcurve)·본문 인라인 —
      사전 등록 **규칙은 고정하되 판정 자체는 재계산**(그록 08-18: "수치만
      교체" 표현은 '결과 보기 전 결론 잠금'으로 읽힘 — 재계산 명시가 정답,
      원고 pending 블록은 이미 recomputed로 서술됨)
- [ ] A3. 서지 확정: prefix/future 계열 arXiv 매핑 재검증(현재 검증된 링크만
      수록), VIP·M2PO 등 추가 인용 여부 결정
- [ ] A4. 발표자료 19덱을 재편 구조로 갱신 (원고 안정화 후 일괄 — 먼저 하지 말 것)
- [ ] A5. (선택) 이론 보강: B8 오차분해 remark, ceiling 명제 해석적 증명 시도,
      group normalization K→∞ 점근 한 줄 (그록 Theory 지적)
- [ ] A6. 제출 직전 문장 단위 스쿱 재검색 재실행 + 검색 로그 appendix 수록
      (그록 D7·B10 — 9/20 수치 동결 직전 실행)
- [ ] A7. 재현 부록: CROPI step-200 checkpoint 버전·라이선스 pin, seed·config
      hash 표 (그록 B14)
- [ ] A8. 자체 리뷰어 통독(8/13) 발견 보강 — 우선순위순:
      (a) **§5.1 신호 체제 주장의 통계 보강**: harvest.sh에 STATS.md 동봉
      완료(8/14) — v1 계열 포함 run별 stats_extra 정확 p·CI 자동 산출,
      수확 시 본문 병기만 남음. reversal_freq도 v1 계열(drift*/gate-*)로
      확장 — MATH-500 강신호 체제의 반전 유병률까지 자동 수확
      (b) ✅ Fisher 결합 독립성 방어 한 줄 (§5.2, 8/14)
      (c) ✅ tab:reversal g11 대비 초과 반전에 짝지은 McNemar p 병기
      (8/18 수확분 본문 병기 — s2 g00 p=0.014 유일 유의, 나머지 p≥0.13)
      (d) ✅ Fig.1 disagreement 상자 실측 지지 표기 (8/14)
- [ ] A9. 제출 포맷 압축: 현재 article 13p → ICLR 본문 9~10p 예산
      (pitfalls §7 일부 appendix 강등이 유력한 절약처, 9/20 이후)
- [x] A10. ✅ v0.4 심사 감사(new-paper-ideas `REVIEWER-AUDIT-v04-2026-08-14.md`)
      언어·정합 수정 반영(8/14): Fig.1 명제 번호 교정, two→three stages,
      GPU-free→artifact-only 전면, "the one"→"a" warning sign, structural
      absence 조건화(epistemic/ontic 구분 명문), certification 표현 하향+
      approximate/indifference-zone 범위 명시, ceiling 모델-조건부 명시,
      spend nothing→withhold 완화, 0.04 방향성 병기, Prop1 증명/탐색 분리,
      판정 재계산 문구. **잔여(문구류)**: 관련 연구 확장(§7 요구), floor
      명칭 각주 검토 — A2 수치 반영 때 일괄
- [ ] A11. 감사 P0-5: prompt-pool clustered bootstrap·cell-vs-full paired
      검정을 STATS 산출에 추가(🤖 스크립트 확장, 수확 편승) + 표에 CI 병기

## 트랙 B — 보강 실험 (약점 제거 효과순)

- [ ] B1. 🤖 v1 run 대응 kcurve·조건부 floor 확장 스크립트 작성 → 🖥 실행 (GPU 0)
      — 판정 표본을 2 run에서 ~7조건(v1 게이트·math500 포함)으로 확대,
      "1/2 표결" 취약점 제거. **최우선 보강.**
- [ ] B2. 🖥 E1 frontier replay를 교정 floor로 재실행 (CPU) — floor-gated
      프로토콜 승리 시 처방(§9)에 양성 결과 추가, 패배 시 진단 프로토콜 유지
- [ ] B3. 🖥 D7/go_35 소형 스윕 (Qwen3.5 0.8/2/4B) — 능력-난이도 위상도.
      floor를 조절하는 손잡이를 보이면 인과 서사 부활. **유일한 GPU 레버,
      9/10 결정점 전 판단.**
- [x] B4. ✅(8/18 확정) 🖥 `bash scripts/reversal_freq.sh` (CPU, 기존 산출물 재집계 —
      GPU·신규 롤아웃 불필요) — 프롬프트 단위 부호반전율·경계 대역 피해자
      비율·불일치 경보 조건부 반전율 → 원고 primary table 후보.
      existence↔prevalence 갭을 실측으로 닫는 최저비용 보강 (그록 D4·B3).
      **1차 수확 판독 완료(8/13, v2 4개 run)** — 신호 run 경계 반전 24~35%,
      불일치→g10 반전 Fisher p=0.015/0.007 (s1·s2 재현). 단 oracle 자기
      불일치 닻이 없던 판이라 수치 확정 보류 — **별도 재실행·재전달 불요**,
      닻 포함판은 s3 수확 harvest.sh에 자동 동봉. **원고 §5 예비 반영 완료**
      (tab:reversal + 단락: 원시 반전율은 상한 명시, 본문 주장은 g11 공유-노이즈
      대비와 run 내부 조건부 경보 2건만, estimand 구분 명시) — 닻 도착 시
      앵커 행 추가·수치 확정 교체.
      "불일치 시 범인은 주로 g10" 풀링 48/79는 이항 양측
      p=0.071 — 방향성 관찰로만, 과판매 금지.
      **✅ 확정(8/18 수확: 닻+McNemar+v1 도착)** — 닻(oracle split-half 자기
      불일치) 전체 39%/43%·대역 27%/20%. 전체 반전은 전 셀이 닻 ±수 pp로
      잡음 지배(실패 증거 불독해로 명문화). 대역은 one-sided 0.34/0.32 >
      닻 0.24 = g11 0.24. McNemar 유일 유의: s2 g00 b=27/c=11 p=0.014
      (나머지 p≥0.13). 불일치 경보 3-run 재현: s1/s2 p=0.015/0.007 +
      v1 gate-7b-math500 81% vs 45% p=0.012; 포화 14B는 무신호 p≥0.44
      (퇴화 서사와 정합). §5 확정 반영 완료(앵커 행·캡션·본문 교체).
- [ ] B5. (9/10 결정점에서 B3와 묶어 판단) GradAlign k_r-matched·full IS(g11)·
      uncorrected 동일 그림 head-to-head — A+B 환원 방어의 마지막 조각
      (그록 D3·B2). GPU 필요, 단독 착수 금지.
- [ ] B6. 🖥 **[감사 P0]** 대표 1조건에서 K'=64(가능하면 128) oracle 직접 관측 —
      Spearman-Brown 외삽의 calibration 검증(예측구간 적중 여부). 실패 시
      결론을 budget-limited absence로 하향 (REVIEWER-AUDIT-v04 §3)
- [ ] B7. 🖥 **[감사 P0]** 고신호 체제(7B MATH-500 또는 drift-400) equal-budget
      반복 downstream: oracle·4셀·pass-rate·random — 실전 피해 입증 실패 시
      논문 범위를 selection-evaluation audit으로 명시 축소 (§6·§P0-4)
- [ ] B8. 🖥 비수학 태스크 1종 외적 타당성 probe — **mbpp 확보 완료** 활용,
      floor·kcurve만 1 seed (§10). check_data.sh mbpp로 위치 확인 후
- [ ] B9. (B3 승격 메모) capability-difficulty sweep은 감사 P0-2로 승격 —
      9/10 결정점에서 "선택"이 아니라 "필수" 취급
- [ ] B10. 🖥 도메인 다각화 (사용자 지시 8/18) — MBPP(코드, 테스트 실행 채점)
      **1-seed 고정**(사용자 결정 8/18: 비수학 도메인은 seed 1개만),
      go_v2 동일 프로토콜, **kk(논리, Logic-RL 근거)와 동시 착수**(8/18 확정):
      `SEEDS="0" DATASETS="mbpp kk" bash scripts/go_v2.sh`
      (착수 전 `check_data.sh mbpp`·`check_data.sh kk`, **별도 노드 — 수학
      5-seed 노드 불가침**, 수량 부족 시 N_TRAIN=400 N_VAL=100). 판정 기준
      사전 등록: v2와 동일 게이트 + 닻 대비 반전율·불일치 경보 재현 여부.
      **단일 seed이므로 원고 표기는 "exploratory, single-seed" 방향성 관찰로
      한정**(통계 주장 금지, 과판매 금지 원칙) — 재현=일반화 시사,
      비재현=도메인 조건성 시사. 완성 시 도메인 3축: 수학(5-seed 통계)·
      코드(MBPP)·논리(kk).
      ※ 아래 "동일 풀 신규 착수 금지"와 구분 — 신규 도메인 풀이므로
      결론 불변 논리에 저촉 없음.
- [ ] **B12. 메인 세대 교체 (사용자 결정 8/19)** — Qwen3.8-27B **5-seed 풀
      매트릭스 본실행**으로 승격 (B11의 1-seed 탐색을 대체). 노드당 seed 1개:
      `SEEDS_NEW="k" bash scripts/go_new.sh` (k=0..4), B300 노드는
      `OM_SKIP_HYBRID=0`으로 hybrid 포함. GSM8K 자연 풀(포화 표본)·DAPO
      hard-slice(유신호)·MATH500 자연 풀. 소요 ~3일(8/22 전후). 수확 시 논문
      재구성: 메인=3.8-27B, Qwen2.5 7B/14B 5-seed는 이전 세대 재현 축으로
      강등 배치(초록·셋업 문구 교체는 수확 후 — 수치 선기록 금지).
- [ ] B11. 🖥 최신 세대 모델 검증 (사용자 지시 8/18: "최신 모델에서도 제대로
      동작함"을 설득) — **Qwen3.8-27B**(2026-08-14 출시, 27.8B dense·hybrid
      Gated DeltaNet·멀티모달·Apache 2.0) 1-seed, 메인 v2 동일 프로토콜:
      `bash scripts/go_new.sh` (스냅샷 자동 fetch → gsm8k+dapo → math500,
      run 폴더 runs/qwen3.8-27b-* 격리). 30분 스모크 실패 시 폴백:
      `REPO27B=Qwen/Qwen3.6-27B bash scripts/go_new.sh` (자산 기확보, 동일
      아키 계열, AIME26 94.1·GSM8K 97.7). 판정 사전 등록: v2 동일 게이트+
      닻+경보, **exploratory 1-seed 표기**. 예상 그림: GSM8K 초포화(97.7%+)
      → 퇴화 체제로 이동, DAPO live↑ 가능 — "체제 지도가 능력축 따라
      이동한다"는 본문 서사의 최신 모델 실증으로 서술. ※ go_27b 폐기와 구분:
      폐기 사유는 '동일 풀 추가 수치로 결론 변경'이었고 이번 목적은 세대
      방어(exploratory) — 저촉 없음.
### 그록 08-18 재감사 반영 (수신 f8c2da2, 판독 8/19)

전제: 그록 문서는 **8/18 수확(닻·McNemar) 반영 전** 기준 — A2(닻 대기)는 기완료,
원고-실험-대조의 철회 요구(21/21·anti-select·역상관)는 v0.5가 기반영. 아래는
그 대조 후 **살아남은 신규 항목**만.

- [x] A12. ✅(8/19 반영) 선행 2건 인용·포지션 수정 — **등급 재조정 기록
      (8/19)**: 2608.01704는 동시 연구(같은 8월)라 인용 의무 없음(주요 학회
      contemporaneous work 관례), 도메인도 무관(crowd reading) — 치명 아닌
      **비용 0 보험**으로 유지. ACE도 방법 철회로 과녁 없음, 들어간 문장은
      서사 보강용. 사용자 재량으로 두 인용 삭제 가능(두 문장). 실질 급소는
      B16 하나로 정리됨. (선행-전수검색) — ① ACE(arXiv
      2601.20989, top-k on a budget): §2·§6에 인용 + "약 oracle이 잡음이
      아니라 **계통 편향**(부호반전)이면 ACE 하한이 부적용 — 인증 실패는
      예산이 아니라 편향 탓" 한 문장. ② Floor/Ceiling/Fusion(arXiv
      2608.01704, 8월 신간): split-half ceiling 용어 선행 — 인용 + ceiling
      기여를 "발명"이 아니라 **RLVR 선택 평가로의 이식+자기 적용**으로 명시
      (QNA Q1의 '문헌 전체 적용' 표현도 하향).
- [ ] A13. 🤖 범위 한정 표(리뷰어-일반화 §7) 최종 반영 — 수확(27B·mbpp·kk)
      후 실범위 확정과 함께: stale='controlled LoRA-RFT twins' 명시,
      'within evaluated pools and this CI procedure' 견지, 도메인·모델
      명사는 실측 있는 것만. (v0.5가 상당 반영, 최종 일괄 점검용)
- [ ] A14. 🤖 new-paper-ideas **문서 정합** (재검증 §5·§8 — "한 레포에 세
      논문") — concept.md 상태 헤더·Abstract 시제, QNA Q16의 '15위' 인용,
      구판 paper/(v0.3) 폴더에 '철회 반영 전 구판' 표지 또는 v0.5 동기화,
      P4 readout·PRECHECK 첨부, **QNA Q1의 'ceiling 문헌 전체 적용' 하향**
      (v0.5 모형-조건부 수용을 QNA가 되돌리고 있음 — 장단점 B15 신규 지적). 순위(2위 vs 그록 권고 4위)는 사용자 판단
      사항으로 기록만.
- [ ] B13. 🤖 pass-rate 베이스라인 확장 (그록 C1·R17·R18 — "C1이 A4보다
      싸다") — 저장 점수만으로 **|p−0.5| (MoPPS형) top-k 재순위**를
      frontier에 추가, 원고에 pass-rate/random/4셀 한 표 승격. alignment가
      pass-rate를 못 이기면 downstream GPU(B7) 집행 금지, 범위 축소가 정답.
- [ ] B14. 🤖 ceiling 경험 분포 재보정 (그록 C4) — Gaussian 가정 공격 +
      2608.01704 대비: 저장 점수 resample로 '모형 천장 vs 경험 천장' 병기.
- [ ] B15. 🤖 verify_theory 정합 (그록 B4) — 샘플 5만→20만, 본문/스크립트
      수치 일치. CPU 분 단위, 즉시 가능.
- [ ] B16. 🖥 drift 풀 ∩ 평가 풀 분리 ablation (그록 R14) — **유일한 실질
      급소(8/19 판정)**. 급소인 이유 = 측정 순환: ① β가 평가 풀 512개에
      rollout 생성 → ② 그 정답 풀이로 LoRA 학습해 π 생성 → ③ 같은 512개를
      π 기준으로 순위 측정. "순위 매길 문제의 정답을 보고 드리프트를
      만들었다"는 타당성 공격이라 문장만으론 안 닫힘. 방어 2카드:
      ⓐ disjoint split 1-run(드리프트 학습 프롬프트 ≠ 평가 프롬프트)으로
      "겹쳐도 결론 불변" 실측 — 종결 조건, ⓑ 반박 문장(실제 RLVR도 선택
      대상 풀에 학습 → 실무 조건 모사) — 보조. 비용 run 1개, 9/10 전
      집행이 정답. B6·B7과 함께 판단.
- 매핑 메모: 그록 A1=기존 B6(대형 K), A5=A11(clustered bootstrap),
  R7=B10(mbpp — **진행 중**), A2(닻)=B4 **✅기완료**, C3(경계 부호 일치율)은
  B4 REVERSAL 대역 분석이 부분 커버. R2(KL 리라벨)는 A2 수치 반영 때 manifest
  divergence로 일괄.
- ⚠ 긴장 기록: 그록 08-18 §D는 27B 본실행을 기간 리스크로 **반대**. B12는
  사용자 결정(8/19)으로 진행 — 스모크 게이트·5노드 병렬·DONE 스킵으로 리스크
  완화, 9/10 결정점에서 중간 점검(미완이면 1-seed 탐색으로 축소 폴백).

- [ ] A15. 🤖 부록 구성 채택 (그록 ICLR-부록-계획) — 본문 9p+부록 22~33p
      구조: App.A(2×2 항등·원문 기호 대응표) B(반례 전문) C(맹점 스코프)
      D(인증 기하·ACE 대조) E(FIRST 사전등록 전문·문턱 민감도) F(**재현
      명세 — 비면 탈락**) G(전체 표, 교정 후만) H(함정 사례연구) I(통계)
      J(일반화 한계 먼저 적기) K(확장 관련연구) L(계산 회계). +한도 비산입
      필수 3종: **AI use statement(미기재 시 desk reject)**·Reproducibility·
      Ethics. A9(포맷 압축)와 통합 진행.
- [ ] A16. 🤖 cor:blind의 cosine 항 스코프 재점검 (그록 수식-검증 §6) —
      pointwise cosine(신선한 gradient 필요)은 stale-computable 지표가
      아니므로 맹점 목록의 cosine을 "평균/noisy cosine 대시보드"로 한정
      하거나 서술 정비. T2 샘플 수(5만 vs 본문 표기)도 B15에서 함께 정합.
- 폐기 확정: go_hard(NO-GO), go_full 신규 확장·go_boost·go_27b(구조적 부재
  판정으로 동일 풀 추가 수치는 결론 불변 — 수확만 받고 신규 착수 금지)

## 트랙 C — 조건부 (트리거 발생 시에만)

- [ ] C1. s3 포함 kcurve가 "확장 권고"로 뒤집힘 → P4-1(FRESH_K 증량 oracle
      재실행) 검토 — 단 s1 r1=0.045가 함정 4호 사정권임을 감안, 보수적으로
- [ ] C2. 클러스터 불능 → D7을 아무 안정 GPU에서 (n 축소 가능)
- [ ] C3. D7에서 고신호 체제(floor ≥ 2×chance) 발견 → stale 붕괴 검정
      본실험 부활 검토 (사전 등록 후)

## 마감 게이트 (ICLR 2027 공식 — **8/19 정정**, 출처 ICLR2027-AUTHOR-GUIDELINES.md)

⚠ 구표(9/18 초록·9/25 본문)는 오기였음 — 그록 08-18 지적으로 검증·정정.

| 날짜 | 게이트 |
|---|---|
| **9/3** | 경로 결정점 (구 9/10에서 당김) — B3·B6·B7·B16 집행 여부 최종 판단 |
| **9/11 (금) AoE** | **abstract 제출 마감** — 제목·초록·저자 확정, placeholder 불가 |
| **9/11** | 내부 수치 동결 (본문 마감 −5일) |
| **9/16 (수) AoE** | **full paper 제출 마감** |

역산(8/19 기준): 초록까지 23일. 27B 5-seed 수확(~8/22)·2.5 완주·mbpp/kk는
일정 내. B-track GPU 실험은 9/3까지 착수분만 본문行.

## 완료 (참고)

- ✅ 본제목 명사 교체(8/14, 사용자 선택): "Wrong Prompts" → "Wrong Training
  Data" — 분야 상위 명칭(data selection) 정합·프롬프트 엔지니어링 오독 제거,
  정밀도는 부제(Prompt Selection)가 유지
- ✅ 제목 부제 보강(8/13, 사용자 선택): "... : A Measurement-First Audit of
  Off-Policy Prompt Selection in RLVR" — wrong prompts 중의성 해소·재편
  스파인 반영·검색 키워드 확보 (본제목은 확정 이력 유지)
- ✅ 그록 리뷰 대조(8/13): 치명 지적 선반영 확인, 잔여 5건 분류(A6·A7·B4·B5·
  안전 단락), 안전/보상 해킹 단락 §9 반영, `src/reversal_freq.py` 작성
  (로컬 산출물 2벌 검증 완주)
- ✅ 판별: P3-0 precheck NO-GO·P4-0 kcurve 구조적 부재 (코드 감사 양쪽 통과 —
  독립 jitter 정상, 사전 등록 규칙 일치)
- ✅ 원고 v0.1: 재편 골격 전면 초안 + PDF (pdflatex 무에러, push됨)
- ✅ 보강 계산: 측정 상한 명제 시뮬·초기하 정확검정·Fisher 결합·함정 4호 합성
  (스크립트는 세션 스크래치 — 원고 부록 수치로 반영됨)
- ✅ 이전: 제목 확정·용어 정규화·서지 검증·go_* 스택·D1~D7 설계·이론 전수 검증·
  발표자료 19덱 (재편 전 기준 — A4에서 갱신 예정)
