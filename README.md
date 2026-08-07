# offpolicy-misranking — 컨셉 #68 실행 레포

**주장**: RLVR 데이터 선택에서 off-policy rollout의 절반짜리 importance 교정(prefix만·suffix만)은
KL이 아무리 작아도 prompt gradient 방향을 뒤집어 top-k 선택을 틀리게 한다.
정본: `new-paper-ideas/68-one-sided-offpolicy-misranking/concept.md`
(수식 반례 검산: 같은 폴더 `verify_theory.py` — PASS 확인 2026-08-07).

## 무엇을 재나

- 2×2 추정량 `g00/g10/g01/g11` (prefix occupancy × continuation outcome 복원 여부)로
  같은 β rollout에서 프롬프트 점수 `s_i = cos(μ_i, v)`를 계산하고, π fresh rollout
  oracle 대비 **top-10% precision/Jaccard**를 잰다 (split-half noise floor 병기).
- **CertaGrad**: 순위 경계에만 fresh micro-group을 배분하는 순차 top-k 인증 —
  균등 배분(GradAlign-matched) 대비 fresh 사용량과 precision을 비교한다.

## 게이트 (concept 10절, 축소판 기준)

통과: one-sided 두 추정량이 각각 noise floor 대비 precision −0.15 이상 낮고,
CertaGrad가 uniform 대비 fresh ≤50%로 precision 차 ≤0.02.
사망: one-sided가 oracle과 noise floor 안에서 같음 / full-IS 안정화만으로 해결 /
uniform 소량으로 이미 복원 / fresh 사용 >50%.

## 구성

```
src/data.py        GSM8K(기본)·MATH-500 로딩, 256 train + 50 val, binary reward
src/rollout.py     β rollout 수집, drift 생성(정답 rollout LoRA RFT: 50/100/200 step), π 로드
src/grads.py       2×2 토큰 가중치·LOO advantage·projected prompt gradient (JL float32)
src/certagrad.py   confidence-ball 순차 top-k 인증 + uniform baseline (pool 시뮬레이션)
src/experiment.py  stage orchestrator (prep→rollout-behavior→drift→oracle→score→report)
```

drift는 CROPI checkpoint 대신 자체 LoRA RFT로 만든다(자족성) — β=base,
π=base+adapter. 본실험에서 CROPI 공개 checkpoint 재현을 병행한다.

## 실행

```bash
pip install -r requirements.txt
# H100 클러스터: GitHub egress 없음 → HF 미러 사용
# export HF_ENDPOINT=<미러 URL>

bash scripts/run_smoke.sh              # 0.5B, 8 prompts — 파이프라인 완주 확인
bash scripts/run_gate.sh outputs/h100-pilot 100   # 1.5B, 256+50, drift 100 step
```

산출물: `outputs/<run>/report.md` — 추정량별 precision 표 + CertaGrad 비교.
drift 스윕은 `run_gate.sh RUN 50|100|200`을 별도 RUN으로.

## 전체 스테이지 (2026-08-07 완성)

`run_gate.sh`가 순서대로 실행: prep → rollout-behavior → drift → oracle → score →
report → **hybrid**(prefix 절단 25/50/75% 2×2 처치) → **downstream**(선택 소스별
200-step GRPO-lite 학습 → val 정확도). H100 일괄 실행은 `scripts/run_h100_all.sh`
(기본 **Qwen2.5-7B-Instruct**, drift 50/100/200 스윕; `HF_ENDPOINT`·`OUT_ROOT` 필수).
데이터셋: 파일럿 GSM8K, `--dataset math500|dapo-math` 지원(DAPO 스키마는 첫 실행 확인).

## 파일럿 근사 (본실험에서 정식화)

- confidence 반경 기본값은 χ² 근사(`--radius-mode gaussian`) — coverage는 게이트
  반복실험으로 실측. `hoeffding`은 보수 판본 비교용.
- downstream은 clip 없는 GRPO-lite(LOO baseline REINFORCE) — 인증 관측치와 동일.
