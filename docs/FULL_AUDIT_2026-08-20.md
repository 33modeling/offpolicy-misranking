# 논문·수식·코드·레퍼런스 전수 감사 (2026-08-20)

## 범위와 판정

- 코드 기준: `9f4fd01`에서 분기한 `audit/p1-integrity`
- 논문 기준: 비공개 `offpolicy-misranking-paper`의 `b1c3219` 이후 교정본
- 점검 범위: 2×2 IS 분해, 반례/하한/ceiling 수식, rollout 생성과 소비,
  oracle/floor/K-curve/frontier/gate/downstream, 실행·수확 스크립트, 본문 주장과
  인용, 2026-08-20 기준 신규 직접 경쟁 문헌

**판정:** 열거 검산에서 알려진 대수 오류는 없다. 그러나 기존 실험 수치는
confirmatory evidence가 아니다. 가장 큰 구현 오류는 stale selection score와 evaluation
oracle이 같은 validation 방향을 공유한 것이며, oracle split-half의 두 candidate 절반도
같은 validation 방향으로 점수화됐다. K-curve와 frontier에도 같은 종류의 validation
재사용이 남아 있었다. 모두 수정했지만 결과 표는 교정 코드로 재생성해야 한다.

## 수식 감사

### 통과한 항목

- trajectory score identity에서 현재 정책 gradient를 prefix occupancy와 continuation
  outcome의 두 축으로 나누는 2×2 셀 `g00/g10/g01/g11`을 직접 열거했다.
- full product인 `g11`만 target-policy trajectory law를 복원하며, prefix-only와
  future-only 반례 모두에서 KL이 `O(epsilon^2)`로 가면서 부호 반전이 유지된다.
- 두 관측 가능 pool이 같지만 target gradient 부호가 다른 indistinguishability
  construction과 “one-sided 정보만으로 certificate 불가” 범위를 확인했다.
- leave-one-out advantage의 baseline 항, full IS identity, ranking flip, angular radius,
  top-k gap의 `Delta^-2 log(1/delta)` 하한을 재검산했다.
- binary group normalization 반례를 일반 group size로 확장한 수치 검산은
  `K=2..64,128,256,512,1024`에서 양의 margin을 유지했다.
- ceiling proposition은 조건부 독립 Gaussian parallel-oracle model 안의 결과다.
  원고에서 일반적인 방법 상한처럼 읽히던 문장을 model-conditional plug-in 결과로
  제한했고 finite-floor uncertainty가 별도로 필요함을 명시했다.

### 남은 수학·통계 경계

- 기본 CertaGrad radius는 isotropic Gaussian model 기반이다. adaptive look 전체에
  delta를 union-bound하지만 distribution-free/time-uniform coverage는 주장하지 않는다.
- Spearman-Brown K extrapolation은 진단 모델이지 증명이 아니다. corrected observed
  curve와 독립 calibration 없이 “structural absence”라고 부를 수 없다.
- `2 x chance`는 운영 휴리스틱이다. 신뢰수준 또는 hypothesis-test threshold가 아니다.
- CountSketch와 마지막 layer subset이 방향 평균은 보존해도 조밀한 top-k 경계를
  보존하는지는 별도 empirical calibration이 필요하다.

## 코드 감사와 수정

| 심각도 | 발견 사항 | 조치 |
|---|---|---|
| Critical | stale selection score와 evaluation oracle이 같은 validation gradient를 공유 | selection은 `val_groups[0::2]`, oracle evaluation은 `val_groups[1::2]`로 분리; 두 프로토콜 마커가 없으면 모든 판정·후처리 거부 |
| Critical | oracle candidate odd/even halves가 같은 validation gradient를 공유 | `val_groups.pt`도 odd/even으로 분리하고 non-shard/merge 경로를 동일 함수로 통일 |
| Critical | K-curve가 candidate만 재분할하고 validation을 고정 | 반복마다 candidate와 validation을 독립 분할 |
| Critical | frontier selection/truth가 validation 표본을 공유 | candidate와 validation을 모두 selection/truth로 분리; truth reliability도 재분할 |
| Critical | hard pool이 모델/정확 K/원천 rollout에 결박되지 않음 | model config hash, source/pool hash, exact rollout coverage, sidecar, 독립 main-run liveness qualification 추가 |
| High | 직접 stage 호출이나 복사된 rollout이 생성 계약 검증 없이 새 score/oracle 마커를 만들 수 있음 | torch 비의존 공통 validator를 score/oracle 단계에 내장; manifest 범위, raw-softmax kwargs, EOS, prompt별 정확 K를 모두 통과해야 마커 생성 |
| High | C1의 두 축 실패와 C1′ 회복을 서로 다른 run/cut에서 합칠 수 있음 | 동일 run joint C1, cut `0.5` joint recovery만 causal witness로 인정 |
| High | 생성 코드 채점이 host Python subprocess에서 실행 | bubblewrap network/home 격리와 CPU/RAM/file/process/output 제한 추가 |
| High | 실행 중 공유 code가 바뀌어 stage별 code version이 섞일 수 있음 | HEAD, staged/unstaged diff, untracked status를 매 stage 전에 확인 |
| High | downstream DAPO run에도 dataset이 `gsm8k`로 고정 | 각 `run_config.json`의 dataset/n_train/n_val 사용 |
| High | frontier family 비교가 서로 다른 fresh budget의 최고값을 직접 비교 | rollout 단위 budget envelope로 변경; shard KL/ESS를 가중·log-sum 집계 |
| High | 수확/판독/retry wrapper가 실패를 숨김 | 실행 오류 전파; 과학적 verdict exit code 3/4만 예외 |
| Medium | 전역 `pkill`이 다른 run/GPU 작업까지 종료 가능 | run path와 기록 PID 범위로 제한 |
| Medium | legacy runner가 manifest/exact merge/code lock 없이 confirmatory 실행 가능 | `run_h100_all.sh`와 전용 `babysit.sh` 기본 비활성화 |
| Medium | CertaGrad를 formal certification처럼 표기 | output에 coverage model을 기록하고 본문/판정명을 boundary diagnostic으로 수정 |

교정 전 코드로 시작한 현재 실행은 중단하지 않는다. 완료 후 공유 checkout을 갱신하고
GPU 환경에서 각 run에 다음 명령을 실행한다.

```bash
python3 src/rescore_completed_run.py RUN_DIR
```

이 명령은 raw-softmax generation manifest와 모든 primary rollout row의 `resp_end`를
먼저 확인한다. 계약 이전 rollout이면 실패하며 우회해서는 안 된다. 통과하면 GPU에서
behavior rollout의 `g00/g10/g01/g11`을 selection validation 절반으로 다시 점수화하고,
저장된 `oracle_micro_groups.pt`와 evaluation validation 절반으로 oracle, 독립
split-half, `report.json`, 입력/output hash가 든 `postprocess_manifest.json`을 만든다.
`score_protocol.json`과 `oracle_protocol.json`은 이 경로에서만 생성한다. 이후
`harvest.sh`로 모든 표를 재생성해야 한다. 이미 새 score 프로토콜이 있는 run의
oracle만 재생성할 때에 한해 `recompute_oracle_scores.py`를 직접 쓸 수 있다.

## 기존 결과의 계보

`/home/kms/Downloads/0818`의 6개 readout은 모두 2026-08-18 08:54 KST에 생성됐다.
생성/EOS 계약 수정 `c6ca013`은 2026-08-19 22:57 KST이므로 이 readout들은 수정된
생성 계약의 결과일 수 없다.

| 파일 | SHA-256 |
|---|---|
| `FRONTIER.md` | `1d9efb4fe98123873ec5c7b272feac133800bf5436714c7bda3624dcf24e7152` |
| `KCURVE.md` | `26cb4bf3d62f9c7e8acfe1a0a0fa1e2e2e78f1194ad9b7467e964d091fd7c607` |
| `READOUT.md` | `4c76a5acb94f9e41da30356320ee30181dcafa34c98b6d226313afbe7169d4f1` |
| `REVERSAL.md` | `f558a3bde0f7b45989906e87e5c17311f988a8d8ac79e4405f9fa5eafff52971` |
| `STATS.md` | `bdfbcad522297ed0377da01bbecba745070f7ecdaf3abce21db57cd8b7c67bec` |
| `TABLES.md` | `6a70540df12ddc1956595d34530407b4387f5664a5e66dfdc105c9051cd25d60` |

원고는 숫자를 삭제하지 않고 “historical exploratory snapshot”으로 일괄 표기했다.
corrected rerun 전에는 abstract/conclusion의 empirical support로 사용하지 않는다.

## 실행 중 GPU 작업

- 이 감사 작업은 별도 worktree이므로 현재 다른 머신의 GPU 작업을 종료할 이유는 없다.
- 공유 code를 쓰는 모든 현재 작업이 끝날 때까지 공유 checkout에서 `git pull`하지 않는다.
  shell이 다음 stage에서 새 Python 프로세스를 시작하면 한 run에 code version이 섞일 수 있다.
- 완료 후 `run_config.json`의 git commit, `git_diff_sha256`, generation manifests,
  rollout row의 `resp_end`, prompt/K coverage를 검사한다.
- 계약이 통과해도 구버전 scalar score는 재사용하지 않는다. 위 GPU 재점수화 후
  oracle/floor/report/readout을 재생성한다. 계약이 실패하거나 generation commit이
  `c6ca013` 이전이면 해당 결과는 탐색용으로만 보존하고 새 `OUT_ROOT`에서 재실행한다.

### v4 clean rerun

생성 계약과 독립 validation 수정이 모두 들어간 clean commit에서는 기존 v2/v3 폴더를
재사용하지 않고 `scripts/go_v4.sh`를 사용한다. 이 경로는 GSM8K와 MATH500을 각각
seed 0..4로 실행해 `runs/v4-s*`와 `results/v4`에 격리한다. 공유 스토리지의 여러
클라우드 머신에서는 seed 하나씩 다음처럼 맡긴다.

```bash
SEEDS_V4="0" bash scripts/go_v4.sh  # 머신별로 seed 0..4
```

worker는 공용 TABLES/FRONTIER를 쓰지 않는다. 전체 10 run이 끝난 뒤 한 머신에서만
다음 명령으로 protocol-complete 행렬을 검증하고 집계한다.

```bash
OM_V4_FINALIZE_ONLY=1 bash scripts/go_v4.sh
bash scripts/harvest.sh
```

## 레퍼런스 감사

기존 off-policy correction, data selection, top-k identification, reliability 원전과 setup
인용을 직접 링크에 대조했다. 이번 감사에서 원고에 추가한 직접 관련 문헌은 다음과 같다.

- DEPO, ICLR 2026: offline/online RLVR data selection
- Learn More with Less, ICLR 2026: subjective/objective uncertainty query selection
- VIP, ICLR 2026 및 VIGOR: variance-aware rollout allocation
- SIS: selective importance sampling for off-policy tokens
- IRDS: verifier-coupled interpretable coverage selection
- SHIFT: single-rollout hidden-state, training-free selection
- InSight: weighted mutual-information selection
- Prompt Replay: medium-difficulty prompt reuse with on-policy trajectories

이들은 downstream 효율 또는 correction/allocation 방법을 제안한다. 본 논문의 고유 범위는
stale one-sided gradient ranking 자체의 identifiability와 noisy top-k measurement audit이다.
“관련 방법이 없다”는 novelty 주장은 사용하지 않는다. 34개 bibliography key는 모두 본문에
인용되고 미사용/누락 key가 없음을 확인했다.

## 검증 결과

- 통과: `test_artifact_contract.py`, `test_hard_pool.py`, `test_judge.py`, `test_frontier.py`,
  `test_pool_qualification.py`, `test_data_sandbox.py`, `test_reversal_freq.py`
- 통과: `scripts/verify_theory.py`의 sign reversal, indistinguishable pool, IS identity,
  LOO, group normalization, angular radius, 50k double-flip 탐색
- 통과: 전체 Python `py_compile`, 전체 shell `bash -n`, Ruff `F/E9/B`, `git diff --check`;
  score/oracle 이중 프로토콜과 재점수화 entry point 정적 검사 포함
- 통과: `latexmk -pdf`; 19 pages; undefined citation/reference 및 overfull box 없음
- 로컬 미실행: `test_contract.py`, `test_core.py`, `test_protocol.py`. 이 머신에
  `torch`와 `transformers`가 없다. GPU/실험 환경에서 반드시 재실행한다.

## 제출 전 필수 작업

1. corrected commit으로 seed matrix를 완주하고 모든 숫자/표/본문 인라인 값을 교체한다.
2. math reward의 문자열/float 판정을 symbolic equivalence 또는 공식 verifier로 교체하고
   수동 표본 오류율을 보고한다.
3. drift-training과 selection-evaluation prompt overlap을 제거한 ablation을 추가한다.
4. projection/full-gradient ranking calibration과 matched downstream 반복을 추가한다.
5. 공식 ICLR 스타일로 변환하고 본문 분량, 익명화, reproducibility appendix를 점검한다.
6. 수치 동결 직전 신규 문헌 검색과 citation-to-claim 대조를 한 번 더 실행한다.
