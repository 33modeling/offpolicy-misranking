# Fresh-audit 비용–품질 frontier

진실=홀수 micro-group 전용, 정책 관측=짝수 그룹 — 표본 비공유 프로토콜. 확률 정책은 20회 평균±sd.

## F1. run별 정책 × 예산
| run | policy | fresh(관측) | 예산% | precision | ±sd | regret |
|---|---|---|---|---|---|---|
| v3-s2-math500 | stale_g00 | 0 | 0% | 0.275 | 0.000 | 0.0625 |
| v3-s2-math500 | stale_g10 | 0 | 0% | 0.275 | 0.000 | 0.0679 |
| v3-s2-math500 | stale_g01 | 0 | 0% | 0.200 | 0.000 | 0.0655 |
| v3-s2-math500 | stale_g11 | 0 | 0% | 0.275 | 0.000 | 0.0637 |
| v3-s2-math500 | passrate_beta | 0 | 0% | 0.265 | 0.013 | 0.0629 |
| v3-s2-math500 | random | 0 | 0% | 0.098 | 0.054 | 0.0706 |
| v3-s2-math500 | floor_gated(제안) | 272 | 17% | 0.212 | 0.078 | 0.0689 |
| v3-s2-math500 | fresh_m1 | 400 | 25% | 0.230 | 0.010 | 0.0685 |
| v3-s2-math500 | fresh_m2 | 800 | 50% | 0.275 | 0.000 | 0.0639 |
| v3-s2-math500 | fresh_m4 | 1600 | 100% | 0.225 | 0.000 | 0.0707 |
| v3-s2-math500 | audit_rand_p1_m2 | 8 | 0% | 0.276 | 0.010 | 0.0623 |
| v3-s2-math500 | audit_bnd_p1_m2 | 8 | 0% | 0.250 | 0.000 | 0.0654 |
| v3-s2-math500 | audit_rand_p5_m2 | 40 | 2% | 0.269 | 0.020 | 0.0630 |
| v3-s2-math500 | audit_bnd_p5_m2 | 40 | 2% | 0.250 | 0.000 | 0.0678 |
| v3-s2-math500 | audit_rand_p10_m2 | 80 | 5% | 0.267 | 0.027 | 0.0635 |
| v3-s2-math500 | audit_bnd_p10_m2 | 80 | 5% | 0.250 | 0.000 | 0.0678 |
| v3-s2-math500 | audit_rand_p25_m2 | 200 | 12% | 0.259 | 0.034 | 0.0651 |
| v3-s2-math500 | audit_bnd_p25_m2 | 200 | 12% | 0.275 | 0.000 | 0.0647 |
| v3-s2-math500 | 2dref_p1 | 0 | 0% | 0.275 | 0.000 | 0.0625 |
| v3-s2-math500 | 2dref_marginonly_p1 | 0 | 0% | 0.275 | 0.000 | 0.0625 |
| v3-s2-math500 | 2dref_disagreeonly_p1 | 0 | 0% | 0.275 | 0.000 | 0.0625 |
| v3-s2-math500 | 2dref_p5 | 40 | 2% | 0.251 | 0.046 | 0.0681 |
| v3-s2-math500 | 2dref_marginonly_p5 | 40 | 2% | 0.258 | 0.039 | 0.0664 |
| v3-s2-math500 | 2dref_disagreeonly_p5 | 40 | 2% | 0.273 | 0.016 | 0.0660 |
| v3-s2-math500 | 2dref_p10 | 80 | 5% | 0.241 | 0.026 | 0.0660 |
| v3-s2-math500 | 2dref_marginonly_p10 | 80 | 5% | 0.250 | 0.039 | 0.0635 |
| v3-s2-math500 | 2dref_disagreeonly_p10 | 80 | 5% | 0.271 | 0.037 | 0.0616 |
| v3-s2-math500 | 2dref_p25 | 200 | 12% | 0.253 | 0.016 | 0.0649 |
| v3-s2-math500 | 2dref_marginonly_p25 | 200 | 12% | 0.243 | 0.034 | 0.0660 |
| v3-s2-math500 | 2dref_disagreeonly_p25 | 200 | 12% | 0.253 | 0.031 | 0.0660 |

## F2. dataset 집계 (seed 평균)
| dataset | policy | precision(seed평균) | seed-sd | seeds |
|---|---|---|---|---|
| v3-math500 | 2dref_disagreeonly_p1 | 0.275 | 0.000 | 1 |
| v3-math500 | 2dref_disagreeonly_p10 | 0.271 | 0.000 | 1 |
| v3-math500 | 2dref_disagreeonly_p25 | 0.253 | 0.000 | 1 |
| v3-math500 | 2dref_disagreeonly_p5 | 0.273 | 0.000 | 1 |
| v3-math500 | 2dref_marginonly_p1 | 0.275 | 0.000 | 1 |
| v3-math500 | 2dref_marginonly_p10 | 0.250 | 0.000 | 1 |
| v3-math500 | 2dref_marginonly_p25 | 0.243 | 0.000 | 1 |
| v3-math500 | 2dref_marginonly_p5 | 0.258 | 0.000 | 1 |
| v3-math500 | 2dref_p1 | 0.275 | 0.000 | 1 |
| v3-math500 | 2dref_p10 | 0.241 | 0.000 | 1 |
| v3-math500 | 2dref_p25 | 0.253 | 0.000 | 1 |
| v3-math500 | 2dref_p5 | 0.251 | 0.000 | 1 |
| v3-math500 | audit_bnd_p10_m2 | 0.250 | 0.000 | 1 |
| v3-math500 | audit_bnd_p1_m2 | 0.250 | 0.000 | 1 |
| v3-math500 | audit_bnd_p25_m2 | 0.275 | 0.000 | 1 |
| v3-math500 | audit_bnd_p5_m2 | 0.250 | 0.000 | 1 |
| v3-math500 | audit_rand_p10_m2 | 0.267 | 0.000 | 1 |
| v3-math500 | audit_rand_p1_m2 | 0.276 | 0.000 | 1 |
| v3-math500 | audit_rand_p25_m2 | 0.259 | 0.000 | 1 |
| v3-math500 | audit_rand_p5_m2 | 0.269 | 0.000 | 1 |
| v3-math500 | floor_gated(제안) | 0.212 | 0.000 | 1 |
| v3-math500 | fresh_m1 | 0.230 | 0.000 | 1 |
| v3-math500 | fresh_m2 | 0.275 | 0.000 | 1 |
| v3-math500 | fresh_m4 | 0.225 | 0.000 | 1 |
| v3-math500 | passrate_beta | 0.265 | 0.000 | 1 |
| v3-math500 | random | 0.098 | 0.000 | 1 |
| v3-math500 | stale_g00 | 0.275 | 0.000 | 1 |
| v3-math500 | stale_g01 | 0.200 | 0.000 | 1 |
| v3-math500 | stale_g10 | 0.275 | 0.000 | 1 |
| v3-math500 | stale_g11 | 0.275 | 0.000 | 1 |

## F3. family 비교 (run별 최고 정책)
| run | gradient(stale) | 2D-REFRESH | predictor(passrate) | audit(random) | audit(boundary) | fresh | random |
|---|---|---|---|---|---|---|---|
| v3-s2-math500 | 0.275 | 0.275 | 0.265 | 0.276 | 0.275 | 0.275 | 0.098 |

## F4. 조건 지표 (Q1 위상도 재료)
| run | clipfrac_g10 | clipfrac_g11 | k | live_frac_beta | n | token_kl_beta_pi | traj_ess_frac_g11 | truth_margin_k | truth_reliability |
|---|---|---|---|---|---|---|---|---|---|
| v3-s2-math500 | 0.1227 | 0.3113 | 40 | 0.273 | 400 | 0.0027 | 0.0079 | 0.0013 | 0.325 |