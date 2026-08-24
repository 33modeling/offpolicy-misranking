# Fresh-audit 비용–품질 frontier

진실=홀수 micro-group 전용, 정책 관측=짝수 그룹 — 표본 비공유 프로토콜. 확률 정책은 20회 평균±sd.

## F1. run별 정책 × 예산
| run | policy | fresh(관측) | 예산% | precision | ±sd | regret |
|---|---|---|---|---|---|---|
| v2-s0 | stale_g00 | 0 | 0% | 0.216 | 0.000 | 0.0868 |
| v2-s0 | stale_g10 | 0 | 0% | 0.235 | 0.000 | 0.0792 |
| v2-s0 | stale_g01 | 0 | 0% | 0.255 | 0.000 | 0.0770 |
| v2-s0 | stale_g11 | 0 | 0% | 0.235 | 0.000 | 0.0810 |
| v2-s0 | passrate_beta | 0 | 0% | 0.340 | 0.026 | 0.0738 |
| v2-s0 | random | 0 | 0% | 0.116 | 0.053 | 0.1017 |
| v2-s0 | floor_gated(제안) | 294 | 14% | 0.177 | 0.057 | 0.0932 |
| v2-s0 | fresh_m1 | 512 | 25% | 0.235 | 0.000 | 0.0888 |
| v2-s0 | fresh_m2 | 1024 | 50% | 0.196 | 0.000 | 0.0897 |
| v2-s0 | fresh_m4 | 2048 | 100% | 0.216 | 0.000 | 0.0855 |
| v2-s0 | audit_rand_p1_m2 | 10 | 0% | 0.224 | 0.012 | 0.0861 |
| v2-s0 | audit_bnd_p1_m2 | 10 | 0% | 0.235 | 0.000 | 0.0866 |
| v2-s0 | audit_rand_p5_m2 | 50 | 2% | 0.220 | 0.018 | 0.0867 |
| v2-s0 | audit_bnd_p5_m2 | 50 | 2% | 0.216 | 0.000 | 0.0881 |
| v2-s0 | audit_rand_p10_m2 | 102 | 5% | 0.217 | 0.022 | 0.0876 |
| v2-s0 | audit_bnd_p10_m2 | 102 | 5% | 0.216 | 0.000 | 0.0874 |
| v2-s0 | audit_rand_p25_m2 | 256 | 12% | 0.227 | 0.037 | 0.0860 |
| v2-s0 | audit_bnd_p25_m2 | 256 | 12% | 0.216 | 0.000 | 0.0833 |
| v2-s0 | 2dref_p1 | 0 | 0% | 0.216 | 0.000 | 0.0868 |
| v2-s0 | 2dref_marginonly_p1 | 0 | 0% | 0.216 | 0.000 | 0.0868 |
| v2-s0 | 2dref_disagreeonly_p1 | 0 | 0% | 0.216 | 0.000 | 0.0868 |
| v2-s0 | 2dref_p5 | 50 | 2% | 0.226 | 0.039 | 0.0846 |
| v2-s0 | 2dref_marginonly_p5 | 50 | 2% | 0.239 | 0.029 | 0.0825 |
| v2-s0 | 2dref_disagreeonly_p5 | 50 | 2% | 0.241 | 0.033 | 0.0818 |
| v2-s0 | 2dref_p10 | 102 | 5% | 0.164 | 0.072 | 0.0944 |
| v2-s0 | 2dref_marginonly_p10 | 102 | 5% | 0.160 | 0.078 | 0.0951 |
| v2-s0 | 2dref_disagreeonly_p10 | 102 | 5% | 0.156 | 0.077 | 0.0960 |
| v2-s0 | 2dref_p25 | 256 | 12% | 0.139 | 0.079 | 0.0977 |
| v2-s0 | 2dref_marginonly_p25 | 256 | 12% | 0.142 | 0.082 | 0.0999 |
| v2-s0 | 2dref_disagreeonly_p25 | 256 | 12% | 0.154 | 0.078 | 0.0952 |

## F2. dataset 집계 (seed 평균)
| dataset | policy | precision(seed평균) | seed-sd | seeds |
|---|---|---|---|---|
| v2 | 2dref_disagreeonly_p1 | 0.216 | 0.000 | 1 |
| v2 | 2dref_disagreeonly_p10 | 0.156 | 0.000 | 1 |
| v2 | 2dref_disagreeonly_p25 | 0.154 | 0.000 | 1 |
| v2 | 2dref_disagreeonly_p5 | 0.241 | 0.000 | 1 |
| v2 | 2dref_marginonly_p1 | 0.216 | 0.000 | 1 |
| v2 | 2dref_marginonly_p10 | 0.160 | 0.000 | 1 |
| v2 | 2dref_marginonly_p25 | 0.142 | 0.000 | 1 |
| v2 | 2dref_marginonly_p5 | 0.239 | 0.000 | 1 |
| v2 | 2dref_p1 | 0.216 | 0.000 | 1 |
| v2 | 2dref_p10 | 0.164 | 0.000 | 1 |
| v2 | 2dref_p25 | 0.139 | 0.000 | 1 |
| v2 | 2dref_p5 | 0.226 | 0.000 | 1 |
| v2 | audit_bnd_p10_m2 | 0.216 | 0.000 | 1 |
| v2 | audit_bnd_p1_m2 | 0.235 | 0.000 | 1 |
| v2 | audit_bnd_p25_m2 | 0.216 | 0.000 | 1 |
| v2 | audit_bnd_p5_m2 | 0.216 | 0.000 | 1 |
| v2 | audit_rand_p10_m2 | 0.217 | 0.000 | 1 |
| v2 | audit_rand_p1_m2 | 0.224 | 0.000 | 1 |
| v2 | audit_rand_p25_m2 | 0.227 | 0.000 | 1 |
| v2 | audit_rand_p5_m2 | 0.220 | 0.000 | 1 |
| v2 | floor_gated(제안) | 0.177 | 0.000 | 1 |
| v2 | fresh_m1 | 0.235 | 0.000 | 1 |
| v2 | fresh_m2 | 0.196 | 0.000 | 1 |
| v2 | fresh_m4 | 0.216 | 0.000 | 1 |
| v2 | passrate_beta | 0.340 | 0.000 | 1 |
| v2 | random | 0.116 | 0.000 | 1 |
| v2 | stale_g00 | 0.216 | 0.000 | 1 |
| v2 | stale_g01 | 0.255 | 0.000 | 1 |
| v2 | stale_g10 | 0.235 | 0.000 | 1 |
| v2 | stale_g11 | 0.235 | 0.000 | 1 |

## F3. family 비교 (run별 최고 정책)
| run | gradient(stale) | 2D-REFRESH | predictor(passrate) | audit(random) | audit(boundary) | fresh | random |
|---|---|---|---|---|---|---|---|
| v2-s0 | 0.255 | 0.226 | 0.340 | 0.227 | 0.235 | 0.235 | 0.116 |

## F4. 조건 지표 (Q1 위상도 재료)
| run | clipfrac_g10 | clipfrac_g11 | k | live_frac_beta | n | token_kl_beta_pi | traj_ess_frac_g11 | truth_margin_k | truth_reliability |
|---|---|---|---|---|---|---|---|---|---|
| v2-s0 | 0.56 | 0.959 | 51 | 1.0 | 512 | -0.5677 | 0.001 | 0.0008 | 0.275 |