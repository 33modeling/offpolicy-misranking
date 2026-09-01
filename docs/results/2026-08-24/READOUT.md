# 판독 보고서

> **폐기된 혼합 revision 중간 산출물:** 이 파일의 자동 결론은 서로 다른
> generation commit을 구분하지 않고 seed를 합쳤으므로 실험 근거로 사용할 수
> 없다. 원시 run별 `run_config.json`도 이 보존본에 없어 여기 적힌 숫자를
> 사후에 안전하게 SHA별 재집계할 수 없다. 처리 원칙은
> [PROVENANCE_STATUS.md](PROVENANCE_STATUS.md)를 따른다.

## 한눈 요약

| run | floor | chance | g00 | g10 | g01 | g11 | one-sided가 더 나쁜가 | hybrid 회복 | mixed-dip |
|---|---|---|---|---|---|---|---|---|---|
| v4-27b-s0-math500 | 0.384† | 0.10 | 0.307 | 0.350 | 0.300 | 0.325 | 아니오 | 데이터 없음 | - |
| v4-27b-s1 | 0.255† | 0.10 | 0.216 | 0.176 | 0.157 | 0.157 | 아니오 | 데이터 없음 | - |
| v4-27b-s1-math500 | 0.475† | 0.10 | 0.306 | 0.236 | 0.220 | 0.195 | 예 (사전 문턱) | 데이터 없음 | - |
| v4-27b-s2-math500 | 0.400† | 0.10 | 0.273 | 0.134 | 0.138 | 0.136 | 예 (사전 문턱) | 데이터 없음 | - |
| v4-27b-s3-math500 | 0.400† | 0.10 | 0.367 | 0.304 | 0.246 | 0.229 | 아니오 | 데이터 없음 | - |
| v4-7b-s0 | 0.196† | 0.10 | 0.118 | 0.137 | 0.157 | 0.098 | 아니오 | C1 미충족 | 아니오 |
| v4-7b-s0-math500 | 0.175† | 0.10 | 0.175 | 0.150 | 0.225 | 0.200 | 아니오 | C1 미충족 | 아니오 |
| v4-7b-s1 | 0.275† | 0.10 | 0.216 | 0.196 | 0.176 | 0.176 | 아니오 | C1 미충족 | 아니오 |
| v4-7b-s1-math500 | 0.300† | 0.10 | 0.325 | 0.250 | 0.250 | 0.200 | 아니오 | C1 미충족 | 일부 |
| v4-7b-s2 | 0.216† | 0.10 | 0.118 | 0.118 | 0.098 | 0.078 | 아니오 | C1 미충족 | 일부 |
| v4-7b-s2-math500 | 0.300† | 0.10 | 0.150 | 0.175 | 0.175 | 0.225 | 아니오 | C1 미충족 | 아니오 |
| v4-7b-s3 | 0.176† | 0.10 | 0.216 | 0.235 | 0.176 | 0.176 | 아니오 | C1 미충족 | 일부 |
| v4-7b-s3-math500 | 0.225† | 0.10 | 0.250 | 0.300 | 0.225 | 0.300 | 아니오 | C1 미충족 | 일부 |
| v4-7b-s4 | 0.235† | 0.10 | 0.176 | 0.216 | 0.157 | 0.176 | 아니오 | C1 미충족 | 아니오 |
| v4-7b-s4-math500 | 0.125† | 0.10 | 0.300 | 0.275 | 0.225 | 0.150 | 아니오 | C1 미충족 | 일부 |

## 자동 결론

- **v4/27b/gsm8k**: one-sided 열세 0/1 run에서 관찰. hybrid 회복: 데이터 없음
- **v4/27b/math500**: one-sided 열세 2/4 run에서 관찰. hybrid 회복: 데이터 없음, 데이터 없음, 데이터 없음, 데이터 없음
- **v4/7b/gsm8k**: one-sided 열세 0/5 run에서 관찰. hybrid 회복: C1 미충족, C1 미충족, C1 미충족, C1 미충족, C1 미충족
- **v4/7b/math500**: one-sided 열세 0/5 run에서 관찰. hybrid 회복: C1 미충족, C1 미충족, C1 미충족, C1 미충족, C1 미충족

(주의: run 수가 적으면 위 관찰은 통계적 확정이 아님 — 5-seed 전승이 유의선)

## 용어 — 표를 읽는 법

- **floor**: oracle 절반끼리의 일치도. †는 원시 점수에서 독립 tie stream으로 재계산한 정본
- **chance**: 아무거나 찍었을 때의 기대 precision
- **g00/g10/g01/g11**: 무보정 / prefix만 / suffix만 / 전부 보정의 top-k precision
- **one-sided가 더 나쁜가**: 동일 run에서 g10·g01 모두 floor보다 0.15 이상 낮으면 '예'
- **hybrid 회복**: C1을 만족한 동일 run의 사전고정 cut=0.5에서 pp가 pb·bp보다 모두 높으면 '예'
- **mixed-dip**: 혼합 셀(bp·pb)이 순수 stale(bb)보다 낮으면 '예'

## 제외된 historical run

- gate-14b: corrected score/oracle protocol 없음
- gate-14b-math500: corrected score/oracle protocol 없음
- gate-7b: corrected score/oracle protocol 없음
- gate-7b-math500: corrected score/oracle protocol 없음
- v2-s0: corrected score/oracle protocol 없음
- v2-s0-math500-math500: corrected score/oracle protocol 없음
- v2-s1: corrected score/oracle protocol 없음
- v2-s1-dapo-math-dapo-math: corrected score/oracle protocol 없음
- v2-s1-math500-math500: corrected score/oracle protocol 없음
- v2-s2: corrected score/oracle protocol 없음
- v2-s2-dapo-math: corrected score/oracle protocol 없음
- v2-s2-math500-math500: corrected score/oracle protocol 없음
- v2-s4: corrected score/oracle protocol 없음
- v3-s2-math500: corrected score/oracle protocol 없음

## 상세 (원시 출력)

<details><summary>v4-27b-s0-math500 원시 출력</summary>

```
=== 게이트 판정: /group-volume/minsoo3.kim/offpolicy-misranking/runs/v4-27b-s0-math500 (파이프라인 1개) ===

[v4-27b-s0-math500] noise_floor=0.384, k=40
    g00: precision=0.307 (Δfloor=-0.076)
    g10: precision=0.350 (Δfloor=-0.034)
    g01: precision=0.300 (Δfloor=-0.084)
    g11: precision=0.325 (Δfloor=-0.059)
    boundary diagnostic: separated=False fresh=1.06× precision=0.375 uniform=0.275 → FAIL

    hybrid joint: 미판정


=== 종합 ===
  FAIL  C1 g10(prefix만) 실패 실증
  FAIL  C1 g01(suffix만) 실패 실증
  FAIL  C1 동일 run에서 두 축 실패
  미판정  C1' 사전고정 cut에서 hybrid 축 교체로 회복
  FAIL  C2 경계 분리·비용 진단
  미판정  C3 downstream 비열등

→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.
```
</details>

<details><summary>v4-27b-s1 원시 출력</summary>

```
=== 게이트 판정: /group-volume/minsoo3.kim/offpolicy-misranking/runs/v4-27b-s1 (파이프라인 1개) ===

[v4-27b-s1] noise_floor=0.255, k=51
    g00: precision=0.216 (Δfloor=-0.039)
    g10: precision=0.176 (Δfloor=-0.078)
    g01: precision=0.157 (Δfloor=-0.098)
    g11: precision=0.157 (Δfloor=-0.098)
    boundary diagnostic: separated=False fresh=1.04× precision=0.255 uniform=0.216 → FAIL

    hybrid joint: 미판정


=== 종합 ===
  FAIL  C1 g10(prefix만) 실패 실증
  FAIL  C1 g01(suffix만) 실패 실증
  FAIL  C1 동일 run에서 두 축 실패
  미판정  C1' 사전고정 cut에서 hybrid 축 교체로 회복
  FAIL  C2 경계 분리·비용 진단
  미판정  C3 downstream 비열등

→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.
```
</details>

<details><summary>v4-27b-s1-math500 원시 출력</summary>

```
=== 게이트 판정: /group-volume/minsoo3.kim/offpolicy-misranking/runs/v4-27b-s1-math500 (파이프라인 1개) ===

[v4-27b-s1-math500] noise_floor=0.475, k=40
    g00: precision=0.306 (Δfloor=-0.169)
    g10: precision=0.236 (Δfloor=-0.239)  ← one-sided 실패 실증
    g01: precision=0.220 (Δfloor=-0.255)  ← one-sided 실패 실증
    g11: precision=0.195 (Δfloor=-0.280)
    C1 joint: 두 one-sided 실패가 이 run에서 동시 성립
    boundary diagnostic: separated=False fresh=1.06× precision=0.475 uniform=0.475 → FAIL

    hybrid joint: 미판정


=== 종합 ===
  PASS  C1 g10(prefix만) 실패 실증
  PASS  C1 g01(suffix만) 실패 실증
  PASS  C1 동일 run에서 두 축 실패
  미판정  C1' 사전고정 cut에서 hybrid 축 교체로 회복
  FAIL  C2 경계 분리·비용 진단
  미판정  C3 downstream 비열등

→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.
```
</details>

<details><summary>v4-27b-s2-math500 원시 출력</summary>

```
=== 게이트 판정: /group-volume/minsoo3.kim/offpolicy-misranking/runs/v4-27b-s2-math500 (파이프라인 1개) ===

[v4-27b-s2-math500] noise_floor=0.400, k=40
    g00: precision=0.273 (Δfloor=-0.128)
    g10: precision=0.134 (Δfloor=-0.266)  ← one-sided 실패 실증
    g01: precision=0.138 (Δfloor=-0.263)  ← one-sided 실패 실증
    g11: precision=0.136 (Δfloor=-0.264)
    C1 joint: 두 one-sided 실패가 이 run에서 동시 성립
    boundary diagnostic: separated=False fresh=1.06× precision=0.400 uniform=0.050 → FAIL

    hybrid joint: 미판정


=== 종합 ===
  PASS  C1 g10(prefix만) 실패 실증
  PASS  C1 g01(suffix만) 실패 실증
  PASS  C1 동일 run에서 두 축 실패
  미판정  C1' 사전고정 cut에서 hybrid 축 교체로 회복
  FAIL  C2 경계 분리·비용 진단
  미판정  C3 downstream 비열등

→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.
```
</details>

<details><summary>v4-27b-s3-math500 원시 출력</summary>

```
=== 게이트 판정: /group-volume/minsoo3.kim/offpolicy-misranking/runs/v4-27b-s3-math500 (파이프라인 1개) ===

[v4-27b-s3-math500] noise_floor=0.400, k=40
    g00: precision=0.367 (Δfloor=-0.033)
    g10: precision=0.304 (Δfloor=-0.096)
    g01: precision=0.246 (Δfloor=-0.154)  ← one-sided 실패 실증
    g11: precision=0.229 (Δfloor=-0.171)
    boundary diagnostic: separated=False fresh=1.06× precision=0.400 uniform=0.300 → FAIL

    hybrid joint: 미판정


=== 종합 ===
  FAIL  C1 g10(prefix만) 실패 실증
  PASS  C1 g01(suffix만) 실패 실증
  FAIL  C1 동일 run에서 두 축 실패
  미판정  C1' 사전고정 cut에서 hybrid 축 교체로 회복
  FAIL  C2 경계 분리·비용 진단
  미판정  C3 downstream 비열등

→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.
```
</details>

<details><summary>v4-7b-s0 원시 출력</summary>

```
=== 게이트 판정: /group-volume/minsoo3.kim/offpolicy-misranking/runs/v4-7b-s0 (파이프라인 1개) ===

[v4-7b-s0] noise_floor=0.196, k=51
    g00: precision=0.118 (Δfloor=-0.078)
    g10: precision=0.137 (Δfloor=-0.059)
    g01: precision=0.157 (Δfloor=-0.039)
    g11: precision=0.098 (Δfloor=-0.098)
    boundary diagnostic: separated=False fresh=1.04× precision=0.196 uniform=0.059 → FAIL

[v4-7b-s0] hybrid cut=0.25: bb=0.12 bp=0.12 pb=0.38 pp=0.38  ← g10 미회복, g01 회복; 진단용 cut 또는 C1 미충족
[v4-7b-s0] hybrid cut=0.5: bb=0.12 bp=0.00 pb=0.31 pp=0.44  ← g10 회복, g01 회복; 진단용 cut 또는 C1 미충족
[v4-7b-s0] hybrid cut=0.75: bb=0.12 bp=0.31 pb=0.25 pp=0.50  ← g10 회복, g01 회복; 진단용 cut 또는 C1 미충족
    hybrid joint: 미판정


=== 종합 ===
  FAIL  C1 g10(prefix만) 실패 실증
  FAIL  C1 g01(suffix만) 실패 실증
  FAIL  C1 동일 run에서 두 축 실패
  미판정  C1' 사전고정 cut에서 hybrid 축 교체로 회복
  FAIL  C2 경계 분리·비용 진단
  미판정  C3 downstream 비열등

→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.
```
</details>

<details><summary>v4-7b-s0-math500 원시 출력</summary>

```
=== 게이트 판정: /group-volume/minsoo3.kim/offpolicy-misranking/runs/v4-7b-s0-math500 (파이프라인 1개) ===

[v4-7b-s0-math500] noise_floor=0.175, k=40
    g00: precision=0.175 (Δfloor=+0.000)
    g10: precision=0.150 (Δfloor=-0.025)
    g01: precision=0.225 (Δfloor=+0.050)
    g11: precision=0.200 (Δfloor=+0.025)
    boundary diagnostic: separated=False fresh=1.06× precision=0.175 uniform=0.225 → FAIL

[v4-7b-s0-math500] hybrid cut=0.25: bb=0.25 bp=0.31 pb=0.19 pp=0.25  ← g10 회복, g01 미회복; 진단용 cut 또는 C1 미충족
[v4-7b-s0-math500] hybrid cut=0.5: bb=0.25 bp=0.25 pb=0.25 pp=0.50  ← g10 회복, g01 회복; 진단용 cut 또는 C1 미충족
[v4-7b-s0-math500] hybrid cut=0.75: bb=0.25 bp=0.25 pb=0.31 pp=0.38  ← g10 회복, g01 회복; 진단용 cut 또는 C1 미충족
    hybrid joint: 미판정


=== 종합 ===
  FAIL  C1 g10(prefix만) 실패 실증
  FAIL  C1 g01(suffix만) 실패 실증
  FAIL  C1 동일 run에서 두 축 실패
  미판정  C1' 사전고정 cut에서 hybrid 축 교체로 회복
  FAIL  C2 경계 분리·비용 진단
  미판정  C3 downstream 비열등

→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.
```
</details>

<details><summary>v4-7b-s1 원시 출력</summary>

```
=== 게이트 판정: /group-volume/minsoo3.kim/offpolicy-misranking/runs/v4-7b-s1 (파이프라인 1개) ===

[v4-7b-s1] noise_floor=0.275, k=51
    g00: precision=0.216 (Δfloor=-0.059)
    g10: precision=0.196 (Δfloor=-0.078)
    g01: precision=0.176 (Δfloor=-0.098)
    g11: precision=0.176 (Δfloor=-0.098)
    boundary diagnostic: separated=False fresh=1.04× precision=0.275 uniform=0.137 → FAIL

[v4-7b-s1] hybrid cut=0.25: bb=0.31 bp=0.38 pb=0.25 pp=0.56  ← g10 회복, g01 회복; 진단용 cut 또는 C1 미충족
[v4-7b-s1] hybrid cut=0.5: bb=0.31 bp=0.38 pb=0.38 pp=0.38  ← g10 미회복, g01 미회복; 진단용 cut 또는 C1 미충족
[v4-7b-s1] hybrid cut=0.75: bb=0.31 bp=0.31 pb=0.31 pp=0.25  ← g10 미회복, g01 미회복; 진단용 cut 또는 C1 미충족
    hybrid joint: 미판정


=== 종합 ===
  FAIL  C1 g10(prefix만) 실패 실증
  FAIL  C1 g01(suffix만) 실패 실증
  FAIL  C1 동일 run에서 두 축 실패
  미판정  C1' 사전고정 cut에서 hybrid 축 교체로 회복
  FAIL  C2 경계 분리·비용 진단
  미판정  C3 downstream 비열등

→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.
```
</details>

<details><summary>v4-7b-s1-math500 원시 출력</summary>

```
=== 게이트 판정: /group-volume/minsoo3.kim/offpolicy-misranking/runs/v4-7b-s1-math500 (파이프라인 1개) ===

[v4-7b-s1-math500] noise_floor=0.300, k=40
    g00: precision=0.325 (Δfloor=+0.025)
    g10: precision=0.250 (Δfloor=-0.050)
    g01: precision=0.250 (Δfloor=-0.050)
    g11: precision=0.200 (Δfloor=-0.100)
    boundary diagnostic: separated=False fresh=1.06× precision=0.300 uniform=0.325 → FAIL

[v4-7b-s1-math500] hybrid cut=0.25: bb=0.25 bp=0.19 pb=0.12 pp=0.19  ← g10 회복, g01 미회복; 진단용 cut 또는 C1 미충족
[v4-7b-s1-math500] hybrid cut=0.5: bb=0.25 bp=0.31 pb=0.38 pp=0.31  ← g10 미회복, g01 미회복; 진단용 cut 또는 C1 미충족
[v4-7b-s1-math500] hybrid cut=0.75: bb=0.25 bp=0.31 pb=0.31 pp=0.25  ← g10 미회복, g01 미회복; 진단용 cut 또는 C1 미충족
    hybrid joint: 미판정


=== 종합 ===
  FAIL  C1 g10(prefix만) 실패 실증
  FAIL  C1 g01(suffix만) 실패 실증
  FAIL  C1 동일 run에서 두 축 실패
  미판정  C1' 사전고정 cut에서 hybrid 축 교체로 회복
  FAIL  C2 경계 분리·비용 진단
  미판정  C3 downstream 비열등

→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.
```
</details>

<details><summary>v4-7b-s2 원시 출력</summary>

```
=== 게이트 판정: /group-volume/minsoo3.kim/offpolicy-misranking/runs/v4-7b-s2 (파이프라인 1개) ===

[v4-7b-s2] noise_floor=0.216, k=51
    g00: precision=0.118 (Δfloor=-0.098)
    g10: precision=0.118 (Δfloor=-0.098)
    g01: precision=0.098 (Δfloor=-0.118)
    g11: precision=0.078 (Δfloor=-0.137)
    boundary diagnostic: separated=False fresh=1.04× precision=0.216 uniform=0.137 → FAIL

[v4-7b-s2] hybrid cut=0.25: bb=0.31 bp=0.19 pb=0.12 pp=0.12  ← g10 미회복, g01 미회복; 진단용 cut 또는 C1 미충족
[v4-7b-s2] hybrid cut=0.5: bb=0.31 bp=0.19 pb=0.31 pp=0.25  ← g10 미회복, g01 회복; 진단용 cut 또는 C1 미충족
[v4-7b-s2] hybrid cut=0.75: bb=0.31 bp=0.31 pb=0.31 pp=0.19  ← g10 미회복, g01 미회복; 진단용 cut 또는 C1 미충족
    hybrid joint: 미판정


=== 종합 ===
  FAIL  C1 g10(prefix만) 실패 실증
  FAIL  C1 g01(suffix만) 실패 실증
  FAIL  C1 동일 run에서 두 축 실패
  미판정  C1' 사전고정 cut에서 hybrid 축 교체로 회복
  FAIL  C2 경계 분리·비용 진단
  미판정  C3 downstream 비열등

→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.
```
</details>

<details><summary>v4-7b-s2-math500 원시 출력</summary>

```
=== 게이트 판정: /group-volume/minsoo3.kim/offpolicy-misranking/runs/v4-7b-s2-math500 (파이프라인 1개) ===

[v4-7b-s2-math500] noise_floor=0.300, k=40
    g00: precision=0.150 (Δfloor=-0.150)
    g10: precision=0.175 (Δfloor=-0.125)
    g01: precision=0.175 (Δfloor=-0.125)
    g11: precision=0.225 (Δfloor=-0.075)
    boundary diagnostic: separated=False fresh=1.06× precision=0.300 uniform=0.300 → FAIL

[v4-7b-s2-math500] hybrid cut=0.25: bb=0.19 bp=0.38 pb=0.31 pp=0.25  ← g10 미회복, g01 미회복; 진단용 cut 또는 C1 미충족
[v4-7b-s2-math500] hybrid cut=0.5: bb=0.19 bp=0.25 pb=0.44 pp=0.25  ← g10 미회복, g01 미회복; 진단용 cut 또는 C1 미충족
[v4-7b-s2-math500] hybrid cut=0.75: bb=0.19 bp=0.38 pb=0.31 pp=0.25  ← g10 미회복, g01 미회복; 진단용 cut 또는 C1 미충족
    hybrid joint: 미판정


=== 종합 ===
  FAIL  C1 g10(prefix만) 실패 실증
  FAIL  C1 g01(suffix만) 실패 실증
  FAIL  C1 동일 run에서 두 축 실패
  미판정  C1' 사전고정 cut에서 hybrid 축 교체로 회복
  FAIL  C2 경계 분리·비용 진단
  미판정  C3 downstream 비열등

→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.
```
</details>

<details><summary>v4-7b-s3 원시 출력</summary>

```
=== 게이트 판정: /group-volume/minsoo3.kim/offpolicy-misranking/runs/v4-7b-s3 (파이프라인 1개) ===

[v4-7b-s3] noise_floor=0.176, k=51
    g00: precision=0.216 (Δfloor=+0.039)
    g10: precision=0.235 (Δfloor=+0.059)
    g01: precision=0.176 (Δfloor=+0.000)
    g11: precision=0.176 (Δfloor=+0.000)
    boundary diagnostic: separated=False fresh=1.04× precision=0.176 uniform=0.078 → FAIL

[v4-7b-s3] hybrid cut=0.25: bb=0.31 bp=0.25 pb=0.50 pp=0.44  ← g10 미회복, g01 회복; 진단용 cut 또는 C1 미충족
[v4-7b-s3] hybrid cut=0.5: bb=0.31 bp=0.19 pb=0.38 pp=0.44  ← g10 회복, g01 회복; 진단용 cut 또는 C1 미충족
[v4-7b-s3] hybrid cut=0.75: bb=0.31 bp=0.25 pb=0.19 pp=0.19  ← g10 미회복, g01 미회복; 진단용 cut 또는 C1 미충족
    hybrid joint: 미판정


=== 종합 ===
  FAIL  C1 g10(prefix만) 실패 실증
  FAIL  C1 g01(suffix만) 실패 실증
  FAIL  C1 동일 run에서 두 축 실패
  미판정  C1' 사전고정 cut에서 hybrid 축 교체로 회복
  FAIL  C2 경계 분리·비용 진단
  미판정  C3 downstream 비열등

→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.
```
</details>

<details><summary>v4-7b-s3-math500 원시 출력</summary>

```
=== 게이트 판정: /group-volume/minsoo3.kim/offpolicy-misranking/runs/v4-7b-s3-math500 (파이프라인 1개) ===

[v4-7b-s3-math500] noise_floor=0.225, k=40
    g00: precision=0.250 (Δfloor=+0.025)
    g10: precision=0.300 (Δfloor=+0.075)
    g01: precision=0.225 (Δfloor=+0.000)
    g11: precision=0.300 (Δfloor=+0.075)
    boundary diagnostic: separated=False fresh=1.06× precision=0.225 uniform=0.300 → FAIL

[v4-7b-s3-math500] hybrid cut=0.25: bb=0.31 bp=0.19 pb=0.31 pp=0.38  ← g10 회복, g01 회복; 진단용 cut 또는 C1 미충족
[v4-7b-s3-math500] hybrid cut=0.5: bb=0.31 bp=0.19 pb=0.25 pp=0.25  ← g10 미회복, g01 회복; 진단용 cut 또는 C1 미충족
[v4-7b-s3-math500] hybrid cut=0.75: bb=0.31 bp=0.19 pb=0.19 pp=0.19  ← g10 미회복, g01 미회복; 진단용 cut 또는 C1 미충족
    hybrid joint: 미판정


=== 종합 ===
  FAIL  C1 g10(prefix만) 실패 실증
  FAIL  C1 g01(suffix만) 실패 실증
  FAIL  C1 동일 run에서 두 축 실패
  미판정  C1' 사전고정 cut에서 hybrid 축 교체로 회복
  FAIL  C2 경계 분리·비용 진단
  미판정  C3 downstream 비열등

→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.
```
</details>

<details><summary>v4-7b-s4 원시 출력</summary>

```
=== 게이트 판정: /group-volume/minsoo3.kim/offpolicy-misranking/runs/v4-7b-s4 (파이프라인 1개) ===

[v4-7b-s4] noise_floor=0.235, k=51
    g00: precision=0.176 (Δfloor=-0.059)
    g10: precision=0.216 (Δfloor=-0.020)
    g01: precision=0.157 (Δfloor=-0.078)
    g11: precision=0.176 (Δfloor=-0.059)
    boundary diagnostic: separated=False fresh=1.04× precision=0.235 uniform=0.118 → FAIL

[v4-7b-s4] hybrid cut=0.25: bb=0.12 bp=0.38 pb=0.25 pp=0.31  ← g10 회복, g01 미회복; 진단용 cut 또는 C1 미충족
[v4-7b-s4] hybrid cut=0.5: bb=0.12 bp=0.25 pb=0.31 pp=0.19  ← g10 미회복, g01 미회복; 진단용 cut 또는 C1 미충족
[v4-7b-s4] hybrid cut=0.75: bb=0.12 bp=0.25 pb=0.19 pp=0.25  ← g10 회복, g01 미회복; 진단용 cut 또는 C1 미충족
    hybrid joint: 미판정


=== 종합 ===
  FAIL  C1 g10(prefix만) 실패 실증
  FAIL  C1 g01(suffix만) 실패 실증
  FAIL  C1 동일 run에서 두 축 실패
  미판정  C1' 사전고정 cut에서 hybrid 축 교체로 회복
  FAIL  C2 경계 분리·비용 진단
  미판정  C3 downstream 비열등

→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.
```
</details>

<details><summary>v4-7b-s4-math500 원시 출력</summary>

```
=== 게이트 판정: /group-volume/minsoo3.kim/offpolicy-misranking/runs/v4-7b-s4-math500 (파이프라인 1개) ===

[v4-7b-s4-math500] noise_floor=0.125, k=40
    g00: precision=0.300 (Δfloor=+0.175)
    g10: precision=0.275 (Δfloor=+0.150)
    g01: precision=0.225 (Δfloor=+0.100)
    g11: precision=0.150 (Δfloor=+0.025)
    boundary diagnostic: separated=False fresh=1.06× precision=0.125 uniform=0.275 → FAIL

[v4-7b-s4-math500] hybrid cut=0.25: bb=0.31 bp=0.25 pb=0.25 pp=0.31  ← g10 회복, g01 회복; 진단용 cut 또는 C1 미충족
[v4-7b-s4-math500] hybrid cut=0.5: bb=0.31 bp=0.31 pb=0.25 pp=0.19  ← g10 미회복, g01 미회복; 진단용 cut 또는 C1 미충족
[v4-7b-s4-math500] hybrid cut=0.75: bb=0.31 bp=0.25 pb=0.31 pp=0.38  ← g10 회복, g01 회복; 진단용 cut 또는 C1 미충족
    hybrid joint: 미판정


=== 종합 ===
  FAIL  C1 g10(prefix만) 실패 실증
  FAIL  C1 g01(suffix만) 실패 실증
  FAIL  C1 동일 run에서 두 축 실패
  미판정  C1' 사전고정 cut에서 hybrid 축 교체로 회복
  FAIL  C2 경계 분리·비용 진단
  미판정  C3 downstream 비열등

→ 핵심 조건 실패 있음 — concept 사망 조건 대조 필요. 수치 원인 분석 권장.
```
</details>
