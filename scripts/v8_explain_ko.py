"""Korean reading of one V8 analysis (analysis.json + policy_outcomes_bank.json), appended under the numbers.

Used by v8_summary.py and v8_recount.py. Standard library only; every sentence is computed from the same
values printed above it, never from fixed text about a past run.
"""

NAMES = {
    "srgc_current": "SR-GC(현재 규칙)",
    "on_continued": "전환 없이 On-policy 계속",
    "srgc_one_negative": "SR-GC 음성 1회 규칙",
    "srgc_two_negative": "SR-GC 음성 2회 연속 규칙",
    "srgc_three_negative": "SR-GC 음성 3회 연속 규칙",
    "srgc_uniform_destination": "SR-GC 시점에 Uniform으로 전환",
    "sr_early": "처음부터 SR",
    "uniform_early": "처음부터 Uniform",
    "sampled_time_sr": "무작위 시점 SR 전환",
    "sampled_time_sr.d0": "무작위 시점 SR 전환",
    "plateau_sr_tuned": "학습 보상 정체 트리거(조정값)",
    "fixed_sr_tuned": "고정 시점 전환(조정값)",
}
GOOD = ("triggered", "not_triggered", "ran")


def name(p):
    if p in NAMES:
        return NAMES[p]
    if p.startswith("fixed_sr_t"):
        return f"고정 시점 SR 전환 τ={int(p[-3:])}"
    if p.startswith("fixed_uniform_t"):
        return f"고정 시점 Uniform 전환 τ={int(p[-3:])}"
    if p.startswith("plateau_sr_eps"):
        return f"학습 보상 정체 트리거 ε={int(p[-4:]) / 1000:g}"
    if p.endswith("_c50") and p[:-4] in NAMES:
        return NAMES[p[:-4]] + ", 검사 간격 50"
    return p


def pp(v):
    return "-" if v is None else f"{v:+.2f}점"


def verdict(p, ci=None):
    """One clause: who is ahead and how sure, from wins/ties/losses and a 95% interval if any."""
    n = p.get("n_training_seeds") or 0
    m = p.get("mean_delta_pp")
    w, t, l = p.get("wins") or 0, p.get("ties") or 0, p.get("losses") or 0
    lo, hi = (ci or p).get("ci_low"), (ci or p).get("ci_high")
    if n == 0 or m is None:
        return "값 없음"
    if n < 3:
        miss = p.get("missing_seeds") or []
        return f"시드 {n}개뿐이라 판단 보류" + (f"(빠진 시드 {len(miss)}개)" if miss else "")
    if t == n:
        return "모든 시드에서 결과가 같음(같은 시점에 전환)"
    head = f"{n}개 시드 중 {w}승 {t}무 {l}패"
    if lo is not None and lo > 0:
        return f"{head}, 95% 구간이 0보다 위라서 앞섬"
    if hi is not None and hi < 0:
        return f"{head}, 95% 구간이 0보다 아래라서 뒤짐"
    if w == n:
        return f"{head}, 모든 시드에서 앞서지만 구간이 0을 포함"
    if l == n:
        return f"{head}, 모든 시드에서 뒤지지만 구간이 0을 포함"
    return f"{head}, 차이가 불확실"


def mean_of(rows, policy, key, budget=False):
    v = [r[key] for r in rows if r.get("policy_id") == policy and r.get("status") in GOOD
         and r.get(key) is not None and (not budget or r.get("budget_status") == "selected")]
    return (sum(v) / len(v), len(v)) if v else (None, 0)


def explain(a, rows, out):
    rows = rows or []
    ep = a.get("endpoints") or {}
    H = ep.get("matched_update")
    b = ep.get("budget") or {}
    out.append("-- 해석 (위 숫자에서 계산) --")
    out.append("모든 차이는 SR-GC 쪽 보상에서 비교 대상 보상을 뺀 값이고 단위는 보상 퍼센트포인트(점)다.")

    pr, pb = a.get("primary_rebuttal") or {}, a.get("primary_rebuttal_budget") or {}
    gb = ((pr.get("loso") or {}).get("grid_best_same_seeds") or {}).get("tau")
    hats = {v.get("tau_hat") for v in ((pr.get("loso") or {}).get("per_seed") or {}).values()}
    if "primary_contrast" in pr:
        c = pr["primary_contrast"]
        out.append(f"같은 업데이트(H={H})에서 SR-GC와 LOSO로 고른 고정 시점 전환의 차이는 {pp(c.get('mean_delta_pp'))}, "
                   f"{verdict(c, c.get('naive_paired_t'))}.")
        if len(hats) == 1 and None not in hats:
            out.append(f"LOSO는 모든 시드에서 τ={hats.pop()}을 골랐다. 고정 시점 선택이 시드에 따라 흔들리지 않는다.")
        elif hats:
            out.append(f"LOSO가 고른 τ가 시드마다 다르다({sorted(h for h in hats if h is not None)}). 조정이 불안정하다.")
    if "primary_contrast" in pb:
        c = pb["primary_contrast"]
        out.append(f"같은 GPU 예산({(b.get('value') or 0) / 3600:.2f} GPU-시간)에서는 {pp(c.get('mean_delta_pp'))}, "
                   f"{verdict(c, c.get('naive_paired_t'))}.")
        if "cold-start incomplete" in str(b.get("scope")):
            out.append("이 예산은 이어 학습 비용만으로 정했다. prefix와 SR 캐시 비용이 다 모이지 않았기 때문이다.")
        costlier = False
        fx = f"fixed_sr_t{gb:03d}" if isinstance(gb, int) else None
        if fx:
            hs, ns = mean_of(rows, "srgc_current", "continuation_gpu_seconds")
            hf, nf = mean_of(rows, fx, "continuation_gpu_seconds")
            us, _ = mean_of(rows, "srgc_current", "budget_endpoint_update", budget=True)
            uf, _ = mean_of(rows, fx, "budget_endpoint_update", budget=True)
            if hs and hf:
                line = (f"H까지 이어 학습 비용은 SR-GC {hs / 3600:.2f}시간, 고정 τ={gb} {hf / 3600:.2f}시간이다"
                        f"(시드 평균).")
                if hs > hf * 1.02:
                    costlier = True
                    line += f" SR-GC가 {100 * (hs / hf - 1):.0f}% 더 쓰는데, 검사 비용 때문이다."
                if us is not None and uf is not None:
                    line += f" 그래서 같은 예산에서 끝나는 업데이트가 SR-GC는 평균 {us:.0f}, 고정 전환은 {uf:.0f}이다."
                out.append(line)
        m1 = (pr.get("primary_contrast") or {}).get("mean_delta_pp")
        m2 = c.get("mean_delta_pp")
        if m1 is not None and m2 is not None and m1 > -0.1 and m2 < -0.5:
            out.append("업데이트 수를 맞추면 비슷하거나 앞서지만 GPU 예산을 맞추면 뒤진다."
                       + (" 전환 시점이 나빠서가 아니라 검사에 드는 비용 때문이다." if costlier else ""))
        out.append("구간은 LOSO 폴드를 독립으로 본 낙관적 t 구간이다. 실제 불확실성은 이보다 넓다.")
        out.append("리뷰 결론 후보(P9): " + p9(c))

    tests = [a[k] for k in ("primary_test", "primary_test_budget") if k in a]
    if tests and all(not t.get("n_training_seeds") for t in tests):
        out.append("새 시드에서 미리 정한 비교(TEST)는 아직 결과가 없다.")
    elif tests:
        for k in ("primary_test", "primary_test_budget"):
            if k in a:
                where = "같은 예산" if k.endswith("budget") else "같은 업데이트"
                out.append(f"새 시드 검증({where}): {pp(a[k].get('mean_delta_pp'))}, {verdict(a[k])}.")

    for key, where in (("secondary", "같은 업데이트"), ("secondary_budget", "같은 예산")):
        items = [p for p in a.get(key) or [] if p.get("status") != "missing_policy"]
        if not items:
            continue
        out.append(f"보조 비교({where}):")
        for p in items:
            lab = p.get("label") or ""
            if lab.startswith("E3 "):
                tau = lab.rsplit("(", 1)[-1].rstrip(")")
                out.append(f"  같은 부모 상태에서 SR 대 Uniform(τ={int(tau)}): {pp(p.get('mean_delta_pp'))}, {verdict(p)}")
                continue
            x, _, y = lab.partition(" - ")
            who = f"{name(y)} 대비" if x == "srgc_current" else f"{name(x)} 대 {name(y)}"
            out.append(f"  {who}: {pp(p.get('mean_delta_pp'))}, {verdict(p)}")

    for key, where in (("reward_fraction", "같은 업데이트"), ("budget_reward_fraction", "같은 예산")):
        acc = {}
        for r in rows:
            if r.get("status") in GOOD and r.get(key) is not None and \
                    (key == "reward_fraction" or r.get("budget_status") == "selected"):
                acc.setdefault(r["policy_id"], []).append(r[key])
        full = max((len(v) for v in acc.values()), default=0)
        ranked = sorted(((sum(v) / len(v), p) for p, v in acc.items() if len(v) == full), reverse=True)
        if ranked:
            top = ", ".join(f"{name(p)} {100 * v:.2f}%" for v, p in ranked[:3])
            pos = next((i + 1 for i, (_, p) in enumerate(ranked) if p == "srgc_current"), None)
            out.append(f"{where} 평균 보상 상위 3개(시드 {full}개가 다 있는 정책만): {top}."
                       + (f" SR-GC는 {len(ranked)}개 중 {pos}위." if pos else ""))

    capped = {}
    for r in rows:
        if r.get("budget_status") == "capped_at_horizon":
            capped.setdefault(r["policy_id"], set()).add(r.get("seed"))
    if capped:
        out.append("예산 끝점이 H를 넘어 같은 예산 값이 없는 런(capped_at_horizon): "
                   + ", ".join(f"{name(p)} {len(s)}개" for p, s in sorted(capped.items()))
                   + ". 이 정책들의 같은 예산 비교는 시드 수가 적다.")

    sc = a.get("status_counts") or {}
    bad = sum(n for d in sc.values() for s, n in d.items() if s not in GOOD)
    if bad:
        out.append(f"쓸 수 없는 런이 {bad}개 있다(censored·unavailable 등). 위 비교에서 그 시드는 빠졌다.")
    pe = a.get("physical_expenditure") or {}
    if pe and not pe.get("complete"):
        out.append(f"클러스터 전체 GPU 합계는 기록 {pe.get('missing_events')}개가 빠져 미완성이다. 합계 숫자는 인용하면 안 된다.")


def p9(c):
    n = c.get("n_training_seeds") or 0
    nt = c.get("naive_paired_t") or {}
    lo, hi = nt.get("ci_low"), nt.get("ci_high")
    if n < 3:
        return "시드가 모자라 아직 고를 수 없다."
    if lo is not None and lo > 0 and c.get("wins") == n:
        return "(a) SR-GC가 같은 예산에서 고정 전환보다 낫다. 시점 주장을 유지한다."
    if (hi is not None and hi < 0) or c.get("losses") == n:
        return ("(c) 같은 예산에서 고정 시점 전환이 SR-GC 이상이다. 시점 주장은 빼고 "
                "기여를 '단계적 선택'(On-policy에서 SR로 한 번 전환)으로 쓴다.")
    return "(b) 같은 예산에서 SR-GC와 고정 전환이 구별되지 않는다. 기여를 단계적 선택으로 쓴다."
