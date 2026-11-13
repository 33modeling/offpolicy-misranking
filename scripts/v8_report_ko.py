"""V8 결과 보고서(한글): 원본 값을 전부 적고, 항목마다 아래에 무슨 실험인지와 결과를 단다.

v8_recount.py가 cohort마다 report()를 부른다. 표준 라이브러리만 쓴다. 모든 문장은 같은 블록에 적힌
원본 값에서 계산한다. 지난 실행에 대한 고정 문구는 없다.
"""
import json
from pathlib import Path

import v8_explain_ko as ko

GOOD = ko.GOOD
ROLES = (("rebuttal_bank", "bank"), ("development", "development"), ("test", "test"))
ROLE_KO = {
    "rebuttal_bank": "학습 시드 bank. 시드마다 On-policy 궤적 하나와 그 궤적에서 갈라진 branch들을 학습해 두고, "
                     "정책마다 그 정책이 골랐을 시점의 branch 결과를 그대로 찾아 쓴다(보간 없음).",
    "development": "개발 시드. 고정 τ, ε 같은 조정값을 고르는 데만 쓴다. dev split으로 평가한다.",
    "test": "새 시드 검증. 개발 시드에서 조정값을 고정한 뒤, 처음 보는 시드에서 정책을 처음부터 실제로 실행했다.",
}
STATUS_ROW = {
    "triggered": "규칙이 발동해 그 시점에 전환",
    "not_triggered": "H까지 발동하지 않음. 전환 없는 On-policy 궤적 결과를 씀",
    "ran": "처음부터 따로 학습해 끝까지 감",
    **ko.STATUS_KO,
}
RULE_KO = {
    "current": "현재 규칙(D가 두 번 연속 음수, 또는 음수·0 이상·음수 순서이고 세 값 합이 음수일 때 전환)",
    "one_negative": "음성 1회 규칙(D가 한 번 음수면 바로 전환)",
    "two_negative": "음성 2회 연속 규칙(D가 두 번 연속 음수일 때 전환)",
    "three_negative": "음성 3회 연속 규칙(D가 세 번 연속 음수일 때 전환)",
}
BUDGET_KO = {
    "selected": "예산 안에서 마지막으로 저장된 체크포인트를 찾음",
    "capped_at_horizon": "예산이 H를 넘어 남는데 연장 학습이 아직 없어 끝점을 못 정함",
    "none": "예산 끝점 계산 안 됨",
}


def pct(v):
    return "-" if v is None else f"{100 * v:.2f}%"


def hrs(v):
    if v is None:
        return "-"
    return f"{v / 3600:.2f}h" if abs(v) >= 36 else f"{v:.2f}초"


def pp(v):
    return "-" if v is None else f"{v:+.2f}점"


def describe(pid, role, H):
    """무슨 실험인지: 정책 하나를 한 문단으로."""
    fresh = role in ("development", "test")
    base = pid.split(".")[0]
    c50 = base.endswith("_c50")
    core = base[:-4] if c50 else base
    if core == "on_continued":
        return ("전환 없이 On-policy 선택기로 H까지 학습했다." + (
            " 새 시드에서 처음부터 따로 학습한 런이다." if fresh else
            " 공유 prefix 뒤 이어간 기준 궤적이다. 25 업데이트마다 SR-GC 신호 D를 기록했고, "
            "다른 정책의 branch는 전부 이 궤적의 스냅샷에서 갈라진다."))
    if core.startswith("fixed_sr_t") and core[-3:].isdigit():
        t = int(core[-3:])
        return (f"update {t}에서 cached SR 선택기로 바꿔 H={H}까지 학습했다. 전환 시점은 미리 정한 고정값이라 "
                "검사 비용이 없다." + ("" if fresh else f" 궤적의 update {t} 스냅샷에서 갈라진 branch다."))
    if core.startswith("fixed_uniform_t"):
        t = int(core[-3:])
        return (f"update {t}에서 Uniform(무작위 선택) 선택기로 바꿔 H까지 학습했다. 이 시점은 그 시드에서 "
                "SR-GC 현재 규칙이 발동한 시점과 같다. 검사 비용은 넣지 않았다.")
    if core == "srgc_uniform_destination":
        return ("SR-GC 현재 규칙이 발동한 시점에 SR 대신 Uniform으로 바꿨다. 검사 비용은 넣었다. "
                "SR-GC의 이득이 SR이라는 목적지 덕인지, 그 시점에 바꾼 것 자체 덕인지 가르는 대조군이다.")
    if core.startswith("srgc_"):
        rule = core[len("srgc_"):]
        every = 50 if c50 else 25
        how = ("새 시드에서 SR-GC를 실제로 켜고 처음부터 학습했다." if fresh else
               "궤적에 기록된 D를 이 규칙으로 다시 읽어, 처음 발동한 시점의 SR branch 결과를 쓴다. "
               "발동하지 않으면 On-policy 궤적 결과를 쓴다.")
        return (f"SR-GC {RULE_KO.get(rule, rule)}. 검사는 {every} 업데이트마다다"
                + ("(25 간격 기록 중 50의 배수만 남겨 다시 읽음)" if c50 else "") + ". "
                + how + " 검사(40문항 두 세트 채점) 비용이 비용에 들어간다. D가 음수면 SR 쪽이 유리하다는 뜻이다.")
    if core.startswith("plateau_sr_eps"):
        eps = int(core[-4:]) / 1000
        return ("평가 데이터 없이 학습 배치 보상만 보는 싼 트리거다. 25 업데이트마다 최근 25 업데이트 평균 보상에서 "
                f"그 전 25 업데이트 평균을 뺀 개선폭이 ε={eps:g}(보상 비율, {100 * eps:g}점) 이하인 검사가 "
                "2번 연속 나오면 SR로 전환한다. 첫 검사는 update 50이다.")
    if core == "plateau_sr_tuned":
        return "학습 보상 정체 트리거를 개발 시드에서 고른 ε로 새 시드에 적용했다."
    if core == "fixed_sr_tuned":
        return "개발 시드에서 고른 고정 τ로 새 시드에서 SR 전환을 실행했다. 새 시드를 보고 다시 조정하지 않았다."
    if core.startswith("sampled_time_sr"):
        return ("개발 시드의 SR-GC 전환 시점 분포에서 학습 전에 한 번 뽑은 시점에 SR로 바꿨다. "
                "SR-GC가 고른 시점에 정보가 있는지 보는 무작위 시점 대조군이다.")
    if core.startswith("sr_early_refreshed_k"):
        k = int(core[-3:])
        return f"prefix 직후부터 cached SR로 학습하고, SR 캐시를 {k} 업데이트마다 현재 모델로 다시 만들었다."
    if core == "sr_early":
        return "prefix 직후부터 cached SR로 학습했다. 전환 시점 없이 처음부터 SR이다."
    if core == "uniform_early":
        return "prefix 직후부터 Uniform(무작위 선택)으로 학습했다."
    return "설명이 등록되지 않은 정책이다."


def row_line(r, H):
    st = r.get("status")
    parts = [f"s{r.get('seed')}: 상태 {st}"]
    if st not in GOOD:
        if r.get("switch_update") is not None:
            parts.append(f"전환 update {r['switch_update']}")
        if r.get("reason"):
            parts.append(f"이유 {r['reason']}")
        if r.get("censored_after") is not None:
            parts.append(f"기록 끝 update {r['censored_after']}")
        return "  " + ", ".join(parts)
    parts.append(f"전환 update {r.get('switch_update') if r.get('switch_update') is not None else '없음'}")
    if r.get("checks_used") is not None:
        parts.append(f"사용한 검사 {r['checks_used']}회(기록 끝 update {r.get('observed_through')})")
    q = f"{r.get('n_questions')}문항×{r.get('responses_per_question')}응답" if r.get("n_questions") else ""
    sp = f"{r['split']} split" if r.get("split") else ""
    parts.append(f"보상@H{r.get('endpoint_update', H)} {pct(r.get('reward_fraction'))}"
                 f"({', '.join(x for x in (q, sp, r.get('eval_status') or '') if x)})")
    parts.append(f"이어 학습 {hrs(r.get('continuation_gpu_seconds'))}")
    cs = r.get("cold_start_gpu_seconds")
    parts.append(f"cold-start {hrs(cs)}" if cs is not None else
                 f"cold-start 모름(빠진 부분 {r.get('cost_missing') or '?'}, 하한 {hrs(r.get('cold_start_lower_bound_gpu_seconds'))})")
    bs = r.get("budget_status")
    if bs:
        b = f"예산 {bs}"
        if r.get("budget_endpoint_update") is not None:
            b += (f" 끝점 update {r['budget_endpoint_update']} 보상 {pct(r.get('budget_reward_fraction'))}"
                  f"({r.get('budget_eval_status')}) 여유 {hrs(r.get('budget_slack_gpu_seconds'))}")
        parts.append(b)
    return "  " + ", ".join(parts)


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def policy_result(pid, rs, all_rows, H, ranks):
    """결과: 정책 하나의 원본 행에서 계산한 요약 문단."""
    seeds = sorted({r.get("seed") for r in rs})
    good = [r for r in rs if r.get("status") in GOOD]
    meas = [r for r in good if r.get("reward_fraction") is not None]
    out = [f"  결과: 시드 {len(seeds)}개 중 {len(good)}개 사용 가능, 그중 평가 끝난 것 {len(meas)}개."]
    if meas:
        v = [r["reward_fraction"] for r in meas]
        out.append(f"    같은 업데이트(H={H}) 평균 보상 {pct(mean(v))}, 가장 낮은 시드 {pct(min(v))}, 가장 높은 시드 {pct(max(v))}."
                   + (f" 시드가 다 있는 정책 {ranks[1]}개 중 {ranks[0]}위." if ranks else ""))
        on = {r["seed"]: r["reward_fraction"] for r in all_rows if r.get("policy_id") == "on_continued"
              and r.get("status") in GOOD and r.get("reward_fraction") is not None}
        d = [100 * (r["reward_fraction"] - on[r["seed"]]) for r in meas if r["seed"] in on]
        if d and pid != "on_continued":
            out.append(f"    같은 시드의 전환 없음(On-policy 계속)보다 평균 {pp(mean(d))}"
                       f"({sum(x > 0 for x in d)}개 시드 높음, {sum(x < 0 for x in d)}개 낮음, {sum(x == 0 for x in d)}개 같음).")
    sw = [r.get("switch_update") for r in good]
    if any(s is not None for s in sw):
        out.append("    전환 시점: " + ", ".join(f"s{r['seed']} {r.get('switch_update') or '없음'}" for r in good) + ".")
    c = mean([r.get("continuation_gpu_seconds") for r in good])
    if c is not None:
        out.append(f"    H까지 이어 학습 비용 시드 평균 {hrs(c)}.")
    bud = [r for r in good if r.get("budget_status") == "selected" and r.get("budget_reward_fraction") is not None]
    if bud:
        eu = mean([r.get("budget_endpoint_update") for r in bud])
        out.append(f"    같은 예산 평균 보상 {pct(mean([r['budget_reward_fraction'] for r in bud]))}(시드 {len(bud)}개)"
                   + (f", 예산 끝점 평균 update {eu:.0f}." if eu is not None else "."))
    bwait = [r for r in good if r.get("budget_status") == "selected" and r.get("budget_reward_fraction") is None]
    if bwait:
        out.append(f"    같은 예산 끝점 평가 대기 시드 {', '.join('s' + str(r.get('seed')) for r in bwait)}"
                   "(끝점 체크포인트는 정해졌고 그 평가가 남음).")
    capped = [r for r in good if r.get("budget_status") == "capped_at_horizon"]
    if capped:
        out.append(f"    시드 {len(capped)}개는 예산이 H 뒤까지 남는데 연장 학습이 없어 같은 예산 값이 비었다.")
    bad = [r for r in rs if r.get("status") not in GOOD]
    for st in sorted({r.get("status") for r in bad}, key=str):
        ss = [r for r in bad if r.get("status") == st]
        out.append(f"    못 쓰는 시드 {', '.join('s' + str(r.get('seed')) for r in ss)}: {st} = "
                   f"{STATUS_ROW.get(st, '분류되지 않은 상태')}.")
    unm = [r for r in good if r.get("reward_fraction") is None]
    if unm:
        out.append(f"    평가 대기 시드 {', '.join('s' + str(r.get('seed')) for r in unm)}: 학습은 끝났고 평가 작업이 남았다.")
    return out


def rank_table(rows):
    acc = {}
    for r in rows:
        if r.get("status") in GOOD and r.get("reward_fraction") is not None:
            acc.setdefault(r["policy_id"], []).append(r["reward_fraction"])
    full = max((len(v) for v in acc.values()), default=0)
    ranked = sorted(((sum(v) / len(v), p) for p, v in acc.items() if len(v) == full), reverse=True)
    return {p: (i + 1, len(ranked)) for i, (_, p) in enumerate(ranked)}


def pair_raw(p):
    s = (f"  원본: 평균 차이 {pp(p.get('mean_delta_pp'))}, 시드 {p.get('n_training_seeds')}개, "
         f"승/무/패 {p.get('wins')}/{p.get('ties')}/{p.get('losses')}")
    per = p.get("per_seed_delta_pp") or {}
    if per:
        s += ", 시드별 " + " ".join(f"s{k} {v:+.2f}" for k, v in sorted(per.items(), key=lambda kv: int(kv[0])))
    if p.get("ci_low") is not None:
        s += f", 95% 구간 [{p['ci_low']:+.2f}, {p['ci_high']:+.2f}]"
    if p.get("p_two_sided") is not None:
        s += f", p={p['p_two_sided']:.3f}"
    if p.get("sample_sd_delta_pp") is not None:
        s += f", 시드 간 표준편차 {p['sample_sd_delta_pp']:.2f}점"
    if p.get("missing_seeds"):
        s += f", 빠진 시드 {p['missing_seeds']}"
    if p.get("tau_mismatch_seeds"):
        s += f", 시점 불일치 시드 {p['tau_mismatch_seeds']}"
    return s


def pair_purpose(lab, where):
    x, _, y = lab.partition(" - ")
    y = y.split(" (")[0]
    if lab.startswith("E3 "):
        tau = lab.rsplit("(", 1)[-1].rstrip(")")
        return (f"E3. SR-GC 현재 규칙이 τ={int(tau)}에서 발동한 시드만 모아, 같은 부모 스냅샷에서 SR branch 보상과 "
                "Uniform branch 보상을 뺐다. 같은 시점에 바꿨을 때 목적지가 SR인 것이 Uniform보다 나은지 본다.")
    head = f"{where}에서 시드마다 [{ko.name(x)}] 보상 - [{ko.name(y)}] 보상을 계산했다. "
    core = y[:-4] if y.endswith("_c50") else y
    if x == "srgc_current" and y == "on_continued":
        why = "전환 자체가 도움이 되는지, 즉 SR-GC로 바꾼 것이 전환 없이 계속한 것보다 나은지 본다."
    elif x == "srgc_current" and y.startswith("fixed_sr_t") and y[-3:].isdigit():
        why = f"SR-GC가 고른 시점이 고정 시점 τ={int(y[-3:])}보다 나은지 본다."
    elif x == "srgc_current" and y == "fixed_sr_tuned":
        why = "새 시드에서, 개발 시드로 고른 고정 시점 전환보다 SR-GC가 나은지 본다. 미리 정한 확인 비교다."
    elif y.startswith("plateau_sr"):
        why = "평가 데이터를 안 보는 학습 보상 정체 트리거와 비교한다. 싼 트리거로도 같은 결과가 나오는지 본다."
    elif y == "srgc_uniform_destination":
        why = "같은 시점에 Uniform으로 바꾼 것과 비교한다. 이득이 SR이라는 목적지 덕인지 본다."
    elif y.endswith("_c50"):
        why = "검사 간격을 25에서 50으로 늘리면 결과가 얼마나 바뀌는지 본다. 검사 비용을 줄여도 되는지와 관련된다."
    elif core.startswith("srgc_") or x.startswith("srgc_") and x != "srgc_current":
        why = "전환 규칙을 바꿨을 때 결과가 얼마나 민감한지 본다(규칙 민감도)."
    elif y.startswith("sampled_time"):
        why = "무작위로 뽑은 시점과 비교한다. SR-GC의 시점 선택에 정보가 있는지 본다."
    elif x.startswith("sr_early_refreshed_k"):
        why = "SR 캐시를 주기적으로 현재 모델로 다시 만드는 것이 도움이 되는지 본다."
    elif y in ("sr_early", "uniform_early") or y.startswith("sr_early"):
        why = "처음부터 한 선택기로만 학습한 것과 비교한다. 중간 전환이 필요한지 본다."
    else:
        why = "두 정책의 보상 차이를 본다."
    return head + why


def pair_result(p):
    per = p.get("per_seed_delta_pp") or {}
    s = f"  결과: {pp(p.get('mean_delta_pp'))}, {ko.verdict(p)}."
    if len(per) >= 2:
        hi = max(per.items(), key=lambda kv: kv[1])
        lo = min(per.items(), key=lambda kv: kv[1])
        if hi[1] != lo[1]:
            s += f" 시드별로는 s{lo[0]} {lo[1]:+.2f}점부터 s{hi[0]} {hi[1]:+.2f}점까지다."
    if p.get("exploratory"):
        s += " 분석 설계에서 탐색용(exploratory)으로 정한 비교다."
    return s


def contrast(title, c, where, out):
    out.append(f"== {title}")
    if not isinstance(c, dict) or "primary_contrast" not in c:
        out.append(f"  원본: {c.get('status') if isinstance(c, dict) else '없음'}")
        out.append("  결과: 계산할 행이 없어 비어 있다.")
        return
    pc = c["primary_contrast"]
    out.append(pair_raw(pc))
    nt = pc.get("naive_paired_t") or {}
    if nt.get("ci_low") is not None:
        out.append(f"  원본: 낙관적 paired t 95% 구간 [{nt['ci_low']:+.2f}, {nt['ci_high']:+.2f}]"
                   + (f", p={nt['p_two_sided']:.3f}" if nt.get("p_two_sided") is not None else ""))
    lo = c.get("loso") or {}
    hats = {k: v.get("tau_hat") for k, v in (lo.get("per_seed") or {}).items()}
    gb = (lo.get("grid_best_same_seeds") or {}).get("tau")
    out.append(f"  원본: SR-GC - 시드별 사후 최적 τ {pp((c.get('minus_oracle') or {}).get('mean_delta_pp'))}, "
               f"SR-GC - 같은 시드 전체 최적 고정 τ({gb}) {pp((c.get('minus_grid_best') or {}).get('mean_delta_pp'))}")
    out.append("  원본: LOSO가 고른 τ " + (" ".join(f"s{k} {'없음' if v is None else v}" for k, v in sorted(hats.items(), key=lambda kv: int(kv[0])))
                                        or "-") + f", 시드마다 달라짐={lo.get('tau_hat_unstable')}")
    out.append(f"  무슨 비교: {where}에서 SR-GC(현재 규칙)와 고정 시점 SR 전환을 비교했다. 고정 시점은 LOSO로 정했다. "
               "시드 하나를 빼고, 나머지 시드에서 평균 보상이 가장 높은 τ를 골라, 뺀 시드에 적용한다. "
               "그래서 그 시드의 결과를 보고 τ를 고르지 않는다. '사후 최적 τ'는 시드마다 결과를 보고 가장 좋은 τ를 "
               "고른 상한이고, '전체 최적 고정 τ'는 같은 시드 전체에서 가장 좋은 τ 하나라서 낙관적인 기준이다.")
    lines = [pair_result(pc)]
    known = [v for v in hats.values() if v is not None]
    if len(set(known)) == 1 and len(known) == len(hats):
        lines.append(f"    LOSO는 모든 시드에서 τ={known[0]}을 골랐다. 고정 시점 선택이 시드에 따라 흔들리지 않는다.")
    elif len(set(known)) > 1:
        lines.append(f"    LOSO가 고른 τ가 시드마다 다르다. 조정 결과가 시드에 따라 흔들린다.")
    lines.append("    구간은 LOSO 폴드를 독립으로 본 낙관적 값이다. 같은 시드가 조정과 평가에 함께 쓰여 실제 불확실성은 더 넓다.")
    out.extend(lines)


def d_block(dobs, rows, out):
    out.append("[시드별 SR-GC 신호 D 원본]")
    out.append("  무슨 값: On-policy 궤적에서 25 업데이트마다 잰 SR-GC 신호다. 40문항 두 세트로 gradient 정렬을 비교한 값이고, "
               "음수면 SR 쪽이 유리하다는 뜻이다. 규칙은 이 값만 보고 전환 시점을 정한다.")
    if not dobs:
        out.append("  원본: 기록 없음.")
        return
    first = {r["seed"]: r.get("switch_update") for r in rows if r.get("policy_id") == "srgc_current"
             and r.get("status") in GOOD}
    for seed, obs in sorted(dobs.items()):
        if obs is None:
            out.append(f"  s{seed}: 궤적 기록을 읽지 못함.")
            continue
        vals = " ".join(f"{u}:{'무효' if v is None else f'{v:+.4f}'}" for u, v in obs)
        out.append(f"  s{seed}: {vals or '아직 검사 없음'}")
        neg = [u for u, v in obs if v is not None and v < 0]
        out.append(f"    결과: 검사 {len(obs)}회 중 음수 {len(neg)}회"
                   + (f", 처음 음수는 update {neg[0]}" if neg else "")
                   + (f", 현재 규칙 전환 update {first[seed]}" if first.get(seed) is not None else
                      (", 현재 규칙은 발동 안 함" if seed in first else "")) + ".")


def report(cohort, cfg, adir, out, *, dobs=None, cold=None, missing=None, diff=None):
    adir = Path(adir)
    a = json.loads((adir / "analysis.json").read_text())
    rows_by = {}
    for role, stem in ROLES:
        p = adir / f"policy_outcomes_{stem}.json"
        rows_by[role] = json.loads(p.read_text()) if p.is_file() else []
    bank = rows_by["rebuttal_bank"]
    ep = a.get("endpoints") or {}
    H = ep.get("matched_update")
    b = ep.get("budget") or {}
    m = (cfg or {}).get("model") or {}
    t = (cfg or {}).get("training") or {}
    seeds = ((cfg or {}).get("seeds") or {})

    out.append("[이 cohort는 무슨 실험인가]")
    out.append(f"  원본: 모델 {m.get('repository')} (어댑터 {m.get('adapter')}), 데이터 {(cfg or {}).get('dataset')}, "
               f"학습 시드 {seeds.get('training')}, prefix {t.get('prefix_updates')} 업데이트, H={H}, "
               f"검사 간격 {t.get('check_interval')}, 업데이트당 문항 {t.get('prompts_per_update')}개×응답 {t.get('responses_per_prompt')}개, "
               f"목적 {t.get('objective')}")
    out.append(f"  원본: 고정 예산 {b.get('value') if b.get('value') is None else round(b['value'], 2)} {b.get('unit')} (= {hrs(b.get('value'))} GPU), 범위 {b.get('scope')}, "
               f"시드 {b.get('n_seeds')}개로 정함" if b else "  원본: 고정 예산 없음")
    out.append("  설명: 모든 런은 같은 공유 prefix에서 시작한다. 질문은 On-policy로 학습하다가 언제 cached SR 선택기로 "
               "바꾸는 게 좋은지다. SR-GC는 학습 중 신호 D를 보고 시점을 정하는 방법이고, 고정 시점 전환·학습 보상 정체 "
               "트리거·무작위 시점·전환 없음과 비교한다. 보상은 평가 문항 정답률이고, 차이는 퍼센트포인트(점)다.")
    out.append("  설명: 비교는 두 가지 끝점에서 한다. 같은 업데이트는 모두 H 업데이트에서 잰다. 같은 예산은 GPU 시간을 맞춘다. "
               "예산은 평가 결과를 보기 전에 SR-GC 현재 규칙이 H까지 쓴 비용의 시드 평균으로 고정했고, 정책마다 그 예산 안에서 "
               "마지막으로 저장된 체크포인트(25 업데이트 간격)를 쓴다. 싼 정책은 H 뒤로 연장 학습해 예산을 채운다.")
    if b and "cold-start incomplete" in str(b.get("scope")):
        out.append("  설명: 이 예산은 이어 학습 비용만으로 정했다. prefix와 SR 캐시 비용 기록이 다 모이지 않았기 때문이다.")
    out.append("")
    out.append("[판정]")
    out.extend("  " + x for x in ko.readiness(a, bank))
    if diff is not None:
        out.append(f"  재집계: 큐가 옛 코드로 낸 값과 현재 코드 값이 다른 행 {diff[0]}개 / 전체 {diff[1]}개. "
                   "아래 값은 전부 현재 코드 값이다. 바뀐 행 상세는 ~/v8-recount.txt에 있다.")
    out.append("")

    d_block(dobs or {}, bank, out)
    out.append("")

    for role, _ in ROLES:
        rs = rows_by[role]
        if not rs:
            continue
        out.append(f"[정책별 원본 행: {role}]")
        out.append(f"  {ROLE_KO[role]}")
        ranks = rank_table(rs)
        by = {}
        for r in rs:
            by.setdefault(r.get("policy_id"), []).append(r)
        order = sorted(by, key=lambda p: (p != "on_continued", not str(p).startswith("srgc_current"), str(p)))
        for pid in order:
            prs = sorted(by[pid], key=lambda r: (r.get("seed") is None, r.get("seed") or 0))
            out.append(f"== {pid} : {ko.name(pid)}")
            out.append(f"  무슨 실험: {describe(pid, role, H)}")
            out.extend(row_line(r, H) for r in prs)
            out.extend(policy_result(pid, prs, rs, H, ranks.get(pid)))
        out.append("")

    out.append("[비교 원본과 결과]")
    out.append("  모든 차이는 앞 정책 보상에서 뒤 정책 보상을 뺀 값이다. 단위는 점(퍼센트포인트). 승/무/패는 시드 단위다.")
    contrast("주 비교, 같은 업데이트: SR-GC - LOSO로 고른 고정 시점", a.get("primary_rebuttal"), f"같은 업데이트(H={H})", out)
    contrast("주 비교, 같은 예산: SR-GC - LOSO로 고른 고정 시점", a.get("primary_rebuttal_budget"),
             f"같은 GPU 예산({hrs(b.get('value'))})", out)
    pb = (a.get("primary_rebuttal_budget") or {}).get("primary_contrast")
    if pb:
        out.append("    리뷰 결론 후보(P9): " + ko.p9(pb))
    ext = a.get("primary_rebuttal_extended")
    if isinstance(ext, dict) and "primary_contrast" in ext:
        contrast("확장 τ 격자(25~250 전부)로 같은 비교", ext, f"같은 업데이트(H={H})", out)
    for k, where in (("primary_test", "같은 업데이트"), ("primary_test_budget", "같은 예산")):
        if k in a:
            p = a[k]
            out.append(f"== 새 시드 확인 비교({where}): {p.get('label')}")
            out.append(pair_raw(p))
            out.append("  무슨 비교: " + pair_purpose("srgc_current - fixed_sr_tuned", where))
            out.append(pair_result(p) if p.get("n_training_seeds") else "  결과: 아직 결과가 없다.")
    for key, where in (("secondary", "같은 업데이트"), ("secondary_budget", "같은 예산"),
                       ("secondary_test", "새 시드, 같은 업데이트"), ("secondary_test_budget", "새 시드, 같은 예산")):
        for p in a.get(key) or []:
            out.append(f"== 보조 비교({where}): {p.get('label')}")
            if p.get("status") == "missing_policy":
                out.append(f"  원본: 정책 행 없음({', '.join(p.get('missing') or [])})")
                out.append("  결과: 그 정책의 행이 없어 계산하지 못했다.")
                continue
            out.append(pair_raw(p))
            out.append("  무슨 비교: " + pair_purpose(p.get("label") or "", where))
            out.append(pair_result(p))
    e3 = a.get("e3") or []
    if e3:
        out.append("== E3 시드별 원본: 같은 부모 상태에서 SR branch 대 Uniform branch")
        for r in e3:
            if r.get("delta_pp") is None:
                out.append(f"  s{r.get('seed')}: {r.get('status')}")
            else:
                out.append(f"  s{r['seed']}: τ={r.get('tau_s')}, SR {pct(r.get('sr'))}, Uniform {pct(r.get('uniform'))}, "
                           f"차이 {r['delta_pp']:+.2f}점")
        dd = [r["delta_pp"] for r in e3 if r.get("delta_pp") is not None]
        if dd:
            out.append(f"  결과: 시드 {len(dd)}개에서 SR이 {sum(x > 0 for x in dd)}번 높고 {sum(x < 0 for x in dd)}번 낮다. "
                       f"평균 {pp(mean(dd))}.")
    out.append("")

    out.append("[비용 원본]")
    for role, _ in ROLES:
        acc = {}
        for r in rows_by[role]:
            if r.get("continuation_gpu_seconds") is not None:
                acc.setdefault(r["policy_id"], []).append(r["continuation_gpu_seconds"])
        if acc:
            out.append(f"  {role}: H까지 이어 학습 GPU 시간 시드 평균(싼 순서): "
                       + ", ".join(f"{p} {hrs(sum(v) / len(v))}(n={len(v)})"
                                   for p, v in sorted(acc.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))))
    out.append("  설명: 이어 학습 비용은 prefix 뒤부터 H까지 학습·검사·캐시 갱신에 쓴 GPU 시간이다. SR-GC는 검사 비용이 "
               "더해지므로, 같은 예산에서는 고정 시점 전환보다 적은 업데이트에서 끝날 수 있다.")
    if cold is not None:
        for s, pre, cache, d in cold:
            lb = d.get("known_lower_bound_gpu_seconds")
            out.append(f"  s{s}: prefix {hrs(pre) if pre is not None else '모름'}, 첫 SR 캐시 "
                       + (hrs(cache) if cache is not None else f"모름({d.get('status')}"
                          + (f", 확인된 것만 {hrs(lb)} 이상" if lb else "") + ")"))
        try:
            import v8_recount
            out.extend("  " + x for x in v8_recount.cold_start_ko(cold, bank, missing or {}))
        except Exception as e:  # noqa: BLE001
            out.append(f"  (cold-start 해석 실패: {type(e).__name__}: {e})")
    pe = a.get("physical_expenditure") or {}
    if pe:
        g = pe.get("gpu_seconds")
        out.append(f"  원본: 클러스터 전체 GPU 합계 {hrs(g)}, 완전={pe.get('complete')}, 빠진 기록 {pe.get('missing_events')}개, "
                   f"중간에 죽은 단계 {pe.get('killed_phase_starts')}")
        out.append("  설명: 실제로 돈 모든 작업(실패·재시도 포함)의 GPU 시간 합이다."
                   + (" 기록이 빠져 있어 합계를 인용하면 안 된다." if not pe.get("complete") else ""))
    out.append("")

    out.append("[종합 해석]")
    tmp = []
    ko.explain(a, bank, tmp)
    out.extend("  " + x for x in tmp[1:])
