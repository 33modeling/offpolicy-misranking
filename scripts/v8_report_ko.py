"""V8 결과 보고서(한글). 원본 값을 표로 전부 적고, 항목마다 바로 아래에 무슨 실험인지와 결과를 단다.

v8_recount.py가 cohort마다 report()를 부른다. 표준 라이브러리만 쓴다. 모든 문장은 같은 블록의 원본 값에서
계산한다. 지난 실행에 대한 고정 문구는 없다. 폰 화면에서 읽도록 줄을 짧게 하고 표는 글자 폭을 맞춘다.
"""
import json
import unicodedata
from pathlib import Path

import v8_explain_ko as ko

GOOD = ko.GOOD
ROLES = (("rebuttal_bank", "bank"), ("development", "development"), ("test", "test"))
ROLE_TITLE = {
    "rebuttal_bank": "학습 시드 bank",
    "development": "개발 시드",
    "test": "새 시드 검증",
}
ROLE_KO = {
    "rebuttal_bank": ["시드마다 On-policy 궤적 하나와, 그 궤적의 스냅샷에서 갈라진 branch들을 학습해 두었다.",
                      "정책마다 그 정책이 골랐을 시점의 branch 결과를 그대로 찾아 쓴다. 보간은 하지 않는다."],
    "development": ["고정 τ, ε 같은 조정값을 고르는 데만 쓰는 시드다. dev split으로 평가했다."],
    "test": ["개발 시드에서 조정값을 고정한 뒤, 처음 보는 시드에서 정책을 처음부터 실제로 실행했다."],
}
STATE = {  # 표에 쓰는 짧은 상태 이름
    "triggered": "전환함", "not_triggered": "전환 안 함", "ran": "실행 완료",
    "censored": "기록 짧음", "unavailable": "branch 미완", "missing_trajectory": "궤적 없음",
}
STATE_LONG = {
    "censored": "기준 궤적의 검사 기록이 마지막 예정 검사 전에 끝나서, 규칙이 발동했을지 알 수 없다",
    "unavailable": "규칙이 고른 시점의 branch 학습이 아직 끝나지 않았다",
    "missing_trajectory": "기준 On-policy 궤적이 아직 없다",
}
GROUPS = (
    ("기준: 전환 없음", ["전환 없이 On-policy로 H까지 학습했다. 다른 정책은 모두 이것과 비교된다."]),
    ("SR-GC 규칙", ["학습 중 25 업데이트마다 신호 D를 재고, 규칙이 발동하면 cached SR로 바꾼다.",
                    "bank에서는 궤적에 기록된 D를 규칙으로 다시 읽어, 처음 발동한 시점의 SR branch 결과를 쓴다.",
                    "발동하지 않으면 전환 없음 결과를 쓴다. D 검사 비용이 비용에 들어간다."]),
    ("고정 시점 SR 전환", ["미리 정한 update τ에서 cached SR로 바꿔 H까지 학습했다. 검사 비용이 없다."]),
    ("학습 보상 정체 트리거", ["평가 데이터 없이 학습 배치 보상만 본다. 25 업데이트마다 최근 25 업데이트 평균 보상에서",
                            "그 전 25 업데이트 평균을 뺀 개선폭이 ε 이하인 검사가 2번 연속이면 SR로 바꾼다.",
                            "첫 검사는 update 50이다."]),
    ("Uniform 대조군", ["SR 대신 Uniform(무작위 선택)으로 바꾼 것이다. SR-GC의 이득이 SR이라는 목적지 덕인지 가른다."]),
    ("처음부터 한 선택기", ["prefix 직후부터 전환 없이 한 선택기로만 학습했다."]),
    ("무작위 시점 대조군", ["개발 시드의 SR-GC 전환 시점 분포에서 학습 전에 한 번 뽑은 시점에 SR로 바꿨다.",
                         "SR-GC가 고른 시점에 정보가 있는지 본다."]),
    ("기타", []),
)
RULE_KO = {
    "current": "현재 규칙. D가 두 번 연속 음수이거나, 음수·0 이상·음수 순서에 세 값 합이 음수이면 전환",
    "one_negative": "음성 1회 규칙. D가 한 번 음수면 바로 전환",
    "two_negative": "음성 2회 규칙. D가 두 번 연속 음수면 전환",
    "three_negative": "음성 3회 규칙. D가 세 번 연속 음수면 전환",
}


# ---------- 형식 도우미 ----------

def width(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def pad(s, w, right=False):
    s = str(s)
    gap = " " * max(0, w - width(s))
    return gap + s if right else s + gap


def table(head, rows, indent="  ", right=()):
    """글자 폭을 맞춘 표. right: 오른쪽 정렬할 열 번호."""
    ws = [max(width(x) for x in col) for col in zip(head, *rows)]
    line = lambda r: indent + "  ".join(pad(x, ws[i], i in right) for i, x in enumerate(r)).rstrip()
    return [line(head)] + [line(r) for r in rows]


def pct(v):
    return "-" if v is None else f"{100 * v:.2f}%"


def hrs(v):
    if v is None:
        return "-"
    return f"{v / 3600:.2f}h" if abs(v) >= 36 else f"{v:.2f}초"


def pp(v):
    return "-" if v is None else f"{v:+.2f}점"


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def seeds_txt(ss):
    return ", ".join(f"s{s}" for s in ss)


def short(pid):
    """비교 제목용 이름. srgc_current는 그냥 SR-GC."""
    return "SR-GC" if pid == "srgc_current" else ko.name(pid)


# ---------- 정책 ----------

def group_of(pid):
    core = pid.split(".")[0]
    core = core[:-4] if core.endswith("_c50") else core
    if core == "on_continued":
        return 0
    if core == "srgc_uniform_destination" or core.startswith("fixed_uniform") or core == "uniform_early":
        return 4
    if core.startswith("srgc_"):
        return 1
    if core.startswith("fixed_sr"):
        return 2
    if core.startswith("plateau"):
        return 3
    if core.startswith("sr_early"):
        return 5
    if core.startswith("sampled_time"):
        return 6
    return 7


def one_line(pid, role):
    """정책 하나의 차이점만 한 줄로. 공통 설명은 묶음 머리에 있다."""
    core = pid.split(".")[0]
    c50 = core.endswith("_c50")
    core = core[:-4] if c50 else core
    fresh = role != "rebuttal_bank"
    if core == "on_continued":
        return "새 시드에서 처음부터 따로 학습한 런이다." if fresh else \
            "공유 prefix 뒤 이어간 기준 궤적이다. D를 기록했고 branch는 모두 여기서 갈라진다."
    if core.startswith("srgc_") and core != "srgc_uniform_destination":
        s = RULE_KO.get(core[5:], core[5:]) + "."
        if c50:
            s += " 검사를 50 업데이트마다만 했다고 보고 다시 읽었다."
        if fresh:
            s += " 실제로 SR-GC를 켜고 처음부터 학습했다."
        return s
    if core.startswith("fixed_sr_t") and core[-3:].isdigit():
        return f"update {int(core[-3:])}에서 전환."
    if core == "fixed_sr_tuned":
        return "개발 시드에서 고른 τ로 전환. 새 시드를 보고 다시 조정하지 않았다."
    if core.startswith("plateau_sr_eps"):
        eps = int(core[-4:]) / 1000
        return f"ε={eps:g}, 즉 개선폭이 {100 * eps:g}점 이하이면 정체로 본다."
    if core == "plateau_sr_tuned":
        return "개발 시드에서 고른 ε를 썼다."
    if core == "srgc_uniform_destination":
        return "SR-GC 현재 규칙이 발동한 시점에 Uniform으로 바꿨다. 검사 비용 포함."
    if core.startswith("fixed_uniform_t"):
        return f"update {int(core[-3:])}에서 Uniform으로 바꿨다. 그 시드의 SR-GC 전환 시점과 같다. 검사 비용 없음."
    if core == "uniform_early":
        return "처음부터 Uniform."
    if core.startswith("sr_early_refreshed_k"):
        return f"처음부터 SR, SR 캐시를 {int(core[-3:])} 업데이트마다 현재 모델로 다시 만들었다."
    if core == "sr_early":
        return "처음부터 cached SR."
    if core.startswith("sampled_time"):
        return "뽑힌 시점에 전환."
    return "설명이 등록되지 않은 정책이다."


def state_of(r):
    st = r.get("status")
    if st in GOOD and r.get("reward_fraction") is None:
        return "평가 대기"
    return STATE.get(st, str(st))


def policy_rows_table(rs):
    head = ["시드", "상태", "전환", "보상@H", "이어학습", "예산끝", "예산보상"]
    body = []
    for r in rs:
        good = r.get("status") in GOOD
        sw = r.get("switch_update")
        bs = r.get("budget_status")
        if not good or not bs:
            bend, brew = "-", "-"
        elif bs == "capped_at_horizon":
            bend, brew = "연장필요", "-"
        elif bs == "selected":
            bend = str(r.get("budget_endpoint_update"))
            brew = pct(r["budget_reward_fraction"]) if r.get("budget_reward_fraction") is not None else "평가대기"
        else:
            bend, brew = str(bs), "-"
        body.append([f"s{r.get('seed')}", state_of(r), "-" if sw is None else str(sw),
                     pct(r.get("reward_fraction")) if good else "-",
                     hrs(r.get("continuation_gpu_seconds")) if good else "-", bend, brew])
    return table(head, body, indent="    ", right=(2, 3, 4, 5, 6))


def policy_result(pid, rs, all_rows, ranks):
    good = [r for r in rs if r.get("status") in GOOD]
    meas = [r for r in good if r.get("reward_fraction") is not None]
    out = []
    if meas:
        v = [r["reward_fraction"] for r in meas]
        s = f"평균 보상 {pct(mean(v))}(시드 {len(meas)}개)"
        if ranks:
            s += f", 시드가 다 있는 {ranks[1]}개 정책 중 {ranks[0]}위"
        out.append(s + ".")
        on = {r["seed"]: r["reward_fraction"] for r in all_rows if r.get("policy_id") == "on_continued"
              and r.get("status") in GOOD and r.get("reward_fraction") is not None}
        d = [100 * (r["reward_fraction"] - on[r["seed"]]) for r in meas if r["seed"] in on]
        if d and pid != "on_continued":
            out.append(f"전환 없음보다 평균 {pp(mean(d))}. "
                       f"{sum(x > 0 for x in d)}개 시드 높고, {sum(x < 0 for x in d)}개 낮고, {sum(x == 0 for x in d)}개 같다.")
    bud = [r for r in good if r.get("budget_status") == "selected" and r.get("budget_reward_fraction") is not None]
    if bud:
        out.append(f"같은 예산 평균 보상 {pct(mean([r['budget_reward_fraction'] for r in bud]))}(시드 {len(bud)}개).")
    wait = [r["seed"] for r in good if r.get("reward_fraction") is None]
    bwait = [r["seed"] for r in good if r.get("budget_status") == "selected" and r.get("budget_reward_fraction") is None]
    capped = [r["seed"] for r in good if r.get("budget_status") == "capped_at_horizon"]
    bad = [r for r in rs if r.get("status") not in GOOD]
    if wait:
        out.append(f"{seeds_txt(wait)}는 학습이 끝났고 H 평가가 남았다.")
    if bwait:
        out.append(f"{seeds_txt(bwait)}는 예산 끝점 체크포인트의 평가가 남았다.")
    if capped:
        out.append(f"{seeds_txt(capped)}는 예산이 H 뒤까지 남는데 연장 학습이 아직 없다.")
    for st in sorted({r.get("status") for r in bad}, key=str):
        ss = [r["seed"] for r in bad if r.get("status") == st]
        out.append(f"{seeds_txt(ss)}는 못 쓴다. {STATE_LONG.get(st, '분류되지 않은 상태다')}.")
    if not out:
        out.append("쓸 수 있는 값이 아직 없다.")
    return ["    결과: " + out[0]] + ["          " + x for x in out[1:]]


def rank_table(rows):
    acc = {}
    for r in rows:
        if r.get("status") in GOOD and r.get("reward_fraction") is not None:
            acc.setdefault(r["policy_id"], []).append(r["reward_fraction"])
    full = max((len(v) for v in acc.values()), default=0)
    ranked = sorted(((sum(v) / len(v), p) for p, v in acc.items() if len(v) == full), reverse=True)
    return {p: (i + 1, len(ranked)) for i, (_, p) in enumerate(ranked)}


def role_section(role, rs, H, out):
    out.append(f"■ 정책별 원본 값: {ROLE_TITLE[role]}")
    out.extend("  " + x for x in ROLE_KO[role])
    nq = next((r for r in rs if r.get("n_questions")), None)
    if nq:
        out.append(f"  평가는 {nq['n_questions']}문항에 문항당 {nq.get('responses_per_question')}개 답을 뽑은 정답률이다.")
    out.append("  표의 '보상@H'는 H 업데이트의 보상, '이어학습'은 prefix 뒤 H까지 쓴 GPU 시간,")
    out.append("  '예산끝'은 같은 예산으로 끝나는 update, '예산보상'은 그때 보상이다.")
    ranks = rank_table(rs)
    by = {}
    for r in rs:
        by.setdefault(r.get("policy_id"), []).append(r)
    order = sorted(by, key=lambda p: (group_of(str(p)), str(p) != "srgc_current", str(p)))
    last = None
    for pid in order:
        g = group_of(str(pid))
        if g != last:
            title, lines = GROUPS[g]
            out.append("")
            out.append(f"  [{title}]")
            out.extend("  " + x for x in lines)
            last = g
        prs = sorted(by[pid], key=lambda r: (r.get("seed") is None, r.get("seed") or 0))
        out.append("")
        out.append(f"  ● {ko.name(pid)}  ({pid})")
        out.append(f"    {one_line(str(pid), role)}")
        out.extend(policy_rows_table(prs))
        out.extend(policy_result(pid, prs, rs, ranks.get(pid)))
    out.append("")


# ---------- 비교 ----------

def purpose(x, y, label):
    if label.startswith("E3 "):
        return "같은 부모 상태에서 SR로 바꾼 것과 Uniform으로 바꾼 것 중 어느 쪽이 나은지"
    core = y[:-4] if y.endswith("_c50") else y
    if y == "on_continued":
        return "전환 자체가 도움이 되는지"
    if y.startswith("fixed_sr_t") and y[-3:].isdigit():
        return f"SR-GC가 고른 시점이 고정 시점 τ={int(y[-3:])}보다 나은지"
    if y == "fixed_sr_tuned":
        return "처음 보는 시드에서도 SR-GC가 조정된 고정 시점 전환보다 나은지(미리 정한 확인 비교)"
    if y.startswith("plateau_sr"):
        return "평가 데이터를 안 보는 싼 트리거로도 같은 결과가 나오는지"
    if y == "srgc_uniform_destination":
        return "이득이 SR이라는 목적지 덕인지(같은 시점에 Uniform으로 바꾼 것과 비교)"
    if y.endswith("_c50"):
        return "검사 간격을 25에서 50으로 늘려도 결과가 같은지"
    if x.startswith("sr_early_refreshed_k"):
        return "SR 캐시를 주기적으로 다시 만드는 것이 도움이 되는지"
    if core.startswith("srgc_"):
        return "전환 규칙을 바꾸면 결과가 얼마나 달라지는지"
    if y.startswith("sampled_time"):
        return "SR-GC가 고른 시점이 무작위 시점보다 나은지"
    if y in ("sr_early", "uniform_early"):
        return "중간 전환이 처음부터 한 선택기로 학습하는 것보다 나은지"
    return "두 정책 중 어느 쪽 보상이 높은지"


def pair_lines(p, title, why):
    out = [f"  ● {title}", f"    보려는 것: {why}"]
    if p.get("status") == "missing_policy":
        out.append(f"    결과: 정책 행이 없어 계산하지 못했다({', '.join(p.get('missing') or [])}).")
        return out
    per = p.get("per_seed_delta_pp") or {}
    if per:
        out.append("    시드별 차이: " + ", ".join(f"s{k} {v:+.2f}" for k, v in sorted(per.items(), key=lambda kv: int(kv[0]))))
    if p.get("missing_seeds"):
        out.append(f"    빠진 시드: {', '.join(str(s) for s in p['missing_seeds'])}")
    s = f"    평균 {pp(p.get('mean_delta_pp'))}, {p.get('wins')}승 {p.get('ties')}무 {p.get('losses')}패"
    if p.get("ci_low") is not None:
        s += f", 95% 구간 {p['ci_low']:+.2f} ~ {p['ci_high']:+.2f}"
    if p.get("p_two_sided") is not None:
        s += f", p={p['p_two_sided']:.3f}"
    out.append(s)
    out.append(f"    결과: {ko.verdict(p)}.")
    return out


def contrast(title, c, out):
    out.append(f"  ● {title}")
    if not isinstance(c, dict) or "primary_contrast" not in c:
        out.append(f"    결과: 계산할 행이 없다({c.get('status') if isinstance(c, dict) else '없음'}).")
        return
    pc = c["primary_contrast"]
    per = pc.get("per_seed_delta_pp") or {}
    lo = c.get("loso") or {}
    hats = {k: v.get("tau_hat") for k, v in (lo.get("per_seed") or {}).items()}
    gb = (lo.get("grid_best_same_seeds") or {}).get("tau")
    if per:
        out.append("    시드별 차이: " + ", ".join(f"s{k} {v:+.2f}" for k, v in sorted(per.items(), key=lambda kv: int(kv[0]))))
    if hats:
        out.append("    LOSO가 고른 τ: " + ", ".join(f"s{k} {'없음' if v is None else v}"
                                                for k, v in sorted(hats.items(), key=lambda kv: int(kv[0]))))
    if pc.get("missing_seeds"):
        out.append(f"    빠진 시드: {', '.join(str(s) for s in pc['missing_seeds'])}")
    out.append(f"    평균 {pp(pc.get('mean_delta_pp'))}, {pc.get('wins')}승 {pc.get('ties')}무 {pc.get('losses')}패")
    nt = pc.get("naive_paired_t") or {}
    if nt.get("ci_low") is not None:
        out.append(f"    낙관적 95% 구간 {nt['ci_low']:+.2f} ~ {nt['ci_high']:+.2f}"
                   + (f", p={nt['p_two_sided']:.3f}" if nt.get("p_two_sided") is not None else ""))
    out.append(f"    참고: 시드별 사후 최적 τ 대비 {pp((c.get('minus_oracle') or {}).get('mean_delta_pp'))}, "
               f"전체 최적 고정 τ={gb} 대비 {pp((c.get('minus_grid_best') or {}).get('mean_delta_pp'))}")
    out.append(f"    결과: {ko.verdict(pc, nt)}.")
    known = [v for v in hats.values() if v is not None]
    if known and len(set(known)) == 1 and len(known) == len(hats):
        out.append(f"          LOSO는 모든 시드에서 τ={known[0]}을 골랐다. 고정 시점 조정이 시드에 따라 흔들리지 않는다.")
    elif len(set(known)) > 1:
        out.append("          LOSO가 고른 τ가 시드마다 다르다. 고정 시점 조정이 시드에 따라 흔들린다.")


def comparisons(a, H, b, out):
    out.append("■ 비교 원본 값과 결과")
    out.append("  차이는 앞 정책 보상에서 뒤 정책 보상을 뺀 값이다. 단위는 점(퍼센트포인트)이다.")
    out.append("  승·무·패는 시드 단위다. 시드가 3개 미만이면 판단을 미룬다.")
    out.append("")
    out.append("  [주 비교: SR-GC 대 고정 시점 SR 전환]")
    out.append("  고정 시점은 LOSO로 골랐다. 시드 하나를 빼고 나머지 시드에서 평균 보상이 가장 높은 τ를 골라")
    out.append("  뺀 시드에 쓴다. 그 시드의 결과를 보고 τ를 고르지 않는다는 뜻이다.")
    out.append("  구간은 폴드를 독립으로 본 낙관적 값이다. 같은 시드가 조정과 평가에 함께 쓰여 실제로는 더 넓다.")
    out.append("  '사후 최적 τ'는 시드마다 결과를 보고 고른 상한, '전체 최적 고정 τ'는 같은 시드 전체에서 가장 좋은 τ다.")
    contrast(f"같은 업데이트(H={H})", a.get("primary_rebuttal"), out)
    contrast(f"같은 예산({hrs(b.get('value'))})", a.get("primary_rebuttal_budget"), out)
    ext = a.get("primary_rebuttal_extended")
    if isinstance(ext, dict) and "primary_contrast" in ext:
        contrast("τ 후보를 25~250 전부로 넓혔을 때(같은 업데이트)", ext, out)
    pb = (a.get("primary_rebuttal_budget") or {}).get("primary_contrast")
    if pb:
        out.append(f"  리뷰 결론 후보(P9): {ko.p9(pb)}")
    tests = [(k, w) for k, w in (("primary_test", "같은 업데이트"), ("primary_test_budget", "같은 예산")) if k in a]
    if tests:
        out.append("")
        out.append("  [새 시드 확인 비교: SR-GC 대 조정된 고정 시점]")
        for k, where in tests:
            p = a[k]
            if not p.get("n_training_seeds"):
                out.append(f"  ● {where}: 아직 결과가 없다.")
                continue
            out.extend(pair_lines(p, where, purpose("srgc_current", "fixed_sr_tuned", "")))
    for key, title in (("secondary", "보조 비교, 같은 업데이트"), ("secondary_budget", "보조 비교, 같은 예산"),
                       ("secondary_test", "새 시드 보조 비교, 같은 업데이트"),
                       ("secondary_test_budget", "새 시드 보조 비교, 같은 예산")):
        items = a.get(key) or []
        if not items:
            continue
        out.append("")
        out.append(f"  [{title}]")
        if all(p.get("exploratory") for p in items):
            out.append("  분석 설계에서 탐색용으로 정한 비교다. 주 결론의 근거로 쓰지 않는다.")
        for p in items:
            lab = p.get("label") or ""
            if lab.startswith("E3 "):
                tau = int(lab.rsplit("(", 1)[-1].rstrip(")"))
                out.extend(pair_lines(p, f"같은 부모 상태 SR 대 Uniform(τ={tau})", purpose("", "", lab)))
                continue
            x, _, y = lab.partition(" - ")
            y = y.split(" (")[0]
            out.extend(pair_lines(p, f"{short(x)} 대 {short(y)}", purpose(x, y, lab)))
    e3 = a.get("e3") or []
    if e3:
        out.append("")
        out.append("  [E3 시드별 원본: 같은 부모 상태에서 SR branch 대 Uniform branch]")
        body = [[f"s{r.get('seed')}", str(r.get("tau_s", "-")), pct(r.get("sr")), pct(r.get("uniform")),
                 "-" if r.get("delta_pp") is None else f"{r['delta_pp']:+.2f}"] for r in e3]
        out.extend(table(["시드", "τ", "SR", "Uniform", "차이"], body, indent="    ", right=(1, 2, 3, 4)))
        dd = [r["delta_pp"] for r in e3 if r.get("delta_pp") is not None]
        if dd:
            out.append(f"    결과: 시드 {len(dd)}개 중 SR이 {sum(x > 0 for x in dd)}번 높고 {sum(x < 0 for x in dd)}번 낮다. "
                       f"평균 {pp(mean(dd))}.")
    out.append("")


# ---------- 나머지 절 ----------

def d_section(dobs, rows, out):
    out.append("■ SR-GC 신호 D 원본 값")
    out.append("  On-policy 궤적에서 25 업데이트마다 잰 값이다. 40문항 두 세트로 gradient 정렬을 비교했다.")
    out.append("  음수면 SR 쪽이 유리하다는 뜻이다. SR-GC 규칙은 이 값만 보고 전환 시점을 정한다.")
    seeds = sorted(s for s, o in dobs.items() if o)
    if not seeds:
        out.append("  기록 없음.")
        out.append("")
        return
    vals = {s: dict(dobs[s]) for s in seeds}
    ups = sorted({u for s in seeds for u in vals[s]})
    cell = lambda v: "무효" if v is None else f"{v:+.4f}"
    body = [[str(u)] + [cell(vals[s][u]) if u in vals[s] else "" for s in seeds] for u in ups]
    first = {r["seed"]: r.get("switch_update") for r in rows if r.get("policy_id") == "srgc_current"}
    stat = {r["seed"]: r.get("status") for r in rows if r.get("policy_id") == "srgc_current"}
    body.append(["전환"] + [str(first[s]) if first.get(s) is not None else STATE.get(stat.get(s), "-") for s in seeds])
    out.extend(table(["update"] + [f"s{s}" for s in seeds], body, right=tuple(range(1, len(seeds) + 1))))
    out.append("  ('전환'은 SR-GC 현재 규칙이 전환한 update)")
    for s in seeds:
        obs = dobs[s]
        neg = [u for u, v in obs if v is not None and v < 0]
        out.append(f"  s{s}: 검사 {len(obs)}회 중 음수 {len(neg)}회" + (f", 처음 음수 update {neg[0]}" if neg else "") + ".")
    bad = [s for s, o in dobs.items() if o is None]
    if bad:
        out.append(f"  {seeds_txt(sorted(bad))}: 궤적 기록을 읽지 못했다.")
    out.append("")


def cost_section(rows_by, cold, missing, a, out):
    out.append("■ 비용 원본 값")
    out.append("  이어학습은 prefix 뒤부터 H까지 학습·검사·캐시 갱신에 쓴 GPU 시간이다.")
    out.append("  SR-GC는 D 검사 비용이 더해져서, 같은 예산이면 고정 시점 전환보다 일찍 끝날 수 있다.")
    for role, _ in ROLES:
        acc = {}
        for r in rows_by[role]:
            if r.get("continuation_gpu_seconds") is not None:
                acc.setdefault(r["policy_id"], []).append(r)
        if not acc:
            continue
        out.append("")
        out.append(f"  [{ROLE_TITLE[role]}] 정책별 시드 평균, 싼 순서")
        body = []
        for p, rs in sorted(acc.items(), key=lambda kv: mean([r["continuation_gpu_seconds"] for r in kv[1]])):
            cs = mean([r.get("cold_start_gpu_seconds") for r in rs])
            lb = mean([r.get("cold_start_lower_bound_gpu_seconds") for r in rs])
            miss = sorted({m for r in rs for m in (r.get("cost_missing") or "").split(",") if m})
            body.append([ko.name(p), str(len(rs)), hrs(mean([r["continuation_gpu_seconds"] for r in rs])),
                         hrs(cs) if cs is not None else (f"{hrs(lb)} 이상" if lb is not None else "-"),
                         ", ".join(miss) or "-"])
        out.extend(table(["정책", "시드", "이어학습", "cold-start", "빠진 비용"], body, indent="    ", right=(1, 2, 3)))
    if cold is not None:
        out.append("")
        out.append("  [시드별 cold-start 입력] 공유 prefix와 첫 SR 캐시 생성 비용")
        body = [[f"s{s}", hrs(pre) if pre is not None else "모름",
                 hrs(cache) if cache is not None else "모름",
                 "-" if cache is not None else (str(d.get("status") or "")
                                                + (f", {hrs(d['known_lower_bound_gpu_seconds'])} 이상"
                                                   if d.get("known_lower_bound_gpu_seconds") else ""))]
                for s, pre, cache, d in cold]
        out.extend(table(["시드", "prefix", "SR 캐시", "모르는 이유"], body, indent="    ", right=(1, 2)))
        full = [s for s, pre, cache, _ in cold if pre is not None and cache is not None]
        if len(full) == len(cold) and cold:
            out.append("    결과: 모든 시드의 cold-start 비용이 다 기록됐다. 리뷰 Q3 답변에 그대로 쓸 수 있다.")
        else:
            out.append(f"    결과: 시드 {len(cold)}개 중 {len(full)}개만 다 기록됐다. "
                       "빈 칸은 0으로 채우지 말고 하한으로 적어야 한다.")
    pe = a.get("physical_expenditure") or {}
    if pe:
        out.append("")
        out.append(f"  [클러스터 전체] 실패·재시도까지 포함해 실제로 쓴 GPU 시간 {hrs(pe.get('gpu_seconds'))}")
        out.append(f"    빠진 기록 {pe.get('missing_events')}개, 중간에 죽은 단계 {pe.get('killed_phase_starts') if pe.get('killed_phase_starts') is not None else '-'}개."
                   + (" 기록이 빠져 있어 이 합계는 인용하면 안 된다." if not pe.get("complete") else " 기록이 완전하다."))
    out.append("")


def glance(a, bank, H, b, out):
    out.append("■ 한눈에 보기")
    out.extend("  " + x for x in ko.readiness(a, bank))
    for k, where in (("primary_rebuttal", f"같은 업데이트(H={H})"), ("primary_rebuttal_budget", "같은 예산")):
        c = (a.get(k) or {}).get("primary_contrast")
        if c:
            out.append(f"  SR-GC 대 고정 시점, {where}: {pp(c.get('mean_delta_pp'))}, {ko.verdict(c, c.get('naive_paired_t'))}.")
    for k, where in (("primary_test", "같은 업데이트"), ("primary_test_budget", "같은 예산")):
        p = a.get(k)
        if p and p.get("n_training_seeds"):
            out.append(f"  새 시드 확인, {where}: {pp(p.get('mean_delta_pp'))}, {ko.verdict(p)}.")
    pb = (a.get("primary_rebuttal_budget") or {}).get("primary_contrast")
    if pb:
        out.append(f"  리뷰 결론 후보(P9): {ko.p9(pb)}")
    acc = {}
    for r in bank:
        if r.get("status") in GOOD and r.get("reward_fraction") is not None:
            acc.setdefault(r["policy_id"], []).append(r["reward_fraction"])
    full = max((len(v) for v in acc.values()), default=0)
    ranked = sorted(((sum(v) / len(v), p) for p, v in acc.items() if len(v) == full), reverse=True)
    if ranked:
        out.append(f"  H 보상 상위 3개(시드 {full}개가 다 있는 정책): "
                   + ", ".join(f"{ko.name(p)} {pct(v)}" for v, p in ranked[:3]) + ".")
        pos = next((i + 1 for i, (_, p) in enumerate(ranked) if p == "srgc_current"), None)
        if pos:
            out.append(f"  SR-GC(현재 규칙)는 {len(ranked)}개 중 {pos}위.")
    out.append("")


def setup(cohort, cfg, a, b, diff, out):
    m = (cfg or {}).get("model") or {}
    t = (cfg or {}).get("training") or {}
    seeds = ((cfg or {}).get("seeds") or {}).get("training")
    H = (a.get("endpoints") or {}).get("matched_update")
    out.append("■ 실험 설정 원본 값")
    body = [["모델", str(m.get("repository"))], ["어댑터", str(m.get("adapter"))],
            ["데이터", str((cfg or {}).get("dataset"))], ["학습 시드", " ".join(str(s) for s in seeds or [])],
            ["학습 목적", str(t.get("objective"))], ["공유 prefix", f"{t.get('prefix_updates')} 업데이트"],
            ["끝점 H", f"{H} 업데이트"], ["D 검사 간격", f"{t.get('check_interval')} 업데이트"],
            ["업데이트당", f"문항 {t.get('prompts_per_update')}개 × 답 {t.get('responses_per_prompt')}개"]]
    if b:
        body.append(["고정 예산", f"{hrs(b.get('value'))} GPU (시드 {b.get('n_seeds')}개로 정함)"])
    out.extend(table(["항목", "값"], body)[1:])
    out.append("")
    out.append("  무슨 실험인가")
    out.append("  모든 런은 같은 공유 prefix에서 시작해 On-policy 선택기로 학습한다. 질문은 언제 cached SR")
    out.append("  선택기로 바꾸는 게 좋은지다. SR-GC는 학습 중 신호 D를 보고 시점을 정한다. 이것을 고정 시점")
    out.append("  전환, 학습 보상 정체 트리거, 무작위 시점, 전환 없음과 비교한다. 보상은 평가 문항 정답률이다.")
    out.append("")
    out.append("  두 가지 끝점")
    out.append("  같은 업데이트는 모든 정책을 H 업데이트에서 잰다. 같은 예산은 GPU 시간을 맞춘다. 예산은 평가를")
    out.append("  보기 전에 SR-GC 현재 규칙이 H까지 쓴 비용의 시드 평균으로 고정했다. 정책마다 그 예산 안에서")
    out.append("  마지막으로 저장된 체크포인트(25 업데이트 간격)를 쓰고, 싼 정책은 H 뒤로 연장 학습한다.")
    if b and "cold-start incomplete" in str(b.get("scope")):
        out.append("  이번 예산은 이어 학습 비용만으로 정했다. prefix와 SR 캐시 비용 기록이 다 모이지 않았다.")
    if diff is not None:
        out.append("")
        out.append(f"  재집계: 큐가 옛 코드로 낸 값과 현재 코드 값이 다른 행 {diff[0]}개(전체 {diff[1]}개).")
        out.append("  이 보고서의 값은 전부 현재 코드 값이다. 바뀐 행 목록은 ~/v8-recount.txt에 있다.")
    out.append("")


# ---------- 보상 곡선과 빠진 것 점검 ----------

def method_policy(mid):
    """런의 method_id를 보고서의 정책 이름으로."""
    if mid == "on_monitored":
        return "on_continued"
    if mid.startswith("bank_sr_t"):
        return "fixed_sr_t" + mid[len("bank_sr_t"):]
    if mid.startswith("bank_uniform_t"):
        return "fixed_uniform_t" + mid[len("bank_uniform_t"):]
    return mid


def load_runs(root):
    runs = []
    for m in sorted((root / "runs").glob("*/manifest.json")):
        try:
            runs.append(json.loads(m.read_text()))
        except (OSError, ValueError):
            runs.append({"run_id": m.parent.name, "status": "manifest 읽기 실패"})
    evals = {}
    for p in sorted((root / "evaluations").glob("*.json")):
        try:
            r = json.loads(p.read_text())
            key = (r["run_id"], r["completed_update"], r["split"])
        except (OSError, ValueError, KeyError):
            continue
        if key not in evals or r.get("evaluated_at", "") < evals[key].get("evaluated_at", ""):
            evals[key] = r
    return runs, evals


SPLIT = {"rebuttal_bank": "final", "development": "dev", "test": "final"}


def curve_section(runs, evals, out):
    out.append("■ 보상 곡선 원본 값")
    out.append("  저장한 체크포인트마다 잰 평가 보상이다(25 업데이트 간격). 학습 배치 보상이 아니라 평가 문항 정답률이다.")
    out.append("  branch와 예산 연장 런은 같은 정책·시드로 합쳐서 update 순서로 적었다.")
    any_role = False
    for role, _ in ROLES:
        split = SPLIT[role]
        cur = {}
        for man in runs:
            if man.get("role") != role or not man.get("method_id"):
                continue
            pol = method_policy(man["method_id"])
            for (rid, u, sp), r in evals.items():
                if rid == man.get("run_id") and sp == split and r.get("reward_fraction") is not None:
                    cur.setdefault(pol, {}).setdefault(man.get("training_seed"), {})[u] = r["reward_fraction"]
        if not cur:
            continue
        any_role = True
        out.append("")
        out.append(f"  [{ROLE_TITLE[role]}] {split} split")
        for pol in sorted(cur, key=lambda p: (group_of(p), p)):
            by_seed = cur[pol]
            seeds = sorted(by_seed)
            ups = sorted({u for s in seeds for u in by_seed[s]})
            body = []
            avg = {}
            for u in ups:
                vs = [by_seed[s].get(u) for s in seeds]
                m = mean(vs) if all(v is not None for v in vs) else None
                avg[u] = m
                body.append([str(u)] + ["" if v is None else f"{100 * v:.2f}" for v in vs]
                            + ["" if m is None else f"{100 * m:.2f}"])
            out.append("")
            out.append(f"  ● {ko.name(pol)}")
            out.extend(table(["update"] + [f"s{s}" for s in seeds] + ["평균"], body, indent="    ",
                             right=tuple(range(1, len(seeds) + 2))))
            full = {u: v for u, v in avg.items() if v is not None}
            if full:
                best = max(full, key=lambda u: full[u])
                last = max(full)
                out.append(f"    결과: 시드가 다 있는 update 중 평균이 가장 높은 곳은 {best}({pct(full[best])}), "
                           f"마지막 {last}은 {pct(full[last])}.")
            else:
                out.append("    결과: 모든 시드가 함께 평가된 update가 없다.")
    if not any_role:
        out.append("  평가 기록이 없다.")
    out.append("")


def gaps_section(runs, evals, rows_by, cold, out, root=None):
    root_of = lambda m: Path(root) / "runs" / str(m.get("run_id"))
    out.append("■ 빠진 것 점검")
    out.append("  학습이 다 돌았어도 결과에 안 나오는 것이 있는지 본다.")
    found = 0
    bad_runs = [m for m in runs if m.get("status") not in ("completed_horizon", "completed_budget")]
    if bad_runs:
        found += len(bad_runs)
        out.append(f"  끝나지 않은 런 {len(bad_runs)}개:")
        body = [[str(m.get("role")), str(m.get("method_id")), f"s{m.get('training_seed')}", str(m.get("status"))]
                for m in sorted(bad_runs, key=lambda m: (str(m.get("role")), str(m.get("method_id")),
                                                          str(m.get("training_seed"))))]
        out.extend(table(["역할", "정책", "시드", "상태"], body, indent="    "))
    H = None
    for m in runs:
        H = (m.get("task") or {}).get("horizon") if (m.get("task") or {}).get("stage") != "budget-extension" else H
        if H:
            break
    real, curve = [], {}
    for m in runs:
        if m.get("status") not in ("completed_horizon", "completed_budget"):
            continue
        split = SPLIT.get(m.get("role"), "final")
        snaps = []
        try:
            for line in (root_of(m) / "snapshots.jsonl").read_text().splitlines():
                if line.strip():
                    snaps.append(json.loads(line)["completed_update"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        for u in sorted(set(snaps)):
            if (m["run_id"], u, split) in evals:
                continue
            if u == H:
                real.append((m, split, u))
            else:
                curve.setdefault((m.get("role"), split), []).append(u)
    if real:
        found += len(real)
        out.append(f"  H={H} 체크포인트가 있는데 H 평가가 없는 런 {len(real)}개:")
        for m, split, u in sorted(real, key=lambda x: (str(x[0].get("role")), str(x[0].get("method_id")),
                                                       str(x[0].get("training_seed")))):
            out.append(f"    {m.get('role')} {m.get('method_id')} s{m.get('training_seed')} ({split})")
    for role, _ in ROLES:
        rs = rows_by[role]
        bad = [r for r in rs if r.get("status") not in GOOD]
        wait = [r for r in rs if r.get("status") in GOOD and r.get("reward_fraction") is None]
        bwait = [r for r in rs if r.get("budget_status") == "selected" and r.get("status") in GOOD
                 and r.get("budget_reward_fraction") is None]
        capped = [r for r in rs if r.get("budget_status") == "capped_at_horizon" and r.get("status") in GOOD]
        for what, xs in (("못 쓰는 행", bad), ("H 평가가 없는 행", wait), ("예산 끝점 평가가 없는 행", bwait),
                         ("예산 연장 학습이 없는 행", capped)):
            if xs:
                found += len(xs)
                out.append(f"  {ROLE_TITLE[role]} {what} {len(xs)}개: "
                           + ", ".join(f"{r.get('policy_id')} s{r.get('seed')}" for r in xs[:12])
                           + (f" 외 {len(xs) - 12}개" if len(xs) > 12 else ""))
    if cold is not None:
        gaps = [s for s, pre, cache, _ in cold if pre is None or cache is None]
        if gaps:
            out.append(f"  cold-start 비용 기록이 빈 시드 {len(gaps)}개: {seeds_txt(gaps)}. "
                       "학습 결과와는 별개로 비용 영수증이 없다는 뜻이다.")
    if not rows_by.get("test"):
        out.append("  새 시드 검증(test)과 개발 시드(development) 런이 없다. 지금 큐는 bank 단계만 돌리므로")
        out.append("  새 시드 확인 비교(TEST)와 무작위 시점·처음부터 SR 같은 새 시드 정책은 결과에 나오지 않는다.")
    for (role, split), us in sorted(curve.items(), key=str):
        out.append(f"  {ROLE_TITLE.get(role, role)}: H가 아닌 중간 체크포인트 {len(us)}개는 저장만 되고 평가되지 않았다({split}).")
    if curve:
        out.append("    큐는 H와 같은 예산 끝점만 평가하도록 짜여 있어서, 보상 곡선에는 평가된 점만 나온다.")
        out.append("    큐 실패는 아니다. 곡선 전체가 필요하면 중간 체크포인트 평가를 따로 돌려야 한다.")
    if not found:
        out.append("  결과: 큐가 만들어야 할 결과 중 빠진 것은 없다. 끝나지 않은 런, H 평가가 빠진 런, 못 쓰는 행이 없다.")
    else:
        out.append(f"  결과: 큐가 만들어야 할 결과 중 위 {found}개가 빠져 있다. 이것들이 채워져야 해당 비교가 완성된다.")
    out.append("")


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
    glance(a, bank, H, b, out)
    setup(cohort, cfg, a, b, diff, out)
    d_section(dobs or {}, bank, out)
    for role, _ in ROLES:
        if rows_by[role]:
            role_section(role, rows_by[role], H, out)
    comparisons(a, H, b, out)
    runs, evals = load_runs(adir.parent)
    curve_section(runs, evals, out)
    cost_section(rows_by, cold, missing, a, out)
    gaps_section(runs, evals, rows_by, cold, out, root=adir.parent)
    out.append(f"  모든 열의 원본은 ~/v8-recount/{cohort}/analysis/policy_outcomes_*.csv 에 있다.")
