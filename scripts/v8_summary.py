#!/usr/bin/env python3
"""Key V8 numbers only, from the analysis the workers already wrote (read-only, standard library).

    sh scripts/v8_summary.sh        -> prints the summary and saves ~/v8-summary.txt

Per cohort: primary contrast (SR-GC minus LOSO-tuned fixed), oracle/grid-best gaps, tau_hat, the same at the
frozen budget, secondary pairs, fresh-seed test if present, mean reward per policy, run status and GPU hours.
"""
import json, os, socket, sys, time
from pathlib import Path

COHORTS = ("olmo_math", "qwen_mbpp", "olmo_mbpp", "qwen_math")


def f(v, d=2, sign=True):
    if v is None:
        return "-"
    return (f"{v:+.{d}f}" if sign else f"{v:.{d}f}")


def pair_line(p):
    if not isinstance(p, dict):
        return "-"
    if p.get("status") == "missing_policy":
        return f"{p.get('label')}: missing {','.join(p.get('missing') or [])}"
    s = f"{p.get('label') or ''}: {f(p.get('mean_delta_pp'))}pp n={p.get('n_training_seeds')} " \
        f"W/T/L={p.get('wins')}/{p.get('ties')}/{p.get('losses')}"
    if p.get("ci_low") is not None:
        s += f" CI[{f(p['ci_low'])},{f(p['ci_high'])}]"
    if p.get("p_two_sided") is not None:
        s += f" p={p['p_two_sided']:.3f}"
    if p.get("missing_seeds"):
        s += f" missing_seeds={p['missing_seeds']}"
    return s


def per_seed(p):
    d = (p or {}).get("per_seed_delta_pp") or {}
    return " ".join(f"s{k}:{f(v, 1)}" for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))) or "-"


def contrast_block(name, c, out):
    if not isinstance(c, dict) or "primary_contrast" not in c:
        out.append(f"{name}: {c.get('status') if isinstance(c, dict) else 'not available'}")
        return
    pc = c["primary_contrast"]
    out.append(f"{name}: {pair_line(pc)}")
    out.append(f"  per seed: {per_seed(pc)}")
    nt = pc.get("naive_paired_t") or {}
    if nt.get("ci_low") is not None:
        out.append(f"  naive paired-t (optimistic) CI[{f(nt['ci_low'])},{f(nt['ci_high'])}]"
                   + (f" p={nt['p_two_sided']:.3f}" if nt.get("p_two_sided") is not None else ""))
    out.append(f"  minus oracle: {f(c.get('minus_oracle', {}).get('mean_delta_pp'))}pp   "
               f"minus grid best: {f(c.get('minus_grid_best', {}).get('mean_delta_pp'))}pp")
    lo = c.get("loso") or {}
    hats = {k: v.get("tau_hat") for k, v in (lo.get("per_seed") or {}).items()}
    gb = lo.get("grid_best_same_seeds") or {}
    out.append(f"  tau_hat per seed: {' '.join(f's{k}:{v}' for k, v in sorted(hats.items(), key=lambda kv: str(kv[0])))}"
               f"  unstable={lo.get('tau_hat_unstable')}  grid_best_tau={gb.get('tau')}"
               + (f"  missing_seeds={lo['missing_seeds']}" if lo.get("missing_seeds") else ""))


def policy_means(rows, key):
    acc = {}
    for r in rows:
        if key.startswith("budget_") and r.get("budget_status") != "selected":
            continue
        if r.get("status") in ("triggered", "not_triggered", "ran") and r.get(key) is not None:
            acc.setdefault(r["policy_id"], []).append(r[key])
    return {p: (sum(v) / len(v), len(v)) for p, v in acc.items()}


def summarize(label, adir, out):
    a = json.loads((adir / "analysis.json").read_text())
    out.append(f"== {label}  ({time.strftime('%Y-%m-%d %H:%M', time.localtime((adir / 'analysis.json').stat().st_mtime))})")
    ep = a.get("endpoints") or {}
    b = ep.get("budget") or {}
    out.append(f"matched update H={ep.get('matched_update')}   frozen budget="
               + (f"{b.get('value')} {b.get('unit')} ({b.get('scope')}, n_seeds={b.get('n_seeds')})" if b else "none"))
    contrast_block("PRIMARY matched-update", a.get("primary_rebuttal"), out)
    contrast_block("PRIMARY frozen-budget", a.get("primary_rebuttal_budget"), out)
    ext = (a.get("primary_rebuttal_extended") or {}).get("primary_contrast")
    if ext:
        out.append(f"extended tau grid: {pair_line(ext)}")
    for k in ("primary_test", "primary_test_budget"):
        if k in a:
            out.append(f"TEST {pair_line(a[k])}")
            out.append(f"  per seed: {per_seed(a[k])}")
    for k in ("secondary", "secondary_budget", "secondary_test", "secondary_test_budget"):
        items = [p for p in a.get(k) or [] if p.get("status") != "missing_policy"]
        if items:
            out.append(f"{k}:")
            out.extend("  " + pair_line(p) for p in items)
    e3 = [r for r in a.get("e3") or [] if r.get("delta_pp") is not None]
    if e3:
        out.append("E3 SR-Uniform same state: " + " ".join(f"s{r['seed']}(tau{r['tau_s']}):{f(r['delta_pp'], 1)}" for r in e3))
    rows_path = adir / "policy_outcomes_bank.json"
    if rows_path.exists():
        rows = json.loads(rows_path.read_text())
        for key, name in (("reward_fraction", "matched update"), ("budget_reward_fraction", "frozen budget")):
            m = policy_means(rows, key)
            if m:
                out.append(f"mean reward per policy ({name}), best first:")
                for p, (v, n) in sorted(m.items(), key=lambda kv: -kv[1][0]):
                    out.append(f"  {p:28s} {100 * v:6.2f}%  n={n}")
    sc = a.get("status_counts") or {}
    bad = {p: {s: n for s, n in d.items() if s not in ("triggered", "not_triggered", "ran")} for p, d in sc.items()}
    bad = {p: d for p, d in bad.items() if d}
    out.append("runs not usable: " + (json.dumps(bad, sort_keys=True) if bad else "none"))
    pe = a.get("physical_expenditure") or {}
    if pe:
        g = pe.get("gpu_seconds")
        out.append(f"GPU hours: {f(g / 3600 if g else None, 1, False)} (complete={pe.get('complete')}, "
                   f"missing_events={pe.get('missing_events')}, killed_phases={pe.get('killed_phase_starts')})")


def main():
    work = os.environ.get("OM_WORK") or f"{os.environ.get('GROUP_VOLUME', '/group-volume')}/" \
                                         f"{os.environ.get('OM_USER', 'minsoo3.kim')}/offpolicy-misranking"
    root = Path(os.environ.get("V8_ROOT") or f"{work}/v8")
    dest = Path(os.environ.get("V8_SUMMARY_OUT") or Path.home() / "v8-summary.txt")
    out = [f"# V8 summary  host={socket.gethostname()}  at={time.strftime('%Y-%m-%d %H:%M %Z')}  root={root}",
           "# delta = SR-GC minus comparison, percentage points of reward; W/T/L over training seeds"]
    for c in COHORTS:
        out += ["", f"################ {c}"]
        adir = root / c / "analysis"
        if not (adir / "analysis.json").is_file():
            out.append("no result yet (analysis/analysis.json missing)")
            continue
        frozen = sorted(adir.glob("matched-update-*/analysis.json"))
        try:
            for p in frozen:
                summarize(f"primary result {p.parent.name} (frozen)", p.parent, out)
            if not frozen or (adir / "analysis.json").read_bytes() != frozen[-1].read_bytes():
                summarize("latest analysis", adir, out)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            out.append(f"could not summarize: {type(e).__name__}: {e}")
    text = "\n".join(out) + "\n"
    tmp = dest.with_name(dest.name + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, dest)
    sys.stdout.write(text + f"\n[v8] saved: {dest}\n")


if __name__ == "__main__":
    main()
