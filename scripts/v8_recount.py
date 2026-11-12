#!/usr/bin/env python3
"""Recount V8 costs and budget endpoints with the CURRENT analysis code over a queue's recorded outputs.

    sh scripts/v8_recount.sh            -> ~/v8-recount/<cohort>/... and ~/v8-recount.txt

A queue keeps running the analysis code it was bound to (v8/queue-runtimes.zip), so queues started before
the SCI-01..18 fixes (f13cd3d) were collected and analyzed without them. This command re-reads the same
recorded runs, ledgers, evaluations and frozen budget, and recomputes rows and the analysis with the code in
this checkout. No training, no evaluation, nothing under V8_ROOT is written: each cohort is mirrored into
~/v8-recount/<cohort>/ by symlinks, and only that mirror's analysis/ folder is new.

Old SR-cache refresh events were labeled cache_generation/cache_build (SCI-01); inside runs whose method
uses a periodic cache they are read as cache_refresh. The frozen budget value is reused as recorded.
"""
import json, os, socket, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent / "v8" / "experiments"
COHORTS = ("olmo_math", "qwen_mbpp", "olmo_mbpp", "qwen_math")
COMPARE = ("status", "switch_update", "reward_fraction", "continuation_gpu_seconds", "cold_start_gpu_seconds",
           "budget_status", "budget_endpoint_update", "budget_reward_fraction")


def relabel_refresh(run_events_fn, read_json):
    """SCI-01: an SR-cache refresh inside a periodic-cache run is cache_refresh, not the initial cache build."""
    count = {"events": 0}

    def wrapped(rdir, *a, **k):
        events = run_events_fn(rdir, *a, **k)
        try:
            man = read_json(Path(rdir) / "manifest.json")
            periodic = (man["task"]["method"].get("cache") or {}).get("policy") == "periodic_current"
        except (OSError, ValueError, KeyError, TypeError):
            periodic = False
        if not periodic:
            return events
        out = []
        for e in events:
            if e.get("component") == "cache_build" and e.get("phase") == "initial_cache_generation" \
                    and e.get("completed_update") is not None:
                e = dict(e, component="cache_refresh", phase="cache_refresh", relabeled_from="cache_build")
                count["events"] += 1
            out.append(e)
        return out
    return wrapped, count


def mirror(src, dst):
    """dst/<cohort>: symlinks to every recorded entry of src except analysis/ (a fresh, empty folder)."""
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name == "analysis":
            continue
        link = dst / p.name
        if link.is_symlink() or link.exists():
            if link.is_symlink() and os.readlink(link) == str(p):
                continue
            raise SystemExit(f"refusing to replace {link}; remove ~/v8-recount first")
        link.symlink_to(p)
    (dst / "analysis").mkdir(exist_ok=True)


def num(v):
    return round(v, 3) if isinstance(v, float) else v


def diff_rows(old, new):
    key = lambda r: (r.get("policy_id"), r.get("seed"))
    o, n = {key(r): r for r in old}, {key(r): r for r in new}
    lines, changed = [], 0
    for k in sorted(set(o) | set(n), key=lambda x: (str(x[0]), str(x[1]))):
        a, b = o.get(k), n.get(k)
        if a is None or b is None:
            lines.append(f"  {k[0]} s{k[1]}: {'new row' if a is None else 'row dropped'}"
                         + (f" ({b.get('status')})" if b else ""))
            changed += 1
            continue
        d = [f"{f} {num(a.get(f))} -> {num(b.get(f))}" for f in COMPARE
             if (a.get(f) != b.get(f)) and not (isinstance(a.get(f), float) and isinstance(b.get(f), float)
                                                and abs(a[f] - b[f]) < 1e-9)]
        if d:
            changed += 1
            lines.append(f"  {k[0]} s{k[1]}: " + "; ".join(d))
    return changed, lines


def gpu_hours_by_policy(rows, field):
    acc = {}
    for r in rows:
        v = r.get(field)
        if v is not None:
            acc.setdefault(r["policy_id"], []).append(v / 3600)
    return {p: (sum(v) / len(v), len(v)) for p, v in acc.items()}


def main():
    work = os.environ.get("OM_WORK") or f"{os.environ.get('GROUP_VOLUME', '/group-volume')}/" \
                                         f"{os.environ.get('OM_USER', 'minsoo3.kim')}/offpolicy-misranking"
    v8 = Path(os.environ.get("V8_ROOT") or f"{work}/v8")
    out_base = Path(os.environ.get("V8_RECOUNT_OUT") or Path.home() / "v8-recount")
    sys.path.insert(0, str(RUNNER))
    sys.path.insert(0, str(HERE))
    os.environ["V8_ALLOW_ANY_ROOT"] = "1"  # the mirror lives in the home folder; sources stay on the group volume
    from switching import bank, cluster, runner, spec
    from switching.common import read_json
    import v8_summary
    wrapped, relabeled = relabel_refresh(runner.run_events, read_json)
    bank.run_events = runner.run_events = wrapped
    # analysis.json is written with sort_keys: a row without budget_status gives a None key next to strings
    from switching import analyze
    counts = analyze.status_counts
    analyze.status_counts = lambda rows: {p: {("none" if s is None else s): n for s, n in d.items()}
                                          for p, d in counts(rows).items()}

    text = [f"# V8 recount with current code  host={socket.gethostname()}  at={time.strftime('%Y-%m-%d %H:%M %Z')}",
            f"# source root={v8} (read only)  output={out_base}",
            "# rows and costs recomputed from recorded runs/ledgers/evaluations; frozen budget reused as recorded"]
    for cohort in COHORTS:
        src = v8 / cohort
        text += ["", f"################ {cohort}"]
        if not (src / "locator.json").is_file() or not (src / "runs").is_dir():
            text.append("no recorded runs")
            continue
        dst = out_base / cohort
        try:
            mirror(src, dst)
            args = type("A", (), {"v8_root": str(out_base), "cohort": cohort})()
            cfg = spec.resolve(cohort, role="rebuttal_bank")
            binding = src / ".multi-node" / "binding.json"
            if binding.is_file():
                bound = read_json(binding).get("config_sha256")
                now = spec.config_hash(cfg)
                text.append("config: same as the queue binding" if bound == now else
                            f"config: DIFFERS from the queue binding ({str(bound)[:12]} vs {now[:12]}); "
                            "rows use the current configs/*.json")
            before = relabeled["events"]
            cluster.cmd_collect(args)
            cluster.cmd_analyze(args)
            text.append(f"SR-cache refresh events relabeled (SCI-01): {relabeled['events'] - before}")
            new = read_json(dst / "analysis" / "policy_outcomes_bank.json")
            old_path = src / "analysis" / "policy_outcomes_bank.json"
            old = read_json(old_path) if old_path.is_file() else []
            changed, lines = diff_rows(old, new)
            text.append(f"rows changed vs the queue's own analysis: {changed} of {len(new)}")
            text += lines[:200]
            if len(lines) > 200:
                text.append(f"  ... {len(lines) - 200} more in {dst / 'analysis' / 'diff.txt'}")
            (dst / "analysis" / "diff.txt").write_text("\n".join(lines) + "\n")
            for field, name in (("continuation_gpu_seconds", "continuation GPU hours (matched update)"),
                                ("cold_start_gpu_seconds", "cold-start GPU hours")):
                h_new, h_old = gpu_hours_by_policy(new, field), gpu_hours_by_policy(old, field)
                if h_new:
                    text.append(f"{name}, mean per seed, old -> new:")
                    for p in sorted(h_new, key=lambda p: h_new[p][0]):
                        o = h_old.get(p)
                        text.append(f"  {p:28s} {('%.3f' % o[0]) if o else '-':>9s} -> {h_new[p][0]:9.3f}  n={h_new[p][1]}")
            v8_summary.summarize("recount (current code)", dst / "analysis", text)
        except Exception as e:  # noqa: BLE001 - report and continue with the next cohort
            import traceback
            traceback.print_exc()
            text.append(f"recount failed: {type(e).__name__}: {e}")
    body = "\n".join(text) + "\n"
    dest = Path(os.environ.get("V8_RECOUNT_TEXT") or Path.home() / "v8-recount.txt")
    tmp = dest.with_name(dest.name + f".tmp.{os.getpid()}")
    tmp.write_text(body)
    os.replace(tmp, dest)
    sys.stdout.write(body + f"\n[v8] saved: {dest}\n")


if __name__ == "__main__":
    main()
