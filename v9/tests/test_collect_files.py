"""collect writes the files the manuscript checklist names, in the formats the figures read."""
import math

from v9lib import collect, costs, env
from v9lib.common import item


def _x1_items():
    its = []
    for c in env.X1_COHORTS[:2]:
        for s in (5, 6, 7, 8, 9):
            for arm, v in (("staged", 0.50), ("sr", 0.47), ("alignment", 0.46), ("random", 0.40)):
                its.append(item("x1", c, f"s{s}.{arm}", "done", value=v + 0.003 * s * (arm == "staged"), seed=s,
                                arm=arm, cost=3600.0 * {"staged": 18, "sr": 20, "alignment": 30, "random": 15}[arm]))
    return its


def test_x1_dat_seeds_and_summary():
    out, rows, tex = collect.x1_tables(75, _x1_items())
    dat = collect.x1_cohorts_dat(out).splitlines()
    assert dat[0] == "y contrast mean lo hi"
    assert dat[1].startswith("6.17 al ") and dat[2].startswith("5.83 sr ") and dat[3].startswith("5.17 al ")
    assert len(dat) == 5  # two measured cohorts x two contrasts; pending cohorts are left out
    assert "% largest_gap_to_best:" in tex and "% staged_ge_sr_mean:" in tex
    seeds = collect.x1_seeds_tex(out)
    assert "Cohort & Method & 5 & 6 & 7 & 8 & 9 & Mean & Staged$-$method" in seeds
    assert seeds.count("Random") >= len(env.X1_COHORTS) and "\\pending{X1.appx}" in seeds


def test_cost_tables_need_every_seed():
    c = {co: {s: {"incremental": {"random": 15 * 3600.0, "alignment": 30 * 3600.0, "sr": 20 * 3600.0,
                                  "staged": None if (co == "olmo_mbpp" and s == 9) else 18 * 3600.0},
                  "physical": {}, "per_refresh_alignment": 0.62 * 3600, "prefix": 3600.0,
                  "cache": None if co != "olmo_math" else 6.52 * 3600, "cache_detail": {"status": "no cost-summary.json"},
                  "cold_start": {"sr": 27.52 * 3600 if co == "olmo_math" else None}}
              for s in (5, 6, 7, 8, 9)} for co in env.X1_COHORTS}
    t = collect.x1_cost_tex(c)
    assert "OLMo-3 7B / MATH & 15.00 & 30.00 & 20.00 & 18.00" in t
    assert "OLMo-3 7B / MBPP & 15.00 & 30.00 & 20.00 & \\pending{X1.cost}" in t
    led = collect.x6_ledger_tex(c)
    assert "& mean & 0.62 &" in led and "6.52" in led and "\\pending{X6.cache}" in led
    assert "% cache olmo_mbpp s5: no cost-summary.json" in led
    x2 = {"olmo_math": {"budget_gpu_seconds": 18 * 3600.0,
                        "arms": {"sr": {"mean": 0.47}, "switch050": {"mean": 0.49}, "staged": {"mean": 0.50},
                                 "alignment": {"mean": 0.46}}}}
    res, txt = collect.x2_extra(75, x2, c)
    assert res["olmo_math"]["best_tau_at_B"] == 75
    assert abs(res["olmo_math"]["staged_cost_percent_of"]["alignment"] - 60.0) < 1e-9
    assert "60.0% of alignment" in txt and "B = 18.00 GPU-h" in txt


def test_outcome_kl_and_drift_files(tmp_path):
    assert collect.bern_kl(0.5, 0.5) == 0.0
    assert math.isinf(collect.bern_kl(0.25, 0.0)) and collect.bern_kl(0.0, 0.0) == 0.0
    p, q = 0.75, 0.5
    want = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    assert abs(collect.bern_kl(p, q) - want) < 1e-12
    assert abs(p - q) <= math.sqrt(want / 2)  # Pinsker on the outcome distribution
    import json
    f = tmp_path / "rows.jsonl"
    f.write_text("\n".join(json.dumps({"prompt_id": f"p{i}", "p_t": pt, "p_beta": pb, "kl_seq_mean": k})
                           for i, (pt, pb, k) in enumerate(((0.75, 0.5, 2.0), (0.25, 0.0, 0.01), (0.5, 0.5, 0.0)))) + "\n")
    pr = collect.x5_prompt_rows(f)
    assert [r["within_dp_le_sqrt_seq"] for r in pr] == [True, False, True]
    prompts = [dict(r, cohort="olmo_math", seed=5, update=100) for r in pr]
    dat = collect.x5_drift_dat(prompts).splitlines()
    assert dat[0].startswith("# olmo_math: 3 prompt-snapshot pairs; x = nan for 1 pairs with D_t = inf")
    assert dat[1] == "seed update x y xseq" and len(dat) == 5
    assert dat[3].split()[2] == "nan" and dat[3].split()[4] == f"{math.sqrt(0.005):.5f}"
    rows = [{"cohort": "olmo_math", "seed": s, "update": 100, "status": "measured",
             "rho_on_split_half_pearson": 0.1 * s, "rho_sr_split_half_pearson": 0.5} for s in (5, 6)]
    rho = collect.x5_rho_dat(rows).splitlines()
    assert rho[0] == "update n al al_se sr sr_se" and rho[1].startswith("100 2 0.5500 0.0500 0.5000 0.0000")


def test_refresh_cost_counts_selection_rounds(monkeypatch, tmp_path):
    (tmp_path / "runs" / "r").mkdir(parents=True)
    ev = []
    for u in (25, 50, 75):
        ev.append({"event_id": f"s{u}", "component": "on_selection", "source_phase": "selection",
                   "completed_update": u, "gpu_seconds": 100.0})
        ev.append({"event_id": f"t{u}", "component": "training", "source_phase": "training",
                   "completed_update": u, "gpu_seconds": 7.0})
    monkeypatch.setattr(costs, "_events", lambda rdir, lineage=True: ev)
    assert costs.refresh_cost(tmp_path, "r", 25, 275) == 100.0
    monkeypatch.setattr(costs, "_family", lambda root, rid: "on_continued")
    assert costs.run_cost(tmp_path, "r", hi=275) == 321.0
    assert costs.run_cost(tmp_path, "r", hi=50) == 107.0
