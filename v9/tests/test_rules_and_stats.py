import math

import numpy as np
import pytest

from v9lib import collect, measure, tau


def test_tau_rule_picks_earliest_within_half_point():
    means = {50: 0.4002, 75: 0.4018, 100: 0.3990, 150: 0.4010, 250: 0.3900}
    chosen, best = tau.choose(means)
    assert best == 0.4018
    assert chosen == 50  # 0.16 pp below the best, earliest inside 0.5 pp


def test_tau_rule_boundary_is_inclusive_and_excludes_outside():
    chosen, _ = tau.choose({50: 0.395, 75: 0.400})
    assert chosen == 50  # exactly 0.5 pp
    chosen, _ = tau.choose({50: 0.3949, 75: 0.400})
    assert chosen == 75


def test_candidate_taus_exclude_the_sr_branch():
    c = {"bank": {"core_grid": [50, 100, 150, 200], "extended_grid": [25, 75, 125, 175, 225, 250]},
         "training": {"prefix_updates": 25}}
    assert tau.candidate_taus(c) == [50, 75, 100, 125, 150, 175, 200, 225, 250]


def test_split_half_statistics():
    rng = np.random.default_rng(0)
    true = rng.normal(size=400)
    a, b = true + rng.normal(scale=0.5, size=400), true + rng.normal(scale=0.5, size=400)
    r = measure.pearson(a, b)
    assert 0.7 < r < 0.9
    assert measure.spearman_brown(r) == pytest.approx(2 * r / (1 + r))
    assert measure.pearson([1, 1, 1], [1, 2, 3]) is None


def test_summary_fields_and_pinsker_flag():
    rows = []
    for i in range(10):
        p = i / 10
        rows.append({"score_a": i, "score_b": i + 0.1, "score_full": i, "p_a": p, "p_b": p, "p_t": p,
                     "p_beta": p, "p_beta_a": p, "p_beta_b": p, "kl_seq_mean": 0.0, "log_ratio_token_sum": 0.0,
                     "tokens": 8})
    s = measure.summarize(rows, v_norm=1.0, meta={"x": 1})
    assert s["rho_on_split_half_pearson"] == pytest.approx(1.0)
    assert s["abs_p_shift_mean"] == 0.0 and s["fraction_within_pinsker_bound"] == 1.0
    assert s["kl_token_mean"] == 0.0 and s["n_prompts"] == 10


def test_contrast_is_pending_until_every_seed_is_measured():
    seeds = [5, 6, 7, 8, 9]
    a = {5: 0.5, 6: 0.4, 7: 0.45, 8: 0.5, 9: None}
    b = {s: 0.4 for s in seeds}
    r = collect.contrast(a, b, seeds, "x")
    assert r["status"] == "pending" and r["missing_seeds"] == [9]
    assert collect.fmt_delta(r, "X1").startswith("\\pending{X1}")
    a[9] = 0.41
    r = collect.contrast(a, b, seeds, "x")
    assert r["status"] == "measured" and r["n_training_seeds"] == 5
    assert r["ci_low"] < r["mean_delta_pp"] < r["ci_high"]
    assert "(4/1/0)" in collect.fmt_delta(r, "X1")  # seed 6 is a tie
    assert math.isfinite(r["mean_delta_pp"])


def test_rankdata_matches_average_ranks_without_scipy():
    from v9lib.measure import rankdata, spearman
    assert list(rankdata([3, 1, 2])) == [3.0, 1.0, 2.0]
    assert list(rankdata([1, 2, 2, 3])) == [1.0, 2.5, 2.5, 4.0]
    assert list(rankdata([5, 5, 5])) == [2.0, 2.0, 2.0]
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12


def test_x1_table_has_random_column_and_contrast():
    from v9lib.common import item
    its = []
    for s in (5, 6, 7, 8, 9):
        for arm, v in (("staged", 0.5), ("sr", 0.45), ("alignment", 0.47)):
            its.append(item("x1", "olmo_math", f"s{s}.{arm}", "done", value=v + 0.001 * s, seed=s, arm=arm))
        its.append(item("x1", "olmo_math", f"s{s}.random.train", "done", seed=s, arm="random"))
        its.append(item("x1", "olmo_math", f"s{s}.random.eval", "done", value=0.40 + 0.002 * s, seed=s, arm="random"))
    out, rows, tex = collect.x1_tables(75, its)
    o = out["olmo_math"]
    assert abs(o["means"]["random"] - 0.414) < 1e-9
    assert o["staged_minus_random"]["status"] == "measured"
    assert "Cohort & Random & Staged & SR & Alignment" in tex and "Staged$-$Random" in tex
    assert sum(1 for r in rows if r["arm"] == "random" and r["cohort"] == "olmo_math") == 5
    assert out["qwen_mbpp"]["staged_minus_random"]["status"] == "pending"
