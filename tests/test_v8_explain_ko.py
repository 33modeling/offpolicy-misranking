"""The Korean reading under the V8 summary/recount numbers is computed from those numbers."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import v8_recount  # noqa: E402
import v8_summary  # noqa: E402

SEEDS = [5, 6, 7, 8, 9]


def _pair(label, deltas, lo=None, hi=None):
    v = list(deltas.values())
    return {"label": label, "n_training_seeds": len(v), "mean_delta_pp": sum(v) / len(v) if v else None,
            "per_seed_delta_pp": {str(k): x for k, x in deltas.items()}, "missing_seeds": [s for s in SEEDS if s not in deltas],
            "wins": sum(x > 0 for x in v), "ties": sum(x == 0 for x in v), "losses": sum(x < 0 for x in v),
            "ci_low": lo, "ci_high": hi}


def _contrast(deltas, lo, hi):
    c = _pair("srgc_current - LOSO tuned fixed", deltas)
    c["naive_paired_t"] = {"ci_low": lo, "ci_high": hi}
    return {"primary_contrast": c, "minus_oracle": {"mean_delta_pp": -0.4}, "minus_grid_best": {"mean_delta_pp": 0.4},
            "loso": {"per_seed": {str(s): {"tau_hat": 50} for s in SEEDS}, "grid_best_same_seeds": {"tau": 50}}}


def _rows():
    rows = []
    for s in SEEDS:
        rows.append({"policy_id": "srgc_current", "seed": s, "status": "triggered", "reward_fraction": .40,
                     "continuation_gpu_seconds": 21.39 * 3600, "budget_status": "selected",
                     "budget_endpoint_update": 225, "budget_reward_fraction": .394})
        rows.append({"policy_id": "fixed_sr_t050", "seed": s, "status": "triggered", "reward_fraction": .396,
                     "continuation_gpu_seconds": 17.82 * 3600, "budget_status": "selected",
                     "budget_endpoint_update": 275, "budget_reward_fraction": .411})
        rows.append({"policy_id": "srgc_uniform_destination", "seed": s, "status": "triggered", "reward_fraction": .385,
                     "budget_status": "selected" if s == 7 else "capped_at_horizon",
                     "budget_reward_fraction": .36 if s == 7 else None})
    return rows


def _write(tmp_path):
    a = {"endpoints": {"matched_update": 275, "budget": {"value": 77008.4, "unit": "gpu_seconds",
                                                         "scope": "continuation_algorithm (cold-start incomplete)", "n_seeds": 5}},
         "primary_rebuttal": _contrast({5: 0.0, 6: 0.0, 7: 1.9, 8: 0.0, 9: 0.0}, -0.68, 1.45),
         "primary_rebuttal_budget": _contrast({5: -0.9, 6: -2.0, 7: -3.0, 8: -1.6, 9: -1.0}, -2.77, -0.63),
         "primary_test": _pair("srgc_current - fixed_sr_tuned (matched update)", {}),
         "secondary_budget": [_pair("srgc_current - on_continued", {s: 2.48 for s in SEEDS}, 1.28, 3.67),
                              _pair("srgc_current - srgc_uniform_destination", {6: 2.5}),
                              _pair("srgc_two_negative - srgc_one_negative", {5: 1, 6: -1, 7: -1, 8: -1, 9: -1}, -3.09, 1.79)],
         "status_counts": {"srgc_current": {"triggered": 5}},
         "physical_expenditure": {"gpu_seconds": None, "complete": False, "missing_events": 20}}
    (tmp_path / "analysis.json").write_text(json.dumps(a))
    (tmp_path / "policy_outcomes_bank.json").write_text(json.dumps(_rows()))


def test_reading_follows_the_numbers(tmp_path):
    _write(tmp_path)
    out = []
    v8_summary.summarize("x", tmp_path, out)
    text = "\n".join(out)
    assert "-- 해석" in text and "해석 생성 실패" not in text
    assert "같은 업데이트(H=275)" in text and "5개 시드 중 1승 4무 0패" in text
    assert "5개 시드 중 0승 0무 5패, 95% 구간이 0보다 아래라서 뒤짐" in text
    assert "LOSO는 모든 시드에서 τ=50을 골랐다" in text
    assert "SR-GC 21.39시간, 고정 τ=50 17.82시간" in text and "검사 비용 때문" in text
    assert "SR-GC는 평균 225, 고정 전환은 275" in text
    assert "리뷰 결론 후보(P9): (c)" in text
    assert "새 시드에서 미리 정한 비교(TEST)는 아직 결과가 없다" in text
    assert "전환 없이 On-policy 계속 대비: +2.48점, 5개 시드 중 5승 0무 0패, 95% 구간이 0보다 위라서 앞섬" in text
    assert "SR-GC 시점에 Uniform으로 전환 대비: +2.50점, 시드 1개뿐이라 판단 보류" in text
    assert "SR-GC 음성 2회 연속 규칙 대 SR-GC 음성 1회 규칙" in text
    assert "SR-GC 시점에 Uniform으로 전환 4개" in text
    assert "기록 20개가 빠져 미완성" in text


def test_cold_start_reading_never_fills_gaps_with_zero():
    cold = [(5, 3600.0, 7200.0, {}), (6, None, None, {"status": "incomplete receipts", "known_lower_bound_gpu_seconds": 1800.0})]
    text = "\n".join(v8_recount.cold_start_ko(cold, [{}], {"prefix_missing": {"srgc_current"}}))
    assert "시드 2개 중 1개는 세 부분이 다 기록" in text and "prefix 1.00시간, SR 캐시 2.00시간" in text
    assert "s6: prefix 비용 기록 없음, SR 캐시 비용 미완성(incomplete receipts, 지금까지 확인된 것만 0.50시간 이상)" in text
    assert "0으로 채우지 말고 하한" in text
    full = "\n".join(v8_recount.cold_start_ko([(5, 3600.0, 7200.0, {})], [{}], {}))
    assert "Q3 답변에 그대로 쓸 수 있다" in full


def test_readiness_says_whether_a_cohort_can_be_used_now():
    import v8_explain_ko as ko
    a = {"primary_rebuttal": {"primary_contrast": {"n_training_seeds": 0}}, "primary_rebuttal_budget": {}}
    rows = [{"policy_id": "srgc_current", "seed": s, "status": "triggered", "reward_fraction": None} for s in SEEDS]
    text = "\n".join(ko.readiness(a, rows))
    assert "판정: 아직 못 씀. 평가가 진행 중이다(평가 안 된 행 5개)" in text and "SR-GC(현재 규칙)" in text
    assert ko.readiness(a, [{"policy_id": "x", "seed": 5, "status": "censored"}])[0].startswith("판정: 아직 못 씀. 학습이 덜 끝나")
    out = []
    ko.explain({"status_counts": {"p": {"censored": 3, "triggered": 2}}}, [], out)
    assert any(line.startswith("쓸 수 없는 행 3개: censored = 기준 On-policy 궤적") for line in out)
