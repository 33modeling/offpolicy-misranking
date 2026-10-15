"""CPU analysis of saved SRGC results, reward curves and durable rollout groups.

No model loading, training, generation or interpolation. Missing records remain
missing. Run ``python -m srgc_research.stage_rollout_analysis --help``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .rollout_distribution import dataset_name, finite, group_report, integer, summary

ARMS = {"random", "sr", "on_policy", "switch", "prefix"}
IDENTITIES = ("plan_sha256", "input_sha256", "implementation_sha256", "prefix_checkpoint_sha256")
MAIN_RECORD = re.compile(r"seed-(\d+)/(random|sr|on_policy|switch)-(endpoint|progress)\.json")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Analysis:
    def __init__(self, *, model=None, dataset=None):
        self.model, self.dataset = model, dataset
        self.sources, self.cohorts, self.groups, self.curves, self.checks, self.selections = [], {}, {}, {}, {}, {}
        self.switches, self.horizons, self.issues = {}, {}, []
        self.shared_prefix_duplicates = 0

    def context(self, envelope, row, source, *, arm=None, anchor=None):
        plan = envelope.get("plan", {})
        model = row.get("model") or plan.get("model") or envelope.get("model") or self.model or "unrecorded"
        if model == "unrecorded" and self.model:
            model = self.model
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a nonempty string")
        if self.model and model != "unrecorded" and model != self.model:
            raise ValueError("--model differs from the recorded model")
        dataset = dataset_name(envelope.get("dataset") or plan.get("dataset") or row.get("dataset") or self.dataset)
        if self.dataset and dataset != self.dataset:
            raise ValueError("--dataset differs from the recorded dataset")
        seed = integer(row.get("seed", envelope.get("seed")), "seed")
        identity = {k: row.get(k, row.get("identity", {}).get(k)) for k in IDENTITIES}
        run = {"model": model, "dataset": dataset, "seed": seed, **identity}
        if not any(identity.values()):
            run["source_sha256"] = envelope.get("run_id") or source["sha256"]
        cohort = digest(run)[:20]
        self.cohorts[cohort] = {**run, "cohort_id": cohort,
                               "identity_complete": all(identity.values())}
        return {"cohort_id": cohort, "model": model, "dataset": dataset, "seed": seed,
                "arm": arm or row.get("arm"), "anchor_step": anchor}

    @staticmethod
    def key(context, *parts):
        return (context["cohort_id"], context["arm"], context.get("anchor_step"), *parts)

    def curve(self, context, step, reward, kind, source, *, shared=False, measurement=None):
        context = {**context, "arm": "prefix"} if shared else context
        step, reward = integer(step, "curve step"), finite(reward, "curve reward")
        key = self.key(context, kind, step)
        row = {**context, "step": step, "reward": reward, "reward_percent": 100 * reward,
               "kind": kind, "measurement": measurement, "source": source}
        old = self.curves.get(key)
        if old and old["reward"] != reward:
            raise ValueError("conflicting rewards for the same cohort, arm and step")
        self.curves.setdefault(key, row)

    def check(self, context, item, source):
        value = item.get("d")
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("SR-GC contrast must be finite")
        step = integer(item.get("checkpoint", item.get("step")), "check step")
        row = {**context, "step": step, "d": value, "source": source}
        key = self.key(context, step)
        if key in self.checks and self.checks[key]["d"] != value:
            raise ValueError("conflicting SR-GC checks")
        self.checks[key] = row

    def history(self, context, history, source):
        for item in history:
            step = integer(item.get("completed_updates"), "completed_updates", 1)
            if item.get("checkpoint") != step - 1:
                raise ValueError("history checkpoint does not precede its completed update")
            metrics = item.get("metrics", {})
            if "sample_reward" in metrics:
                self.curve(context, step, metrics["sample_reward"], "training", source,
                           measurement="selected_training_batch")
            selected = item.get("train_ids")
            if selected is not None:
                if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
                    raise ValueError("training prompt IDs must be a nonempty unique list")
                self.selections[self.key(context, step)] = {**context, "step": step,
                    "selector": item.get("selector"), "selected_ids": selected,
                    "candidate_ids": item.get("on_ids", item.get("training_candidate_ids")), "source": source}
            self.check(context, item, source)

    def group(self, context, group, source):
        phase = group.get("phase")
        if phase not in {"training", "selection", "diagnostic", "selected_diagnostic"}:
            raise ValueError("raw group phase is missing or invalid")
        step = integer(group.get("update"), "group update", 1 if phase == "training" else 0)
        prompt = group.get("prompt_id")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("raw group prompt_id is required")
        measured = group_report(group)
        lengths = measured.pop("response_token_lengths")
        if lengths is None:
            supplied = [r.get("completion_tokens") for r in group["responses"]]
            if all(v is not None for v in supplied):
                lengths = [integer(v, "completion_tokens", 1) for v in supplied]
            elif any(v is not None for v in supplied):
                raise ValueError("partial completion length coverage within a group")
        cached = group.get("cached_success_rate")
        if cached is not None:
            cached = finite(cached, "cached_success_rate")
        row = {**context, "phase": phase, "update": step,
               "draw": integer(group.get("draw", 0), "draw"), "prompt_id": prompt, **measured,
               "response_token_lengths": lengths, "cached_success_rate": cached,
               "current_minus_cached": measured["observed_success_fraction"] - cached if cached is not None else None,
               "selected_for_training": group.get("selected_for_training"), "source": source}
        key = self.key(context, phase, step, row["draw"], prompt)
        if key in self.groups:
            old = {k: v for k, v in self.groups[key].items() if k != "source"}
            if context["arm"] == "prefix" and old == {k: v for k, v in row.items() if k != "source"}:
                self.shared_prefix_duplicates += 1
                return
            raise ValueError("duplicate raw group; do not combine retries or overlapping exports")
        self.groups[key] = row

    def endpoint(self, envelope, record, source, *, arm=None):
        context = self.context(envelope, record, source, arm=arm)
        if context["arm"] not in ARMS:
            return
        horizon = record.get("total_updates", envelope.get("plan", {}).get("total_updates"))
        if horizon is not None:
            self.horizons[self.key(context)] = integer(horizon, "total_updates")
        boundary = record.get("switched_at")
        if boundary is not None and context["arm"] == "switch":
            boundary = integer(boundary, "switch boundary")
            if context["cohort_id"] in self.switches and self.switches[context["cohort_id"]] != boundary:
                raise ValueError("conflicting switch boundaries")
            self.switches[context["cohort_id"]] = boundary
        self.history(context, record.get("history", []), source["path"])
        for item in record.get("checks", []):
            self.check(context, item, source["path"])
        if record.get("reward") is not None and horizon is not None:
            self.curve(context, horizon, record["reward"], "held_out", source["path"], measurement="final_evaluation")
        return context

    def mechanism(self, envelope, document, source):
        for measurement in document.get("measurements", []):
            stage = integer(measurement["stage"], "mechanism stage")
            for draw, phase in (("A", "selection"), ("B", "diagnostic")):
                rewards = measurement.get(draw, {}).get("rewards", {})
                context = self.context(envelope, document, source, arm="candidate_pool", anchor=stage)
                for prompt, values in rewards.items():
                    self.group(context, {"phase": phase, "update": stage, "prompt_id": prompt,
                        "responses": [{"reward": v} for v in values],
                        "cached_success_rate": measurement.get("cached_success_rates", {}).get(prompt)}, source["path"])
            for arm, selected in measurement.get("selected", {}).items():
                context = self.context(envelope, document, source, arm=arm, anchor=stage)
                for prompt in selected:
                    values = measurement.get("B", {}).get("rewards", {}).get(prompt)
                    if values is None:
                        continue
                    self.group(context, {"phase": "selected_diagnostic", "update": stage, "prompt_id": prompt,
                        "responses": [{"reward": v} for v in values],
                        "cached_success_rate": measurement.get("cached_success_rates", {}).get(prompt)}, source["path"])
        for item in document.get("training_records", []):
            context = self.context(envelope, document, source, arm=item["mode"], anchor=item["stage"])
            for prompt, values in item.get("rewards", {}).items():
                self.group(context, {"phase": "training", "update": item["update"], "prompt_id": prompt,
                           "responses": [{"reward": v} for v in values]}, source["path"])
        for item in document.get("rows", []):
            context = self.context(envelope, document, source, arm=item["mode"], anchor=item["stage"])
            self.curve(context, 0, item["baseline_reward"], "held_out", source["path"], measurement="branch_baseline")
            self.curve(context, item["updates"], item["reward"], "held_out", source["path"], measurement="branch_evaluation")

    def document(self, document, source, parent=None):
        envelope = {**(parent or {}), **document}
        if document.get("schema") == "srgc-rollout-trace-v1":
            context = self.endpoint(envelope, document, source) or self.context(envelope, document, source)
            if document.get("history_record"):
                self.history(context, [document["history_record"]], source["path"])
            for group in document.get("groups", []):
                if group.get("arm") not in ARMS:
                    raise ValueError("trace arm is invalid")
                self.group({**context, "arm": group["arm"]}, group, source["path"])
            return
        if "datasets" in document:
            for name, snapshot in document["datasets"].items():
                self.document({**snapshot, "dataset": snapshot.get("dataset", name)}, source, envelope)
            return
        if "source_results" in document:
            for name, record in document["source_results"].items():
                if not isinstance(record, dict):
                    continue
                if record.get("measurements") is not None and record.get("training_records") is not None:
                    self.mechanism(envelope, record, {**source, "path": source["path"] + "#" + name})
                elif record.get("arm") in ARMS:
                    match = MAIN_RECORD.fullmatch(name)
                    if match is None:
                        self.issues.append(f"excluded separate schedule/replicate: {name}")
                        continue
                    if record.get("seed") != int(match[1]) or record.get("arm") != match[2]:
                        raise ValueError("endpoint identity differs from its archive path")
                    self.endpoint(envelope, record, {**source, "path": source["path"] + "#" + name})
            return
        if "endpoints" in document or "reward_curves" in document:
            endpoints = {(e["seed"], e["arm"]): e["endpoint"] for e in document.get("endpoints", [])}
            for record in endpoints.values():
                self.endpoint(envelope, record, source)
            for curve in document.get("reward_curves", []):
                record = {"identity": curve.get("identity", {}),
                          **endpoints.get((curve["seed"], curve["arm"]), {}), "seed": curve["seed"], "arm": curve["arm"]}
                context = self.context(envelope, record, source)
                horizon = envelope.get("plan", {}).get("total_updates")
                if horizon is not None:
                    self.horizons[self.key(context)] = integer(horizon, "total_updates")
                for kind, field in (("training", "training_rewards"), ("held_out", "held_out_rewards")):
                    for point in curve.get(field, []):
                        self.curve(context, point["step"], point["reward"], kind, source["path"],
                                   shared=point.get("shared_prefix", False), measurement=point.get("kind"))
                for group in curve.get("rollout_groups", []):
                    self.group({**context, "arm": group["arm"]}, group, source["path"])
            self.issues.extend(str(error) for error in document.get("errors", []))
            return
        if "measurements" in document and "training_records" in document:
            self.mechanism(envelope, document, source)
            return
        if document.get("arm") in ARMS:
            self.endpoint(envelope, document, source)
            return
        raise ValueError("unsupported input; use SRGC results, mechanism endpoint, or rollout-trace-v1")

    def result(self):
        for row in [*self.groups.values(), *self.curves.values()]:
            boundary = self.switches.get(row["cohort_id"]) if row["anchor_step"] is None else None
            step = row.get("update", row.get("step"))
            row["updates_relative_to_switch"] = step - boundary if boundary is not None else None
            row["switch_period"] = (None if boundary is None else
                "pre_switch" if step < boundary or step == boundary and row.get("phase") == "training" else
                "boundary" if step == boundary else "post_switch")
        windows = defaultdict(list)
        for row in self.groups.values():
            step = row["update"]
            end = ((step - 1) // 25 + 1) * 25 if row["phase"] == "training" and step else (step // 25) * 25
            windows[self.key(row, row["phase"], end, row["responses"])].append(row)
        window_rows = []
        for rows in windows.values():
            first = rows[0]
            phase, n = first["phase"], first["responses"]
            step = first["update"]
            end = ((step - 1) // 25 + 1) * 25 if phase == "training" and step else (step // 25) * 25
            lengths = [v for r in rows for v in r["response_token_lengths"] or []]
            gaps = [r["current_minus_cached"] for r in rows if r["current_minus_cached"] is not None]
            histogram = Counter(r["successes"] for r in rows)
            outcomes = Counter(r["outcome"] for r in rows)
            window_rows.append({**{k: first[k] for k in ("cohort_id", "model", "dataset", "seed", "arm", "anchor_step")},
                "phase": phase, "window_start": end - 24 if phase == "training" else end,
                "window_end": end if phase == "training" else end + 24, "responses_per_group": n,
                "group_count": len(rows), "response_count": n * len(rows),
                "recorded_updates": len({r["update"] for r in rows}),
                "distinct_prompts": len({r["prompt_id"] for r in rows}),
                "mean_observed_success": sum(r["successes"] for r in rows) / (n * len(rows)),
                "all_wrong_groups": outcomes["all_wrong"], "mixed_groups": outcomes["mixed"],
                "all_correct_groups": outcomes["all_correct"], "mixed_group_fraction": outcomes["mixed"] / len(rows),
                "success_count_histogram": {str(i): histogram[i] for i in range(n + 1)},
                "response_length": summary(lengths),
                "switch_periods": sorted({r["switch_period"] for r in rows if r["switch_period"] is not None}),
                "token_sequence_uniqueness": summary([r["unique_token_sequence_fraction"] for r in rows
                                                       if r["unique_token_sequence_fraction"] is not None]),
                "token_bigram_distance": summary([r["mean_token_bigram_jaccard_distance"] for r in rows
                                                   if r["mean_token_bigram_jaccard_distance"] is not None]),
                "token_length_coverage_groups": sum(r["response_token_lengths"] is not None for r in rows),
                "cache_gap": summary(gaps), "absolute_cache_gap": summary([abs(v) for v in gaps])})
        selections = list(self.selections.values())
        coverage = []
        observed = {self.key(r): r for r in [*self.groups.values(), *self.curves.values(), *selections, *self.checks.values()]}
        for key, row in observed.items():
            groups = [g for g in self.groups.values() if self.key(g) == key]
            selected = [s for s in selections if self.key(s) == key]
            held = sorted(p["step"] for p in self.curves.values() if self.key(p) == key and p["kind"] == "held_out")
            horizon = self.horizons.get(key)
            expected = list(range(25, horizon + 1, 25)) if horizon is not None else None
            exposures = [g["prompt_id"] for g in groups if g["phase"] == "training"]
            training_steps = sorted(p["step"] for p in self.curves.values() if self.key(p) == key and p["kind"] == "training")
            raw_steps = sorted({g["update"] for g in groups if g["phase"] == "training"})
            coverage.append({**{k: row[k] for k in ("cohort_id", "model", "dataset", "seed", "arm", "anchor_step")},
                "raw_groups": len(groups), "training_group_exposures": len(exposures),
                "distinct_training_prompts": len(set(exposures)), "recorded_selection_updates": len(selected),
                "distinct_selected_prompts": len({pid for s in selected for pid in s["selected_ids"]}),
                "recorded_training_reward_steps": training_steps,
                "recorded_training_rollout_steps": raw_steps,
                "missing_rollouts_for_recorded_training_steps": [s for s in training_steps if s not in raw_steps],
                "recorded_held_out_steps": held, "expected_held_out_steps": expected,
                "missing_held_out_steps": [s for s in expected if s not in held] if expected is not None else None,
                "recorded_switch_boundary": self.switches.get(row["cohort_id"]),
                "switch_arm_first_sr_update": self.switches[row["cohort_id"]] + 1 if row["cohort_id"] in self.switches else None})
        if not observed and not self.checks:
            raise ValueError("no usable recorded observations")
        return {"schema": "srgc-stage-rollout-analysis-v1", "sources": self.sources,
                "cohorts": list(self.cohorts.values()), "coverage": coverage,
                "windows_25": sorted(window_rows, key=lambda r: (r["cohort_id"], r["arm"], r["phase"], r["anchor_step"] or 0, r["window_start"])),
                "groups": list(self.groups.values()), "reward_points": list(self.curves.values()),
                "switch_checks": list(self.checks.values()), "selections": selections,
                "shared_prefix_duplicates_removed": self.shared_prefix_duplicates, "issues": self.issues,
                "interpretation": {"training_reward_is_held_out": False, "missing_values_interpolated": False,
                    "semantic_diversity_claimed": False, "mechanism_branch_updates_are_global_updates": False}}


def csv_file(path, rows):
    if not rows:
        path.write_text("record_count\n0\n")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def markdown(report):
    lines = ["# SRGC 저장 기록 분석", "", "학습 배치 보상과 독립 평가 보상은 별도 CSV에 저장한다. 누락 구간은 보간하지 않는다.",
             "메커니즘의 anchor_step은 분기 출발 상태이며, update는 분기 내 업데이트다.", "",
             "| 모델 | 데이터 | seed | 방법 | 원시 group | 독립 평가 step | 누락 step |", "|---|---|---:|---|---:|---|---|"]
    for row in report["coverage"]:
        lines.append(f"| {row['model']} | {row['dataset']} | {row['seed']} | {row['arm']} | {row['raw_groups']} | "
                     f"{row['recorded_held_out_steps']} | {row['missing_held_out_steps']} |")
    lines += ["", "## 25-update 분포", "", "| 데이터 | seed | 방법 | phase | anchor | 구간 | group | mixed 비율 | 캐시와 현재 성공률의 평균 절대 차이 |",
              "|---|---:|---|---|---:|---|---:|---:|---:|"]
    for row in report["windows_25"]:
        gap = row["absolute_cache_gap"]["mean"]
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['arm']} | {row['phase']} | {row['anchor_step']} | "
                     f"{row['window_start']}–{row['window_end']} | {row['group_count']} | {row['mixed_group_fraction']:.4f} | "
                     f"{gap if gap is not None else '미측정'} |")
    lines += ["", "최종점만 있는 입력에서는 중간 곡선과 정답 0~8개 분포를 복원하지 않는다.",
              "cohort_id가 다른 실행·backbone·seed는 합산하지 않는다. 체크포인트 파일은 역직렬화하지 않는다.", ""]
    return "\n".join(lines)


def plots(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for cohort in report["cohorts"]:
        points = [p for p in report["reward_points"] if p["cohort_id"] == cohort["cohort_id"] and p["anchor_step"] is None]
        if not points:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        maximum = max([p["step"] for p in points] +
                      [max(r["expected_held_out_steps"] or [0]) for r in report["coverage"]
                       if r["cohort_id"] == cohort["cohort_id"]])
        for axis, kind in zip(axes, ("training", "held_out"), strict=True):
            for arm in sorted({p["arm"] for p in points}):
                rows = sorted((p for p in points if p["kind"] == kind and p["arm"] == arm), key=lambda p: p["step"])
                if rows:
                    axis.scatter([p["step"] for p in rows], [p["reward_percent"] for p in rows], s=14, label=arm)
            boundary = next((r["recorded_switch_boundary"] for r in report["coverage"]
                             if r["cohort_id"] == cohort["cohort_id"] and r["recorded_switch_boundary"] is not None), None)
            if boundary is not None:
                axis.axvline(boundary, color="black", linestyle="--", linewidth=1, label="switch boundary")
            axis.set(xlabel="Completed updates", ylabel="Reward (%)", title=kind)
            axis.set_xlim(0, max(1, maximum) * 1.02)
            if axis.collections:
                axis.legend(fontsize=8)
            else:
                axis.text(.5, .5, "No recorded rewards", ha="center", va="center", transform=axis.transAxes)
                axis.set_yticks([])
        figure.suptitle(f"{cohort['model']} / {cohort['dataset']} / seed {cohort['seed']} — recorded points only")
        figure.savefig(output / f"rewards-{cohort['cohort_id']}.png", dpi=180)
        figure.savefig(output / f"rewards-{cohort['cohort_id']}.pdf")
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="results JSON, mechanism endpoint, or rollout trace JSON[.gz]")
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory; existing outputs are never overwritten")
    parser.add_argument("--model", help="declare model identity only when absent from the source")
    parser.add_argument("--dataset", choices=("math", "mbpp"), help="declare dataset only when absent from the source")
    parser.add_argument("--plots", action="store_true", help="optional PNG/PDF figures; requires requirements-switch-plots.txt")
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise ValueError("output directory already exists")
        analysis, seen = Analysis(model=args.model, dataset=args.dataset), set()
        for path in args.inputs:
            payload = path.read_bytes()
            source = {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}
            if source["sha256"] in seen:
                raise ValueError("duplicate source payload")
            seen.add(source["sha256"])
            document = json.loads(gzip.decompress(payload) if path.suffix == ".gz" else payload)
            if not isinstance(document, dict):
                raise TypeError("input must be a JSON object")
            analysis.sources.append(source)
            analysis.document(document, source)
        report = analysis.result()
        report["analyzer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if args.plots:
            import importlib.util
            if importlib.util.find_spec("matplotlib") is None:
                raise ValueError("--plots requires requirements-switch-plots.txt; analysis also works without --plots")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        for name, rows in (("groups", report["groups"]), ("windows25", report["windows_25"]),
                           ("training_rewards", [p for p in report["reward_points"] if p["kind"] == "training"]),
                           ("held_out_rewards", [p for p in report["reward_points"] if p["kind"] == "held_out"]),
                           ("switch_checks", report["switch_checks"]), ("selections", report["selections"]),
                           ("coverage", report["coverage"])):
            csv_file(args.output_dir / f"{name}.csv", rows)
        (args.output_dir / "README.md").write_text(markdown(report))
        if args.plots:
            plots(report, args.output_dir)
    except (OSError, ValueError, TypeError, KeyError, EOFError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "raw_groups": len(report["groups"]),
                      "reward_points": len(report["reward_points"]), "switch_checks": len(report["switch_checks"])}))


if __name__ == "__main__":
    main()
