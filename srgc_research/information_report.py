"""Problem/response/update reports; old result files remain partial evidence."""

import csv
import hashlib
import html
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from srgc_rebuttal.runtime import atomic_json

PROTOCOL = "srgc-selection-information-v1"
METHODS = ("on_policy", "sr")
PHASES = ("score-A", "score-B", "probe", *METHODS)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def result_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_object(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def correlation(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    if any(not math.isfinite(v) for v in [*left, *right]):
        raise ValueError("nonfinite correlation inputs")
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return statistics.correlation(left, right)


def ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result, start = [0.] * len(values), 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        for i in ordered[start:stop]:
            result[i] = (start + stop - 1) / 2
        start = stop
    return result


def group(rewards):
    if not rewards or any(type(v) not in (int, float) or v not in (0, 1) for v in rewards):
        raise ValueError("response rewards must be nonempty binary arrays")
    k = int(sum(rewards))
    return {"responses": len(rewards), "successes": k, "success_rate": k / len(rewards),
            "mixed": 0 < k < len(rewards), "all_wrong": k == 0, "all_correct": k == len(rewards)}


def finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite recorded measurement")


def question(rid, records):
    r = records.get(rid, {})
    return {"id": rid, "question": r.get("question", r.get("problem")),
            "prompt": r.get("prompt"), "answer": r.get("answer"),
            "subject": r.get("subject", r.get("topic")), "dataset_level": r.get("level"),
            "prompt_characters": len(r["prompt"]) if isinstance(r.get("prompt"), str) else None}


def check_response_record(row):
    samples = row["samples"]
    summary = group([s["reward"] for s in samples])
    if any(row.get(k) != v for k, v in summary.items()):
        raise ValueError("raw responses disagree with recorded success summary")
    if [s["response"] for s in samples] != list(range(len(samples))):
        raise ValueError("duplicate or unordered responses")
    for sample in samples:
        if (not 0 < sample["prompt_tokens"] < len(sample["sequence_ids"])
                or len(sample["sequence_ids"]) - sample["prompt_tokens"] != sample["completion_tokens"]
                or any(type(t) is not int or t < 0 for t in sample["sequence_ids"])):
            raise ValueError("raw token lengths disagree")
        for field in ("logps_before", "logps_after"):
            if field in sample and (len(sample[field]) != sample["completion_tokens"]
                    or any(not math.isfinite(x) for x in sample[field])):
                raise ValueError("invalid per-token log probabilities")
        if "logps_after" in sample:
            for field, expected in (("mean_logp_change", mean(sample["logps_after"]) - mean(sample["logps_before"])),
                                    ("sum_logp_change", sum(sample["logps_after"]) - sum(sample["logps_before"]))):
                if not math.isclose(sample[field], expected, abs_tol=1e-10):
                    raise ValueError("incorrect response probability change")


def read_measurement(folder):
    folder = Path(folder)
    endpoint = read_object(folder / "endpoint.json")
    finite_tree(endpoint)
    identity = endpoint["identity"]
    if identity.get("protocol") != PROTOCOL or endpoint.get("status") != "complete":
        raise ValueError("unknown or unfinished information measurement")
    if endpoint.get("methods") != list(METHODS) or set(endpoint.get("phases", {})) != set(PHASES):
        raise ValueError("missing information phases or selectors")
    manifest = read_object(folder / "manifest.json")
    if manifest["identity"] != identity:
        raise ValueError("endpoint and measurement manifest identity differ")
    for filename, field in (("inputs.json", "input_sha256"), ("plan.json", "plan_sha256"),
                            ("source-checkpoint.pt", "source_checkpoint_sha256")):
        if identity.get(field) is not None and digest(folder / filename) != identity[field]:
            raise ValueError("frozen source hash differs")
    for field in ("cost_receipts", "invocation_receipts"):
        if field in endpoint:
            from .storage import validate_costs
            validate_costs(endpoint[field])
    phases = {}
    for name in PHASES:
        path = (folder / endpoint["phases"][name]).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("phase path escapes measurement")
        if digest(path) != endpoint["phase_sha256"][name]:
            raise ValueError("phase receipt hash differs")
        receipt = read_object(path)
        finite_tree(receipt)
        if receipt.get("identity") != identity or receipt.get("phase") != name:
            raise ValueError("information phase identity differs")
        if result_digest(receipt["result"]) != receipt["result_sha256"]:
            raise ValueError("phase result hash differs")
        if not receipt.get("artifacts"):
            raise ValueError("missing measured tensors")
        for artifact in receipt["artifacts"]:
            target = (folder / artifact["file"]).resolve()
            if not target.is_relative_to(folder.resolve()) or digest(target) != artifact["sha256"]:
                raise ValueError("measured tensor hash differs")
        for row in receipt["result"]["responses"].values():
            check_response_record(row)
        phases[name] = receipt["result"]
    ids = endpoint["candidate_ids"]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("duplicate or missing candidate IDs")
    inputs = read_object(folder / "inputs.json")
    if not set(ids) <= set(inputs["candidate_ids"]) or endpoint["reference_ids"] != inputs["ranking_validation_ids"]:
        raise ValueError("measurement IDs differ from source inputs")
    if (not set(endpoint["probe_ids"]) <= set(inputs["validation_pool_ids"]) or
            set(endpoint["probe_ids"]) & set(inputs["evaluation_ids"])):
        raise ValueError("probe is outside independent validation split")
    if endpoint["questions"] != {rid: inputs["records"][rid] for rid in ids}:
        raise ValueError("problem text differs from immutable inputs")
    for block in ("score-A", "score-B"):
        if phases[block]["candidate_ids"] != ids or set(phases[block]["scores"]) != set(ids):
            raise ValueError("candidate draw differs between independent scoring blocks")
        if (phases[block]["reference_ids"] != endpoint["reference_ids"] or
                set(phases[block]["responses"]) != set(ids) | set(endpoint["reference_ids"])):
            raise ValueError("scoring response/reference IDs differ")
        if any(not math.isfinite(v) or abs(v) > 1.00001 for v in phases[block]["scores"].values()):
            raise ValueError("invalid cosine score")
    if set(ids) & (set(endpoint["reference_ids"]) | set(endpoint["probe_ids"])):
        raise ValueError("candidate and diagnostic/reference IDs overlap")
    if set(endpoint["reference_ids"]) & set(endpoint["probe_ids"]):
        raise ValueError("selection reference leaks into independent probe")
    if (phases["probe"]["ids"] != endpoint["probe_ids"] or
            set(phases["probe"]["responses"]) != set(endpoint["probe_ids"])):
        raise ValueError("probe response IDs differ")
    for mode in METHODS:
        r = phases[mode]
        if not set(r["selected_ids"]) <= set(ids) or len(r["selected_ids"]) != len(set(r["selected_ids"])):
            raise ValueError("invalid selected IDs")
        if r["selected_ids"] != endpoint["selected"][mode] or set(r["responses"]) != set(r["selected_ids"]):
            raise ValueError("training response IDs differ from selected problems")
        if r["metrics"].get("updates") != 1:
            raise ValueError("information update is not a one-step measurement")
        for key, field in (("gradient_before_clip", "gradient_norm_before_clip"),
                           ("gradient_after_clip", "gradient_norm_after_clip"), ("update", "update_norm")):
            total = math.sqrt(sum(p[field] ** 2 for p in r["parameters"]))
            if not math.isclose(total, r["metrics"][f"{key}_norm"], rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError("parameter-level and whole-vector norms disagree")
    # Audit against the original selector's seeded tie rules, including global SR ties.
    from srgc_rebuttal.srgc import Config, cached_sr_set, stream_seed, top_ids
    c = Config(**endpoint["configuration"])
    plan = read_object(folder / "plan.json")
    for key, expected in (("responses", plan["responses"]), ("training_prompts", plan["training_prompts"]),
                          ("scoring_prompts", plan.get("scoring_prompts_per_set", plan.get("scoring_prompts"))),
                          ("projection_dim", plan["projection_dim"]), ("seed", identity["seed"]), ("objective", "grpo")):
        if getattr(c, key) != expected:
            raise ValueError("measurement configuration differs from frozen plan")
    if len(ids) != c.scoring_prompts:
        raise ValueError("candidate count differs from configuration")
    expected_on = list(top_ids(ids, [phases["score-A"]["scores"][i] for i in ids], c.training_prompts,
                               stream_seed(c.seed + 1000, identity["stage"], "online-ties")))
    global_sr = cached_sr_set(inputs["candidate_ids"], inputs["cached_rewards"], len(inputs["candidate_ids"]),
                              c.seed, c.responses)
    expected_sr = [i for i in global_sr if i in set(ids)][:c.training_prompts]
    if endpoint["selected"] != {"on_policy": expected_on, "sr": expected_sr}:
        raise ValueError("selection differs from original On-policy/SR ranking and tie rules")
    if any(r["responses"] != c.responses for p in phases.values() for r in p["responses"].values()):
        raise ValueError("measured response count differs from configuration")
    return endpoint, phases


def measurement_rows(folder):
    endpoint, phases = read_measurement(folder)
    identity, records = endpoint["identity"], endpoint["questions"]
    a, b = phases["score-A"], phases["score-B"]
    candidate_rows = []
    # cached rewards are retained in the immutable inputs, not reconstructed from score.
    manifest = read_object(Path(folder) / "manifest.json")
    if any(identity.get(k) != value for k, value in manifest["identity"].items()):
        raise ValueError("endpoint and measurement manifest identity differ")
    inputs = read_object(Path(folder) / "inputs.json")
    if digest(Path(folder) / "inputs.json") != manifest["input_sha256"]:
        raise ValueError("measurement input hash differs")
    for rid in endpoint["candidate_ids"]:
        cached = group(inputs["cached_rewards"][rid])
        candidate_rows.append({**identity, **question(rid, records),
            "cached_success_rate": cached["success_rate"], "gradient_score_A": a["scores"][rid],
            "gradient_score_B": b["scores"][rid], "current_success_A": a["responses"][rid]["success_rate"],
            "current_success_B": b["responses"][rid]["success_rate"],
            "selected_on_policy": rid in endpoint["selected"]["on_policy"],
            "selected_sr": rid in endpoint["selected"]["sr"]})
    rows, batches, parameters = [], [], []
    for mode in METHODS:
        result = phases[mode]
        for rid in result["selected_ids"]:
            responses = result["responses"][rid]
            samples = responses["samples"]
            rows.append({**identity, "method": mode, **question(rid, records),
                "cached_success_rate": group(inputs["cached_rewards"][rid])["success_rate"],
                "current_success_A": a["responses"][rid]["success_rate"],
                "independent_success_B": b["responses"][rid]["success_rate"],
                "independent_cosine_B": b["scores"][rid],
                "training_success_rate": responses["success_rate"], "training_mixed_fraction": float(responses["mixed"]),
                "training_updates": 1, "correct_mean_logp_change": mean(s["mean_logp_change"] for s in samples if s["reward"]),
                "incorrect_mean_logp_change": mean(s["mean_logp_change"] for s in samples if not s["reward"]),
                "raw_response_source": str(Path(folder) / f"{mode}.json"), "raw_samples": samples,
                "weight_update_available": True})
            rows[-1].update(result["problem_gradients"][rid])
        batches.append({**identity, "method": mode, **result["metrics"],
                        "weight_tensor_source": str(Path(folder) / f"{mode}.pt")})
        parameters.extend({**identity, "method": mode, **p} for p in result["parameters"])
    return rows, batches, candidate_rows, parameters


def legacy_rows(paths, input_paths=()):
    """Deduplicate identical exports; never invent missing weights or responses."""
    inputs = {digest(p): read_object(p) for p in input_paths}
    rows, batches, candidates, seen_files, seen_endpoints = [], [], [], set(), {}
    pending, complete_keys = set(), set()
    for path in paths:
        sha = digest(path)
        if sha in seen_files:
            continue
        seen_files.add(sha)
        bundle = read_object(path)
        finite_tree(bundle)
        if bundle.get("errors"):
            raise ValueError(f"source collection contains errors: {path}")
        for row in bundle["rows"]:
            v = row.get("result")
            if v is None:
                pending.add((row["dataset"], row["seed"]))
                continue
            if v.get("arm") != "stage_mechanism" or v.get("protocol") != "grpo-stage-interventions-v1":
                raise ValueError("expected a stage mechanism endpoint")
            key = (row["dataset"], row["seed"], v["input_sha256"], v["implementation_sha256"])
            fingerprint = hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
            if key in seen_endpoints:
                if seen_endpoints[key] != fingerprint:
                    raise ValueError("conflicting duplicate endpoint; refusing to combine")
                continue
            seen_endpoints[key] = fingerprint
            complete_keys.add((row["dataset"], row["seed"]))
            records = inputs.get(v["input_sha256"], {}).get("records", {})
            if len({m["stage"] for m in v["measurements"]}) != len(v["measurements"]):
                raise ValueError("duplicate stage measurements")
            for m in v["measurements"]:
                ids = m["candidate_ids"]
                if len(ids) != 40 or len(set(ids)) != 40:
                    raise ValueError("legacy measurement needs 40 distinct candidates")
                identity = {"dataset": row["dataset"], "seed": row["seed"], "stage": m["stage"],
                            "input_sha256": v["input_sha256"], "source": str(path)}
                for block in ("A", "B"):
                    if len(m[block]["cosines"]) != len(ids) or any(not math.isfinite(c) or abs(c) > 1.00001 for c in m[block]["cosines"]):
                        raise ValueError("invalid legacy cosine scores")
                for index, rid in enumerate(ids):
                    cached = m["cached_success_rates"][rid]
                    a, b = group(m["A"]["rewards"][rid]), group(m["B"]["rewards"][rid])
                    if not math.isfinite(cached) or not 0 <= cached <= 1 or a["responses"] != 8 or b["responses"] != 8:
                        raise ValueError("invalid legacy cache/rewards")
                    candidates.append({**identity, **question(rid, records), "cached_success_rate": cached,
                        "gradient_score_A": m["A"]["cosines"][index], "gradient_score_B": m["B"]["cosines"][index],
                        "current_success_A": a["success_rate"], "current_success_B": b["success_rate"],
                        "selected_on_policy": rid in m["selected"]["on_policy"],
                        "selected_sr": rid in m["selected"]["sr"]})
                for mode in METHODS:
                    selected = m["selected"][mode]
                    if len(selected) != 4 or len(set(selected)) != 4 or not set(selected) <= set(ids):
                        raise ValueError("invalid legacy selection")
                    training = [t for t in v["training_records"] if t["stage"] == m["stage"] and t["mode"] == mode]
                    if sorted(t["update"] for t in training) != list(range(1, v["horizon"] + 1)):
                        raise ValueError("incomplete or duplicated physical training updates")
                    for t in training:
                        if t["train_ids"] != selected or set(t["rewards"]) != set(selected):
                            raise ValueError("training problem IDs differ from selection")
                    identity = {"dataset": row["dataset"], "seed": row["seed"], "stage": m["stage"],
                                "method": mode, "input_sha256": v["input_sha256"], "source": str(path)}
                    for rid in selected:
                        groups = [group(t["rewards"][rid]) for t in training]
                        if any(g["responses"] != 8 for g in groups):
                            raise ValueError("legacy training response count differs")
                        rows.append({**identity, **question(rid, records),
                            "cached_success_rate": m["cached_success_rates"][rid],
                            "current_success_A": group(m["A"]["rewards"][rid])["success_rate"],
                            "independent_success_B": group(m["B"]["rewards"][rid])["success_rate"],
                            "independent_cosine_B": m["B"]["cosines"][m["candidate_ids"].index(rid)],
                            "training_success_rate": mean(g["success_rate"] for g in groups),
                            "training_mixed_fraction": mean(g["mixed"] for g in groups),
                            "training_updates": len(training), "correct_mean_logp_change": None,
                            "incorrect_mean_logp_change": None, "raw_response_source": None,
                            "raw_samples": None,
                            "weight_update_available": False,
                            "success_count_histogram": dict(Counter(g["successes"] for g in groups))})
                    batches.append({**identity, "updates": len(training),
                        "gradient_norm_mean_before_clip": mean(t["metrics"]["gradient_norm"] for t in training),
                        "update_norm": None, "probe_ascent_update_cosine": None,
                        "weight_tensor_source": None, "status": "legacy-norm-only"})
    pending_rows = [{"dataset": d, "seed": s} for d, s in sorted(pending - complete_keys)]
    return rows, batches, candidates, pending_rows, len(seen_files)


def information_summary(candidates):
    """Within the same draw, compare information held by the chosen problems."""
    grouped = {}
    for row in candidates:
        key = (row["dataset"], row["seed"], row["stage"], row.get("input_sha256"),
               row.get("source_checkpoint_sha256"), row.get("source"), row.get("measurement_sha256"))
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key, pool in grouped.items():
        a, b = [r["gradient_score_A"] for r in pool], [r["gradient_score_B"] for r in pool]
        cached_balance = [-abs(r["cached_success_rate"] - .5) for r in pool]
        current_balance = [-abs(r["current_success_B"] - .5) for r in pool]
        pool_success = mean(r["current_success_B"] for r in pool)
        pool_mixed = mean(0 < r["current_success_B"] < 1 for r in pool)
        overlap = sum(r["selected_on_policy"] and r["selected_sr"] for r in pool)
        for method in METHODS:
            chosen = [r for r in pool if r[f"selected_{method}"]]
            success = mean(r["current_success_B"] for r in chosen)
            mixed = mean(0 < r["current_success_B"] < 1 for r in chosen)
            summaries.append({"dataset": key[0], "seed": key[1], "stage": key[2], "method": method,
                "input_sha256": key[3], "source_checkpoint_sha256": key[4], "source": key[5],
                "measurement_sha256": key[6], "pool_size": len(pool), "selected_count": len(chosen),
                "selected_overlap": overlap, "pool_success_B": pool_success, "selected_success_B": success,
                "pool_mixed_B": pool_mixed, "selected_mixed_B": mixed, "mixed_enrichment_B": mixed - pool_mixed,
                "selected_cosine_B": mean(r["gradient_score_B"] for r in chosen),
                "pool_cosine_B": mean(b), "cosine_enrichment_B": mean(r["gradient_score_B"] for r in chosen) - mean(b),
                "score_AB_pearson": correlation(a, b), "score_AB_spearman": correlation(ranks(a), ranks(b)),
                "gradient_cached_SR_spearman": correlation(ranks(a), ranks(cached_balance)),
                "gradient_current_SR_B_spearman": correlation(ranks(a), ranks(current_balance)),
                "cache_to_current_success_gap_B": mean(r["current_success_B"] - r["cached_success_rate"] for r in chosen),
                "selected_subject_counts": dict(Counter(r["subject"] for r in chosen if r["subject"] is not None)),
                "selected_level_counts": dict(Counter(str(r["dataset_level"]) for r in chosen if r["dataset_level"] is not None))})
    return summaries


def write_report(output, *, folders=(), legacy=(), inputs=(), dataset=None):
    output = Path(output).resolve()
    if any(output == Path(p).resolve() or output.is_relative_to(Path(p).resolve())
           or Path(p).resolve().is_relative_to(output) for p in folders):
        raise ValueError("report directory must be separate from measurement sources")
    targets = [output / name for name in ("information.json", "report.html", "selected-problems.csv",
               "batch-updates.csv", "candidate-information.csv", "parameter-updates.csv", "information-summary.csv")]
    if any(target.resolve() == Path(p).resolve() for target in targets for p in (*legacy, *inputs)):
        raise ValueError("report would overwrite an input source")
    selected, batches, candidates, parameters = [], [], [], []
    seen = set()
    for folder in folders:
        folder = Path(folder).resolve()
        if folder in seen:
            continue
        seen.add(folder)
        r, b, c, p = measurement_rows(folder)
        selected.extend(r)
        batches.extend(b)
        candidates.extend(c)
        parameters.extend(p)
    old, old_batches, old_candidates, pending, unique_files = legacy_rows(legacy, inputs)
    selected.extend(old)
    batches.extend(old_batches)
    candidates.extend(old_candidates)
    if dataset is not None:
        selected, batches, candidates, parameters, pending = [
            [r for r in rows if r["dataset"] == dataset] for rows in (selected, batches, candidates, parameters, pending)]
    summaries = information_summary(candidates)
    result = {"protocol": PROTOCOL, "selected_problems": selected, "batch_updates": batches,
              "candidate_information": candidates, "parameter_updates": parameters,
              "information_summary": summaries,
              "pending": pending, "unique_legacy_files": unique_files,
              "interpretation": "same-state observations; response samples and methods are not independent training seeds"}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "information.json", result)
    for name, values in (("selected-problems", selected), ("batch-updates", batches),
                         ("candidate-information", candidates), ("parameter-updates", parameters),
                         ("information-summary", summaries)):
        fields = list(dict.fromkeys(k for row in values for k in row))
        with (output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else v
                              for k, v in row.items()} for row in values)
    render_html(output / "report.html", result)
    return result


def render_html(path, report):
    def fmt(value):
        return "미기록" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)

    def table(headers, data):
        return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+html.escape(h)+'</th>' for h in headers) + \
            '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(fmt(v))+'</td>' for v in row)+'</tr>'
                                            for row in data) + '</tbody></table></div>'

    body = '<h1>SRGC: 선택 문제 → 응답 신호 → 실제 업데이트</h1><p>같은 checkpoint에서 On-policy와 SR이 잡는 정보를 비교한다. 학습 seed·문제·응답은 서로 다른 반복 단위다.</p>'
    body += table(["dataset", "seed", "시점", "방법", "B 성공률", "B mixed", "후보 대비 mixed 차이", "B cosine", "A/B 상관", "선택 중복"],
        [[r.get(k) for k in ("dataset", "seed", "stage", "method", "selected_success_B", "selected_mixed_B",
                            "mixed_enrichment_B", "selected_cosine_B", "score_AB_pearson", "selected_overlap")]
         for r in report["information_summary"]])
    body += '<p>B는 선택에 사용하지 않은 독립 응답이다. 후보 대비 차이는 같은 후보 집합을 기준으로 계산한다. 상관과 차이만으로 초기 우위의 원인을 확정하지 않는다.</p>'
    body += table(["dataset", "seed", "시점", "방법", "updates", "gradient norm", "실제 update norm", "probe 정렬"],
        [[r.get(k) for k in ("dataset", "seed", "stage", "method", "updates")] +
         [r.get("gradient_before_clip_norm", r.get("gradient_norm_mean_before_clip")), r.get("update_norm"),
          r.get("probe_ascent_update_cosine")] for r in report["batch_updates"]])
    body += '<p>기존 mechanism export에는 실제 weights·응답 token이 없어 해당 칸을 미기록으로 표시한다. 25-update 평균 gradient norm은 초기 모델의 첫 update나 실제 parameter 이동량이 아니다.</p>'
    body += table(["시점", "방법", "probe loss 이전", "probe loss 이후", "1차 예측 변화", "momentum 이동", "batch 추가 이동"],
        [[r.get(k) for k in ("stage", "method", "probe_loss_before", "probe_loss_after", "predicted_probe_loss_change",
                            "zero_gradient_update_norm", "batch_incremental_update_norm")]
         for r in report["batch_updates"] if r.get("weight_tensor_source")])
    for row in report["selected_problems"]:
        body += '<details><summary>'+html.escape(f"{row['dataset']} seed {row['seed']} · t{row['stage']} · {row['method']} · {row['id']}")+'</summary>'
        body += '<pre>'+html.escape(row.get("question") or "원문은 input bundle을 제공해야 표시할 수 있다.")+'</pre>'
        body += table(["cached SR", "현재 A", "독립 B", "학습 SR", "mixed group", "B cosine", "정답 logp 변화", "오답 logp 변화"],
                      [[row.get(k) for k in ("cached_success_rate", "current_success_A", "independent_success_B", "training_success_rate",
                                             "training_mixed_fraction", "independent_cosine_B", "correct_mean_logp_change", "incorrect_mean_logp_change")]])
        body += '<p>'+html.escape(f"subject={fmt(row.get('subject'))} · dataset level={fmt(row.get('dataset_level'))} · measured updates={row['training_updates']}")+'</p>'
        if row.get("weight_update_available"):
            body += table(["문제 GRPO gradient norm", "정답·오답 비율 계수", "probe gradient 정렬", "실제 update 정렬"],
                [[row.get(k) for k in ("loss_gradient_norm", "reward_mix_factor", "probe_ascent_cosine", "ascent_update_cosine")]])
        for sample in row.get("raw_samples") or []:
            body += '<details><summary>'+html.escape(f"응답 {sample['response']} · reward={sample['reward']} · tokens={sample['completion_tokens']} · mean logp Δ={sample['mean_logp_change']:.6g}")+'</summary>'
            body += '<pre>'+html.escape(sample["text"])+'</pre></details>'
        body += '</details>'
    body += '<p>CSV: <a href="information-summary.csv">시점별 요약</a> · <a href="selected-problems.csv">선택 문제</a> · <a href="candidate-information.csv">공통 후보</a> · <a href="batch-updates.csv">batch 업데이트</a> · <a href="parameter-updates.csv">parameter별 변화</a> · <a href="information.json">JSON</a></p>'
    body += '<p>단일 update의 probe loss 변화와 장기 평가 성능은 구분한다. 선택용 projected dense cosine과 실제 LoRA/AdamW update도 구분한다. 미수집 결과를 0으로 채우지 않는다.</p>'
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SRGC 선택 정보 분석</title><style>body{font-family:system-ui,sans-serif;color:#233022;background:#f5f6f1;margin:0;line-height:1.7}main{max-width:1150px;margin:30px auto;padding:30px;background:white;border:1px solid #dce1d5;border-radius:12px}h1{font-size:26px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #dce1d5;text-align:right;white-space:nowrap}th{background:#edf2e7}details{margin:12px 0;padding:14px;border:1px solid #dce1d5;border-radius:8px}summary{cursor:pointer;font-weight:600;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}a{color:#35652d}@media(max-width:600px){main{margin:0;padding:16px;border-radius:0}}</style><main>'+body+'</main></html>'
    Path(path).write_text(page)
