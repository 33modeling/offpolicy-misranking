"""Reviewer-facing sentences from verified analysis outputs, with claim guards (plan Code §13.7)."""

from __future__ import annotations

BANNED = ("universally better", "statistically equivalent", "end-to-end savings", "all seeds improve",
          "verified across four backbones", "significantly better", "proves")


def guard(text: str, *, costs_partial: bool, any_loss: bool, equivalence_procedure: bool = False) -> str:
    low = text.lower()
    for phrase in BANNED:
        if phrase in low:
            if phrase == "statistically equivalent" and equivalence_procedure:
                continue
            if phrase == "end-to-end savings" and not costs_partial:
                continue
            if phrase == "all seeds improve" and not any_loss:
                continue
            raise ValueError(f"unsupported claim phrase in report: {phrase!r}")
    return text


def primary_sentence(contrast: dict, *, budget, unit, scope_components, missing_summary, costs_partial):
    if contrast.get("n_training_seeds", 0) == 0 or contrast.get("mean_delta_pp") is None:
        return "Primary contrast not available: no complete paired seeds (" + missing_summary + ")."
    ci = ("CI withheld: " + contrast["interval_method"]) if contrast.get("ci_low") is None else (
        f"{int(100 * contrast.get('confidence', 0.95))}% CI: {contrast['ci_low']:.2f}, {contrast['ci_high']:.2f}")
    text = (f"At budget {budget} {unit}, SR-GC switching differs from the comparison schedule by "
            f"{contrast['mean_delta_pp']:+.2f} percentage points ({ci}; {contrast['n_training_seeds']} training seeds; "
            f"{contrast['wins']}/{contrast['ties']}/{contrast['losses']} wins/ties/losses). "
            f"The accounting scope includes {scope_components}. Missing or failed runs: {missing_summary}.")
    return guard(text, costs_partial=costs_partial, any_loss=contrast["losses"] > 0)


def markdown(analysis: dict) -> str:
    lines = ["# V8 switching analysis", "", f"status: {analysis.get('status', 'generated from recorded outputs')}", ""]
    for section, body in analysis.items():
        if section in ("status",):
            continue
        lines.append(f"## {section}")
        lines.append("")
        if isinstance(body, str):
            lines.append(body)
        elif isinstance(body, list):
            for item in body[:200]:
                lines.append(f"- {item}")
        else:
            import json
            lines.append("```json")
            lines.append(json.dumps(body, indent=1, sort_keys=True, default=str)[:20000])
            lines.append("```")
        lines.append("")
    text = "\n".join(lines)
    return text
