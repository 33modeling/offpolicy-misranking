"""A math verifier that never aborts training on a gold answer math-verify fails to parse.

``srgc_rebuttal.run_experiment.math_reward`` raises when ``math_verify.parse``
returns nothing for the gold string. On a loaded node the parser's 5-second
alarm can expire on a long gold (matrices, intervals), which surfaced as
``ValueError: gold answer could not be parsed`` and burnt every prefix
attempt. This replacement, installed at runtime by the training child
(``srgc_step_checkpoints.py``) so the hashed package stays untouched:

* parses each distinct gold once (cached, 5 s bound) and never retries;
* when the gold still does not parse, scores the response with the original
  experiment's normalized exact match on its last ``Answer:`` line instead
  of raising, and prints one ``VERIFY fallback`` line per such gold;
* keeps math-verify scoring for every gold that parses.
"""

import re

ANSWER_LINE_RE = re.compile(r"(?im)^\s*Answer:\s*(.+?)\s*$")
_THOUSANDS_RE = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_TEXT_RE = re.compile(r"\\(?:text|mathrm|mbox)\{([^{}]*)\}")
_BOXED_RE = re.compile(r"\\boxed\{(.*)\}\s*$", re.S)
GOLD_TIMEOUT = 5  # one bounded attempt per distinct gold; a miss falls back at once, never stalls a rank
_gold_cache = {}
_reported = set()


def normalize(answer: str) -> str:
    """The original experiment's canonical answer string (src/data.py), plus ``\\boxed`` unwrapping."""
    s = answer.strip().rstrip(".").replace("\\$", "").replace("$", "").strip()  # \$18.90 and $x$ alike
    s = s.replace("\\%", "%")
    match = _BOXED_RE.search(s)
    if match:
        s = match.group(1)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = _TEXT_RE.sub(lambda m: m.group(1), s)
    if _THOUSANDS_RE.fullmatch(s):
        s = s.replace(",", "")
    s = re.sub(r"\s*,\s*", ",", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def exact_match(gold: str, response: str) -> float:
    lines = ANSWER_LINE_RE.findall(response)
    if not lines:
        return 0.0
    prediction, target = normalize(lines[-1]), normalize(gold)
    if prediction == target:
        return 1.0
    try:
        return 1.0 if abs(float(prediction) - float(target)) < 1e-6 else 0.0
    except ValueError:
        return 0.0


def _extraction():
    from math_verify import ExprExtractionConfig, LatexExtractionConfig
    return [LatexExtractionConfig(), ExprExtractionConfig()]


def parse_gold(answer: str):
    """Parsed gold (cached per distinct string); an empty list when math-verify cannot parse it."""
    if answer in _gold_cache:
        return _gold_cache[answer]
    from math_verify import parse
    try:
        gold = parse(answer, extraction_config=_extraction(), parsing_timeout=GOLD_TIMEOUT)
    except Exception:  # noqa: BLE001 - parser failures are treated as unparsable, never fatal
        gold = []
    _gold_cache[answer] = gold
    return gold


def tolerant_math_reward(record: dict, response: str) -> float:
    from math_verify import parse, verify
    answer = str(record["answer"])
    gold = parse_gold(answer)
    if gold:
        try:
            return float(bool(verify(gold, parse(response, extraction_config=_extraction()))))
        except Exception:  # noqa: BLE001 - verifier errors are invalid answers
            return 0.0
    if answer not in _reported:
        _reported.add(answer)
        print(f"VERIFY fallback exact-match gold={answer!r}", flush=True)
    return exact_match(answer, response)


def install():
    """Replace the package's math verifier under both names a plan may use."""
    from srgc_rebuttal import run_experiment, verifiers
    run_experiment.math_reward = tolerant_math_reward
    verifiers.math_reward = tolerant_math_reward
    return tolerant_math_reward
