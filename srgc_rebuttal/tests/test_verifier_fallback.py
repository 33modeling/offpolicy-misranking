import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts import srgc_verifier_fallback as fallback

try:
    import math_verify  # noqa: F401
    HAVE_MATH_VERIFY = True
except ImportError:
    HAVE_MATH_VERIFY = False


class NormalizationTest(unittest.TestCase):
    def test_exact_match_mirrors_the_original_experiment(self):
        gold = r"$\boxed{\text{(C)}}$"
        self.assertEqual(fallback.exact_match(gold, "reasoning\nAnswer: (C)"), 1.0)
        self.assertEqual(fallback.exact_match(gold, "Answer: (B)"), 0.0)
        self.assertEqual(fallback.exact_match(gold, "no answer line"), 0.0)
        self.assertEqual(fallback.exact_match(r"$\boxed{1,234}$", "Answer: 1234"), 1.0)
        self.assertEqual(fallback.exact_match(r"$\boxed{(8,38)}$", "Answer: (8, 38)"), 1.0)
        self.assertEqual(fallback.exact_match(r"$\boxed{3}$", "Answer: 3.0"), 1.0)
        self.assertEqual(fallback.exact_match(r"$\boxed{\dfrac{1}{2}}$", r"Answer: \frac{1}{2}"), 1.0)


@unittest.skipUnless(HAVE_MATH_VERIFY, "math-verify not installed")
class TolerantRewardTest(unittest.TestCase):
    def setUp(self):
        fallback._gold_cache.clear()
        fallback._reported.clear()

    def test_parsable_gold_uses_math_verify(self):
        record = {"answer": r"$\boxed{\frac{1}{2}}$"}
        self.assertEqual(fallback.tolerant_math_reward(record, "Answer: 0.5"), 1.0)
        self.assertEqual(fallback.tolerant_math_reward(record, "Answer: 0.4"), 0.0)

    def test_unparsable_gold_falls_back_instead_of_raising_and_reports_once(self):
        record = {"answer": r"$\boxed{\text{(C)}}$"}
        with patch.object(fallback, "parse_gold", return_value=[]), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(fallback.tolerant_math_reward(record, "Answer: (C)"), 1.0)
            self.assertEqual(fallback.tolerant_math_reward(record, "Answer: (D)"), 0.0)
        self.assertEqual(out.getvalue().count("VERIFY fallback"), 1)

    def test_gold_parse_is_cached_and_never_retried(self):
        calls = []
        def fake_parse(text, extraction_config, **kwargs):
            calls.append(kwargs.get("parsing_timeout"))
            return []
        with patch("math_verify.parse", fake_parse):
            self.assertEqual(fallback.parse_gold("g"), [])
            self.assertEqual(fallback.parse_gold("g"), [])
        self.assertEqual(calls, [fallback.GOLD_TIMEOUT])

    def test_install_replaces_both_verifier_names(self):
        from srgc_rebuttal import run_experiment, verifiers
        originals = (run_experiment.math_reward, verifiers.math_reward)
        try:
            fallback.install()
            self.assertIs(run_experiment.math_reward, fallback.tolerant_math_reward)
            self.assertIs(verifiers.math_reward, fallback.tolerant_math_reward)
        finally:
            run_experiment.math_reward, verifiers.math_reward = originals


if __name__ == "__main__":
    unittest.main()
