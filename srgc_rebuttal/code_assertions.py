"""Trusted MBPP checks: expected values never enter the candidate interpreter."""

import ast
from functools import lru_cache


@lru_cache(maxsize=4096)
def prepare_assertions(source):
    try:
        statements = ast.parse(source, filename="<mbpp-tests>").body
        if not statements or any(not isinstance(row, ast.Assert) for row in statements):
            raise ValueError("expected test assertions")
        expressions, expected = [], []
        constants_pass = True
        for row in statements:
            test = row.test
            if isinstance(test, ast.Constant) and type(test.value) is bool:
                constants_pass = constants_pass and test.value
            elif (isinstance(test, ast.Compare) and len(test.ops) == 1
                  and isinstance(test.ops[0], ast.Eq)):
                expected.append(ast.literal_eval(test.comparators[0]))
                expressions.append(ast.unparse(test.left))
            else:
                raise ValueError("expected equality against a literal value")
        return tuple(expressions), tuple(expected), constants_pass
    except (SyntaxError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"unsupported MBPP test assertions; fix input before training: {exc}") from exc


def check_returned_values(report, expected, constants_pass):
    """Decode only literal data. Never unpickle or execute candidate output."""
    if not isinstance(report, dict) or set(report) != {"values"}:
        return False
    values = report["values"]
    if not isinstance(values, list) or len(values) != len(expected):
        return False
    if not all(isinstance(value, str) for value in values):
        return False
    try:
        return constants_pass and all(ast.literal_eval(value) == gold for value, gold in zip(values, expected))
    except (SyntaxError, TypeError, ValueError, RecursionError, MemoryError):
        return False
