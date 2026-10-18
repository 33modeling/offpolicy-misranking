"""Map frozen plan labels to CLI phases without changing experiment identities."""


def dataset_name(plan):
    name = plan["dataset"]
    if name in {"math", "math_train"}:
        return "math"
    if name == "mbpp":
        return "mbpp"
    raise ValueError(f"unsupported dispatch dataset: {name!r}")
