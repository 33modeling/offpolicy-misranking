"""Run the canonical RLOO update while explicitly permitting a GRPO parent.

Only the parent-objective check changes. The optimizer, absolute step counter,
rollouts, loss implementation, checkpoints and final lineage stay canonical.
The existing trainer file is left untouched for running Pair experiments.
"""

import ast
import copy
import inspect
from pathlib import Path

import train_policy_grpo as canonical

ORIGINAL_TRAIN = canonical.train


def handoff_tree():
    tree = ast.parse(inspect.getsource(ORIGINAL_TRAIN))
    blocks = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and ast.unparse(node.test) == "local_checkpoint is None and completed_steps"]
    if len(blocks) != 1:
        raise ValueError("canonical parent validation changed; refusing an unverified handoff")
    block = blocks[0]
    checks = [node for node in ast.walk(block)
              if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
              and node.value.id == "args" and node.attr == "objective"]
    calls = [node for node in ast.walk(block) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "validate_policy_manifest"]
    if len(checks) != 2 or len(calls) != 1:
        raise ValueError("canonical objective validation changed; refusing an unverified handoff")
    # Validate the parent's real GRPO objective, never relabel its manifest.
    for node in checks:
        node.attr = "parent_objective"
    return ast.fix_missing_locations(tree)


def handoff_train():
    tree = handoff_tree()
    namespace = copy.copy(vars(canonical))
    namespace["validate_policy_lineage"] = validate_policy_lineage
    exec(compile(tree, str(Path(__file__).resolve()), "exec"), namespace)
    return namespace["train"]


def lineage_tree():
    tree = ast.parse(inspect.getsource(canonical.validate_policy_lineage))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "validate_policy_manifest"
             and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "parent"]
    if len(calls) != 1:
        raise ValueError("canonical parent lineage changed; refusing an unverified handoff")
    keywords = [kw for kw in calls[0].keywords if kw.arg == "training_objective"]
    if len(keywords) != 1 or ast.unparse(keywords[0].value) != "training_objective":
        raise ValueError("canonical parent objective changed; refusing an unverified handoff")
    keywords[0].value = ast.Constant(value="grpo")
    return ast.fix_missing_locations(tree)


def validate_policy_lineage(adapter_dir, **kwargs):
    if kwargs["training_objective"] != "rloo":
        raise ValueError("RLOO child objective required")
    parent = kwargs.get("expected_parent")
    if parent is not None:
        canonical.validate_policy_manifest(
            parent, target_steps=kwargs["expected_start_step"], world_size=kwargs["world_size"],
            training_objective="grpo", require_complete_hashes=True)
    namespace = copy.copy(vars(canonical))
    exec(compile(lineage_tree(), str(Path(__file__).resolve()), "exec"), namespace)
    return namespace["validate_policy_lineage"](adapter_dir, **kwargs)


def train(args):
    if args.objective != "rloo":
        raise ValueError("this entry point requires --objective rloo")
    args.parent_objective = "grpo"
    if args.start_step > 0:
        canonical.validate_policy_manifest(
            Path(args.resume_adapter), target_steps=args.start_step,
            world_size=args.expected_world_size, training_objective="grpo",
            require_complete_hashes=True)
    handoff_train()(args)


if __name__ == "__main__":
    canonical.train = train
    canonical.main()
