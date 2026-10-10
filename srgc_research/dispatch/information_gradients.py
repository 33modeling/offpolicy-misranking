"""Capture problem derivatives during the actual GRPO backward pass."""

import ast
import hashlib
import inspect
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

ADAPTER_SOURCE = Path(__file__).read_text()
ADAPTER_SHA256 = hashlib.sha256(ADAPTER_SOURCE.encode()).hexdigest()


class BackwardContributions:
    """Observe incoming parameter gradients without changing backward or AdamW."""

    def __init__(self, backend, records):
        import torch
        self.backend, self.records = backend, records
        self.local_ids = list(records)[backend.rank::backend.world]
        self.parts = {
            rid: [torch.zeros(p.shape, dtype=torch.float32) for _, p in backend.train_parameters]
            for rid in self.local_ids
        }
        self.prompt, self.visited, self.stack = None, [], ExitStack()

    def __enter__(self):
        original_rollout = self.backend._rollout

        def rollout(rid, responses, seed):
            if rid not in self.parts or rid in self.visited:
                raise ValueError("actual training prompt differs from the measured batch")
            self.prompt = rid
            self.visited.append(rid)
            return original_rollout(rid, responses, seed)

        def hook(index):
            def record(gradient):
                if self.prompt is None:
                    raise ValueError("backward contribution has no training prompt")
                with self.backend.cost_meter.stage("information_gradient_capture"):
                    self.parts[self.prompt][index].add_(
                        gradient.detach().to(device="cpu", dtype=self.parts[self.prompt][index].dtype)
                    )
                # Returning None preserves the exact gradient passed to the optimizer.
            return record

        try:
            self.stack.enter_context(patch.object(self.backend, "_rollout", rollout))
            for index, (_, parameter) in enumerate(self.backend.train_parameters):
                self.stack.callback(parameter.register_hook(hook(index)).remove)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *error):
        return self.stack.__exit__(*error)

    def gradients(self):
        import torch
        if self.visited != self.local_ids:
            raise ValueError("actual backward did not visit the complete local batch")
        local = {rid: torch.cat([part.flatten() for part in parts]) * len(self.records)
                 for rid, parts in self.parts.items()}
        gathered = self.backend._gather(local)
        if set(gathered) != set(self.records):
            raise ValueError("actual backward contributions are missing problems")
        if any(not torch.isfinite(value).all() for value in gathered.values()):
            raise FloatingPointError("nonfinite actual backward contribution")
        return {rid: gathered[rid] for rid in self.records}


def instrument_inspect(original):
    """Move only derivative collection; retain the frozen readouts and checks."""
    tree = ast.parse(inspect.getsource(original))
    function = tree.body[0]
    body = next(node for node in function.body if isinstance(node, ast.Try))
    # Fail closed if a different scientific runtime needs a different adapter.
    expected = ("local_gradients", None, "gathered", "problem_gradients")
    if len(body.body) < 4 or any(
        (node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        != name for node, name in zip(body.body[:4], expected)
    ) or not isinstance(body.body[1], ast.For):
        raise ValueError("unsupported frozen information derivative collection")
    body.body[:4] = ast.parse("collector = _ActualBackwardContributions(backend, records)").body
    matching = []
    for index, node in enumerate(body.body):
        if isinstance(node, ast.Try) and len(node.body) == 1 and isinstance(node.body[0], ast.With):
            training = node.body[0]
            if len(training.items) == 1 and ast.unparse(training.items[0].context_expr) == "backend.replaying(records)":
                matching.append((index, training))
    if len(matching) != 1:
        raise ValueError("unsupported frozen information training call")
    index, training = matching[0]
    training.items.insert(0, ast.withitem(context_expr=ast.Name(id="collector", ctx=ast.Load())))
    body.body[index + 1:index + 1] = ast.parse("problem_gradients = collector.gradients()").body
    for node in ast.walk(function):
        if (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and node.exc.args
                and isinstance(node.exc.args[0], ast.Constant) and node.exc.args[0].value ==
                "per-problem gradients do not reconstruct the actual batch gradient"):
            node.exc = ast.parse(
                "FloatingPointError(f'per-problem gradients do not reconstruct the actual batch gradient "
                "(error={reconstruction_error:.8g}, tolerance={reconstruction_tolerance:.8g}, rank={backend.rank})')"
            ).body[0].value
    source = ast.unparse(ast.fix_missing_locations(tree))
    namespace = {}
    # Only instrument the trusted frozen function, never caller-provided text.
    exec(compile(source, "<information-actual-backward>", "exec"), original.__globals__, namespace)  # noqa: S102
    return namespace[original.__name__], source


def problem_gradient(backend, records, *, batch_prompts=None):
    """Historical separate-backward calculation, retained for regression tests."""
    import torch

    from srgc_rebuttal.objectives import grpo_advantages
    from srgc_rebuttal.progress import record as progress
    batch_prompts = len(records) if batch_prompts is None else batch_prompts
    if not records or type(batch_prompts) is not int or batch_prompts < len(records):
        raise ValueError("problem gradients require a valid global batch size")
    saved = [None if p.grad is None else p.grad.detach().clone() for _, p in backend.train_parameters]
    try:
        backend.optimizer.zero_grad(set_to_none=True)
        for rid, record in records.items():
            sequences, rewards, start = record[:3]
            advantages = grpo_advantages(rewards)
            active = [i for i, advantage in enumerate(advantages) if advantage != 0]
            for offset in range(0, len(active), backend.logprob_micro_batch):
                indices = active[offset:offset + backend.logprob_micro_batch]
                values = backend._logps_batch([sequences[i].to(backend.device) for i in indices], start)
                losses = []
                for index, logps in zip(indices, values):
                    # The real one-step update uses its own current logps as
                    # old policy, not a separate single-response forward pass.
                    advantage = float(advantages[index])
                    ratio = torch.exp(logps - logps.detach())
                    weighted = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
                    losses.append(-weighted.mean())
                # Match normalization *inside* backward as well. Scaling a
                # unit-loss gradient afterwards is different in BF16 kernels.
                (sum(losses) / (len(rewards) * batch_prompts)).backward()
                progress("information_problem_gradient", prompt=rid)
        parts = [torch.zeros_like(p).float().cpu().flatten() if p.grad is None else
                 p.grad.detach().float().cpu().flatten() for _, p in backend.train_parameters]
        gradient = torch.cat(parts) * (batch_prompts / len(records))
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite problem gradient")
        return gradient
    finally:
        for (_, parameter), gradient in zip(backend.train_parameters, saved):
            parameter.grad = gradient


@contextmanager
def aligned_problem_gradients():
    from srgc_research import information
    original_inspect = information.inspect_update
    instrumented, measured_source = instrument_inspect(original_inspect)

    def inspect(backend, records, *args, **kwargs):
        result, tensors = instrumented(backend, records, *args, **kwargs)
        result["metrics"]["problem_gradient_definition"] = "actual-GRPO-backward-contributions"
        result["metrics"]["problem_gradient_adapter_sha256"] = ADAPTER_SHA256
        tensors["problem_gradient_adapter"] = {
            "sha256": ADAPTER_SHA256, "source": ADAPTER_SOURCE,
            "measured_inspect_source": measured_source,
            "measured_inspect_sha256": hashlib.sha256(measured_source.encode()).hexdigest(),
        }
        return result, tensors

    with patch.object(information, "_ActualBackwardContributions", BackwardContributions, create=True), \
            patch.object(information, "inspect_update", inspect):
        yield
