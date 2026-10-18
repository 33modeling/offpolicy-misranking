"""Record elastic rank failures around unchanged model/measurement entries."""

import argparse
import faulthandler
import json
import os
import runpy
import sys
from pathlib import Path

MODULE = "srgc_research.dispatch.rank_runtime"
MODELS = ("qwen35", "llama31", "gemma4")


def diagnostic_command(command, model):
    from .reward_curves import recording_command

    command = recording_command(command)
    if MODULE in command or "torch.distributed.run" not in command:
        return command
    modules = {"srgc_research.dispatch.reward_curve_rank",
               "srgc_research.dispatch.llama31.rank", "srgc_research.dispatch.gemma4.rank"}
    scripts = {"srgc_qwen35_rank.py", "srgc_qwen35_rank_runtime.py",
               "srgc_qwen35_reward_curve_rank.py", "srgc_qwen35_smoke.py"}
    for index, argument in enumerate(command):
        if argument in modules and index > 0 and command[index - 1] in {"-m", "--module"}:
            return [*command[:index], MODULE, "--model", model,
                    "--entry-module", argument, "--", *command[index + 1:]]
        if Path(argument).name in scripts:
            return [*command[:index], "--module", MODULE, "--model", model,
                    "--entry-script", argument, "--", *command[index + 1:]]
    return command


def diagnostic_child(original, model):
    def run(command, log_path, environment, **kwargs):
        command = diagnostic_command(command, model)
        if MODULE not in command:
            return original(command, log_path, environment, **kwargs)
        environment = dict(environment)
        # Keep elastic's per-rank error.json on shared attempt storage, rather
        # than losing it with the node's temporary directory after eviction.
        root = Path(environment.get("SRGC_PROGRESS_DIR", Path(log_path).parent))
        root = root / "elastic-errors"
        root.mkdir(parents=True, exist_ok=True)
        index = command.index("torch.distributed.run") + 1
        command[index:index] = ["--log-dir", str(root)]
        environment["PYTHONFAULTHANDLER"] = "1"
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(log_path).open("a") as handle:
            handle.write(f"SRGC CHILD START model={model} elastic_errors={root}\n")
        return original(command, log_path, environment, **kwargs)
    return run


def main():
    from torch.distributed.elastic.multiprocessing.errors import record

    @record
    def run():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--model", choices=MODELS, required=True)
        entry = parser.add_mutually_exclusive_group(required=True)
        entry.add_argument("--entry-module")
        entry.add_argument("--entry-script")
        # A literal separator prevents interpreting the frozen rank's --model
        # or other arguments as this operational wrapper's own options.
        index = sys.argv.index("--")
        args = parser.parse_args(sys.argv[1:index])
        arguments = sys.argv[index + 1:]
        faulthandler.enable()
        print("SRGC RANK START " + json.dumps({
            "model": args.model, "rank": os.environ.get("RANK"),
            "local_rank": os.environ.get("LOCAL_RANK"),
            "entry": args.entry_module or args.entry_script,
            "arguments": arguments, "error_file": os.environ.get("TORCHELASTIC_ERROR_FILE"),
        }, sort_keys=True), flush=True)
        sys.argv = [args.entry_module or args.entry_script, *arguments]
        if args.entry_module:
            runpy.run_module(args.entry_module, run_name="__main__", alter_sys=True)
        else:
            # Match Python's direct-script sys.path[0] for legacy Qwen imports.
            sys.path.insert(0, str(Path(args.entry_script).resolve().parent))
            runpy.run_path(args.entry_script, run_name="__main__")

    return run()


if __name__ == "__main__":
    main()
