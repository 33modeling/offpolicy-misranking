"""Recheck saved tokens, data, source pins and verifier rewards without generation.

Run from the repository with the Gemma runtime/tokenizer dependencies installed:
python docs/evidence/gemma-it-audit-2026-10-22/verify_probes.py --snapshot PATH
The stdout JSON contains verification evidence, not new sampled measurements.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO)


def source_digests(commit):
    adapter_path = "srgc_research/dispatch/gemma4_it/adapter.py"
    tree = ast.parse(git("show", f"{commit}:{adapter_path}"))
    shared = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "SHARED_FILES"
                          for target in node.targets))
    paths = git("ls-tree", "-r", "--name-only", commit).decode().splitlines()
    adapter_files = [p for p in paths if Path(p).parent.as_posix() in {
        "srgc_research/dispatch/gemma4_it", "srgc_research/dispatch/gemma4"}
        and p.endswith(".py")]
    adapter_hash = hashlib.sha256()
    for name in sorted((*shared, *adapter_files)):
        adapter_hash.update(name.encode())
        adapter_hash.update(git("show", f"{commit}:{name}"))
    engine_hash = hashlib.sha256()
    for name in sorted(p for p in paths if Path(p).parent.as_posix() == "srgc_rebuttal"
                       and p.endswith(".py")):
        engine_hash.update(Path(name).name.encode())
        engine_hash.update(git("show", f"{commit}:{name}"))
    return adapter_hash.hexdigest(), engine_hash.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=HERE)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from scripts.srgc_verifier_fallback import tolerant_math_reward
    from srgc_rebuttal.build_cache import candidate_seed
    from srgc_research.dispatch.gemma4_it.probe import canary_bundle
    from srgc_research.dispatch.gemma4_it.protocol import EOS_TOKENS, SUPPRESS_TOKENS
    from srgc_research.dispatch.gemma4_it.verifier import code_reward

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    source = json.loads((args.evidence / "source-verification.json").read_text())
    output = {"scope": "CPU replay checks of saved responses; no new generations or updates",
              "verification_source_commit": git("rev-parse", "HEAD").decode().strip(),
              "verification_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "runs": []}
    for pin in source["runs"]:
        path = args.evidence / pin["file"]
        raw = path.read_bytes()
        run = json.loads(raw)
        assert run["source_commit"] == pin["reported_base_commit"]
        adapter_hash, engine_hash = source_digests(pin["verified_implementation_commit"])
        assert adapter_hash == run["adapter_sha256"] == pin["adapter_sha256"]
        assert engine_hash == run["engine_sha256"] == pin["engine_sha256"]
        result = {"artifact": pin["file"], "sha256": hashlib.sha256(raw).hexdigest(),
                  "source_digests_verified": True, "complete": run["complete"], "cases": []}
        for case in run["cases"]:
            dataset = case["dataset"]
            if case["source_bundle"] is None:
                bundle_bytes = json.dumps(canary_bundle(dataset), sort_keys=True).encode()
            else:
                bundle_bytes = (REPO / case["source_bundle"]).read_bytes()
            assert hashlib.sha256(bundle_bytes).hexdigest() == case["source_bundle_sha256"]
            bundle = json.loads(bundle_bytes)
            assert case["candidate"] == bundle["candidate_ids"][run.get("candidate_offset", 0)]
            record = bundle["records"][case["candidate"]]
            assert case["gold"] == record["answer"]
            assert case["source_prompt"] == record["prompt"]
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": record["prompt"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            assert prompt == case["prompt"]
            assert tokenizer.encode(prompt, add_special_tokens=False).count(2) == 1
            assert case["sampling_seed"] == candidate_seed(run["seed"], case["candidate"])
            assert len(case["responses"]) == len(case["rewards"]) == len(case["token_ids"])
            verified = []
            for text, reward, tokens in zip(case["responses"], case["rewards"], case["token_ids"]):
                assert tokens and len(tokens) <= run["max_new_tokens"]
                assert not any(token in SUPPRESS_TOKENS for token in tokens)
                assert not any(token in EOS_TOKENS for token in tokens[:-1])
                assert tokenizer.decode(tokens, skip_special_tokens=True) == text
                current = float((tolerant_math_reward if dataset == "math" else code_reward)(record, text))
                assert current == reward
                verified.append({"recorded_reward": reward, "rescored_reward": current,
                                 "tokens": len(tokens), "stopped": tokens[-1] in EOS_TOKENS,
                                 "empty": not text.strip()})
            result["cases"].append({"dataset": dataset, "candidate": case["candidate"],
                                    "profile": case["profile"], "responses": verified})
        output["runs"].append(result)
    output["all_checks_passed"] = True
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
