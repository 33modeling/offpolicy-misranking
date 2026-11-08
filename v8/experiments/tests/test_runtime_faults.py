"""Runtime fault regressions (2026-11-08 audit RT-01..RT-12): kills, torn writes, rank-local faults,
stop signals, exit codes through torchrun, seed-run protections. CPU only; torchrun cases use gloo."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

os.environ.setdefault("V8_FSYNC", "0")
from helpers import HAS_CODE, HERE, tmpdir  # noqa: E402
from switching import common, ledger, runner  # noqa: E402
from switching.common import JsonlLog, read_json, read_jsonl, sha256_file  # noqa: E402

TESTS = Path(__file__).resolve().parent
HELPER = TESTS / "fault_helpers.py"
try:
    import torch  # noqa: F401
    import torch.distributed  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def helper(mode, out, *args, check=True, timeout=600):
    env = dict(os.environ, V8_FSYNC="0", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, str(HELPER), mode, str(out), *args], env=env, capture_output=True, text=True,
                       timeout=timeout)
    if check and r.returncode != 0:
        raise AssertionError(f"{mode} {args}: rc={r.returncode}\n{r.stdout}\n{r.stderr}")
    return r


def run_dir(out):
    return next(p.parent for p in (Path(out) / "runs").glob("*/manifest.json"))


def torchrun(mode, out, env_extra=None, *, timeout=240):
    env = dict(os.environ, V8_FSYNC="0", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               **(env_extra or {}))
    return subprocess.Popen([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                             "--max_restarts=0", str(HELPER), mode, str(out)], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)


@unittest.skipUnless(HAS_CODE, "code repository not available")
class KilledAttempts(unittest.TestCase):
    def test_rt02_kill_between_snapshot_file_and_index_is_reindexed(self):
        from switching import matrix as mx
        out = tmpdir()
        r = helper("crash_index", out, "crash", check=False)
        self.assertEqual(r.returncode, 137, r.stdout + r.stderr)
        rdir = run_dir(out)
        self.assertTrue((rdir / "snapshots" / "step-000050.pt").is_file())
        self.assertNotIn(50, mx.snapshot_index(out, rdir.name))
        r = helper("crash_index", out, "resume")
        self.assertIn("indexed retained snapshot(s) [50]", r.stdout)
        index = mx.snapshot_index(out, rdir.name)
        self.assertEqual(sorted(index), [50, 75, 100])
        self.assertEqual(index[50]["sha256"], sha256_file(rdir / "snapshots" / "step-000050.pt"))
        parent = mx.parent_from_snapshot(out, rdir.name, 50)
        self.assertIsNotNone(parent)
        self.assertIsNotNone(index[50]["own_continuation_gpu_seconds"])
        self.assertEqual(read_json(rdir / "manifest.json")["status"], "completed_horizon")

    def test_rt08_torn_events_line_does_not_block_resume(self):
        out = tmpdir()
        r = helper("torn_line", out, "crash", check=False)
        self.assertEqual(r.returncode, 137)
        r = helper("torn_line", out, "resume")
        self.assertIn("RESULT completed_horizon", r.stdout)

    def test_rt04_killed_phase_is_unknown_cost_not_complete(self):
        out = tmpdir()
        r = helper("killed_cost", out, "crash", check=False)
        self.assertEqual(r.returncode, 137)
        helper("killed_cost", out, "resume")
        rdir = run_dir(out)
        events, starts = runner.run_events(rdir, lineage=False), runner.run_starts(rdir)
        killed = ledger.unfinished_starts(events, starts)
        self.assertTrue(killed)
        self.assertTrue(all(s["source_phase"] == "training" and s["completed_update"] == 30 for s in killed))
        phys = ledger.physical_gpu_seconds(events, starts=starts)
        self.assertFalse(phys["complete"])
        self.assertIsNone(phys["gpu_seconds"])
        self.assertEqual(phys["killed_phase_starts"], len(killed))
        # algorithm views keep only the surviving lineage: the redone update is not double counted
        cost, missing = runner._phase_cost(None, rdir, "on_continued", 0.0)
        self.assertIsNotNone(cost)


class JsonlTornTail(unittest.TestCase):
    def test_rt08_reader_skips_torn_tail_and_writer_moves_it_aside(self):
        d = tmpdir()
        p = d / "x.jsonl"
        log = JsonlLog(p, ("k",))
        log.append({"k": 1, "v": "a"})
        with open(p, "a") as h:
            h.write('{"k": 2, "v": "tor')
        self.assertEqual(read_jsonl(p), [{"k": 1, "v": "a"}])
        log = JsonlLog(p, ("k",))
        log.append({"k": 2, "v": "b"})
        self.assertEqual(read_jsonl(p), [{"k": 1, "v": "a"}, {"k": 2, "v": "b"}])
        aside = list(d.glob("x.jsonl.torn-*"))
        self.assertEqual(len(aside), 1)
        self.assertEqual(aside[0].read_text(), '{"k": 2, "v": "tor')

    def test_rt08_complete_tail_without_newline_is_a_record(self):
        d = tmpdir()
        p = d / "x.jsonl"
        p.write_text('{"k":1}\n{"k":2}')
        self.assertEqual(len(read_jsonl(p)), 2)
        log = JsonlLog(p, ("k",))
        self.assertFalse(log.append({"k": 2}))
        log.append({"k": 3})
        self.assertEqual([r["k"] for r in read_jsonl(p)], [1, 2, 3])

    def test_rt08_damage_before_the_last_line_still_raises(self):
        d = tmpdir()
        p = d / "x.jsonl"
        p.write_text('{"k":1}\n{"k": broken\n{"k":3}\n')
        with self.assertRaises(json.JSONDecodeError):
            read_jsonl(p)
        with self.assertRaises(json.JSONDecodeError):
            JsonlLog(p, ("k",))


class LockErrors(unittest.TestCase):
    def test_rt09_lock_permission_error_is_not_busy(self):
        if os.geteuid() == 0:
            self.skipTest("root bypasses permissions")
        d = tmpdir() / "runs" / "r1"
        d.mkdir(parents=True)
        d.chmod(0o555)
        try:
            with self.assertRaises(PermissionError):
                runner._acquire(d)
        finally:
            d.chmod(0o755)

    def test_rt09_held_lock_is_busy(self):
        d = tmpdir() / "runs" / "r1"
        self.assertEqual(runner._acquire(d), "ok")
        code = (f"import sys; sys.path.insert(0, {str(HERE)!r}); from pathlib import Path; "
                f"from switching import runner; print(runner._acquire(Path({str(d)!r})))")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        runner._release(d)
        self.assertEqual(r.stdout.strip(), "busy", r.stderr)


@unittest.skipUnless(HAS_CODE, "code repository not available")
class EvaluationVerifier(unittest.TestCase):
    def test_rt10_record_carries_verifier_and_refuses_a_different_one(self):
        from switching import backends, coderepo
        from switching.engine import make_engine_class
        from switching.runner import Context, execute_eval
        from switching import snapshots
        srgc = coderepo.srgc()
        feat, ans, data = backends.toy_problem(0)
        out = tmpdir()

        def ctx(identity):
            c = Context(srgc=srgc, engine_class=make_engine_class(srgc), backend=None, data=data,
                        config=srgc.Config(seed=5, projection_dim=64), out_root=out, verifier_identity=identity)
            c.backend = backends.toy_backend(feat, ans, seed=5)
            return c
        a = ctx({"plan_verifier": "v:a", "packages": {"math-verify": "0.9.0"}})
        snap = out / "snap.pt"
        snapshots.save(snap, {"engine": {"backend": a.backend.state_dict()}}, immutable=False)
        ids = data["evaluation_ids"]
        task = {"run_id": "r", "snapshot_sha256": sha256_file(snap), "snapshot_path": str(snap),
                "evaluation_spec_hash": "e" * 64, "split": "final", "question_ids_hash": common.sha256_json(sorted(ids)),
                "responses": 8, "seed_purpose": "reporting-evaluation", "training_seed": 5, "completed_update": 25}
        res = execute_eval(task, a)
        self.assertEqual(res["status"], "evaluated")
        self.assertEqual(read_json(res["path"])["verifier"], a.verifier_identity)
        self.assertEqual(execute_eval(task, a)["status"], "skipped_complete")
        with self.assertRaisesRegex(ValueError, "different reward function"):
            execute_eval(task, ctx({"plan_verifier": "v:b", "packages": {"math-verify": "0.9.1"}}))
        self.assertEqual(execute_eval(task, ctx(None))["status"], "skipped_complete")


class ChildRuntimeChecks(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(HERE))
        import cluster_child
        self.child = cluster_child

    def test_rt05_verifier_bootstrap_failure_is_fatal_on_every_rank(self):
        import types
        fake = types.ModuleType("scripts.srgc_sr_refresh")

        def boom():
            raise ValueError("existing verifier bootstrap returned no runtime directory")
        fake.prepare_verifier_runtime = boom
        saved = sys.modules.get("scripts.srgc_sr_refresh")
        sys.modules["scripts.srgc_sr_refresh"] = fake
        try:
            with self.assertRaisesRegex(RuntimeError, r"verifier runtime .*rank\(s\) \[0\]"):
                self.child.prepare_runtime("olmo", gather=lambda v: [v])
            fake.prepare_verifier_runtime = lambda: None
            # this rank succeeded, another rank failed: every rank must raise the same error
            with self.assertRaisesRegex(RuntimeError, r"rank\(s\) \[2\].*rank 2: OSError: no space"):
                self.child.prepare_runtime("olmo", gather=lambda v: [v, None, "OSError: no space", None])
        finally:
            if saved is None:
                sys.modules.pop("scripts.srgc_sr_refresh", None)
            else:
                sys.modules["scripts.srgc_sr_refresh"] = saved

    def test_rt06_qwen_package_pins_are_checked_on_every_rank(self):
        import types
        refresh = types.ModuleType("scripts.srgc_sr_refresh")
        refresh.prepare_verifier_runtime = lambda: None
        qwen = types.ModuleType("srgc_qwen35")

        def pins():
            raise ValueError("Qwen runtime requires fla-core==0.5.2; found None")
        qwen.runtime_packages = pins
        saved = {k: sys.modules.get(k) for k in ("scripts.srgc_sr_refresh", "srgc_qwen35")}
        sys.modules.update({"scripts.srgc_sr_refresh": refresh, "srgc_qwen35": qwen})
        try:
            with self.assertRaisesRegex(RuntimeError, "Qwen runtime packages .*fla-core"):
                self.child.prepare_runtime("qwen", gather=lambda v: [v])
            self.assertIsNone(self.child.prepare_runtime("olmo", gather=lambda v: [v]))
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    @unittest.skipUnless(HAS_CODE and HAS_TORCH, "code repository and torch required")
    def test_rt05_oom_in_evaluation_rollout_is_retried_once(self):
        import numpy as np
        import torch
        state = {"calls": 0}

        class Base:
            device = torch.device("cpu")

            def _rollout(self, prompt_id, responses, seed):
                state["calls"] += 1
                if state["calls"] == 1:
                    raise torch.OutOfMemoryError("CUDA out of memory (injected)")
                return [torch.tensor([1, 2, 3])] * responses, np.ones(responses), 1

            def evaluate(self, ids, *, seed, responses=8):
                return {i: float(self._rollout(i, responses, seed)[1].mean()) for i in ids}
            def train(self, ids, **kw):
                return self._rollout(ids[0], 2, 0)
        cls = self.child.protected_backend_class(Base)
        cls._oom_retry_sleep = 0.0
        self.assertEqual(cls().evaluate(["q1"], seed=3, responses=2), {"q1": 1.0})
        self.assertEqual(state["calls"], 2)
        state["calls"] = 0  # training rollouts are not retried (as in the seed runs)
        with self.assertRaises(torch.OutOfMemoryError):
            cls().train(["q1"])

    def test_rt07_fresh_seed_uses_source_runtime_and_attention(self):
        self.assertEqual(self.child.runtime_seed({"runtime_source_seed": 5}, {"training_seed": 31}), 5)
        self.assertEqual(self.child.runtime_seed({}, {"training_seed": 7}), 7)
        self.assertEqual(self.child.attention_prefix({"attention_prefix_path": "/p.pt"}), "/p.pt")
        self.assertEqual(self.child.attention_prefix({"prefix_path": "/q.pt", "attention_prefix_path": "/p.pt"}),
                         "/q.pt")


@unittest.skipUnless(HAS_CODE, "code repository not available")
class PreparedSeeds(unittest.TestCase):
    def make(self):
        from switching import cluster, coderepo, seeds
        d = tmpdir()
        root = d / "v8" / "olmo_math"
        root.mkdir(parents=True)
        code = coderepo.code_root()
        current = coderepo.implementation_digest(code)
        selection = coderepo.source_runtime(code, "olmo", current)
        bundle = d / "seed-5.json"
        bundle.write_text(json.dumps({"provenance": {"experiment_seed": 5}, "records": {}}))
        src = {"plan_path": str(d / "plan.json"), "plan_sha256": "p", "model": "m", "model_revision": "r",
               "folder": str(d / "seed-5"), "input_bundle": str(bundle), "run_implementation_sha256": current,
               "implementation_matches_code_root": True, "runtime_selection": selection,
               "prefix_path": str(d / "prefix.pt"), "prefix_sha256": "x", "prefix_completed_updates": 25}
        (root / "locator.json").write_text(json.dumps({"seeds": {"5": src}}))
        saved = (seeds.registry, seeds.choose)
        seeds.registry = lambda *a, **k: ([], 0)
        seeds.choose = lambda *a, **k: {"development": [31], "test": []}
        try:
            import argparse
            env = dict(os.environ)
            os.environ.update(V8_ROOT=str(d / "v8"), V8_ALLOW_ANY_ROOT="1")
            try:
                cluster.cmd_prepare_seeds(argparse.Namespace(cohort="olmo_math", v8_root=None, code_root=None,
                                                             source_seed=None))
            finally:
                os.environ.clear()
                os.environ.update(env)
        finally:
            seeds.registry, seeds.choose = saved
        return d, root, read_json(root / "locator.json")["seeds"]["31"], selection

    def test_rt07_prepare_seeds_copies_runtime_selection(self):
        _, _, fresh, selection = self.make()
        self.assertEqual(fresh["runtime_selection"], selection)
        self.assertEqual(fresh["runtime_source_seed"], 5)
        self.assertTrue(fresh["attention_prefix_path"].endswith("prefix.pt"))

    @unittest.skipUnless(HAS_TORCH, "torch required to reach distributed initialization")
    def test_rt07_prepared_seed_child_passes_runtime_verification(self):
        d, root, fresh, _ = self.make()
        matrix = d / "m.json"
        matrix.write_text(json.dumps({"tasks": [{"kind": "train", "run_id": "olmo_math.development.s31.x",
                                                 "training_seed": 31, "role": "development"}]}))
        plan = {"world_size": 4}
        Path(fresh["plan_path"]).write_text(json.dumps(plan))
        env = dict(os.environ, V8_ROOT=str(d / "v8"), V8_ALLOW_ANY_ROOT="1", CUDA_VISIBLE_DEVICES="",
                   V8_EXIT_FILE=str(d / "exit.json"))
        r = subprocess.run([sys.executable, str(HERE / "cluster_child.py"), "--cohort", "olmo_math",
                            "--matrix", str(matrix), "--task", "olmo_math.development.s31.x"], env=env,
                           capture_output=True, text=True, timeout=300)
        self.assertNotIn("Verified source runtime selection changed", r.stdout + r.stderr)
        self.assertNotEqual(r.returncode, 0)  # stops later: no 4-GPU node / no real plan on CPU


class ShellLaunchers(unittest.TestCase):
    FAKE = """#!/bin/sh
echo "$0 $*" >> "$FAKE_LOG"
case " $* " in
  *" pending "*) [ -f "$FAKE_STATE.pending-done" ] || { echo t1; echo t2; touch "$FAKE_STATE.pending-done"; } ;;
  *"torch.distributed.run"*)
    case " $* " in *" t1 "*)
      printf '{"exit_code":75,"status":"busy","task":"t1"}\\n' > "$V8_EXIT_FILE"; exit 1 ;; esac ;;
esac
exit 0
"""

    def fake(self, d, name="python"):
        p = d / name
        p.write_text(self.FAKE)
        p.chmod(0o755)
        return p

    def run_h100(self, d, *args, **env):
        e = dict(os.environ, V8_ROOT=str(d / "root"), V8_CODE_ROOT=str(d / "code"), FAKE_LOG=str(d / "argv.log"),
                 FAKE_STATE=str(d / "state"), CUDA_VISIBLE_DEVICES="", **env)
        for k in ("V8_PYTHON", "QWEN_PYTHON", "SWITCH_PYTHON", "PAIR_PYTHON"):
            if k not in env:
                e.pop(k, None)
        return subprocess.run(["sh", str(HERE / "run_h100.sh"), *args], env=e, capture_output=True, text=True,
                              timeout=60)

    def test_rt03_busy_marker_survives_torchrun_exit_1(self):
        d = tmpdir()
        r = self.run_h100(d, "olmo_math", "trajectory", V8_PYTHON=str(self.fake(d)))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("busy or interrupted: t1", r.stdout)
        self.assertNotIn("task failed", r.stdout)
        calls = (d / "argv.log").read_text()
        self.assertIn("--task t2", calls)  # moved on to the next task instead of aborting the stage

    def test_rt06_qwen_python_is_used_for_qwen_cohorts(self):
        d = tmpdir()
        qwen, switch = self.fake(d, "qwen-python"), self.fake(d, "switch-python")
        r = self.run_h100(d, "qwen_mbpp", "check", QWEN_PYTHON=str(qwen), SWITCH_PYTHON=str(switch))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        calls = (d / "argv.log").read_text().splitlines()
        self.assertTrue(calls and all(c.startswith(str(qwen)) for c in calls), calls)
        r = self.run_h100(d, "olmo_math", "check", QWEN_PYTHON=str(qwen), PAIR_PYTHON=str(switch))
        self.assertTrue((d / "argv.log").read_text().splitlines()[-1].startswith(str(switch)))

    def test_rt06_missing_configured_python_is_an_error(self):
        d = tmpdir()
        r = self.run_h100(d, "qwen_mbpp", "status", QWEN_PYTHON=str(d / "missing-python"))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("configured Python is not executable", r.stderr)

    def test_rt12_harvest_includes_rank_stack_dumps(self):
        d = tmpdir()
        root = d / "v8"
        (root / "olmo_math" / ".multi-node" / "stacks").mkdir(parents=True)
        (root / "olmo_math" / ".multi-node" / "stacks" / "task-rank2.txt").write_text("stack\n")
        (root / "olmo_math" / "locator.json").write_text("{}")
        env = dict(os.environ, V8_ROOT=str(root), TMPDIR=str(d))
        r = subprocess.run(["sh", str(HERE / "harvest.sh")], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        archive = r.stdout.splitlines()[0]
        names = subprocess.run(["tar", "-tzf", archive], capture_output=True, text=True).stdout
        self.assertIn("olmo_math/.multi-node/stacks/task-rank2.txt", names)


@unittest.skipUnless(HAS_CODE and HAS_TORCH and shutil.which("true"), "code repository and torch required")
class Distributed(unittest.TestCase):
    """torchrun --nproc_per_node=2 on gloo with the real runner and srgc_rebuttal.distributed.primary."""

    def test_rt01_rank_local_fault_ends_every_rank_promptly(self):
        for fault_rank in (1, 0):
            with self.subTest(fault_rank=fault_rank):
                out = tmpdir()
                started = time.time()
                p = torchrun("gloo_fault", out, {"FAULT_RANK": str(fault_rank), "GLOO_TIMEOUT": "60"})
                text, _ = p.communicate(timeout=200)
                elapsed = time.time() - started
                self.assertNotEqual(p.returncode, 0)
                self.assertLess(elapsed, 30, f"ranks waited for the collective timeout ({elapsed:.0f}s)\n{text}")
                self.assertIn("simulated rank-local fault", text)
                if fault_rank == 0:
                    m = read_json(out / "runs" / "r1" / "manifest.json")
                    self.assertIn("simulated rank-local fault", m["errors"][-1]["error"])

    def stop(self, mode):
        out = tmpdir()
        exit_file = out / "exit.json"
        p = torchrun("gloo_stop", out, {"STOP_MODE": mode, "V8_EXIT_FILE": str(exit_file),
                                        "V8_STOP_GRACE_SECONDS": "3"})
        ready = out / "slow-update-started"
        deadline = time.time() + 180
        while not ready.exists() and time.time() < deadline and p.poll() is None:
            time.sleep(0.2)
        self.assertTrue(ready.exists(), p.communicate()[0] if p.poll() is not None else "no slow update")
        time.sleep(0.5)
        sent = time.time()
        os.killpg(p.pid, signal.SIGTERM)  # as srgc_v8_nodes.Runtime.process / the supervisor stop
        while not exit_file.exists() and time.time() - sent < 60:
            time.sleep(0.1)
        marker_after = time.time() - sent
        text, _ = p.communicate(timeout=120)
        self.assertTrue(exit_file.exists(), text)
        self.assertLess(marker_after, 10, text)
        marker = read_json(exit_file)
        self.assertEqual((marker["exit_code"], marker["status"]), (75, "interrupted"), marker)
        m = read_json(out / "runs" / "r1" / "manifest.json")
        self.assertEqual(m["status"], "interrupted", m)
        self.assertEqual(m["last_completed_update"], 3)
        return out, text

    def test_rt03_stop_inside_an_update_exits_75_before_sigkill_and_resumes(self):
        out, text = self.stop("python")
        self.assertIn("INTERRUPTED", text)
        from helpers import toy_setup
        from switching.runner import Context, execute_train
        srgc, PE, data, cfg, backend, _ = toy_setup()
        ctx = Context(srgc=srgc, engine_class=PE, backend=backend(), data=data, config=cfg, out_root=out)
        task = read_json(out / "runs" / "r1" / "manifest.json")["task"]
        res = execute_train(task, ctx)
        self.assertEqual(res["status"], "completed_horizon")
        m = read_json(out / "runs" / "r1" / "manifest.json")
        self.assertEqual([a["start"] for a in m["attempts"]], ["initial", "resume@3"])

    def test_rt03_stop_while_rank0_is_blocked_in_a_collective(self):
        out, text = self.stop("blocked")
        self.assertIn("did not exit within 3s", text)


if __name__ == "__main__":
    unittest.main()
