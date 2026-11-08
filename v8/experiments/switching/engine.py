"""PolicyEngine: selector / transition-policy / cache-policy separation over the frozen Engine.

The code-repo Engine (srgc_rebuttal.srgc.Engine) keeps every computation of the
original method: candidate draws, gradient scoring, cosine ranking, 40-vs-40 D,
SR ranking, GRPO training. This subclass only chooses, per update, which base
arm label to run under, routes the base TemporalRule call to a v8 controller,
splits the SR-comparison scoring call for cost attribution (bit-identical
gradients, see `_vectors`), and maps the post-transition selector.

Base arm labels per v8 phase:
  on_policy phase, signal due -> "switch" (scores On 40 + SR 40, computes D)
  on_policy phase, no signal  -> "on_policy" (no SR scoring)
  sr phase                    -> "sr"      (draw 40, SR-cache top 4, every update)
  uniform phase               -> "random"  (draw 40, uniform 4, every update)
"""

from __future__ import annotations

import time

import numpy as np

from .cache import cache_key, refresh_due, rewards_from_means
from .policies import NeverPolicy, TrainingRewardStats
from .rules import SIGNAL_DEFINITION

SELECTED_SIGNAL = "selected4-vs-sr4-validation-dot-v1"  # E5 diagnostic only, never used to decide
V8_PROTOCOL = "v8-policy-engine-v1"
BASE_ARM = {"on_policy": "on_policy", "sr": "sr", "uniform": "random"}


def stats_from_history(history, responses):
    stats = TrainingRewardStats()
    for r in history or []:
        m = (r.get("metrics") or {}).get("sample_reward")
        n = len(r.get("train_ids") or [])
        if m is None or not n:
            stats.add(r["completed_updates"], None, None)
        else:
            stats.add(r["completed_updates"], float(m) * n * responses, n * responses)
    return stats


def make_engine_class(srgc):
    Engine, TemporalRule, stream_seed, cached_sr_set = (srgc.Engine, srgc.TemporalRule, srgc.stream_seed,
                                                       srgc.cached_sr_set)

    class ObserverRule(TemporalRule):
        """Receives the base engine's D at a check and answers with the v8 decision."""

        def observe(self, step, d):
            callback = getattr(self, "_callback", None)
            result = bool(callback(step, d)) if callback else False
            self.last_step = step
            self.switched = self.switched or result
            return result

    class PolicyEngine(Engine):
        def __init__(self, backend, candidate_ids, validation_ids, cached_rewards, *, method, policy=None,
                     config, monitor=False, cache_store=None, cache_context=None, run_id=None):
            initial = method.get("initial_selector", "on_policy")
            super().__init__(backend, candidate_ids, validation_ids, cached_rewards, arm=BASE_ARM[initial],
                             config=config)
            self.method, self.policy = method, policy or NeverPolicy()
            self.monitor = bool(monitor)
            self.phase, self.destination = initial, method.get("destination")
            self.entry_update = None
            self.transition_update = None
            self.stats = TrainingRewardStats()
            self.signal_log, self.decision_log, self.cache_log = [], [], []
            self.cache_store, self.cache_context, self.run_id = cache_store, cache_context or {}, run_id
            self.cache_state = {"version": "initial", "source_policy_update": 0, "cache_id": None}
            self._check_step, self._stash, self._sr_tail = None, {}, None
            c = config
            interval = method.get("check_interval", c.check_interval)
            first = method.get("first_check", c.first_check)
            if interval % c.check_interval or first < c.first_check or first % c.check_interval:
                raise ValueError("v8 check cadence must coincide with the engine's scheduled checks")
            self._install_rule()

        # ---- helpers ----
        def _install_rule(self):
            fields = dict(interval=self.rule.interval, last_step=self.rule.last_step,
                          window=list(self.rule.window), switched=self.rule.switched)
            self.rule = ObserverRule(**fields)
            self.rule._callback = self._on_check

        def _monitor_due(self, t):
            c = self.config
            return t >= c.first_check and t % c.check_interval == 0

        def _enter(self, target, step, decision):
            if target not in ("sr", "uniform"):
                raise ValueError(f"invalid destination {target!r}")
            self.phase, self.transition_update, self.entry_update = target, step, step
            decision = dict(decision, destination=target)
            return decision

        def _log_decision(self, decision):
            row = dict(decision)
            row.setdefault("policy_id", self.policy.policy_id)
            row["v8_phase_before"] = self.phase
            self.decision_log.append(row)

        # ---- base-engine overrides ----
        def _vectors(self, ids, group_size, purpose):
            n = self.config.scoring_prompts
            if purpose == "selection" and self._check_step is not None and len(ids) > n:
                world = int(getattr(self.backend, "world", 1) or 1)
                head, tail = tuple(ids[:n]), tuple(ids[n:])
                if n % world == 0:
                    # Prompt position decides its rank (ids[rank::world]) and hence its draw seed;
                    # with n % world == 0 the tail keeps every prompt's rank, so gradients are identical
                    # to the single union call while the SR-comparison cost is timed separately.
                    a = super()._vectors(head, group_size, purpose)
                    started = time.perf_counter()
                    with self._timing_scope("sr_comparison", section=True), self._timing_scope("scoring"):
                        b = super()._vectors(tail, group_size, purpose)
                    self._sr_tail = {"split": True, "prompts": len(tail),
                                     "wall_seconds": time.perf_counter() - started}
                    result = {**a, **b}
                else:
                    result = super()._vectors(ids, group_size, purpose)
                    self._sr_tail = {"split": False, "prompts": len(tail), "wall_seconds": None}
            else:
                result = super()._vectors(ids, group_size, purpose)
            self._stash[purpose] = result
            return result

        def _training_batch(self, selector, candidates):
            actual = {"sr": "sr", "uniform": "random"}.get(self.phase, selector)
            return super()._training_batch(actual, candidates)

        def _on_check(self, step, d):
            sr_ids = self._sr_comparison()
            sel, val = self._stash.get("selection", {}), self._stash.get("validation", {})
            measurement = {"completed_update": step, "signal_name": "srgc_d",
                           "signal_definition_version": SIGNAL_DEFINITION, "value": float(d), "valid": True,
                           "reason": None, "sr_set_hash": None, "cache_version": self.cache_state["version"],
                           "split_attribution": (self._sr_tail or {}).get("split")}
            try:
                v = np.stack([val[i] for i in self.validation]).mean(axis=0)
                train4 = list(self.active_selection["train_ids"])
                sr4 = list(sr_ids[:self.config.training_prompts])
                on_ids = list(self.active_selection["on_ids"])
                mean = lambda ids: np.stack([sel[i] for i in ids]).mean(axis=0)  # noqa: E731
                measurement.update(
                    on_mean_dot=float(v @ mean(on_ids)), sr_mean_dot=float(v @ mean(sr_ids)),
                    selected_batch_contrast=float(v @ (mean(train4) - mean(sr4))),
                    selected_signal_definition_version=SELECTED_SIGNAL,
                    validation_norm=float(np.linalg.norm(v)), sr_ids=list(sr_ids))
            except (KeyError, ValueError, TypeError):
                measurement["selected_batch_contrast"] = None
            self.signal_log.append(measurement)
            if self.phase == "on_policy" and self.policy.uses_signal and self.policy.needs_signal(step):
                decision = self.policy.on_signal(step, float(d))
                self._log_decision(decision)
                if str(decision["action"]).startswith("switch"):
                    self._enter(self.destination, step, decision)
                    return True
            return False

        def _maybe_refresh_cache(self, t):
            cp = self.method.get("cache", {})
            if cp.get("policy", "static_initial") != "periodic_current" or self.phase != "sr":
                return
            entry = self.entry_update if self.entry_update is not None else self.method.get("prefix_updates", 25)
            if not refresh_due(t, entry_update=entry, interval=cp.get("refresh_interval"),
                               origin=cp.get("refresh_origin", "sr_entry_update")):
                return
            c = self.config
            meter = getattr(self.backend, "cost_meter", None)
            attempts = int(cp.get("max_attempts", 1))
            for attempt in range(1, attempts + 1):
                try:
                    if meter is not None:
                        with meter.phase("cache_generation", t, self.backend.gpu_count):
                            means = dict(self.backend.evaluate(self.candidates, responses=c.responses,
                                         seed=stream_seed(c.seed, t, "sr-cache-refresh")))
                    else:
                        means = dict(self.backend.evaluate(self.candidates, responses=c.responses,
                                     seed=stream_seed(c.seed, t, "sr-cache-refresh")))
                    rewards = rewards_from_means(means, c.responses)
                    ranked = cached_sr_set(self.candidates, rewards, len(self.candidates), c.seed, c.responses)
                    key = None
                    if self.cache_store is not None:
                        info = cache_key(source_checkpoint_hash=self.cache_context.get("state_hash_at", {}).get(t,
                                         f"{self.run_id}@{t}"), pool_ids=self.candidates,
                                         prompt_hash=self.cache_context.get("prompt_hash"),
                                         generation_spec=self.cache_context.get("generation_spec"),
                                         verifier_hash=self.cache_context.get("verifier_hash"),
                                         responses=c.responses, rng_spec=f"stream_seed(seed,{t},'sr-cache-refresh')")
                        self.cache_store.commit(info, rewards, self.candidates, source_policy_update=t)
                        key = info["key"]
                    self.sr_ranked_ids = tuple(ranked)
                    self.cache_state = {"version": f"refresh@{t}", "source_policy_update": t, "cache_id": key}
                    self.cache_log.append({"completed_update": t, "event": "committed", "cache_id": key,
                                           "attempt": attempt})
                    return
                except Exception as exc:  # noqa: BLE001 - policy: retain previous cache and log
                    self.cache_log.append({"completed_update": t, "event": "failed", "attempt": attempt,
                                           "error": f"{type(exc).__name__}: {exc}"[:300]})
                    if cp.get("refresh_failure", "retain_previous_and_log") != "retain_previous_and_log":
                        raise

        def update(self):
            t, c = self.step, self.config
            if self.phase == "on_policy":
                decision = self.policy.before_update(t, self.stats)
                if decision is not None:
                    self._log_decision(decision)
                    if decision["action"] == "switch":
                        self._enter(self.destination, t, decision)
            needs_d = False
            if self.phase == "on_policy" and t % c.selection_interval == 0:
                needs_d = (self.policy.uses_signal and self.policy.needs_signal(t)) or (
                    self.monitor and self._monitor_due(t))
            self.arm = "switch" if needs_d else BASE_ARM[self.phase]
            self._check_step, self._stash, self._sr_tail = (t if needs_d else None), {}, None
            self._maybe_refresh_cache(t)
            record = super().update()
            # The base names "sr" for any non-On training; report the v8 destination instead.
            record.update(selector="on_policy" if record.get("selector") == "on_policy" else self.phase,
                          v8_phase=self.phase, transition_update=self.transition_update,
                          sr_comparison=self._sr_tail, cache_version=self.cache_state["version"])
            m = (record.get("metrics") or {}).get("sample_reward")
            n = len(record.get("train_ids") or [])
            self.stats.add(self.step, None if m is None else float(m) * n * c.responses,
                           None if m is None else n * c.responses)
            self._check_step = None
            return record

        def state_dict(self):
            s = super().state_dict()
            s["v8"] = {"protocol": V8_PROTOCOL, "method": self.method, "phase": self.phase,
                       "destination": self.destination, "transition_update": self.transition_update,
                       "entry_update": self.entry_update, "policy": self.policy.state_dict(),
                       "monitor": self.monitor, "stats": self.stats.state_dict(),
                       "signals": list(self.signal_log), "decisions": list(self.decision_log),
                       "cache_log": list(self.cache_log), "cache_state": dict(self.cache_state)}
            return s

        def load_state_dict(self, state, *, fork_arm=None):
            v8 = state.get("v8")
            parent_stats = (TrainingRewardStats.from_state(v8["stats"]) if v8 else
                            stats_from_history(state.get("history"), self.config.responses))
            resume = fork_arm is None
            if resume and v8 and v8["cache_state"]["version"] != "initial":
                self.sr_ranked_ids = tuple(state["sr_ranked_ids"])  # refreshed ranking (verified by caller)
            super().load_state_dict(state, fork_arm=fork_arm)
            if resume:
                if not v8 or v8["protocol"] != V8_PROTOCOL:
                    raise ValueError("resume requires a v8 checkpoint; fork from code-repo checkpoints")
                if v8["method"] != self.method:
                    raise ValueError("method specification changed between save and resume")
                self.phase, self.destination = v8["phase"], v8["destination"]
                self.transition_update, self.entry_update = v8["transition_update"], v8["entry_update"]
                self.policy.load_state_dict(v8["policy"])
                self.monitor = v8["monitor"]
                self.stats = parent_stats
                self.signal_log, self.decision_log = list(v8["signals"]), list(v8["decisions"])
                self.cache_log, self.cache_state = list(v8["cache_log"]), dict(v8["cache_state"])
                self.arm = BASE_ARM[self.phase]
            else:
                self.stats = parent_stats  # training rewards before the branch are observable history
                self.signal_log, self.decision_log, self.cache_log = [], [], []
                self.phase = self.method.get("initial_selector", "on_policy")
                if self.phase != "on_policy":
                    self.entry_update = self.step
            self._install_rule()

    PolicyEngine.ObserverRule = ObserverRule
    return PolicyEngine


def fork_arm_for(method):
    return BASE_ARM[method.get("initial_selector", "on_policy")]
