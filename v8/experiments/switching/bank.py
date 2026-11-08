"""Branch Bank collection and table lookup (rebuttal plan v2 §5.2-5.6).

Every one-way rule that fires at a 25-update boundary tau has outcome = the SR branch
from the continued On-policy trajectory's snapshot at tau (identical model, optimizer,
selector state and RNG; the check at tau mutates none of them). Lookup is EXACT: a
trigger without a branch at that tau is `unavailable` (no interpolation, no nearest
grid). Rules replayed on a log that ends early are `censored`, reported separately.
Costs are composed from the trajectory ledger (updates up to tau, plus the check at tau
for signal-based rules) and the branch ledger.
"""

from __future__ import annotations

from pathlib import Path
import re

from .budget import select_endpoint
from .common import read_json, read_jsonl
from .ledger import REQUIRED, check_schedule_filter, component_totals, views
from .matrix import monitored_observations
from .policies import TrainingRewardStats, replay_plateau
from .rules import replay, subsample_cadence
from .runner import run_events


def eval_index(out_root):
    """(run_id, completed_update, split) -> first committed evaluation record (pre-specified rule)."""
    idx = {}
    for p in sorted((Path(out_root) / "evaluations").glob("*.json")):
        r = read_json(p)
        key = (r["run_id"], r["completed_update"], r["split"])
        if key not in idx or r["evaluated_at"] < idx[key]["evaluated_at"]:
            idx[key] = r
    return idx


def runs_for(out_root, cohort, seed, role):
    out = {}
    for m in sorted((Path(out_root) / "runs").glob("*/manifest.json")):
        man = read_json(m)
        if (man["cohort_id"], man["training_seed"], man["role"]) == (cohort, seed, role):
            out.setdefault(man["method_id"], []).append(man)
    return out


def _snapshots(out_root, run_id):
    return {r["completed_update"]: r for r in read_jsonl(Path(out_root) / "runs" / run_id / "snapshots.jsonl")}


class SeedBank:
    def __init__(self, out_root, cfg, cohort, seed, role, *, prefix_cost=None, cache_cost=None, evals=None):
        self.out_root, self.cfg, self.cohort, self.seed, self.role = Path(out_root), cfg, cohort, seed, role
        t = cfg["training"]
        self.H, self.ci, self.fc = t["matched_update_endpoint"], t["check_interval"], t["first_check"]
        self.last_check = self.fc + ((self.H - 1 - self.fc) // self.ci) * self.ci
        self.runs = runs_for(out_root, cohort, seed, role)
        self.evals = evals if evals is not None else eval_index(out_root)
        self.prefix_cost, self.cache_cost = prefix_cost, cache_cost
        traj = [m for m in self.runs.get("on_monitored", []) if m["task"]["parent"]["kind"] != "snapshot"]
        if len(traj) > 1:
            raise ValueError(f"{cohort} seed {seed} ({role}): {len(traj)} On trajectories "
                             f"{sorted(m['run_id'] for m in traj)}; D values and branches would mix -- move the "
                             "superseded run directory out of runs/")
        self.traj = traj[0] if traj else None
        self.traj_complete = bool(self.traj) and self.traj["status"] in ("completed_horizon", "completed_budget")
        self._traj_events = run_events(self.out_root / "runs" / self.traj["run_id"]) if self.traj else []
        self.branches, self.extensions = {}, {}
        for mid, mans in self.runs.items():
            hit = re.fullmatch(r"bank_(sr|uniform)_t(\d{3})", mid)
            if hit:
                for man in mans:
                    if man["task"]["method"].get("extension_of"):
                        continue
                    if self.traj is not None and not self._from_trajectory(man):
                        continue  # forked from another trajectory: never combined with this one's D values
                    key = (hit.group(1), int(hit.group(2)))
                    if key in self.branches and self.branches[key]["run_id"] != man["run_id"]:
                        raise ValueError(f"seed {seed}: two {mid} branches of one trajectory "
                                         f"({self.branches[key]['run_id']}, {man['run_id']})")
                    self.branches[key] = man
        by_id = {m["run_id"]: k for k, m in self.branches.items()}
        for mans in self.runs.values():
            for man in mans:
                src = man["task"]["method"].get("extension_of")
                if src in by_id:
                    self.extensions[by_id[src]] = man

    def _from_trajectory(self, man):
        parent = man["task"]["parent"]
        if parent.get("kind") == "snapshot":
            return parent.get("run_id") == self.traj["run_id"]
        if parent.get("kind") == "prefix":  # tau = prefix boundary: same prefix state as the trajectory
            return parent.get("sha256") == self.traj["task"]["parent"].get("sha256")
        return False

    # ---------------- observations
    def d_observations(self):
        if not self.traj:
            return [], None
        obs = monitored_observations(self.out_root, self.traj["run_id"])
        through = self.last_check if self.traj_complete else (max((s for s, _ in obs), default=0))
        return obs, through

    def training_stats(self):
        stats = TrainingRewardStats()
        if self.traj:
            for r in read_jsonl(self.out_root / "runs" / self.traj["run_id"] / "training_stats.jsonl"):
                stats.add(r["completed_update"], r["reward_sum"], r["valid_response_count"])
        return stats

    # ---------------- costs
    def _traj_totals(self, upto, include_check, schedule=None):
        keep = check_schedule_filter(*schedule) if schedule else None
        return component_totals(self._traj_events, max_update=upto, include_selection_at=upto if include_check else None,
                                keep=keep)

    def composed(self, family, tau, branch=None, *, include_check, upto=None, schedule=None):
        """Component totals for a logical policy: trajectory to tau (+check at tau) + branch (to `upto`).

        schedule=(first_check, interval) charges SR-GC comparison cost only at that policy's own checks.
        """
        t_tot, t_missing = self._traj_totals(tau, include_check, schedule)
        totals, missing = dict(t_tot), list(t_missing)
        if branch is not None:
            b_tot, b_missing = component_totals(run_events(self.out_root / "runs" / branch["run_id"]), max_update=upto)
            for k, v in b_tot.items():
                totals[k] = totals.get(k, 0.0) + v
            missing += b_missing
        v = views(totals, family, prefix_gpu_seconds=self.prefix_cost, cache_gpu_seconds=self.cache_cost)
        required = {"on_selection", "sr_comparison", "training", "cache_refresh", "cheap_signal", "sr_preparation"}
        if any(m.get("component") in required for m in missing):
            v.update(continuation_algorithm=None, cold_start_algorithm=None,
                     cold_start_lower_bound=_known_lower_bound(v),
                     cold_start_missing=[*v["cold_start_missing"], "unmeasured_events"],
                     cost_missing_events=len(missing))
        return v

    def _reward(self, run_id, update, split="final"):
        r = self.evals.get((run_id, update, split))
        return (r["reward_fraction"], r["n_questions"], r["responses_per_question"]) if r else (None, None, None)

    def budget_endpoint(self, family, tau, branch, budget, *, include_check, schedule=None):
        """Eligible snapshots of the composed policy with cumulative cost at availability (metadata only).

        If the latest snapshot is chosen and the run simply stopped at the matched horizon (no budget
        extension), the budget endpoint is unknown: status `capped_at_horizon`, never `selected`.
        """
        rows, ext = [], None
        traj_snaps = _snapshots(self.out_root, self.traj["run_id"]) if self.traj else {}
        for u, snap in traj_snaps.items():
            if tau is None or u <= tau:
                c = self.composed(family, u, None, include_check=False, schedule=schedule)["continuation_algorithm"]
                rows.append({"completed_update": u, "cumulative_cost": c, "run_id": self.traj["run_id"],
                             "snapshot_hash": snap["sha256"]})
        if branch is not None:
            for u, snap in _snapshots(self.out_root, branch["run_id"]).items():
                if u > tau:
                    c = self.composed(family, tau, branch, include_check=include_check, upto=u,
                                      schedule=schedule)["continuation_algorithm"]
                    rows.append({"completed_update": u, "cumulative_cost": c, "run_id": branch["run_id"],
                                 "snapshot_hash": snap["sha256"]})
            key = next((k for k, m in self.branches.items() if m["run_id"] == branch["run_id"]), None)
            ext = self.extensions.get(key)
            if ext is not None:
                base = self.composed(family, tau, branch, include_check=include_check, upto=self.H,
                                     schedule=schedule)["continuation_algorithm"]
                ext_events = run_events(self.out_root / "runs" / ext["run_id"])
                for u, snap in _snapshots(self.out_root, ext["run_id"]).items():
                    if u > self.H:
                        tot, miss = component_totals(ext_events, max_update=u)
                        req = REQUIRED[family] - {"prefix", "cache_build"}
                        own = None if any(m.get("component") in req for m in miss) else sum(
                            v for k, v in tot.items() if k in req)
                        rows.append({"completed_update": u, "run_id": ext["run_id"], "snapshot_hash": snap["sha256"],
                                     "cumulative_cost": None if base is None or own is None else base + own})
        # a finished extension ran as far as B allowed (it may stop at 275 when no further update fits)
        finished = ext is not None and ext["status"] in ("completed_horizon", "completed_budget")
        return capped(select_endpoint(rows, budget), rows, stopped_by_budget=finished)

    # ---------------- logical policies
    def policy_rows(self, *, rules=None, cadences=(25, 50), plateau_eps=(0.0, 0.005, 0.01), sampled=None,
                    budget=None):
        rows = []
        if not self.traj:
            return [{"cohort": self.cohort, "seed": self.seed, "policy_id": "on_continued", "status": "missing_trajectory"}]
        rules = rules or ["one_negative", "two_negative", "current", "three_negative"]

        def emit(policy_id, family, status, tau, *, include_check, dest="sr", extra=None, schedule=None):
            row = {"cohort": self.cohort, "seed": self.seed, "role": self.role, "policy_id": policy_id,
                   "family": family, "status": status, "switch_update": tau, "destination": dest}
            if extra:
                row.update(extra)
            branch = None
            if status == "triggered":
                branch = self.branches.get((dest, tau))
                if branch is None or branch["status"] not in ("completed_horizon", "completed_budget"):
                    row.update(status="unavailable", reason=f"no completed {dest} branch at {tau} (exact lookup only)")
                    rows.append(row)
                    return
                run_id = branch["run_id"]
            elif status == "not_triggered":
                run_id = self.traj["run_id"]
            else:
                rows.append(row)
                return
            reward, nq, resp = self._reward(run_id, self.H)
            v = self.composed(family, tau if tau is not None else self.H, branch, include_check=include_check,
                              upto=self.H, schedule=schedule)
            row.update(run_id=run_id, endpoint_type="matched_update", endpoint_update=self.H,
                       reward_fraction=reward, n_questions=nq, responses_per_question=resp,
                       eval_status="measured" if reward is not None else "unmeasured",
                       continuation_gpu_seconds=v["continuation_algorithm"],
                       cold_start_gpu_seconds=v["cold_start_algorithm"],
                       cold_start_lower_bound_gpu_seconds=v["cold_start_lower_bound"],
                       cost_missing=",".join(v["cold_start_missing"]))
            if budget is not None:
                be = self.budget_endpoint(family, tau, branch, budget, include_check=include_check, schedule=schedule)
                row["budget_status"] = be["status"]
                if be["status"] in ("selected", "capped_at_horizon"):
                    ep = be["endpoint"]
                    br, _, _ = self._reward(ep["run_id"], ep["completed_update"])
                    row.update(budget_endpoint_update=ep["completed_update"], budget_reward_fraction=br,
                               budget_eval_status="measured" if br is not None else "unmeasured",
                               budget_slack_gpu_seconds=be["slack"])
            rows.append(row)

        emit("on_continued", "on_continued", "not_triggered", None, include_check=False)
        for (dest, tau), _ in sorted(self.branches.items()):
            if dest == "sr":
                emit(f"fixed_sr_t{tau:03d}", "fixed_sr", "triggered", tau, include_check=False)
        obs, through = self.d_observations()
        for cadence in cadences:
            cobs = subsample_cadence(obs, cadence, cadence if cadence != self.ci else self.fc)
            first = cadence if cadence != self.ci else self.fc
            last = first + ((self.last_check - first) // cadence) * cadence
            for rule in rules:
                r = replay(cobs, rule, first_check=first, interval=cadence, last_scheduled_check=last,
                           observed_through=through)
                pid = f"srgc_{r['rule_id']}" + ("" if cadence == self.ci else f"_c{cadence}")
                emit(pid, "srgc", r["status"], r["switch_update"], include_check=True,
                     extra={"replay_source": "on_trajectory_monitor", "observed_through": r["observed_through"],
                            "checks_used": r["checks_used"]},
                     schedule=None if cadence == self.ci else (first, cadence))
                if rule == "current" and cadence == self.ci and r["status"] == "triggered":
                    emit("srgc_uniform_destination", "srgc_uniform_destination", "triggered", r["switch_update"],
                         include_check=True, dest="uniform")
                    emit(f"fixed_uniform_t{r['switch_update']:03d}", "fixed_uniform", "triggered",
                         r["switch_update"], include_check=False, dest="uniform")
        stats = self.training_stats()
        observed = max(stats.rows, default=0) if not self.traj_complete else self.last_check
        # a trajectory update absent from training_stats.jsonl is unknown (not a warmup reset): stop there
        contiguous = self.traj["task"]["parent"].get("completed_update") or 0
        while contiguous + 1 in stats.rows:
            contiguous += 1
        observed = min(observed, contiguous)
        p = self.cfg["plateau"]
        for eps in plateau_eps:
            r = replay_plateau(stats, window=p["window_updates"], epsilon=eps, consecutive=p["consecutive_checks"],
                               check_interval=p["check_interval"], last_check=self.last_check, observed_through=observed)
            emit(f"plateau_sr_eps{int(round(eps * 1000)):04d}", "plateau_sr", r["status"], r["switch_update"],
                 include_check=False, extra={"replay_source": "on_trajectory_training_rewards"})
        for draw in (sampled or {}).get(str(self.seed), []):
            tau = draw["sampled_update"]
            emit(f"sampled_time_sr.d{draw['draw']}", "sampled_time_sr", "not_triggered" if tau is None else "triggered",
                 tau, include_check=False, extra={"schedule_seed": draw["schedule_seed"],
                                                  "distribution_hash": draw["distribution_hash"]})
        return rows

    def table_a_prime(self):
        """Table A': rewards at the matched horizon for every bank branch (rebuttal plan v2 §5.2)."""
        out = {"seed": self.seed, "on_continued": self._reward(self.traj["run_id"], self.H)[0] if self.traj else None}
        for (dest, tau), man in sorted(self.branches.items()):
            out[f"{'F' if dest == 'sr' else 'U'}{tau}"] = self._reward(man["run_id"], self.H)[0]
        return out


def tau_distribution(rows, *, source_role="development"):
    """E9: empirical categorical distribution of srgc_current tau (incl. no-switch mass) from development rows."""
    from collections import Counter
    from .common import sha256_json
    picks = [r for r in rows if r["policy_id"] == "srgc_current" and r.get("role") == source_role]
    outcomes = []
    for r in picks:
        st = r["status"]
        if st == "ran":  # prospective run: completed; its manifest says whether and when it switched
            st = ("triggered" if r.get("switch_update") is not None else
                  "not_triggered" if r.get("no_switch") is True else "unknown")
        if st == "triggered" and r.get("switch_update") is not None:
            outcomes.append(r["switch_update"])
        elif st == "not_triggered":
            outcomes.append(None)
        else:  # censored / unavailable / failed / interrupted / active / unknown: the outcome is not observed
            raise ValueError(f"cannot freeze a tau distribution: seed {r.get('seed')} srgc_current status "
                             f"{r['status']!r} is not an observed switch/no-switch outcome")
    counts = Counter(outcomes)
    n = sum(counts.values())
    if not n:
        raise ValueError("no development srgc_current outcomes")
    support = [{"switch_update": k, "meaning": "no_switch" if k is None else "switch", "probability": v / n}
               for k, v in sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))]
    dist = {"source_split": source_role, "source_seeds": sorted({r["seed"] for r in picks}), "support": support,
            "n": n, "frozen_before_test": True}
    dist["distribution_id"] = sha256_json(dist)[:16]
    return dist


def sample_schedules(dist, seeds, *, draws=1, base_seed=20261108):
    """Draw tau once per (training seed, draw) BEFORE any test run; resume never re-draws."""
    import numpy as np
    from .common import sha256_json
    h = sha256_json(dist)
    out = {}
    for seed in seeds:
        out[str(seed)] = []
        for d in range(draws):
            schedule_seed = int.from_bytes(bytes.fromhex(sha256_json([base_seed, seed, d, h])[:16]), "little") % (2**31 - 1)
            rng = np.random.default_rng(schedule_seed)
            probs = [s["probability"] for s in dist["support"]]
            pick = dist["support"][int(rng.choice(len(probs), p=probs))]["switch_update"]
            out[str(seed)].append({"draw": d, "schedule_seed": schedule_seed, "sampled_update": pick,
                                   "distribution_hash": h})
    return out


def prospective_rows(out_root, cfg, cohort, role, seeds, *, evals=None, prefix_cost=None, cache_cost=None,
                     budget=None, split="final"):
    """One row per full-policy run started from the prefix (development or test role)."""
    from .ledger import views
    evals = evals if evals is not None else eval_index(out_root)
    H = cfg["training"]["matched_update_endpoint"]
    rows = []
    matrix_aliases = _matrix_aliases(out_root, role)
    for seed in seeds:
        for mid, mans in runs_for(out_root, cohort, seed, role).items():
            for man in mans:
                if man["task"]["method"].get("extension_of") or man["task"]["parent"]["kind"] == "snapshot":
                    continue
                rdir = Path(out_root) / "runs" / man["run_id"]
                done = man["status"] in ("completed_horizon", "completed_budget")
                fam = man["task"]["method"]["family"]
                tot, missing = component_totals(run_events(rdir))
                v = views(tot, fam, prefix_gpu_seconds=prefix_cost, cache_gpu_seconds=cache_cost)
                if missing:  # an unmeasured event: cold-start is unknown too, only a lower bound is known
                    v.update(cold_start_algorithm=None, cold_start_lower_bound=_known_lower_bound(v),
                             cold_start_missing=[*v["cold_start_missing"], "unmeasured_events"])
                r = evals.get((man["run_id"], H, split))
                row = {"cohort": cohort, "seed": seed, "role": role, "policy_id": mid, "family": fam,
                       "run_id": man["run_id"], "status": "ran" if done else man["status"],
                       "switch_update": man.get("transition_update"), "no_switch": man.get("no_switch"),
                       "endpoint_type": "matched_update", "endpoint_update": H, "split": split,
                       "reward_fraction": r["reward_fraction"] if r else None,
                       "eval_status": "measured" if r else "unmeasured",
                       "continuation_gpu_seconds": None if missing else v["continuation_algorithm"],
                       "cold_start_gpu_seconds": v["cold_start_algorithm"],
                       "cold_start_lower_bound_gpu_seconds": v["cold_start_lower_bound"],
                       "cost_missing": ",".join(v["cold_start_missing"])}
                if budget is not None and done:
                    snaps = [{"completed_update": u, "cumulative_cost": s["own_continuation_gpu_seconds"],
                              "snapshot_hash": s["sha256"]} for u, s in _snapshots(out_root, man["run_id"]).items()]
                    be = capped(select_endpoint(snaps, budget), snaps,
                                stopped_by_budget=man["status"] == "completed_budget")
                    row["budget_status"] = be["status"]
                    if be["status"] in ("selected", "capped_at_horizon"):
                        u = be["endpoint"]["completed_update"]
                        br = evals.get((man["run_id"], u, split))
                        row.update(budget_endpoint_update=u, budget_reward_fraction=br["reward_fraction"] if br else None,
                                   budget_eval_status="measured" if br else "unmeasured", budget_slack_gpu_seconds=be["slack"])
                rows.append(row)
                # deduplicated arms (e.g. fixed_sr_tuned at tau=25 = sr_early, a no-switch sampled draw = On)
                for alias in run_aliases(man, matrix_aliases):
                    rows.append(dict(row, policy_id=alias, alias_of=mid))
    return rows


def _matrix_aliases(out_root, role):
    """run_id -> method ids merged into it by matrix.dedupe, from the stage matrices of this role."""
    out = {}
    for p in sorted((Path(out_root) / "matrix").glob(f"*.{role}.json")):
        for t in read_json(p).get("tasks", []):
            if t.get("aliases") and t.get("run_id"):
                out.setdefault(t["run_id"], [])
                out[t["run_id"]] += [a for a in t["aliases"] if a not in out[t["run_id"]]]
    return out


def run_aliases(man, matrix_aliases=None):
    """Aliases of a run: its manifest task (runner path), else the stage matrix (coordinator strips them)."""
    aliases = list(man["task"].get("aliases") or [])
    aliases += [a for a in (matrix_aliases or {}).get(man["run_id"], []) if a not in aliases]
    return [a for a in aliases if a != man["method_id"]]


def capped(selection, snapshots, *, stopped_by_budget):
    """A 'selected' latest snapshot of a run that stopped at its horizon is not a budget endpoint."""
    if selection.get("status") != "selected" or stopped_by_budget:
        return selection
    last = max(s["completed_update"] for s in snapshots if s.get("eligible", True))
    if selection["endpoint"]["completed_update"] == last:
        return dict(selection, status="capped_at_horizon",
                    reason="run ended at the matched horizon with budget left; the budget endpoint is not observed")
    return selection


def _known_lower_bound(v):
    return v["cold_start_algorithm"] if v.get("cold_start_algorithm") is not None else v.get("cold_start_lower_bound")
