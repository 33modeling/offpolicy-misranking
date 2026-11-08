"""Method registry: selector x transition policy x destination x cache policy (plan Code §5.2)."""

from __future__ import annotations

import re

from .common import sha256_json

PREFIX = 25


def method(method_id: str, *, frozen: dict | None = None, interval: int = 25, prefix: int = PREFIX) -> dict:
    """Resolve a method id to its specification. `frozen` supplies tuned values (read-only).

    `interval` is the selection/check cadence (25 in every cohort; smaller only in the tiny smoke).
    """
    frozen = frozen or {}
    m = None
    if method_id in ("on_continued", "on_monitored"):
        m = dict(initial_selector="on_policy", destination=None, policy={"policy": "never"},
                 family="on_continued", monitor=method_id == "on_monitored")
    elif method_id == "sr_early":
        m = dict(initial_selector="sr", destination=None, policy={"policy": "never"}, family="sr_early")
    elif method_id == "uniform_early":
        m = dict(initial_selector="uniform", destination=None, policy={"policy": "never"}, family="uniform_early")
    elif re.fullmatch(r"fixed_sr_t\d{3}", method_id):
        m = dict(initial_selector="on_policy", destination="sr",
                 policy={"policy": "fixed", "fixed_update": int(method_id[-3:])}, family="fixed_sr")
    elif method_id == "fixed_sr_tuned":
        tau = frozen.get("fixed_sr_tuned_tau")
        if tau is None:
            raise ValueError("fixed_sr_tuned requires frozen_tuning.json (development-selected tau)")
        m = dict(initial_selector="on_policy", destination="sr", policy={"policy": "fixed", "fixed_update": tau},
                 family="fixed_sr", tuned_from=frozen.get("tuning_hash"))
    elif re.fullmatch(r"bank_(sr|uniform)_t\d{3}", method_id):
        dest = method_id.split("_")[1]
        m = dict(initial_selector=dest, destination=None, policy={"policy": "never"},
                 family="fixed_sr" if dest == "sr" else "fixed_uniform", branch_update=int(method_id[-3:]))
    elif re.fullmatch(r"srgc_(current|one_negative|two_negative|three_negative)(_c50)?", method_id):
        rule = re.sub(r"_c50$", "", method_id[len("srgc_"):])
        every = 2 * interval if method_id.endswith("_c50") else interval
        m = dict(initial_selector="on_policy", destination="sr",
                 policy={"policy": "sr_gc", "rule": rule, "check_interval": every, "first_check": every},
                 family="srgc", check_interval=every, first_check=every)
    elif method_id == "srgc_uniform_destination":
        m = dict(initial_selector="on_policy", destination="uniform",
                 policy={"policy": "sr_gc", "rule": "current", "check_interval": interval,
                         "first_check": interval}, family="srgc_uniform_destination")
    elif re.fullmatch(r"plateau_sr_eps\d{4}", method_id):
        eps = int(method_id[-4:]) / 1000.0
        m = dict(initial_selector="on_policy", destination="sr",
                 policy={"policy": "reward_plateau", "plateau": {"window_updates": interval, "epsilon": eps,
                         "consecutive_checks": 2, "check_interval": interval}}, family="plateau_sr")
    elif method_id == "plateau_sr_tuned":
        eps = frozen.get("plateau_epsilon")
        if eps is None:
            raise ValueError("plateau_sr_tuned requires a frozen epsilon")
        m = method(f"plateau_sr_eps{int(round(eps * 1000)):04d}", interval=interval, prefix=prefix)
        m["tuned_from"] = frozen.get("tuning_hash")
    elif method_id == "sampled_time_sr":
        m = dict(initial_selector="on_policy", destination="sr",
                 policy={"policy": "sampled_time", "sampled_update": None, "distribution_hash": None,
                         "schedule_seed": None}, family="sampled_time_sr")
    elif re.fullmatch(r"sr_early_refreshed_k\d{3}", method_id):
        m = dict(initial_selector="sr", destination=None, policy={"policy": "never"}, family="sr_refreshed",
                 cache={"policy": "periodic_current", "refresh_interval": int(method_id[-3:]),
                        "refresh_origin": "sr_entry_update", "refresh_failure": "retain_previous_and_log",
                        "max_attempts": 2})
    elif re.fullmatch(r"srgc_current_refreshed_k\d{3}", method_id):
        m = method("srgc_current", interval=interval, prefix=prefix)
        m["cache"] = {"policy": "periodic_current", "refresh_interval": int(method_id[-3:]),
                      "refresh_origin": "sr_entry_update", "refresh_failure": "retain_previous_and_log",
                      "max_attempts": 2, "scope": "after_transition_only"}
        m["family"] = "srgc"
    elif method_id == "switch_repeat_external":
        m = dict(initial_selector="on_policy", destination="sr", policy={"policy": "bidirectional"},
                 family="srgc", external_arm="switch_repeat")
    if m is None:
        raise ValueError(f"unknown method {method_id!r}")
    m.setdefault("cache", {"policy": "static_initial"})
    m.setdefault("monitor", False)
    m["method_id"] = method_id
    m["prefix_updates"] = prefix
    return m


def equivalent_key(m: dict, *, start_update: int):
    """Two specs with the same key are the same computation (dedupe; plan Code §8 E1)."""
    pol = m["policy"]
    if pol["policy"] == "fixed" and pol["fixed_update"] == start_update:
        core = {"selector": m["destination"], "start": start_update, "cache": m["cache"]}
    elif pol["policy"] == "never" or (pol["policy"] == "sampled_time" and pol.get("sampled_update") is None
                                      and pol.get("distribution_hash")):
        core = {"selector": m["initial_selector"], "start": start_update, "cache": m["cache"]}
    else:
        core = {k: m[k] for k in ("initial_selector", "destination", "policy", "cache")}
        core["start"] = start_update
        core["check"] = (m.get("check_interval"), m.get("first_check"))
    return sha256_json(core)


EXECUTION_FIELDS = ("initial_selector", "destination", "policy", "cache", "monitor", "check_interval",
                    "first_check", "prefix_updates")


def method_hash(m: dict) -> str:
    return sha256_json({k: m.get(k) for k in EXECUTION_FIELDS})
