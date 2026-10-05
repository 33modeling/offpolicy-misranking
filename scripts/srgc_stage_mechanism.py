"""Paired GRPO interventions at fixed states, separate from deployed Switch runs."""

from contextlib import contextmanager
from dataclasses import asdict
import copy
import hashlib
from pathlib import Path

import numpy as np

from srgc_rebuttal.srgc import Engine, cosine_scores, stream_seed, top_ids

ARM = "stage_mechanism"
PROTOCOL = "grpo-stage-interventions-v1"
STAGES = (0, 100, 400)
HORIZON = 25
MODES = ("on_policy", "direction_shuffle", "sr", "sr_shuffle", "sr_fresh", "random")
TOTAL_WORK = STAGES[-1] + len(STAGES) * len(MODES) * HORIZON


def code_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def select_sets(ids, cached, fresh, scores, *, seed, step, k, tie_order):
    """Every intervention ranks the SAME candidate draw; B never selects data."""
    ids = tuple(ids)
    rng = np.random.default_rng(stream_seed(seed, step, "mechanism-permutation"))
    sr = np.array([-abs(cached[i] - .5) for i in ids])
    rankings = dict(on_policy=np.asarray(scores), direction_shuffle=rng.permutation(scores),
                    sr=sr, sr_shuffle=rng.permutation(sr),
                    sr_fresh=np.array([-abs(fresh[i] - .5) for i in ids]))
    selected = {}
    for mode, values in rankings.items():
        by_id = dict(zip(ids, values))
        selected[mode] = sorted(ids, key=lambda i: (-by_id[i], tie_order[i]))[:k]
    selected["random"] = rng.choice(ids, k, replace=False).tolist()
    return selected


@contextmanager
def capture_rewards(backend):
    """Observe existing TorchBackend rollouts, including resumed rollout files."""
    original = backend._rollout
    local = {}

    def rollout(prompt, responses, seed):
        result = original(prompt, responses, seed)
        rewards = np.asarray(result[1], dtype=float)
        if rewards.shape != (responses,) or not np.isin(rewards, [0., 1.]).all():
            raise ValueError("mechanism diagnostics require binary rollout rewards")
        local[prompt] = rewards.tolist()
        return result

    backend._rollout = rollout
    try:
        yield local
    finally:
        backend._rollout = original


def gather_rewards(backend, local, ids):
    values = backend._gather(local)
    if set(values) != set(ids):
        raise ValueError("incomplete captured rewards across ranks")
    return values


class StageStudy:
    """Resumable state machine: one action/update per checkpoint, no branch leakage."""

    def __init__(self, backend, data, config, *, stages=STAGES, horizon=HORIZON):
        if config.objective != "grpo":
            raise ValueError("stage mechanism is a GRPO experiment, not RLOO")
        if not stages or stages[0] != 0 or tuple(sorted(set(stages))) != tuple(stages) or horizon < 1:
            raise ValueError("invalid stage/horizon protocol")
        if (set(data["evaluation_ids"]) &
                (set(data["candidate_ids"]) | set(data["ranking_validation_ids"]))):
            raise ValueError("reporting evaluation must be disjoint from scoring/training")
        self.backend, self.data, self.config = backend, data, config
        self.stages, self.horizon = tuple(stages), horizon
        self.carrier = Engine(backend, data["candidate_ids"], data["ranking_validation_ids"],
                              data["cached_rewards"], arm="on_policy", config=config)
        self.tie_order = {i: j for j, i in enumerate(top_ids(
            self.carrier.candidates, [0.] * len(self.carrier.candidates),
            len(self.carrier.candidates), config.seed + 1000))}
        self.step = 0  # Physical updates across carrier AND counterfactual branches.
        self.stage_index = self.branch_index = self.local_updates = 0
        self.phase = "carrier"
        self.anchor = None
        self.measurement = None
        self.baseline = None
        self.rows, self.measurements, self.charges, self.training_records = [], [], [], []

    @property
    def done(self):
        return self.phase == "done"

    @property
    def stage(self):
        return self.stages[min(self.stage_index, len(self.stages) - 1)]

    @property
    def mode(self):
        return MODES[min(self.branch_index, len(MODES) - 1)]

    def timed(self, phase, label, action):
        meter = self.backend.cost_meter
        meter.begin_phase(phase, self.stage, self.backend.gpu_count)
        try:
            with meter.section(label):
                result = action()
        except BaseException:
            meter.active = False
            raise
        event = meter.end_phase()
        self.charges.append(dict(stage=self.stage, component=label,
                                 wall_seconds=event["wall_seconds"], gpu_seconds=event["gpu_seconds"]))
        return result

    def acquire(self, block):
        c, backend = self.config, self.backend
        ids = self.measurement["candidate_ids"]
        reference = self.data["ranking_validation_ids"]

        def score():
            with backend.cost_meter.section("candidate"):
                with capture_rewards(backend) as local:
                    gradients = backend.score_gradients(ids, responses=c.responses,
                        group_size=c.candidate_group_size,
                        seed=stream_seed(c.seed, self.stage, f"mechanism-{block}-candidate"))
                rewards = gather_rewards(backend, local, ids)
            with backend.cost_meter.section("reference"):
                values = backend.score_gradients(reference, responses=c.responses, group_size=c.responses,
                    seed=stream_seed(c.seed, self.stage, f"mechanism-{block}-reference"))
            if set(gradients) != set(ids) or set(values) != set(reference):
                raise ValueError("incomplete scoring gradient set")
            stack = np.stack([gradients[i] for i in ids])
            v = np.stack([values[i] for i in reference]).mean(axis=0)
            if (stack.shape != (len(ids), c.projection_dim) or v.shape != (c.projection_dim,)
                    or not np.isfinite(stack).all() or not np.isfinite(v).all()):
                raise ValueError("invalid projected gradients")
            return dict(cosines=cosine_scores(stack, v).tolist(),
                        dots=(stack @ v).tolist(), norms=np.linalg.norm(stack, axis=1).tolist(),
                        validation_norm=float(np.linalg.norm(v)), rewards=rewards,
                        success_rates={i: float(np.mean(rewards[i])) for i in ids})

        self.measurement[block] = self.timed("selection", f"diagnostic.{block}", score)

    def finalize_measurement(self):
        m, c = self.measurement, self.config
        ids, a, b = m["candidate_ids"], m["A"], m["B"]
        cached = {i: float(np.mean(self.data["cached_rewards"][i])) for i in ids}
        m["selected"] = select_sets(ids, cached, a["success_rates"], a["cosines"],
            seed=c.seed, step=self.stage, k=c.training_prompts, tie_order=self.tie_order)
        repeat_top = select_sets(ids, cached, b["success_rates"], b["cosines"],
            seed=c.seed, step=self.stage, k=c.training_prompts, tie_order=self.tie_order)["on_policy"]
        m["score_correlation"] = correlation(a["cosines"], b["cosines"])
        m["top4_overlap_fraction"] = len(set(m["selected"]["on_policy"]) & set(repeat_top)) / c.training_prompts
        m["cached_success_rates"] = cached
        m["selected_diagnostics"] = {}
        for mode, chosen in m["selected"].items():
            indices = [ids.index(i) for i in chosen]
            m["selected_diagnostics"][mode] = dict(
                independent_cosine=float(np.mean([b["cosines"][j] for j in indices])),
                independent_dot=float(np.mean([b["dots"][j] for j in indices])),
                independent_norm=float(np.mean([b["norms"][j] for j in indices])),
                current_success_rate=float(np.mean([b["success_rates"][i] for i in chosen])),
                current_mixed_group_fraction=float(np.mean([0 < sum(b["rewards"][i]) < c.responses for i in chosen])),
                cache_current_absolute_gap=float(np.mean([abs(cached[i] - b["success_rates"][i]) for i in chosen])))

    def evaluate(self, label):
        values = self.timed("evaluation", label, lambda: self.backend.evaluate(
            self.data["evaluation_ids"], responses=self.config.responses,
            seed=stream_seed(self.config.seed, self.stage, "mechanism-evaluation")))
        if (set(values) != set(self.data["evaluation_ids"]) or not values or
                any(not np.isfinite(v) or not 0 <= v <= 1 for v in values.values())):
            raise ValueError("invalid mechanism evaluation")
        return values

    def advance(self):
        if self.phase == "carrier":
            if self.carrier.step < self.stage:
                with self.backend.cost_meter.section("carrier"):
                    self.carrier.update()
                self.step += 1
                return
            self.anchor = self.carrier.state_dict()
            self.measurement = dict(stage=self.stage, candidate_ids=list(self.carrier._draw_candidates()))
            self.phase = "acquire_A"
        elif self.phase == "acquire_A":
            self.acquire("A")
            self.phase = "acquire_B"
        elif self.phase == "acquire_B":
            self.acquire("B")
            self.timed("preparation", "diagnostic.ranking", self.finalize_measurement)
            self.measurements.append(copy.deepcopy(self.measurement))
            self.phase = "baseline"
        elif self.phase == "baseline":
            self.baseline = self.evaluate("diagnostic.baseline")
            self.phase = "restore_branch"
        elif self.phase == "restore_branch":
            self.timed("checkpoint_load", f"branch.{self.mode}.restore",
                       lambda: self.backend.load_state_dict(self.anchor["backend"]))
            self.local_updates = 0
            self.phase = "train"
        elif self.phase == "train":
            ids = self.measurement["selected"][self.mode]

            def train():
                with capture_rewards(self.backend) as local:
                    metrics = self.backend.train(ids, responses=self.config.responses, objective="grpo",
                        seed=stream_seed(self.config.seed, self.stage + self.local_updates, "mechanism-training"))
                rewards = gather_rewards(self.backend, local, ids)
                return dict(metrics=metrics, rewards=rewards,
                    mixed_group_fraction=float(np.mean([0 < sum(rewards[i]) < self.config.responses for i in ids])))

            record = self.timed("training", f"branch.{self.mode}.train", train)
            self.local_updates += 1
            self.step += 1
            self.training_records.append(dict(stage=self.stage, mode=self.mode, update=self.local_updates,
                                               train_ids=ids, **record))
            if self.local_updates == self.horizon:
                self.phase = "evaluate_branch"
        elif self.phase == "evaluate_branch":
            after = self.evaluate(f"branch.{self.mode}.evaluation")
            before_mean, after_mean = float(np.mean(list(self.baseline.values()))), float(np.mean(list(after.values())))
            self.rows.append(dict(stage=self.stage, mode=self.mode, updates=self.horizon,
                selected_ids=self.measurement["selected"][self.mode], baseline_reward=before_mean,
                reward=after_mean, gain_pp=100 * (after_mean - before_mean),
                baseline_per_question=self.baseline, per_question_reward=after))
            self.branch_index += 1
            self.phase = "restore_carrier" if self.branch_index == len(MODES) else "restore_branch"
        elif self.phase == "restore_carrier":
            self.timed("checkpoint_load", "carrier.restore", lambda: self.carrier.load_state_dict(self.anchor))
            self.stage_index += 1
            self.branch_index = self.local_updates = 0
            self.anchor = self.measurement = self.baseline = None
            self.phase = "done" if self.stage_index == len(self.stages) else "carrier"
        else:
            raise ValueError(f"cannot advance phase {self.phase}")

    def state_dict(self):
        return copy.deepcopy(dict(protocol=PROTOCOL, study_code_sha256=code_hash(), config=asdict(self.config),
            sampling_protocol=self.carrier.SAMPLING_PROTOCOL, arm=ARM, step=self.step,
            stages=self.stages, horizon=self.horizon, modes=MODES, phase=self.phase,
            stage_index=self.stage_index, branch_index=self.branch_index, local_updates=self.local_updates,
            carrier=self.anchor if self.anchor is not None else self.carrier.state_dict(),
            anchor=self.anchor, backend=self.backend.state_dict(), measurement=self.measurement,
            baseline=self.baseline, rows=self.rows, measurements=self.measurements,
            charges=self.charges, training_records=self.training_records))

    def load_state_dict(self, state):
        expected = dict(protocol=PROTOCOL, study_code_sha256=code_hash(), config=asdict(self.config),
                        sampling_protocol=self.carrier.SAMPLING_PROTOCOL, arm=ARM,
                        stages=self.stages, horizon=self.horizon, modes=MODES)
        if any(state.get(k) != v for k, v in expected.items()):
            raise ValueError("mechanism checkpoint protocol/code/config changed")
        self.carrier.load_state_dict(state["carrier"])
        self.backend.load_state_dict(state["backend"])
        for name in ("step", "phase", "stage_index", "branch_index", "local_updates", "anchor",
                     "measurement", "baseline", "rows", "measurements", "charges", "training_records"):
            setattr(self, name, copy.deepcopy(state[name]))


def run_study(backend, data, config, *, out, expected, policy, state, prefix_hash):
    from srgc_rebuttal.cost_ledger import PhaseLedger
    from srgc_rebuttal.distributed import primary
    from srgc_rebuttal.runtime import atomic_json
    from scripts.srgc_step_checkpoints import save_checkpoint
    with backend.cost_meter.phase("preparation", gpu_count=backend.gpu_count):
        with backend.cost_meter.stage("study_setup"):
            study = StageStudy(backend, data, config)
    if state is not None:
        with backend.cost_meter.phase("checkpoint_load", gpu_count=backend.gpu_count):
            with backend.cost_meter.stage("study_restore"):
                study.load_state_dict(state)
    while not study.done:
        primary(lambda: print(f"MECHANISM seed={config.seed} stage={study.stage} mode={study.mode} "
                              f"phase={study.phase} branch_step={study.local_updates}/{HORIZON} "
                              f"work_updates={study.step}/{TOTAL_WORK}", flush=True))
        study.advance()
        save_checkpoint(study, out, ARM, metadata={"checkpoint_policy": policy, **expected})
        primary(lambda: atomic_json(out / f"{ARM}-progress.json", dict(seed=config.seed, arm=ARM,
            step=study.step, total_work_updates=TOTAL_WORK, stage=study.stage, mode=study.mode,
            branch_updates=study.local_updates, phase=study.phase, protocol=PROTOCOL)))
    ledger = primary(lambda: PhaseLedger(out / "cost-receipts" / ARM).totals())
    primary(lambda: atomic_json(out / f"{ARM}-endpoint.json", dict(**expected, arm=ARM,
        protocol=PROTOCOL, study_code_sha256=code_hash(), stages=list(STAGES), horizon=HORIZON,
        total_work_updates=study.step, carrier_updates=STAGES[-1], modes=list(MODES),
        prefix_checkpoint_sha256=prefix_hash, initial_state="fresh-seeded-base-model-not-prefix",
        sampling_protocol=study.carrier.SAMPLING_PROTOCOL, checkpoint_policy=policy,
        rows=study.rows, measurements=study.measurements, training_records=study.training_records,
        charges=study.charges, cost_receipts=ledger, cost_measurement_complete=ledger["complete"])))
    marker = out / f"{ARM}-run.json"
    def finish():
        import json
        atomic_json(marker, {**json.loads(marker.read_text()), "status": "complete"})
    primary(finish)
