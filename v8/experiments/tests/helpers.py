import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from switching import backends, coderepo  # noqa: E402

try:
    coderepo.activate()
    HAS_CODE = True
except FileNotFoundError:
    HAS_CODE = False


def toy_setup(seed=5, prefix_updates=25):
    from switching.engine import make_engine_class
    srgc = coderepo.srgc()
    PE = make_engine_class(srgc)
    feat, ans, data = backends.toy_problem(0)
    cfg = srgc.Config(seed=seed, projection_dim=64)

    def backend(record=None):
        return backends.toy_backend(feat, ans, seed=seed, record=record)
    base = srgc.Engine(backend(), data["candidate_ids"], data["ranking_validation_ids"], data["cached_rewards"],
                       arm="on_policy", config=cfg)
    base.run_until(prefix_updates)
    return srgc, PE, data, cfg, backend, base.state_dict()


def tmpdir():
    return Path(tempfile.mkdtemp(prefix="v8-test-"))
