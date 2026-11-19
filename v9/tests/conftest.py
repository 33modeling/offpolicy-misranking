import os
from pathlib import Path
import sys

V9 = Path(__file__).resolve().parents[1]
REPO = V9.parent
sys.path.insert(0, str(V9))
os.environ.setdefault("V8_CODE_ROOT", str(REPO))
os.environ.setdefault("V8_FSYNC", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from v9lib import env  # noqa: E402

env.activate()
