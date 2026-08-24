#!/usr/bin/env bash
# 데이터셋 자가진단 — 로더가 어디를 찾아보는지·실제로 로드되는지 한 방에.
# "다운받았는데 동작 안함"이면 이것부터: 받은 위치와 찾는 위치의 불일치가 보인다.
#   bash scripts/check_data.sh mbpp
#   bash scripts/check_data.sh dapo-math 64 16
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
DS="${1:?사용법: bash scripts/check_data.sh <gsm8k|math500|mbpp|kk|arc-challenge|dapo-math|apps> [n_train n_val]}"
"$PY" - "$DS" "${2:-32}" "${3:-16}" <<'PYEOF'
import os
import sys
import time

sys.path.insert(0, "src")
from data import _dataset_bases, load_prompts, reward

ds, nt, nv = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
print(f"[check] DATASET={ds}")
print(f"[check] DATASETS_DIR={os.environ.get('DATASETS_DIR')}  OM_DATA={os.environ.get('OM_DATA')}")
print(f"[check] HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} "
      f"HF_DATASETS_OFFLINE={os.environ.get('HF_DATASETS_OFFLINE')} "
      f"(오버라이드: {ds.upper().replace('-', '_')}_DIR={os.environ.get(ds.upper().replace('-', '_') + '_DIR')})")
for b in _dataset_bases():
    if not b.exists():
        print(f"  base[X] {b}")
        continue
    print(f"  base[O] {b}")
    sub = b / ds
    if sub.exists():
        files = sorted(p.name for p in sub.rglob("*") if p.is_file())[:8]
        print(f"    → {ds}/ 있음, 파일: {files}")
    else:
        have = sorted(p.name for p in b.iterdir() if p.is_dir())[:12]
        print(f"    → {ds}/ 없음 (이 base에 있는 폴더: {have})")
t0 = time.time()
try:
    r = load_prompts(ds, nt, nv, seed=0)
except Exception as e:
    print(f"[check] 로드 실패 ({time.time() - t0:.1f}s): {type(e).__name__}")
    print(str(e)[:900])
    sys.exit(1)
print(f"[check] 로드 OK ({time.time() - t0:.1f}s) — train {len(r['train'])} / val {len(r['val'])}")
q0 = r["train"][0]
print("[check] 첫 question 100자:", q0["question"][:100].replace("\n", " "))
print("[check] 첫 answer 60자:", str(q0["answer"])[:60].replace("\n", " "))
if ds == "mbpp":  # 실행 채점 경로까지 확인 (sandbox subprocess)
    t0 = time.time()
    print(f"[check] reward(빈 코드) = {reward('```python\npass\n```', q0['answer'])} "
          f"({time.time() - t0:.1f}s) — 0.0이면 채점기 동작 정상")
print("[check] 전부 정상 — 이 데이터셋으로 실행 가능")
PYEOF
