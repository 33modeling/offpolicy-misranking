#!/usr/bin/env bash
# 실행 전 셋팅 일괄 점검 — 코드 버전·venv·GPU·디스크·모델·데이터(목표 n 기준).
#   bash scripts/preflight.sh
# 전부 OK가 아니면 본실행을 시작하지 말 것.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "== 코드: $(git -c safe.directory='*' log --oneline -1 2>/dev/null || echo 'git 정보 없음')"
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
if [ -x "$PY" ]; then echo "== venv OK: $VENV_DIR"; else echo "✘ venv 없음 — bash scripts/provision.sh"; exit 1; fi
echo "== GPU: $(nvidia-smi -L 2>/dev/null | wc -l)장"
df -h "$OM_WORK" 2>/dev/null | tail -1 | awk '{print "== 디스크("$6"): 남음 "$4" (사용 "$5")"}'

ok7=0
for m in "$MODELS_DIR/Qwen2.5-7B-Instruct" "$MODELS_DIR/Qwen2.5-14B-Instruct" "$MODELS_DIR/Qwen3.6-27B"; do
  if [ -f "$m/config.json" ]; then
    echo "== 모델 OK: $(basename "$m")"
    [ "$(basename "$m")" = "Qwen2.5-7B-Instruct" ] && ok7=1
  else
    echo "   (모델 없음: $(basename "$m") — 7B만 필수, 14B/27B는 boost/G블록용)"
  fi
done
[ "$ok7" = 1 ] || { echo "✘ 7B 스냅샷 없음 — provision.sh 먼저"; exit 1; }

"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, "src")
from data import load_prompts
fail = 0
for d, n, v in [("gsm8k", 512, 100), ("dapo-math", 512, 100),
                ("math500", 400, 100), ("mbpp", 512, 100)]:
    try:
        r = load_prompts(d, n, v)
        print(f"== 데이터 OK: {d} (train {len(r['train'])}/val {len(r['val'])})")
    except Exception as e:
        print(f"✘ 데이터 실패: {d} — {str(e)[:90]}")
        fail = d in ("gsm8k", "dapo-math") or fail
sys.exit(1 if fail else 0)
PYEOF
rc=$?
[ "$rc" = 0 ] || { echo "✘ 필수 데이터(gsm8k/dapo) 문제 — fetch_datasets.sh 후 재점검"; exit 1; }

echo "== 전부 OK — 실행: bash scripts/go_full.sh 2>&1 | tee go_full.console.log"
