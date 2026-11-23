#!/bin/sh
# V8 로그 모으기 (읽기 전용, 실행 중인 worker 는 건드리지 않음)
#   sh scripts/v8_collect_logs.sh            qwen_mbpp 기준으로 이 노드의 로그를 모음
#   sh scripts/v8_collect_logs.sh olmo_math  다른 cohort
# 결과: $WORK/v8-logs/<노드>-<시각>.tar.gz 파일 하나 (노드마다 한 번씩 돌리면 같은 폴더에 쌓임)
set -u
COHORT=${1:-qwen_mbpp}
case "$COHORT" in mbpp) COHORT=qwen_mbpp;; math) COHORT=olmo_math;; esac
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
V8R=${V8_ROOT:-$WORK/v8}
HOST=$(hostname 2>/dev/null || echo node)
STAMP=$(date +%Y%m%d-%H%M%S)
NAME="$HOST-$COHORT-$STAMP"
OUTDIR=${V8_LOG_OUT:-$WORK/v8-logs}
mkdir -p "$OUTDIR" 2>/dev/null || OUTDIR=$HOME/v8-logs; mkdir -p "$OUTDIR"
T=$(mktemp -d "${TMPDIR:-/tmp}/v8-collect.XXXXXX"); D="$T/$NAME"; mkdir -p "$D"
PY=python3; command -v "$PY" >/dev/null 2>&1 || PY=${VENV_DIR:-$WORK/.venv-cu126}/bin/python
TO=""; command -v timeout >/dev/null 2>&1 && TO="timeout"
run() { # run <파일> <초> <명령...>
    f=$1; s=$2; shift 2
    { echo "\$ $*"; if [ -n "$TO" ]; then $TO "$s" "$@"; else "$@"; fi; echo "[exit $?]"; } > "$D/$f" 2>&1
}
tailcp() { # tailcp <원본> <대상> <줄수>
    [ -f "$1" ] || return 0; mkdir -p "$(dirname "$2")"
    sz=$(wc -c < "$1" 2>/dev/null || echo 0)
    if [ "$sz" -gt 5000000 ]; then { echo "[앞부분 생략: 원본 $sz bytes, 마지막 $3줄]"; tail -n "$3" "$1"; } > "$2"; else cp "$1" "$2"; fi
}
echo "V8 로그 모으는 중: cohort=$COHORT node=$HOST (몇 분 걸릴 수 있음)"

# 1) 환경
{
  echo "host=$HOST date=$(date '+%F %T %Z') user=$(id -un) uid=$(id -u)"
  echo "WORK=$WORK"; echo "V8_ROOT=$V8R"; echo "code(here)=$HERE"
  env | grep -E '^(V8_|OM_|GROUP_VOLUME|VENV_DIR|CUDA|NCCL|HF_|TRANSFORMERS|VLLM|PYTORCH|TORCH|PATH=|PYTHON)' | sort
} > "$D/env.txt" 2>&1
run nvidia-smi.txt 60 nvidia-smi
run gpu-apps.txt 60 nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
run disk.txt 60 df -h "$WORK" /tmp /dev/shm "$HOME"
run mem.txt 30 free -h
PYV=${VENV_DIR:-$WORK/.venv-cu126}/bin/python; [ -x "$PYV" ] || PYV=$PY
run python.txt 120 "$PYV" -c "import sys,platform;print(sys.version);import importlib
for m in ('torch','transformers','vllm','accelerate','datasets','flash_attn','deepspeed'):
  try: x=importlib.import_module(m);print(m,getattr(x,'__version__','?'))
  except Exception as e: print(m,'IMPORT FAIL',type(e).__name__,e)
try:
  import torch;print('cuda',torch.version.cuda,'available',torch.cuda.is_available(),'count',torch.cuda.device_count())
except Exception as e: print('torch cuda check fail',e)"
ps -eo pid,ppid,user,etime,pcpu,rss,stat,args 2>/dev/null | grep -E 'v8_|srgc|vllm|torchrun|python' | grep -v grep | cut -c1-400 > "$D/processes.txt"
( cd "$HERE/.." 2>/dev/null && { ls -la; cat SHA256SUMS.txt 2>/dev/null | head -5; git log -1 --oneline 2>/dev/null; } ) > "$D/bundle.txt" 2>&1

# 2) 기존 읽기 전용 보고서
[ -f "$HERE/v8_status.py" ] && run status.txt 300 "$PY" "$HERE/v8_status.py"
[ -f "$HERE/v8_status.py" ] && run status-cohort.txt 300 "$PY" "$HERE/v8_status.py" --cohort "$COHORT"
[ -f "$HERE/run_v8_nodes.sh" ] && run queue-state.json.txt 300 sh "$HERE/run_v8_nodes.sh" --status
[ -f "$HERE/v8_idle_report.sh" ] && V8_IDLE_REPORT="$D/idle-report.txt" run idle-report.console.txt 600 sh "$HERE/v8_idle_report.sh"
[ -f "$HERE/v8_crash_report.sh" ] && V8_CRASH_REPORT="$D/crash-report.txt" run crash-report.console.txt 600 sh "$HERE/v8_crash_report.sh"

# 3) 이 노드의 콘솔 로그 (최근 20개)
CON="${TMPDIR:-/tmp}/v8-console-$(id -u)"
[ -d "$CON" ] || CON="/tmp/v8-console-$(id -u)"
if [ -d "$CON" ]; then
  ls -lt "$CON" > "$D/console-list.txt" 2>&1
  for f in $(ls -t "$CON"/*.log 2>/dev/null | head -20); do tailcp "$f" "$D/console/$(basename "$f")" 20000; done
else echo "콘솔 로그 폴더 없음: $CON" > "$D/console-list.txt"; fi
for f in "$HOME"/v8-*.txt "$HOME"/v8-*.log "$WORK"/*.log; do [ -f "$f" ] && tailcp "$f" "$D/home/$(basename "$f")" 20000; done

# 4) cohort 큐 (state·lease·작업 로그·run manifest)
C="$V8R/$COHORT"
{ echo "== $C"; ls -la "$C" 2>&1; echo; echo "== V8_ROOT"; ls -la "$V8R" 2>&1; } > "$D/queue-tree.txt"
if [ -d "$C" ]; then
  find "$C" -maxdepth 4 \( -name '*.json' -o -name '*.jsonl' \) -size -5M 2>/dev/null | head -400 | while read -r f; do
    rel=${f#"$V8R"/}; mkdir -p "$D/queue/$(dirname "$rel")"; cp "$f" "$D/queue/$rel" 2>/dev/null; done
  find "$C" -maxdepth 4 -type d 2>/dev/null | head -300 >> "$D/queue-tree.txt"
  # 작업 로그: 최근 40개 + native fault
  find "$C" -maxdepth 5 -type f \( -name '*.log' -o -name '*.err' -o -name '*.out' \) -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -60 | cut -d' ' -f2- | while read -r f; do
    rel=${f#"$V8R"/}; tailcp "$f" "$D/queue/$rel" 5000; done
fi
for o in "$V8R"/*/; do [ -d "$o" ] || continue; c=$(basename "$o"); [ "$c" = "$COHORT" ] && continue
  for f in "$o"locator.json "$o"*/state.json; do [ -f "$f" ] && { rel=${f#"$V8R"/}; mkdir -p "$D/queue/$(dirname "$rel")"; cp "$f" "$D/queue/$rel"; }; done; done

# 5) 묶기
tar czf "$OUTDIR/$NAME.tar.gz" -C "$T" "$NAME" && rm -rf "$T"
SZ=$(du -h "$OUTDIR/$NAME.tar.gz" | cut -f1)
echo "완료: $OUTDIR/$NAME.tar.gz ($SZ)"
echo "다른 노드에서도 같은 명령을 돌리면 같은 폴더($OUTDIR)에 쌓여요. 그 폴더를 통째로 주세요."
