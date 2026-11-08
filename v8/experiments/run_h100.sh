#!/bin/sh
# V8 switching experiments on the H100 cluster. One command, resumable and idempotent:
#   sh v8/experiments/run_h100.sh <cohort> <stage> [--seeds 5,6]
# Run the same command on every free 4xH100 node; tasks are leased per run (no duplicates),
# completed tasks are skipped, interrupted ones resume from their last saved update.
# Logs stream to this terminal unchanged and are also kept under $V8_ROOT/<cohort>/logs/.
#
# cohorts: olmo_math (primary) | qwen_mbpp (replication) | olmo_mbpp | qwen_math
# stages (rebuttal Branch Bank on the inspected seeds 5-9; `rebuttal` runs them in this order:
#   locate audit profile trajectory branches extended destination freeze-budget budget-extension
#   eval-final eval-budget collect analyze):
#   check        CPU: code identity, configs, seed registry, GPU-h estimates
#   locate       CPU: resolve plans/run roots/prefix hashes from the code repo (writes locator.json)
#   audit        CPU: E0 provenance audit of the recorded seed 5-9 endpoints + split overlap
#   profile      CPU: cost profile from existing endpoint receipts (no rewards read)
#   trajectory   GPU: continued On-policy from the verified prefix, SR-GC D monitored at every check, snapshots /25
#   branches     GPU: SR from trajectory snapshots at 50/100/150/200            (X1 / E1, E2, E4, E9 lookup)
#   destination  GPU: Uniform + SR from the current rule's trigger state         (X5 / E3)
#   eval-final   GPU: final-split evaluation of every run at the matched horizon (275)
#   freeze-budget CPU: budget B from costs only (write-once)
#   budget-extension GPU: cheap branches continued past 275 inside B (hard cap 400)
#   eval-budget  GPU: final-split evaluation of the budget endpoints selected from metadata
#   eval-dev     GPU (optional, queued offline): dev-split curves every 25 updates (E5/E7); own cost ledger
#   extended     GPU: SR branches at 25/75/125/175/225/250 (exact lookup for every trigger)
#   collect | analyze  CPU: tables, LOSO, sign tests, Pareto plots, report
#   rebuttal     = the full sequence above (freeze-budget runs before any evaluation)
# prospective protocol (fresh seeds from the registry; needed for a confirmatory test):
#   prepare-seeds CPU | prefix GPU | dev GPU | tune-freeze (NOT IMPLEMENTED as a stage: exits 2; the
#   sequence exists only in switching/smoke.py toy()) | test GPU | eval-dev / eval-final GPU
#   e8           GPU (optional): code repo switch_repeat arm via its own launcher
#   history-export CPU (optional): code repo exporter scripts/srgc_v8_history.py (commit 6cfe1920…) when
#                EXPORT_CODE_ROOT is set (may be the latest master; V8_CODE_ROOT stays the frozen checkout)
#   status       CPU: pending/complete counts per stage
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
COHORT=${1:?usage: run_h100.sh <cohort> <stage> [--seeds a,b]}
STAGE=${2:?usage: run_h100.sh <cohort> <stage> [--seeds a,b]}
SEEDS=
if [ "${3:-}" = --seeds ]; then SEEDS=${4:?--seeds needs a list}; fi
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export V8_ROOT=${V8_ROOT:-$WORK/v8}
export V8_CODE_ROOT=${V8_CODE_ROOT:-${OM_CODE_ROOT:-}}
if [ -z "$V8_CODE_ROOT" ]; then
    echo "[v8] set V8_CODE_ROOT to the offpolicy-misranking checkout (user volume) that produced the seed 5-9 runs" >&2
    exit 2
fi
case "$COHORT" in *mbpp) EXPLICIT=${SWITCH_PYTHON:-} ;; *) EXPLICIT=${PAIR_PYTHON:-} ;; esac
PY=${V8_PYTHON:-${EXPLICIT:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}}
command -v "$PY" >/dev/null 2>&1 || PY=python3
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$V8_ROOT/$COHORT/logs"
LOG="$V8_ROOT/$COHORT/logs/$STAGE-$(hostname)-$(date +%Y%m%dT%H%M%S).log"
cd "$HERE"
RCFILE="$V8_ROOT/$COHORT/logs/.rc.$(hostname).$$"
cpu() { CUDA_VISIBLE_DEVICES="" "$PY" -m switching.cli "$@"; }
# cpu_log: stream output to the terminal and the log, and FAIL with python's exit code
# (POSIX sh has no pipefail; `python | tee` would otherwise report tee's status).
cpu_log() {
    ( set +e; CUDA_VISIBLE_DEVICES="" "$PY" -m switching.cli "$@"; echo $? > "$RCFILE.cpu" ) 2>&1 | tee -a "$LOG"
    rc=$(cat "$RCFILE.cpu" 2>/dev/null || echo 1)
    rm -f "$RCFILE.cpu"
    if [ "$rc" -ne 0 ]; then echo "[v8] CPU step failed rc=$rc: $*" | tee -a "$LOG"; exit "$rc"; fi
}

gpu_stage() {  # $1 matrix stage, $2 role, $3 optional comma-separated methods (prospective)
    make_matrix "$@"
    matrix="$V8_ROOT/$COHORT/matrix/$1.$2.json"
    while :; do
        progressed=0
        tasks=$(cpu pending --cohort "$COHORT" --stage "$1" --role "$2") || exit $?
        for task in $tasks; do
            echo "[v8] $(date -Is) $(hostname) stage=$1 task=$task" | tee -a "$LOG"
            # Foreground: Ctrl-C reaches torchrun, which stops every rank at a safe update boundary (exit 75).
            ( set +e
              CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0,1,2,3} "$PY" -m torch.distributed.run --standalone \
                  --nproc_per_node="${V8_NPROC:-4}" --max_restarts=0 "$HERE/cluster_child.py" \
                  --cohort "$COHORT" --matrix "$matrix" --task "$task"
              echo $? > "$RCFILE" ) 2>&1 | tee -a "$LOG"
            rc=$(cat "$RCFILE" 2>/dev/null || echo 1)
            rm -f "$RCFILE"
            case "$rc" in
                0) progressed=1 ;;
                75) echo "[v8] busy or interrupted: $task (another node, or a stop request)" | tee -a "$LOG" ;;
                *) echo "[v8] task failed rc=$rc: $task (attempt kept; rerunning resumes)" | tee -a "$LOG"; return "$rc" ;;
            esac
        done
        [ "$progressed" -eq 1 ] || break
        # dependencies may have unlocked new tasks (e.g. destination after trajectory)
        make_matrix "$@" >/dev/null
    done
}

make_matrix() {  # same arguments as gpu_stage; methods and seeds are always forwarded
    set -- --cohort "$COHORT" --stage "$1" --role "$2" ${3:+--methods "$3"}
    if [ -n "$SEEDS" ]; then set -- "$@" --seeds "$SEEDS"; fi
    cpu make-matrix "$@" || exit $?
}

run_stage() {
    case "$1" in
        check)
            cpu_log inspect-repo
            cpu_log validate-config --cohort "$COHORT" --mode rebuttal_bank
            cpu_log seed-registry --output "$V8_ROOT/$COHORT/seed_registry.csv"
            cpu_log estimate "$COHORT" ;;
        locate) cpu_log locate --cohort "$COHORT" ;;
        audit)
            paths=$(CUDA_VISIBLE_DEVICES="" "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); r=next(iter(d['seeds'].values())); print(r['folder'].rsplit('/seed-',1)[0]); [print(x['input_bundle']) for x in d['seeds'].values() if x.get('folder')]" "$V8_ROOT/$COHORT/locator.json") || exit $?
            set -- $paths
            root=$1; shift
            args=; for b in "$@"; do args="$args --bundle $b"; done
            cpu_log audit --run-root "$COHORT=$root" $args --out "$V8_ROOT/audit/$COHORT" ;;
        profile) cpu_log profile --cohort "$COHORT" ;;
        trajectory) gpu_stage bank-trajectory rebuttal_bank ;;
        branches) gpu_stage bank-branches rebuttal_bank ;;
        extended) gpu_stage bank-extended rebuttal_bank ;;
        destination) gpu_stage destination rebuttal_bank ;;
        eval-final) gpu_stage eval-final rebuttal_bank ;;
        eval-dev) gpu_stage eval-dev rebuttal_bank ;;
        freeze-budget) cpu_log freeze-budget --cohort "$COHORT" ;;
        budget-extension) gpu_stage budget-extension rebuttal_bank ;;
        eval-budget) gpu_stage eval-budget rebuttal_bank ;;
        collect) cpu_log collect --cohort "$COHORT" ;;
        analyze) cpu_log analyze --cohort "$COHORT" ;;
        rebuttal)
            # B is frozen from costs BEFORE any evaluation exists (no outcome peeking); extended gives an
            # exact bank lookup for every 25-update trigger time.
            for s in locate audit profile trajectory branches extended destination freeze-budget \
                     budget-extension eval-final eval-budget collect analyze; do
                echo "[v8] ===== stage $s =====" | tee -a "$LOG"
                run_stage "$s"
            done ;;
        prepare-seeds) cpu_log prepare-seeds --cohort "$COHORT" ;;
        prefix) gpu_stage prefix development; gpu_stage prefix test ;;
        dev)
            METHODS=${V8_DEV_METHODS:-fixed_sr_t025,fixed_sr_t050,fixed_sr_t100,fixed_sr_t150,fixed_sr_t200,plateau_sr_eps0000,plateau_sr_eps0005,plateau_sr_eps0010,sr_early_refreshed_k050,sr_early_refreshed_k100,srgc_current,on_continued}
            gpu_stage prospective development "$METHODS" ;;
        tune-freeze)
            echo "[v8] tune-freeze is NOT IMPLEMENTED as a cluster stage (exit 2). The tuning/freeze sequence exists" \
                 "only in switching/smoke.py toy(); the supported path this time is the rebuttal Branch Bank." | tee -a "$LOG"
            exit 2 ;;
        test)
            METHODS=${V8_TEST_METHODS:-on_continued,sr_early,uniform_early,fixed_sr_tuned,srgc_current,plateau_sr_tuned,srgc_uniform_destination,srgc_one_negative,srgc_two_negative,sampled_time_sr}
            gpu_stage prospective test "$METHODS" ;;
        e8)
            for s in $(echo "${SEEDS:-5,6,7,8,9}" | tr , ' '); do
                (cd "$V8_CODE_ROOT" && sh scripts/run_srgc_sr_refresh.sh "${COHORT#*_}" "$s" switch_repeat) 2>&1 | tee -a "$LOG"
            done ;;
        history-export)
            if [ -z "${EXPORT_CODE_ROOT:-}" ]; then
                echo "[v8] history-export skipped: EXPORT_CODE_ROOT is not set (code repo with scripts/srgc_v8_history.py)" | tee -a "$LOG"
                return 0
            fi
            set -- "$EXPORT_CODE_ROOT/scripts/srgc_v8_history.py" --root "$V8_ROOT" --output "${V8_ROOT}-history-exports"
            if [ -n "${V8_EXPORT_STRICT:-}" ]; then set -- "$@" --verify-checkpoints --strict; fi
            ( set +e; CUDA_VISIBLE_DEVICES="" "$PY" "$@"; echo $? > "$RCFILE.cpu" ) 2>&1 | tee -a "$LOG"
            rc=$(cat "$RCFILE.cpu" 2>/dev/null || echo 1); rm -f "$RCFILE.cpu"
            [ "$rc" -eq 0 ] || { echo "[v8] history export failed rc=$rc" | tee -a "$LOG"; exit "$rc"; } ;;
        status)
            for f in "$V8_ROOT/$COHORT"/matrix/*.json; do
                [ -f "$f" ] || continue
                b=$(basename "$f" .json); st=${b%.*}; role=${b##*.}
                n=$(cpu pending --cohort "$COHORT" --stage "$st" --role "$role") || exit $?
                n=$(printf '%s\n' $n | grep -c . || :)
                echo "$b pending_runnable=$n"
            done ;;
        *) echo "unknown stage: $1" >&2; exit 2 ;;
    esac
}
run_stage "$STAGE"
echo "[v8] stage $STAGE finished on $(hostname); outputs under $V8_ROOT/$COHORT" | tee -a "$LOG"
