#!/bin/bash
# Force REAL contention: only 1-2 GPUs.
# User hypothesis: cluster is too large relative to inference load.
# At 1-2 GPUs, queue depth becomes meaningful and scheduler order
# DECIDES which jobs see big queues.
set -u
cd "$(dirname "$0")"
source venv/bin/activate

cleanup() {
    pkill -f simulator_simple.py 2>/dev/null || true
    pkill -f run_scheduler.py 2>/dev/null || true
    sleep 2
}
cleanup
trap cleanup EXIT
mkdir -p experiment_logs

# 2 GPUs + load 200 jobs/hr (200/3600 = 0.056 arrival/s)
# Service: 2 GPUs × 1/23 = 0.087 jobs/s capacity
# ρ = 0.056 / 0.087 ≈ 0.64 — moderately loaded, not saturated
# Track range that fully fits into the experiment
export BLOX_NUM_MACHINES=1
export BLOX_GPUS_PER_MACHINE=2   # 2 GPUs total
LOAD=200
START=10
STOP=110   # 100 jobs

CONFIGS=(
    "co_fifo|Fifo|"
    "co_las|Las|"
    "co_srtf|Srtf|"
    "co_sjf|SjfTotal|"
    "co_hrrn|Hrrn|"
    "co_metasrtf|MetaSrtf|"
    "co_lasslo60|LasSlo|LAS_SLO_TARGET=60|LAS_SLO_THETA=0.7"
    "co_srtfslo60|SrtfSlo|SRTF_SLO_TARGET=60|SRTF_SLO_THETA=0.7"
    "co_metaslo60|MetaLasSlo|META_SLO_TARGET=60|META_SLO_THETA=0.7"
    "co_lasslo120|LasSlo|LAS_SLO_TARGET=120|LAS_SLO_THETA=0.7"
    "co_metaslo_m3|MetaLasSlo|META_SLO_MULT=3|META_SLO_THETA=0.7"
)

echo "Contention: 2 GPUs, load=$LOAD, ${#CONFIGS[@]} configs"
idx=0
for row in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    exp=${row%%|*}
    rest=${row#*|}
    sched=${rest%%|*}
    rest=${rest#*|}
    e1=$(echo "$rest" | cut -d'|' -f1)
    e2=$(echo "$rest" | cut -d'|' -f2)
    [ "$e2" = "$rest" ] && e2=""
    if ls "${exp}_${START}_${STOP}_${sched}_accept_all_load_${LOAD}.0_job_stats.json" >/dev/null 2>&1; then
        echo "[$idx] $exp SKIP"; continue
    fi
    pkill -9 -f simulator 2>/dev/null; pkill -9 -f run_scheduler 2>/dev/null; sleep 1
    rm -f philly_jobs*.pickle 2>/dev/null

    [ -n "$e1" ] && eval "export $e1"
    [ -n "$e2" ] && eval "export $e2"

    echo "[$idx/${#CONFIGS[@]}] $exp ($sched)  e1='$e1' e2='$e2'"
    SCHED="$sched" EXP_PREFIX="$exp" PORT_BASE=50050 \
        LOAD=$LOAD START=$START STOP=$STOP ROUND_DURATION=10 \
        bash run_one_experiment.sh
    unset LAS_SLO_TARGET LAS_SLO_THETA SRTF_SLO_TARGET SRTF_SLO_THETA META_SLO_TARGET META_SLO_THETA META_SLO_MULT
done

echo "CONTENTION DONE"
