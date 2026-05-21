#!/bin/bash
# Stress test: force queueing by aggressive over-load + small cluster.
# Goal: create scenarios where scheduler ordering MATTERS.
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

# Small cluster + heavy over-load.
export BLOX_NUM_MACHINES=4
export BLOX_GPUS_PER_MACHINE=2   # 8 GPUs total
LOAD=15000  # >> capacity (8 GPU × 144 jobs/hr ≈ 1150/hr) → ~13× over

START=3000
STOP=3300

CONFIGS=(
    # Baselines
    "ss_fifo|Fifo|"
    "ss_las|Las|"
    "ss_srtf|Srtf|"
    # SLO-aware variants — multiple SLO targets
    "ss_lasslo30|LasSlo|LAS_SLO_TARGET=30 LAS_SLO_THETA=0.7"
    "ss_lasslo60|LasSlo|LAS_SLO_TARGET=60 LAS_SLO_THETA=0.7"
    "ss_lasslo120|LasSlo|LAS_SLO_TARGET=120 LAS_SLO_THETA=0.7"
    "ss_lasslo300|LasSlo|LAS_SLO_TARGET=300 LAS_SLO_THETA=0.7"
    # SRTF base + SLO
    "ss_srtfslo60|SrtfSlo|SRTF_SLO_TARGET=60 SRTF_SLO_THETA=0.7"
    "ss_srtfslo120|SrtfSlo|SRTF_SLO_TARGET=120 SRTF_SLO_THETA=0.7"
    # theta sweep
    "ss_lasslo60_t5|LasSlo|LAS_SLO_TARGET=60 LAS_SLO_THETA=0.5"
    "ss_lasslo60_t9|LasSlo|LAS_SLO_TARGET=60 LAS_SLO_THETA=0.9"
)

echo "Stress test: 8 GPUs, load=$LOAD (>10x over capacity), 12 configs"
idx=0
for row in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    exp=${row%%|*}; rest=${row#*|}
    sched=${rest%%|*}; extra=${rest#*|}
    if ls "${exp}_${START}_${STOP}_${sched}_accept_all_load_${LOAD}.0_job_stats.json" >/dev/null 2>&1; then
        echo "[$idx] $exp SKIP"; continue
    fi
    echo "[$idx/${#CONFIGS[@]}] $exp ($sched)"
    env $extra \
        SCHED="$sched" EXP_PREFIX="$exp" PORT_BASE=50050 \
        LOAD=$LOAD START=$START STOP=$STOP ROUND_DURATION=10 \
        bash run_one_experiment.sh
    cleanup
done
echo "STRESS TEST DONE"
