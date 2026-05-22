#!/bin/bash
# Mixed workload sweep — find setups where bucket variants WIN.
# Strategy:
# - Mild ρ so most inference jobs are in safe bucket (not all critical)
# - SLO target tuned so training jobs are bucket 0, inference is bucket 2
# - SRTF in safe bucket should differentiate short inference from long
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
mkdir -p experiment_logs results/mixed

# Algorithms — focus on FIFO baseline vs our bucket variants at different SLOs
ALGOS=(
    "fifo:Fifo:"
    "hrrn:Hrrn:"
    # Bucket with SLO=120 (inference can be safe-zone, training is bucket 0)
    "srtfslo120:SrtfSlo:SRTF_SLO_TARGET=120:SRTF_SLO_THETA=0.7"
    "metaslo120:MetaSrtfSlo:META_SLO_TARGET=120:META_SLO_THETA=0.7"
    # Bucket with longer SLO=300 (more training jobs also in bucket 2)
    "srtfslo300:SrtfSlo:SRTF_SLO_TARGET=300:SRTF_SLO_THETA=0.7"
    "metaslo300:MetaSrtfSlo:META_SLO_TARGET=300:META_SLO_THETA=0.7"
)

# Mild ρ — inference dominates queue, training jobs are critical
# Mean mixed = 502s. capacity = N_GPU × 7.2 jobs/hr
SETUPS=(
    "1:2:7"     # 2 GPU, ρ ≈ 0.5×  (under-loaded, brief queue)
    "1:2:10"    # 2 GPU, ρ ≈ 0.7×  (near capacity)
    "1:4:14"    # 4 GPU, ρ ≈ 0.5×
    "1:4:20"    # 4 GPU, ρ ≈ 0.7×
    "1:4:25"    # 4 GPU, ρ ≈ 0.9×  (very near capacity)
)

START=200
STOP=350    # 150 jobs
export TRACE=./cluster_job_log_mixed

total=$((${#ALGOS[@]} * ${#SETUPS[@]}))
echo "Mixed sweet-spot sweep: ${#SETUPS[@]} setups × ${#ALGOS[@]} algos = $total runs"
idx=0
for setup in "${SETUPS[@]}"; do
    NM=$(echo $setup | cut -d: -f1)
    GPM=$(echo $setup | cut -d: -f2)
    LOAD=$(echo $setup | cut -d: -f3)
    export BLOX_NUM_MACHINES=$NM
    export BLOX_GPUS_PER_MACHINE=$GPM
    setup_tag="m${NM}g${GPM}_l${LOAD}"

    for algo in "${ALGOS[@]}"; do
        idx=$((idx + 1))
        short=$(echo $algo | cut -d: -f1)
        sched=$(echo $algo | cut -d: -f2)
        e1=$(echo $algo | cut -d: -f3)
        e2=$(echo $algo | cut -d: -f4)
        exp="mx2_${setup_tag}_${short}"

        if ls results/mixed/${exp}_${START}_${STOP}_${sched}_accept_all_load_${LOAD}.0_job_stats.json >/dev/null 2>&1; then
            continue
        fi

        pkill -9 -f simulator 2>/dev/null; pkill -9 -f run_scheduler 2>/dev/null; sleep 1
        rm -f philly_jobs*.pickle 2>/dev/null
        for k in LAS_SLO_TARGET LAS_SLO_THETA SRTF_SLO_TARGET SRTF_SLO_THETA META_SLO_TARGET META_SLO_THETA; do
            unset $k
        done
        [ -n "$e1" ] && eval "export $e1"
        [ -n "$e2" ] && eval "export $e2"

        echo "[$idx/$total] $exp ($sched)"
        SCHED="$sched" EXP_PREFIX="$exp" PORT_BASE=50050 \
            LOAD=$LOAD START=$START STOP=$STOP ROUND_DURATION=10 \
            bash run_one_experiment.sh > /dev/null 2>&1 &
        PID=$!
        for i in $(seq 1 240); do  # 4 min timeout
            if ! kill -0 $PID 2>/dev/null; then break; fi
            sleep 1
        done
        if kill -0 $PID 2>/dev/null; then
            echo "  TIMEOUT — killing"
            pkill -9 -f simulator 2>/dev/null
            pkill -9 -f run_scheduler 2>/dev/null
            sleep 2
        fi
    done
done

# Add mx2 to run_one_experiment.sh routing temporarily
mv results/misc/mx2_*.json results/mixed/ 2>/dev/null
mv results/*/mx2_*.json results/mixed/ 2>/dev/null
rmdir results/misc 2>/dev/null

echo "MIXED SWEET-SPOT SWEEP DONE"
ls results/mixed/mx2_*_accept_all_*_job_stats.json 2>/dev/null | wc -l | xargs -I{} echo "result files: {}"
