#!/bin/bash
# HrrnSlo focused sweep on mixed workload, same setups as mixed_sweet
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
mkdir -p results/mixed

ALGOS=(
    "hrrnslo2_meta1500:HrrnSlo:HRRN_SLO_TARGET=1500:HRRN_USE_META=1"
    "hrrnslo2_meta3000:HrrnSlo:HRRN_SLO_TARGET=3000:HRRN_USE_META=1"
    "hrrnslo2_oracle1500:HrrnSlo:HRRN_SLO_TARGET=1500:HRRN_USE_META=0"
)

SETUPS=(
    "1:2:10"    # repeated for verification
    "1:4:14"
    "1:4:20"
    "1:4:25"
)

START=200
STOP=350
export TRACE=./cluster_job_log_mixed

total=$((${#ALGOS[@]} * ${#SETUPS[@]}))
echo "HrrnSlo sweep: ${#SETUPS[@]} × ${#ALGOS[@]} = $total runs"
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
        for k in HRRN_SLO_TARGET HRRN_SLO_THETA HRRN_USE_META; do unset $k; done
        [ -n "$e1" ] && eval "export $e1"
        [ -n "$e2" ] && eval "export $e2"

        echo "[$idx/$total] $exp ($sched, $e1, $e2)"
        SCHED="$sched" EXP_PREFIX="$exp" PORT_BASE=50050 \
            LOAD=$LOAD START=$START STOP=$STOP ROUND_DURATION=10 \
            bash run_one_experiment.sh > /dev/null 2>&1 &
        PID=$!
        for i in $(seq 1 240); do
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

echo "HrrnSlo SWEEP DONE"
