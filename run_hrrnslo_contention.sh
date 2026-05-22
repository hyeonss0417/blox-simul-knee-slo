#!/bin/bash
# HrrnSlo on pure-inference contention sweep — fills in the missing algorithm
# from run_sweep_contention.sh so sweep_avg_by_setup.png includes HrrnSlo.
#
# 3 setups × 2 ranges × 1 algorithm = 6 runs (~10 min)
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

# Pure-inference SLO target = 60 s (matches other bucket variants in this sweep).
# Pure-inference workload has small waits (~100s), so SLO=1500 (mixed target)
# would never trip and would degenerate to plain HRRN.
ALGOS=(
    "hrrnslo:HrrnSlo:HRRN_SLO_TARGET=60:HRRN_SLO_THETA=0.7"
)

SETUPS=(
    "1:1:100"   # 1 GPU mild   (~1.3× over)
    "1:1:200"   # 1 GPU HEAVY  (~2.6× over)
    "1:2:400"   # 2 GPU HEAVY  (~2.6× over)
)

RANGES=("10:110" "1000:1100")

total=$((${#ALGOS[@]} * ${#SETUPS[@]} * ${#RANGES[@]}))
echo "HrrnSlo contention sweep: $total runs"
idx=0
for setup in "${SETUPS[@]}"; do
    NM=$(echo $setup | cut -d: -f1)
    GPM=$(echo $setup | cut -d: -f2)
    LOAD=$(echo $setup | cut -d: -f3)
    export BLOX_NUM_MACHINES=$NM
    export BLOX_GPUS_PER_MACHINE=$GPM
    setup_tag="m${NM}g${GPM}_l${LOAD}"

    for rng in "${RANGES[@]}"; do
        START=$(echo $rng | cut -d: -f1)
        STOP=$(echo $rng | cut -d: -f2)
        range_tag="r${START}_${STOP}"

        for algo in "${ALGOS[@]}"; do
            idx=$((idx + 1))
            short=$(echo $algo | cut -d: -f1)
            sched=$(echo $algo | cut -d: -f2)
            e1=$(echo $algo | cut -d: -f3)
            e2=$(echo $algo | cut -d: -f4)
            exp="sw_${setup_tag}_${range_tag}_${short}"

            if ls results/contention_sweep/${exp}_${START}_${STOP}_${sched}_accept_all_load_${LOAD}.0_job_stats.json >/dev/null 2>&1; then
                echo "[$idx/$total] $exp — already exists, skipping"
                continue
            fi

            pkill -9 -f simulator 2>/dev/null; pkill -9 -f run_scheduler 2>/dev/null; sleep 1
            rm -f philly_jobs*.pickle 2>/dev/null
            for k in HRRN_SLO_TARGET HRRN_SLO_THETA HRRN_USE_META; do unset $k; done
            [ -n "$e1" ] && eval "export $e1"
            [ -n "$e2" ] && eval "export $e2"

            echo "[$idx/$total] $exp"
            SCHED="$sched" EXP_PREFIX="$exp" PORT_BASE=50050 \
                LOAD=$LOAD START=$START STOP=$STOP ROUND_DURATION=10 \
                bash run_one_experiment.sh > /dev/null 2>&1 &
            PID=$!
            for i in $(seq 1 180); do
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
done

echo "HrrnSlo CONTENTION SWEEP DONE"
ls results/contention_sweep/sw_*_hrrnslo_*_job_stats.json 2>/dev/null | wc -l | xargs -I{} echo "HrrnSlo result files: {}"
