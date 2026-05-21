#!/bin/bash
# Mixed inference + training workload sweep.
# Hypothesis: with high CoV (2.46 vs 0.73 pure inference), SJF/SRTF
# gain over FIFO should be much larger.
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

ALGOS=(
    "fifo:Fifo:"
    "las:Las:"
    "srtf:Srtf:"
    "metasrtf:MetaSrtf:"
    "srtfslo:SrtfSlo:SRTF_SLO_TARGET=600:SRTF_SLO_THETA=0.7"
    "metasrtfslo:MetaSrtfSlo:META_SLO_TARGET=600:META_SLO_THETA=0.7"
)

# Mixed workload mean = 502s → 2 GPU capacity ≈ 14 jobs/hr
# Use various ρ
SETUPS=(
    "1:2:20"      # 2 GPU, mild (ρ≈1.4×)
    "1:2:30"      # 2 GPU, moderate (ρ≈2.1×)
    "1:4:50"      # 4 GPU (ρ≈1.7×)
)

START=100
STOP=200    # 100 jobs (~20 training + 80 inference)
export TRACE=./cluster_job_log_mixed

total=$((${#ALGOS[@]} * ${#SETUPS[@]}))
echo "Mixed sweep: ${#SETUPS[@]} setups × ${#ALGOS[@]} algos = $total runs"
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
        exp="mx_${setup_tag}_${short}"

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
        # 4 min timeout
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

# Move output from results/misc to results/mixed (case statement won't catch mx_)
mv results/misc/mx_*.json results/mixed/ 2>/dev/null || true
mv results/*/mx_*.json results/mixed/ 2>/dev/null || true
rmdir results/misc 2>/dev/null

echo "MIXED SWEEP DONE"
ls results/mixed/mx_*_accept_all_*_job_stats.json 2>/dev/null | wc -l | xargs -I{} echo "result files: {}"
