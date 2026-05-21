#!/bin/bash
# Contention sweep: 1/2 GPU × multiple loads × multiple track ranges.
# Goal: confirm 2-GPU finding holds, find sweet spot, gather statistics.
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

# Algorithm set (kept small for sweep size)
ALGOS=(
    "fifo:Fifo:"
    "las:Las:"
    "srtf:Srtf:"
    "metasrtf:MetaSrtf:"
    "srtfslo:SrtfSlo:SRTF_SLO_TARGET=60:SRTF_SLO_THETA=0.7"
    "metasrtfslo:MetaSrtfSlo:META_SLO_TARGET=60:META_SLO_THETA=0.7"
)

# (machines, gpus_per_machine, load) tuples — focus on contention sweet spot
SETUPS=(
    "1:1:100"   # 1 GPU mild   (~1.3× over)
    "1:1:200"   # 1 GPU HEAVY  (~2.6× over)
    "1:2:400"   # 2 GPU HEAVY  (~2.6× over)
)

# 2 track ranges (different positions in trace) for stat robustness
RANGES=("10:110" "1000:1100")

total=$((${#ALGOS[@]} * ${#SETUPS[@]} * ${#RANGES[@]}))
echo "Sweep: ${#ALGOS[@]} algos × ${#SETUPS[@]} setups × ${#RANGES[@]} ranges = $total runs"
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

            if ls "${exp}_${START}_${STOP}_${sched}_accept_all_load_${LOAD}.0_job_stats.json" >/dev/null 2>&1; then
                continue
            fi

            pkill -9 -f simulator 2>/dev/null; pkill -9 -f run_scheduler 2>/dev/null; sleep 1
            rm -f philly_jobs*.pickle 2>/dev/null
            for k in LAS_SLO_TARGET LAS_SLO_THETA SRTF_SLO_TARGET SRTF_SLO_THETA META_SLO_TARGET META_SLO_THETA META_SLO_MULT; do
                unset $k
            done
            [ -n "$e1" ] && eval "export $e1"
            [ -n "$e2" ] && eval "export $e2"

            echo "[$idx/$total] $exp"
            # Timeout to avoid runaway thrashing (LAS/SRTF/MetaSrtf can stall)
            SCHED="$sched" EXP_PREFIX="$exp" PORT_BASE=50050 \
                LOAD=$LOAD START=$START STOP=$STOP ROUND_DURATION=10 \
                bash run_one_experiment.sh > /dev/null 2>&1 &
            PID=$!
            # 3 min timeout per experiment
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
echo "SWEEP COMPLETE"
ls sw_*_accept_all_*_job_stats.json 2>/dev/null | wc -l | xargs -I{} echo "result files: {}"
