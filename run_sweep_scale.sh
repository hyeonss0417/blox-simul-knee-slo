#!/bin/bash
# Scale-up sweep: keep ρ constant, vary absolute cluster size.
# Tests whether scheduling regime is determined by ρ alone or by
# absolute cluster size as well.
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

ALGOS=(
    "fifo:Fifo:"
    "las:Las:"
    "srtf:Srtf:"
    "metasrtf:MetaSrtf:"
    "srtfslo:SrtfSlo:SRTF_SLO_TARGET=60:SRTF_SLO_THETA=0.7"
    "metasrtfslo:MetaSrtfSlo:META_SLO_TARGET=60:META_SLO_THETA=0.7"
)

# Same ρ ≈ 2.6× over capacity, scaled up
# Cluster capacity ≈ 16/23 jobs/s per GPU = ~156/hr.  2.6× ρ → load ≈ N_gpu × 405
SETUPS=(
    "1:1:200"     # 1 GPU, baseline (already done in sweep, regenerate)
    "1:4:800"     # 4 GPU
    "2:4:1600"    # 8 GPU
)

START=10
STOP=110

total=$((${#ALGOS[@]} * ${#SETUPS[@]}))
echo "Scale-up sweep: ${#SETUPS[@]} setups × ${#ALGOS[@]} algos = $total runs"
idx=0
for setup in "${SETUPS[@]}"; do
    NM=$(echo $setup | cut -d: -f1)
    GPM=$(echo $setup | cut -d: -f2)
    LOAD=$(echo $setup | cut -d: -f3)
    export BLOX_NUM_MACHINES=$NM
    export BLOX_GPUS_PER_MACHINE=$GPM
    setup_tag="g$((NM * GPM))_l${LOAD}"

    for algo in "${ALGOS[@]}"; do
        idx=$((idx + 1))
        short=$(echo $algo | cut -d: -f1)
        sched=$(echo $algo | cut -d: -f2)
        e1=$(echo $algo | cut -d: -f3)
        e2=$(echo $algo | cut -d: -f4)
        exp="sc_${setup_tag}_${short}"

        if ls "results/contention_sweep/${exp}_${START}_${STOP}_${sched}_accept_all_load_${LOAD}.0_job_stats.json" >/dev/null 2>&1; then
            continue
        fi

        pkill -9 -f simulator 2>/dev/null; pkill -9 -f run_scheduler 2>/dev/null; sleep 1
        rm -f philly_jobs*.pickle 2>/dev/null
        for k in LAS_SLO_TARGET LAS_SLO_THETA SRTF_SLO_TARGET SRTF_SLO_THETA META_SLO_TARGET META_SLO_THETA META_SLO_MULT; do
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
echo "SCALE SWEEP DONE"
# Move sc_ files into contention_sweep folder
mv sc_*.json results/contention_sweep/ 2>/dev/null || true
ls results/contention_sweep/sc_*_accept_all_*_job_stats.json 2>/dev/null | wc -l | xargs -I{} echo "result files: {}"
