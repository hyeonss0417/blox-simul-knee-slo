#!/bin/bash
# v2 grid runner — SEQUENTIAL.
# Parallel execution introduced state-leakage between runs (verified by
# v2_fifo coming out as 14.75h instead of v1's 29.23h), so we run one at
# a time using the default port triplet. Baselines (FIFO/LAS/SRTF) reuse
# the validated v1 results — only the new configs run here.
set -u
cd "$(dirname "$0")"
source venv/bin/activate

# Skip baselines we already validated.
CONFIGS=(
    # --- New baselines ---
    "v2_sjf|SjfTotal|"
    "v2_edf|Edf|EDF_SLO_TARGET=21600"
    "v2_llf|Llf|LLF_SLO_TARGET=21600"

    # --- Knee-SLO grid ---
    "v2_knee_t3_q|KneeSlo|KNEE_THETA=0.3 KNEE_GAMMA=2 KNEE_RISK_FN=knee_quadratic"
    "v2_knee_t5_q|KneeSlo|KNEE_THETA=0.5 KNEE_GAMMA=2 KNEE_RISK_FN=knee_quadratic"
    "v2_knee_t7_q|KneeSlo|KNEE_THETA=0.7 KNEE_GAMMA=2 KNEE_RISK_FN=knee_quadratic"
    "v2_knee_t7_q3|KneeSlo|KNEE_THETA=0.7 KNEE_GAMMA=3 KNEE_RISK_FN=knee_quadratic"
    "v2_knee_t5_q3|KneeSlo|KNEE_THETA=0.5 KNEE_GAMMA=3 KNEE_RISK_FN=knee_quadratic"
    "v2_knee_t7_lin|KneeSlo|KNEE_THETA=0.7 KNEE_RISK_FN=linear"
    "v2_knee_t7_sig|KneeSlo|KNEE_THETA=0.7 KNEE_RISK_FN=sigmoid"
    "v2_knee_t5_sig|KneeSlo|KNEE_THETA=0.5 KNEE_RISK_FN=sigmoid"
)

cleanup() {
    pkill -f simulator_simple.py 2>/dev/null || true
    pkill -f run_scheduler.py 2>/dev/null || true
    sleep 2
}

cleanup
trap cleanup EXIT

mkdir -p experiment_logs
echo "Total experiments: ${#CONFIGS[@]} (sequential)"

idx=0
for row in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    exp=${row%%|*}; rest=${row#*|}
    sched=${rest%%|*}; extra=${rest#*|}

    # Skip if result already exists (resume support)
    if ls "${exp}_3000_3100_${sched}_accept_all_load_8.0_job_stats.json" >/dev/null 2>&1; then
        echo "[$idx/${#CONFIGS[@]}] $exp ($sched)  SKIP (result exists)"
        continue
    fi

    echo "[$idx/${#CONFIGS[@]}] $exp ($sched)  extra='$extra'"
    env $extra \
        SCHED="$sched" EXP_PREFIX="$exp" \
        PORT_BASE=50050 \
        LOAD=8 START=3000 STOP=3100 \
        bash run_one_experiment.sh
    cleanup
done

echo "ALL EXPERIMENTS COMPLETE"
ls -la v2_*_load_8.0_job_stats.json 2>/dev/null | head -40
