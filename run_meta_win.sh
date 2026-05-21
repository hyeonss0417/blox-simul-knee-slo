#!/bin/bash
# Goal: beat baselines using metadata-based prediction + SLO bucket.
# Moderate over-load (queue exists but not pathological).
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

# 16 GPUs, load 4000 jobs/hr.  Cluster capacity ≈ 16 × 144 = 2300/hr.  Load ≈ 1.7× over.
# This should produce queueing but not pathological saturation.
export BLOX_NUM_MACHINES=4
export BLOX_GPUS_PER_MACHINE=4   # 16 GPUs
LOAD=4000
START=3000
STOP=3300

CONFIGS=(
    # ── Baselines ─────────────────────────────────────────────
    "mw_fifo|Fifo|"
    "mw_las|Las|"
    "mw_srtf|Srtf|"                      # ORACLE — uses true job_duration

    # ── LAS + SLO bucket (flat) ──────────────────────────────
    "mw_lasslo60|LasSlo|LAS_SLO_TARGET=60 LAS_SLO_THETA=0.7"
    "mw_lasslo120|LasSlo|LAS_SLO_TARGET=120 LAS_SLO_THETA=0.7"
    "mw_lasslo300|LasSlo|LAS_SLO_TARGET=300 LAS_SLO_THETA=0.7"

    # ── SRTF + SLO bucket ────────────────────────────────────
    "mw_srtfslo120|SrtfSlo|SRTF_SLO_TARGET=120 SRTF_SLO_THETA=0.7"

    # ── METADATA-based predictor (the new contribution) ──────
    "mw_metasrtf|MetaSrtf|"              # SRTF using meta prediction, no oracle
    "mw_metaslo60|MetaLasSlo|META_SLO_TARGET=60 META_SLO_THETA=0.7"
    "mw_metaslo120|MetaLasSlo|META_SLO_TARGET=120 META_SLO_THETA=0.7"
    "mw_metaslo_m3|MetaLasSlo|META_SLO_MULT=3 META_SLO_THETA=0.7"   # SLO = pred × 3
    "mw_metaslo_m5|MetaLasSlo|META_SLO_MULT=5 META_SLO_THETA=0.7"   # SLO = pred × 5
)

echo "Meta-Win: 16 GPUs, load=$LOAD (~1.7× over), ${#CONFIGS[@]} configs"
idx=0
for row in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    exp=${row%%|*}; rest=${row#*|}
    sched=${rest%%|*}; extra=${rest#*|}
    if ls "${exp}_${START}_${STOP}_${sched}_accept_all_load_${LOAD}.0_job_stats.json" >/dev/null 2>&1; then
        echo "[$idx/${#CONFIGS[@]}] $exp SKIP"; continue
    fi
    echo "[$idx/${#CONFIGS[@]}] $exp ($sched)  extra='$extra'"
    env $extra \
        SCHED="$sched" EXP_PREFIX="$exp" PORT_BASE=50050 \
        LOAD=$LOAD START=$START STOP=$STOP ROUND_DURATION=10 \
        bash run_one_experiment.sh
    cleanup
done
echo "META-WIN COMPLETE"
