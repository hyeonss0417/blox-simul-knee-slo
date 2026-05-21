#!/bin/bash
# Wave R: real inference workload (exponential=False, real exec_time).
# Job durations are now 12-33s (actual Stable Diffusion times) not the
# gavel-like 30min-150h that the v1 default produced.
#
# SLO targets are recalibrated to inference scale.
#
# All experiments use:
#   - prefix `inf_<config>` so they don't collide with the synthetic v2/w3 set
#   - --round-duration 10 (scheduler decision tick = 10s)
set -u
cd "$(dirname "$0")"
source venv/bin/activate

# Inference setup:
#   - smaller cluster (8 machines × 4 GPU = 32 GPU) — a realistic inference
#     serving deployment, not the 128-GPU training cluster.
#   - load=8000 jobs/hour gives ~1.7x over capacity → meaningful queueing.
#   - smoke check confirmed: median 28s, p99 66s, max 151s with FIFO.
LOAD=8000
export BLOX_NUM_MACHINES=8
export BLOX_GPUS_PER_MACHINE=4

# Track a larger window for statistical robustness on short jobs.
START=3000
STOP=3300

CONFIGS=(
    # ── Baselines ───────────────────────────────────────────────────
    "inf_fifo|Fifo|"
    "inf_las|Las|"
    "inf_srtf|Srtf|"
    "inf_sjf|SjfTotal|"
    "inf_hrrn|Hrrn|"

    # ── EDF / LLF with inference-scale SLO target (1 min) ─────────────
    "inf_edf_60s|Edf|EDF_SLO_TARGET=60"
    "inf_llf_60s|Llf|LLF_SLO_TARGET=60"

    # ── Knee-SLO @ SLO=30s (tight) ──────────────────────────────────
    "inf_knee_t7_30s|KneeSlo|KNEE_THETA=0.7 KNEE_SLO_TARGET_SECONDS=30 KNEE_RISK_FN=knee_quadratic"
    "inf_knee_t5_30s|KneeSlo|KNEE_THETA=0.5 KNEE_SLO_TARGET_SECONDS=30 KNEE_RISK_FN=knee_quadratic"

    # ── Knee-SLO @ SLO=60s (typical user-facing) ────────────────────
    "inf_knee_t7_60s|KneeSlo|KNEE_THETA=0.7 KNEE_SLO_TARGET_SECONDS=60 KNEE_RISK_FN=knee_quadratic"
    "inf_knee_t5_60s|KneeSlo|KNEE_THETA=0.5 KNEE_SLO_TARGET_SECONDS=60 KNEE_RISK_FN=knee_quadratic"
    "inf_knee_t7_60s_lin|KneeSlo|KNEE_THETA=0.7 KNEE_SLO_TARGET_SECONDS=60 KNEE_RISK_FN=linear"
    "inf_knee_t7_60s_sig|KneeSlo|KNEE_THETA=0.7 KNEE_SLO_TARGET_SECONDS=60 KNEE_RISK_FN=sigmoid"

    # ── Knee-SLO @ SLO=300s (5 min — batch-friendly) ────────────────
    "inf_knee_t7_5min|KneeSlo|KNEE_THETA=0.7 KNEE_SLO_TARGET_SECONDS=300 KNEE_RISK_FN=knee_quadratic"

    # ── Extensions ───────────────────────────────────────────────────
    "inf_knee_np_60s|KneeSloNonPreempt|KNEE_THETA=0.7 KNEE_SLO_TARGET_SECONDS=60"
    "inf_knee_cdur_m3|KneeSloClassDur|KNEE_THETA=0.7 KNEE_SLO_MULTIPLIER=3.0"
    "inf_knee_cdur_m5|KneeSloClassDur|KNEE_THETA=0.7 KNEE_SLO_MULTIPLIER=5.0"
)

cleanup() {
    pkill -f simulator_simple.py 2>/dev/null || true
    pkill -f run_scheduler.py 2>/dev/null || true
    sleep 2
}
cleanup
trap cleanup EXIT

mkdir -p experiment_logs
echo "Wave R (real inference) — Total: ${#CONFIGS[@]}  load=$LOAD jobs/hr  track=$START-$STOP"

idx=0
for row in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    exp=${row%%|*}; rest=${row#*|}
    sched=${rest%%|*}; extra=${rest#*|}

    if ls "${exp}_${START}_${STOP}_${sched}_accept_all_load_${LOAD}.0_job_stats.json" >/dev/null 2>&1; then
        echo "[R $idx/${#CONFIGS[@]}] $exp ($sched)  SKIP"
        continue
    fi

    echo "[R $idx/${#CONFIGS[@]}] $exp ($sched)  extra='$extra'"
    env $extra \
        SCHED="$sched" EXP_PREFIX="$exp" PORT_BASE=50050 \
        LOAD=$LOAD START=$START STOP=$STOP \
        bash run_one_experiment.sh
    cleanup
done

echo "WAVE R COMPLETE"
ls -la inf_*_load_${LOAD}.0_job_stats.json 2>/dev/null | head
