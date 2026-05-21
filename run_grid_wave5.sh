#!/bin/bash
# Wave 5: SLO recalibration.
# Wave 1 used SLO=6h which is TIGHTER than even FIFO's median JCT (19.5h),
# causing every job to be in the late-zone urgency branch.  As a result
# Wave-1's Knee θ sweep showed almost no variation between θ=0.3 and θ=0.5.
#
# We re-run with SLO=24h (FIFO miss ≈ 34%) — a realistic operating point
# where Knee scoring can actually differentiate safe / danger / late.
set -u
cd "$(dirname "$0")"
source venv/bin/activate

# B = 24h = 86400 s.
B=86400

CONFIGS=(
    "v2_knee24_t3|KneeSlo|KNEE_THETA=0.3 KNEE_GAMMA=2 KNEE_RISK_FN=knee_quadratic KNEE_SLO_TARGET_SECONDS=$B"
    "v2_knee24_t5|KneeSlo|KNEE_THETA=0.5 KNEE_GAMMA=2 KNEE_RISK_FN=knee_quadratic KNEE_SLO_TARGET_SECONDS=$B"
    "v2_knee24_t7|KneeSlo|KNEE_THETA=0.7 KNEE_GAMMA=2 KNEE_RISK_FN=knee_quadratic KNEE_SLO_TARGET_SECONDS=$B"
    "v2_knee24_t9|KneeSlo|KNEE_THETA=0.9 KNEE_GAMMA=2 KNEE_RISK_FN=knee_quadratic KNEE_SLO_TARGET_SECONDS=$B"
    "v2_knee24_t7_sig|KneeSlo|KNEE_THETA=0.7 KNEE_RISK_FN=sigmoid KNEE_SLO_TARGET_SECONDS=$B"
    "v2_knee24_t7_lin|KneeSlo|KNEE_THETA=0.7 KNEE_RISK_FN=linear KNEE_SLO_TARGET_SECONDS=$B"
    "v2_knee24_np|KneeSloNonPreempt|KNEE_THETA=0.7 KNEE_RISK_FN=knee_quadratic KNEE_SLO_TARGET_SECONDS=$B"
    "v2_knee24_age|KneeSlo|KNEE_THETA=0.7 KNEE_W_AGE=0.5 KNEE_SLO_TARGET_SECONDS=$B"

    # Class-aware SLO with category-mean × multiplier  (the key improvement
    # that addresses "all jobs in same SLO tier ⇒ degenerates to FIFO").
    "v2_knee24_cdur_m2|KneeSloClassDur|KNEE_THETA=0.7 KNEE_RISK_FN=knee_quadratic KNEE_SLO_MULTIPLIER=2.0"
    "v2_knee24_cdur_m3|KneeSloClassDur|KNEE_THETA=0.7 KNEE_RISK_FN=knee_quadratic KNEE_SLO_MULTIPLIER=3.0"
    "v2_knee24_cdur_m5|KneeSloClassDur|KNEE_THETA=0.7 KNEE_RISK_FN=knee_quadratic KNEE_SLO_MULTIPLIER=5.0"
)

cleanup() { pkill -f simulator_simple.py 2>/dev/null || true; pkill -f run_scheduler.py 2>/dev/null || true; sleep 2; }
cleanup; trap cleanup EXIT
mkdir -p experiment_logs
echo "Wave 5 (SLO=${B}s=$((B/3600))h) — Total: ${#CONFIGS[@]} (sequential)"

idx=0
for row in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    exp=${row%%|*}; rest=${row#*|}
    sched=${rest%%|*}; extra=${rest#*|}
    if ls "${exp}_3000_3100_${sched}_accept_all_load_8.0_job_stats.json" >/dev/null 2>&1; then
        echo "[W5 $idx/${#CONFIGS[@]}] $exp SKIP"; continue
    fi
    echo "[W5 $idx/${#CONFIGS[@]}] $exp ($sched)  extra='$extra'"
    env $extra SCHED="$sched" EXP_PREFIX="$exp" PORT_BASE=50050 \
        LOAD=8 START=3000 STOP=3100 bash run_one_experiment.sh
    cleanup
done
echo "WAVE 5 COMPLETE"
