#!/bin/bash
# Wave 4: Stress test + extreme variants.
# Run AFTER Wave 3 to test edge cases and gather final ablation data.
set -u
cd "$(dirname "$0")"
source venv/bin/activate

# Read best Knee config from Wave 2 summary (fallback to default).
BEST_THETA="0.7"
BEST_RISK="knee_quadratic"

CONFIGS=(
    # Extreme theta / gamma combos
    "v2_knee_t9_q|KneeSlo|KNEE_THETA=0.9 KNEE_GAMMA=2 KNEE_RISK_FN=knee_quadratic"
    "v2_knee_t1_q|KneeSlo|KNEE_THETA=0.1 KNEE_GAMMA=2 KNEE_RISK_FN=knee_quadratic"
    "v2_knee_g4|KneeSlo|KNEE_THETA=0.7 KNEE_GAMMA=4 KNEE_RISK_FN=knee_quadratic"

    # Pure variants — only urgency, only size, only wait
    "v2_knee_only_urg|KneeSlo|KNEE_THETA=0.7 KNEE_W_SIZE=0.0 KNEE_W_URG=2.0 KNEE_W_AGE=0.0"
    "v2_knee_only_size|KneeSlo|KNEE_THETA=0.7 KNEE_W_SIZE=2.0 KNEE_W_URG=0.0 KNEE_W_AGE=0.0"

    # Combined: NonPreempt + age_high
    "v2_knee_np_age|KneeSloNonPreempt|KNEE_THETA=0.7 KNEE_W_AGE=0.5"

    # Class-aware with different theta
    "v2_knee_cls_t3|KneeSloClass|KNEE_THETA=0.3 KNEE_GAMMA=2 KNEE_RISK_FN=knee_quadratic"
)

cleanup() {
    pkill -f simulator_simple.py 2>/dev/null || true
    pkill -f run_scheduler.py 2>/dev/null || true
    sleep 2
}
cleanup
trap cleanup EXIT

mkdir -p experiment_logs
echo "Wave 4 — Total experiments: ${#CONFIGS[@]} (sequential)"

idx=0
for row in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    exp=${row%%|*}; rest=${row#*|}
    sched=${rest%%|*}; extra=${rest#*|}
    if ls "${exp}_3000_3100_${sched}_accept_all_load_8.0_job_stats.json" >/dev/null 2>&1; then
        echo "[W4 $idx/${#CONFIGS[@]}] $exp ($sched)  SKIP"
        continue
    fi
    echo "[W4 $idx/${#CONFIGS[@]}] $exp ($sched)  extra='$extra'"
    env $extra \
        SCHED="$sched" EXP_PREFIX="$exp" PORT_BASE=50050 \
        LOAD=8 START=3000 STOP=3100 \
        bash run_one_experiment.sh
    cleanup
done
echo "WAVE 4 COMPLETE"
