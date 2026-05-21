#!/bin/bash
# Wave 2: Algorithmic extensions and refined KneeSlo grid.
# Run AFTER Wave 1 finishes.
set -u
cd "$(dirname "$0")"
source venv/bin/activate

CONFIGS=(
    # New baselines
    "v2_hrrn|Hrrn|"

    # Non-preemptive variant (with the best-so-far theta=0.7 quadratic)
    "v2_knee_np_t7|KneeSloNonPreempt|KNEE_THETA=0.7 KNEE_GAMMA=2 KNEE_RISK_FN=knee_quadratic"
    "v2_knee_np_t5|KneeSloNonPreempt|KNEE_THETA=0.5 KNEE_GAMMA=2 KNEE_RISK_FN=knee_quadratic"

    # Class-aware SLO targets
    "v2_knee_cls_t7|KneeSloClass|KNEE_THETA=0.7 KNEE_GAMMA=2 KNEE_RISK_FN=knee_quadratic"
    "v2_knee_cls_t5|KneeSloClass|KNEE_THETA=0.5 KNEE_GAMMA=2 KNEE_RISK_FN=knee_quadratic"

    # Aging emphasis (LAS-like protection): w_age large vs small
    "v2_knee_age_hi|KneeSlo|KNEE_THETA=0.7 KNEE_GAMMA=2 KNEE_W_AGE=0.5 KNEE_RISK_FN=knee_quadratic"
    "v2_knee_age_lo|KneeSlo|KNEE_THETA=0.7 KNEE_GAMMA=2 KNEE_W_AGE=0.0 KNEE_RISK_FN=knee_quadratic"

    # Urgency vs Size dominance
    "v2_knee_urg_hi|KneeSlo|KNEE_THETA=0.7 KNEE_W_SIZE=0.5 KNEE_W_URG=2.0 KNEE_RISK_FN=knee_quadratic"
    "v2_knee_size_hi|KneeSlo|KNEE_THETA=0.7 KNEE_W_SIZE=2.0 KNEE_W_URG=0.5 KNEE_RISK_FN=knee_quadratic"

    # SLO budget sweep (tight vs loose)
    "v2_knee_b3h|KneeSlo|KNEE_THETA=0.7 KNEE_SLO_TARGET_SECONDS=10800 KNEE_RISK_FN=knee_quadratic"
    "v2_knee_b12h|KneeSlo|KNEE_THETA=0.7 KNEE_SLO_TARGET_SECONDS=43200 KNEE_RISK_FN=knee_quadratic"

    # Late-zone penalty c3 sweep
    "v2_knee_c3_lo|KneeSlo|KNEE_THETA=0.7 KNEE_C3=5 KNEE_RISK_FN=knee_quadratic"
    "v2_knee_c3_hi|KneeSlo|KNEE_THETA=0.7 KNEE_C3=100 KNEE_RISK_FN=knee_quadratic"

    # Adaptive theta
    "v2_knee_adapt|KneeSloAdaptive|KNEE_GAMMA=2 KNEE_RISK_FN=knee_quadratic"
)

cleanup() {
    pkill -f simulator_simple.py 2>/dev/null || true
    pkill -f run_scheduler.py 2>/dev/null || true
    sleep 2
}

cleanup
trap cleanup EXIT

mkdir -p experiment_logs
echo "Wave 2 — Total experiments: ${#CONFIGS[@]} (sequential)"

idx=0
for row in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    exp=${row%%|*}; rest=${row#*|}
    sched=${rest%%|*}; extra=${rest#*|}

    if ls "${exp}_3000_3100_${sched}_accept_all_load_8.0_job_stats.json" >/dev/null 2>&1; then
        echo "[W2 $idx/${#CONFIGS[@]}] $exp ($sched)  SKIP"
        continue
    fi

    echo "[W2 $idx/${#CONFIGS[@]}] $exp ($sched)  extra='$extra'"
    env $extra \
        SCHED="$sched" EXP_PREFIX="$exp" \
        PORT_BASE=50050 \
        LOAD=8 START=3000 STOP=3100 \
        bash run_one_experiment.sh
    cleanup
done

echo "WAVE 2 COMPLETE"
