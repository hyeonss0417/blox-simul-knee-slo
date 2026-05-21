#!/bin/bash
# Wave 3: Load sensitivity — compare the winning Knee config against
# FIFO/LAS/SRTF at multiple loads (4, 12, 16 jobs/hr).
# Default load is 8 (already done in Wave 1).
set -u
cd "$(dirname "$0")"
source venv/bin/activate

# Fixed best config (will be set after Wave 2 analysis).
# For now, use KneeSlo with the design-doc defaults: theta=0.7 q gamma=2.
KNEE_CONFIG_NAME="best"
KNEE_ENV="KNEE_THETA=0.7 KNEE_GAMMA=2 KNEE_RISK_FN=knee_quadratic"

LOADS=(4 12 16)
SCHEDS=(Fifo Las Srtf KneeSlo)

cleanup() {
    pkill -f simulator_simple.py 2>/dev/null || true
    pkill -f run_scheduler.py 2>/dev/null || true
    sleep 2
}
cleanup
trap cleanup EXIT

mkdir -p experiment_logs
echo "Wave 3 — Load sensitivity: ${#LOADS[@]} loads × ${#SCHEDS[@]} scheds = $((${#LOADS[@]} * ${#SCHEDS[@]})) experiments"

idx=0
for L in "${LOADS[@]}"; do
    for S in "${SCHEDS[@]}"; do
        idx=$((idx + 1))
        exp="w3_l${L}_$(echo $S | tr A-Z a-z)"
        if ls "${exp}_3000_3100_${S}_accept_all_load_${L}.0_job_stats.json" >/dev/null 2>&1; then
            echo "[W3 $idx] $exp ($S, load=$L)  SKIP"
            continue
        fi
        echo "[W3 $idx] $exp sched=$S load=$L"
        if [ "$S" = "KneeSlo" ]; then
            env $KNEE_ENV \
                SCHED="$S" EXP_PREFIX="$exp" PORT_BASE=50050 \
                LOAD=$L START=3000 STOP=3100 \
                bash run_one_experiment.sh
        else
            env SCHED="$S" EXP_PREFIX="$exp" PORT_BASE=50050 \
                LOAD=$L START=3000 STOP=3100 \
                bash run_one_experiment.sh
        fi
        cleanup
    done
done

echo "WAVE 3 COMPLETE"
