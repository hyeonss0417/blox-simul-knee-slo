#!/bin/bash
# Master orchestrator: wait for Wave 1, then run Wave 2, then Wave 3,
# then plotting + report regen.
# Idempotent — uses SKIP on already-done experiments.
set -u
cd "$(dirname "$0")"
mkdir -p experiment_logs

WAIT_LOG=experiment_logs/all_waves.log
exec > >(tee -a "$WAIT_LOG") 2>&1

echo "[$(date)] master orchestrator starting"

# 1) Wait for Wave 1 (run_grid.sh) to be done.
#    It's already running in background; we just block until its sentinel
#    appears OR until 11 result files exist.
while true; do
    if grep -q "ALL EXPERIMENTS COMPLETE" experiment_logs/grid_master.log 2>/dev/null; then
        echo "[$(date)] Wave 1 complete (sentinel)"
        break
    fi
    sleep 30
done

# Activate venv early — plot/summary scripts need it.
source venv/bin/activate

# Generate intermediate plots/summary right after Wave 1 (so the
# report has at least the Wave 1 results visible while Wave 2 runs).
echo "[$(date)] post-Wave-1 plots/summary"
python plot_v2_results.py 2>&1 || true
python plot_slo_curves.py 2>&1 || true
python generate_summary.py 2>&1 || true

# 2) Wave 2.
echo "[$(date)] starting Wave 2"
bash run_grid_wave2.sh

echo "[$(date)] post-Wave-2 plots/summary"
python plot_v2_results.py 2>&1 || true
python plot_slo_curves.py 2>&1 || true
python generate_summary.py 2>&1 || true

# 3) Wave 3.
echo "[$(date)] starting Wave 3"
bash run_grid_wave3.sh

echo "[$(date)] post-Wave-3 plots/summary"
python plot_v2_results.py 2>&1 || true
python plot_w3_loadsweep.py 2>&1 || true
python generate_summary.py 2>&1 || true

# 4) Wave 4 — extreme + combined variants
echo "[$(date)] starting Wave 4"
bash run_grid_wave4.sh

echo "[$(date)] post-Wave-4 plots/summary"
python plot_v2_results.py 2>&1 || true
python plot_slo_curves.py 2>&1 || true
python generate_summary.py 2>&1 || true

# 5) Wave 5 — SLO recalibration to 24h (proper operating point).
echo "[$(date)] starting Wave 5"
bash run_grid_wave5.sh

# Final pass plots + summary + load sweep.
echo "[$(date)] final plots"
python plot_v2_results.py 2>&1 || true
python plot_w3_loadsweep.py 2>&1 || true
python generate_summary.py 2>&1 || true

# Compile final exec summary at the top of the report.
echo "[$(date)] compiling final report"
python compile_final_report.py 2>&1 || true

echo "[$(date)] ALL WAVES + PLOTS COMPLETE"
touch experiment_logs/ALL_WAVES_COMPLETE
