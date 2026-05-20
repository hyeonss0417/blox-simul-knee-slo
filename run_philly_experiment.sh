#!/bin/bash
# Run all 4 schedulers with real Philly Trace at multiple load levels
set -e
cd /Users/ericpark/Desktop/Projects/blox
source venv/bin/activate

TRACE="./trace-data/cluster_job_log"
EXP_PREFIX="philly"
LOADS="6 10 14"
SCHEDULERS="Fifo Las Srtf SloScoring"

for LOAD in $LOADS; do
  for SCHED in $SCHEDULERS; do
    echo "=========================================="
    echo "Scheduler: $SCHED | Load: $LOAD jobs/hr"
    echo "=========================================="

    # Kill any leftover processes
    pkill -f simulator_simple.py 2>/dev/null || true
    pkill -f run_scheduler.py 2>/dev/null || true
    sleep 2

    # Start simulator
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python simulator_simple.py \
        --cluster-job-log "$TRACE" \
        --sim-type trace-synthetic \
        --jobs-per-hour "$LOAD" \
        --exp-prefix "$EXP_PREFIX" \
        --scheduler "$SCHED" \
        --start-job-track 3000 \
        --end-job-track 4000 &
    SIM_PID=$!
    sleep 3

    # Start scheduler
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python run_scheduler.py \
        --simulate --load "$LOAD" --exp-prefix "$EXP_PREFIX" \
        --scheduler-name "$SCHED" 2>&1 | tail -5

    wait $SIM_PID 2>/dev/null || true

    echo "Done: $SCHED at load $LOAD"
    echo ""
  done
done

echo "All experiments complete!"
ls -la ${EXP_PREFIX}_*_job_stats.json 2>/dev/null
