#!/bin/bash
# Fast experiment: track 200 jobs (3000-3200) at load=10
set -e
cd /Users/ericpark/Desktop/Projects/blox
source venv/bin/activate

TRACE="./trace-data/cluster_job_log"
EXP="philly10"
LOAD=10
START=3000
STOP=3200

for SCHED in Fifo Las Srtf SloScoring; do
    pkill -f simulator_simple.py 2>/dev/null || true
    pkill -f run_scheduler.py 2>/dev/null || true
    sleep 2

    echo "=== $SCHED load=$LOAD ==="
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python simulator_simple.py \
        --cluster-job-log "$TRACE" --sim-type trace-synthetic \
        --jobs-per-hour $LOAD --exp-prefix "$EXP" --scheduler "$SCHED" \
        --start-job-track $START --end-job-track $STOP &
    SIM_PID=$!
    sleep 4

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python run_scheduler.py \
        --simulate --load $LOAD --exp-prefix "$EXP" \
        --scheduler-name "$SCHED" --start-id-track $START --stop-id-track $STOP \
        2>&1 | grep -E "Avg JCT|Avg responsiveness|Terminate"
    wait $SIM_PID 2>/dev/null || true
    echo "--- $SCHED done ---"
done
echo "ALL DONE"
