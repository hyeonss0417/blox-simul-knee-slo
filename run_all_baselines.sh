#!/bin/bash
# Run FIFO, LAS, SRTF baselines sequentially
# Each run: start simulator -> start scheduler -> wait for completion

cd /Users/ericpark/Desktop/Projects/blox
source venv/bin/activate

for SCHED in Fifo Las Srtf; do
    echo "=========================================="
    echo "Running scheduler: $SCHED"
    echo "=========================================="

    # Kill any leftover processes
    pkill -f simulator_simple.py 2>/dev/null
    pkill -f run_scheduler.py 2>/dev/null
    sleep 2

    # Start simulator with this scheduler name
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python simulator_simple.py \
        --cluster-job-log ./cluster_job_log \
        --sim-type trace-synthetic \
        --jobs-per-hour 1 \
        --exp-prefix baseline \
        --scheduler "$SCHED" &
    SIM_PID=$!
    sleep 3

    # Start scheduler
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python run_scheduler.py \
        --simulate --load 1 --exp-prefix baseline \
        --scheduler-name "$SCHED" 2>&1 | grep -E "Avg JCT|Avg responsiveness|Running Scheduler|Terminate"

    # Wait for simulator to finish
    wait $SIM_PID 2>/dev/null

    echo "Done: $SCHED"
    echo ""
done

echo "All baselines complete!"
ls -la baseline_*_job_stats.json
