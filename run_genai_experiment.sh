#!/bin/bash
# Run all 4 schedulers on Alibaba 2026 GenAI (Stable Diffusion) trace
set -e
cd "$(dirname "$0")"
source venv/bin/activate

TRACE="./cluster_job_log"
EXP="genai"
LOAD=8
START=3000
STOP=3100

for SCHED in Fifo Las Srtf SloScoring; do
    pkill -f simulator_simple.py 2>/dev/null || true
    pkill -f run_scheduler.py 2>/dev/null || true
    sleep 2

    echo "=========================================="
    echo " Scheduler: $SCHED | Load: $LOAD | Track: $START-$STOP"
    echo "=========================================="

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python simulator_simple.py \
        --cluster-job-log "$TRACE" --sim-type trace-synthetic \
        --jobs-per-hour $LOAD --exp-prefix "$EXP" --scheduler "$SCHED" \
        --start-job-track $START --end-job-track $STOP &
    SIM_PID=$!
    sleep 4

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python run_scheduler.py \
        --simulate --load $LOAD --exp-prefix "$EXP" \
        --scheduler-name "$SCHED" --start-id-track $START --stop-id-track $STOP

    wait $SIM_PID 2>/dev/null || true
    echo "--- $SCHED done ---"
    echo ""
done

echo "ALL EXPERIMENTS COMPLETE"
ls -la ${EXP}_*_job_stats.json 2>/dev/null
