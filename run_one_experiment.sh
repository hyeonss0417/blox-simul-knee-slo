#!/bin/bash
# Run a single (simulator + scheduler) experiment with isolated ports.
#
# Args (env vars):
#   SCHED          scheduler name (e.g. KneeSlo, Fifo, Las, ...)
#   EXP_PREFIX     unique prefix for output files (e.g. "kneeQ7g2")
#   PORT_BASE      base port (will use BASE, BASE+1, BASE+2)
#   LOAD           jobs per hour (default 8)
#   START, STOP    job track range
#   KNEE_THETA, KNEE_GAMMA, KNEE_C2, KNEE_C3, KNEE_RISK_FN  Knee-SLO knobs
#
# Each batch should call this with disjoint PORT_BASE values
# (e.g. 50050, 50060, 50070, 50080).
set -e
cd "$(dirname "$0")"

# venv activate is optional — caller may already have activated
if [ -z "$VIRTUAL_ENV" ] && [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

: "${SCHED:?need SCHED}"
: "${EXP_PREFIX:?need EXP_PREFIX}"
: "${PORT_BASE:=50050}"
: "${LOAD:=8}"
: "${START:=3000}"
: "${STOP:=3100}"
: "${TRACE:=./cluster_job_log}"
: "${ROUND_DURATION:=10}"

SIM_PORT=$PORT_BASE
RM_PORT=$((PORT_BASE + 1))
NM_PORT=$((PORT_BASE + 2))

LOGDIR=experiment_logs
mkdir -p "$LOGDIR"
SIM_LOG="$LOGDIR/${EXP_PREFIX}_sim.log"
SCHED_LOG="$LOGDIR/${EXP_PREFIX}_sched.log"

echo "[$EXP_PREFIX] sched=$SCHED  ports sim=$SIM_PORT rm=$RM_PORT nm=$NM_PORT"

# Clean stale pickles between runs (same trace, different scheduler)
# but keep them per-experiment so parallel runs don't trample each other.
# The simulator caches workload state in *.pickle in cwd; the workload
# generator keys it by trace name, so concurrent runs reading the SAME
# trace file CAN share the cache safely (read-only after generation).

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python simulator_simple.py \
    --cluster-job-log "$TRACE" --sim-type trace-synthetic \
    --jobs-per-hour "$LOAD" --exp-prefix "$EXP_PREFIX" \
    --scheduler "$SCHED" \
    --start-job-track "$START" --end-job-track "$STOP" \
    --simulator-rpc-port "$SIM_PORT" \
    --round-duration "$ROUND_DURATION" \
    > "$SIM_LOG" 2>&1 &
SIM_PID=$!

sleep 4

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python run_scheduler.py \
    --simulate --load "$LOAD" --exp-prefix "$EXP_PREFIX" \
    --scheduler-name "$SCHED" \
    --start-id-track "$START" --stop-id-track "$STOP" \
    --simulator-rpc-port "$SIM_PORT" \
    --central-scheduler-port "$RM_PORT" \
    --node-manager-port "$NM_PORT" \
    --round-duration "$ROUND_DURATION" \
    > "$SCHED_LOG" 2>&1

# Simulator never self-exits (grpc server.wait_for_termination loops).
# Now that the scheduler has printed "Terminate: <name>" and exited,
# we must actively kill the simulator before the next iteration.
kill $SIM_PID 2>/dev/null || true
sleep 1
kill -9 $SIM_PID 2>/dev/null || true
wait $SIM_PID 2>/dev/null || true

# Auto-organize result files into results/<category>/ based on prefix.
case "$EXP_PREFIX" in
    v2_*)        DEST="results/v2_training" ;;
    inf_*)       DEST="results/inference_singlepool" ;;
    co_*)        DEST="results/contention" ;;
    sw_*|sc_*)   DEST="results/contention_sweep" ;;
    mx_*|mx2_*)  DEST="results/mixed" ;;
    bb_*)        DEST="results/beat_baseline" ;;
    w3_*)        DEST="results/load_sweep" ;;
    mw_*|mw2_*)  DEST="results/meta_win" ;;
    closed_*)    DEST="results/closed_batch" ;;
    *)           DEST="results/misc" ;;
esac
mkdir -p "$DEST"
mv ${EXP_PREFIX}_*.json "$DEST/" 2>/dev/null || true

echo "[$EXP_PREFIX] done → $DEST"
