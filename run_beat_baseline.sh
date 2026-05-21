#!/bin/bash
# Beat-baseline experiment: LAS-SLO and SRTF-SLO against LAS / SRTF on
# both workloads (A = training-like, B = real inference).
#
# Goal: match baseline Avg JCT but lower SLO miss rate.
set -u
cd "$(dirname "$0")"
source venv/bin/activate

cleanup() {
    pkill -f simulator_simple.py 2>/dev/null || true
    pkill -f run_scheduler.py 2>/dev/null || true
    sleep 2
}
cleanup
trap cleanup EXIT

mkdir -p experiment_logs

# ─────────────────────────────────────────────────────────────────
# WORKLOAD A — training-like (exponential=True, hours)
# ─────────────────────────────────────────────────────────────────
# We'll need exponential=True for this — flip it back temporarily.
# Currently default is exponential=False, but for workload A
# experiments we want gavel-like synthetic duration.
# Use a separate run with manual override via env var.

echo "=== Workload A: training-like (synthetic durations, SLO=24h) ==="

# For workload A we need exponential=True. The simulator currently has
# it as False (our inference default).  We'll use a wrapper env that
# the simulator picks up at runtime.  Quick hack: edit and revert.

# Actually simpler — we already have v2 / w3 results from when default
# was True.  For LAS-SLO / SRTF-SLO comparison, use the SAME workload
# (synthetic). Run them with cluster=128 GPU, load=8 jobs/hr, SLO=24h.

# We need to flip the default back temporarily.  Patch the file.
python -c "
import re
p='simulator_simple.py'
t=open(p).read()
t=re.sub(r'^\s*exponential=False,.*?#.*?inference\.', '        exponential=True,', t, flags=re.MULTILINE|re.DOTALL)
open(p,'w').write(t)
print('Flipped exponential=True for workload A')
"

rm -f *.pickle
# Re-use the v2 prefix layout but with our new schedulers.
CONFIGS_A=(
    "bb_a_las|Las|"
    "bb_a_srtf|Srtf|"
    "bb_a_lasslo24|LasSlo|LAS_SLO_TARGET=86400 LAS_SLO_THETA=0.7"
    "bb_a_srtfslo24|SrtfSlo|SRTF_SLO_TARGET=86400 SRTF_SLO_THETA=0.7"
    "bb_a_lasslo36|LasSlo|LAS_SLO_TARGET=129600 LAS_SLO_THETA=0.7"
    "bb_a_lasslo48|LasSlo|LAS_SLO_TARGET=172800 LAS_SLO_THETA=0.7"
)
idx=0
for row in "${CONFIGS_A[@]}"; do
    idx=$((idx + 1))
    exp=${row%%|*}; rest=${row#*|}
    sched=${rest%%|*}; extra=${rest#*|}
    if ls "${exp}_3000_3100_${sched}_accept_all_load_8.0_job_stats.json" >/dev/null 2>&1; then
        echo "[A $idx] $exp SKIP"; continue
    fi
    echo "[A $idx/${#CONFIGS_A[@]}] $exp ($sched)  extra='$extra'"
    env $extra \
        SCHED="$sched" EXP_PREFIX="$exp" PORT_BASE=50050 \
        LOAD=8 START=3000 STOP=3100 ROUND_DURATION=300 \
        bash run_one_experiment.sh
    cleanup
done

# Restore exponential=False for inference experiments
python -c "
import re
p='simulator_simple.py'
t=open(p).read()
t=re.sub(r'^\s*exponential=True,.*$', '        exponential=False,', t, flags=re.MULTILINE)
open(p,'w').write(t)
print('Restored exponential=False')
"
rm -f *.pickle

# ─────────────────────────────────────────────────────────────────
# WORKLOAD B — real inference (exponential=False, seconds)
# ─────────────────────────────────────────────────────────────────
# Smaller cluster, higher load to actually force queueing.

echo ""
echo "=== Workload B: real inference (32 GPU, load 8000, SLO=60s) ==="

export BLOX_NUM_MACHINES=8
export BLOX_GPUS_PER_MACHINE=4

CONFIGS_B=(
    "bb_b_las|Las|"
    "bb_b_srtf|Srtf|"
    "bb_b_lasslo60|LasSlo|LAS_SLO_TARGET=60 LAS_SLO_THETA=0.7"
    "bb_b_srtfslo60|SrtfSlo|SRTF_SLO_TARGET=60 SRTF_SLO_THETA=0.7"
    "bb_b_lasslo120|LasSlo|LAS_SLO_TARGET=120 LAS_SLO_THETA=0.7"
)
idx=0
for row in "${CONFIGS_B[@]}"; do
    idx=$((idx + 1))
    exp=${row%%|*}; rest=${row#*|}
    sched=${rest%%|*}; extra=${rest#*|}
    if ls "${exp}_3000_3300_${sched}_accept_all_load_8000.0_job_stats.json" >/dev/null 2>&1; then
        echo "[B $idx] $exp SKIP"; continue
    fi
    echo "[B $idx/${#CONFIGS_B[@]}] $exp ($sched)  extra='$extra'"
    env $extra \
        SCHED="$sched" EXP_PREFIX="$exp" PORT_BASE=50050 \
        LOAD=8000 START=3000 STOP=3300 ROUND_DURATION=10 \
        bash run_one_experiment.sh
    cleanup
done

echo ""
echo "BEAT-BASELINE EXPERIMENT COMPLETE"
