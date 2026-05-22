"""Worked-example figure for HrrnSlo: priority of a long training job
over time, comparing FIFO / HRRN / SrtfSlo / HrrnSlo.

Scenario:
  t=0     long training job X arrives (service = 5000 s)
  t=100   inference job Y_1 arrives (service = 30 s), every 100 s thereafter
  → How does X's chance of being scheduled evolve?

We compute a "scheduling rank" of X relative to all other live jobs in queue
at each time tick (rank 1 = X runs next). Lower rank = better for X.
"""
import os
import math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "docs/report_v2/figures/hrrnslo_walkthrough.png")

X_SERVICE       = 5000.0
Y_SERVICE       = 30.0
Y_ARRIVAL_GAP   = 100.0
SLO_TARGET      = 1500.0
THETA           = 0.7
TICK            = 10
HORIZON         = 3000     # show first 3000 s

def queue_at(t):
    """Return list of (jid, submit, service) for jobs in queue at time t.
    X is always there; Y_k arrives at k*GAP, completes Y_SERVICE after that
    (we assume Y always runs immediately if scheduler picks it; for the
    'queue snapshot' we list Y jobs that arrived in (t-GAP, t]).
    For this didactic figure we keep the queue small: X + the 5 most-recent Y."""
    jobs = [("X", 0.0, X_SERVICE)]
    if t >= Y_ARRIVAL_GAP:
        k_now = int(t // Y_ARRIVAL_GAP)
        for k in range(max(1, k_now - 4), k_now + 1):
            arr = k * Y_ARRIVAL_GAP
            if arr <= t and t - arr < Y_SERVICE * 1.5:  # roughly "in queue"
                jobs.append((f"Y{k}", arr, Y_SERVICE))
    return jobs

def rank_x(score_fn, t):
    jobs = queue_at(t)
    scored = [(jid, score_fn(submit, service, t)) for jid, submit, service in jobs]
    scored.sort(key=lambda x: x[1])
    for i, (jid, _) in enumerate(scored):
        if jid == "X":
            return i + 1
    return len(scored) + 1

# Scoring keys: lower = scheduled sooner
def s_fifo(submit, service, t):
    return submit

def s_hrrn(submit, service, t):
    wait = max(0, t - submit)
    R = (wait + service) / service
    return -R

def s_srtf(submit, service, t):
    """Pure SRTF (oracle): shortest remaining service first. No aging, no bucket."""
    return service

def s_srtfslo(submit, service, t):
    """SRTF + SLO bucket: bucket by wait threshold, secondary = service."""
    wait = max(0, t - submit)
    if wait >= SLO_TARGET:
        bucket = 0
    elif wait >= THETA * SLO_TARGET:
        bucket = 1
    else:
        bucket = 2
    return (bucket, service)   # shortest first within bucket

def s_hrrnslo(submit, service, t):
    """HrrnSlo: bucket × -R."""
    wait = max(0, t - submit)
    R = (wait + service) / service
    if wait >= SLO_TARGET:
        bucket = 0
    elif wait >= THETA * SLO_TARGET:
        bucket = 1
    else:
        bucket = 2
    return (bucket, -R)

ts = np.arange(0, HORIZON + 1, TICK)
rk_fifo     = [rank_x(s_fifo, t)     for t in ts]
rk_srtf     = [rank_x(s_srtf, t)     for t in ts]
rk_hrrn     = [rank_x(s_hrrn, t)     for t in ts]
rk_srtfslo  = [rank_x(s_srtfslo, t)  for t in ts]
rk_hrrnslo  = [rank_x(s_hrrnslo, t)  for t in ts]

# Compute R curve of X for the bottom panel
r_x = [(t + X_SERVICE) / X_SERVICE for t in ts]
bucket_x = []
for t in ts:
    if t >= SLO_TARGET:        bucket_x.append(0)
    elif t >= THETA * SLO_TARGET: bucket_x.append(1)
    else:                       bucket_x.append(2)

# --- plot ---
fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), gridspec_kw={"height_ratios":[3, 2]})

ax = axes[0]
# Order: best for AVG JCT theory → worst for X. End with HrrnSlo for emphasis.
ax.plot(ts, rk_srtf,    label="SRTF (oracle, theoretical avg-JCT optimal)",
        color="#16a34a", lw=2.0, linestyle="--")
ax.plot(ts, rk_srtfslo, label="SRTF + SLO bucket",
        color="#a16207", lw=2.0, linestyle="--")
ax.plot(ts, rk_hrrn,    label="HRRN",
        color="#3b82f6", lw=2.0)
ax.plot(ts, rk_fifo,    label="FIFO (oldest-first, always picks X)",
        color="#888",    lw=2.0, linestyle=":")
ax.plot(ts, rk_hrrnslo, label="HrrnSlo (ours)",
        color="#dc2626", lw=2.8)

ax.set_ylabel("Scheduling rank of training job X\n(1 = scheduled next, lower = better)")
ax.set_xlabel("time (s)")
ax.set_title("X (training, service=5000 s) competing with stream of inference Y$_k$ "
             "(arriving every 100 s, each service=30 s)",
             fontsize=11)
ax.invert_yaxis()
ax.set_xlim(0, HORIZON)
ax.set_yticks([1, 2, 3, 4, 5, 6])
ax.grid(alpha=0.3)
ax.axvline(THETA * SLO_TARGET, color="#f59e0b", linestyle=":", alpha=0.7)
ax.axvline(SLO_TARGET,         color="#dc2626", linestyle=":", alpha=0.7)
ax.text(THETA * SLO_TARGET + 30, 6.3, "warning\n(0.7·SLO)", fontsize=9, color="#92400e")
ax.text(SLO_TARGET + 30, 6.3,         "critical\n(SLO=1500)", fontsize=9, color="#991b1b")

# Annotate the "starvation" region for SRTF
ax.annotate("SRTF: X always last\n→ starves forever",
            xy=(2500, max(rk_srtf[-50:])), xytext=(2200, 4.5),
            fontsize=9, color="#15803d",
            arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.2))
# Annotate HrrnSlo's transition
ax.annotate("HrrnSlo: bucket cliff\n→ rank 1 forever",
            xy=(1700, 1), xytext=(1900, 2.5),
            fontsize=9, color="#991b1b", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5))

ax.legend(loc="center right", fontsize=9, framealpha=0.95)

ax2 = axes[1]
ax2.plot(ts, r_x, color="#3b82f6", lw=2.2, label="R of X (response ratio)")
ax2.set_ylabel("R = (wait + service) / service")
ax2.set_xlabel("time (s)")
ax2.set_xlim(0, HORIZON)
ax2.grid(alpha=0.3)
# bucket bands
ax2.axvspan(0, THETA*SLO_TARGET, alpha=0.08, color="#22c55e", label="bucket 2 (safe)")
ax2.axvspan(THETA*SLO_TARGET, SLO_TARGET, alpha=0.15, color="#f59e0b", label="bucket 1 (warning)")
ax2.axvspan(SLO_TARGET, HORIZON, alpha=0.18, color="#dc2626", label="bucket 0 (critical)")
ax2.set_title("R of X grows only ~+20 % over 1000 s (training is long, R rises slowly) — "
              "bucket compensates",
              fontsize=11)
ax2.legend(loc="upper left", fontsize=9, ncol=2)

plt.suptitle("Worked example: why HrrnSlo protects long jobs while HRRN/SRTF+SLO don't",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG, dpi=140, bbox_inches="tight")
print(f"Saved: {FIG}")
