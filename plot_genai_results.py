"""
Generate charts from Alibaba 2026 GenAI trace experiment results.
Metrics: Avg JCT, JCT CDF, JCT Percentiles, Responsiveness, SLO Fulfillment Rate.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams["font.size"] = 12

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

EXP = "genai"
START, STOP = 3000, 3100
LOAD = 8.0

SCHEDS = ["Fifo", "Las", "Srtf", "SloScoring"]
LABELS = ["FIFO", "LAS", "SRTF", "SloScoring"]
COLORS = ["#6b7280", "#e67e22", "#3498db", "#0d1b2a"]

jct_data = {}
resp_data = {}
runtime_data = {}

for s in SCHEDS:
    job_file = f"{EXP}_{START}_{STOP}_{s}_accept_all_load_{LOAD}_job_stats.json"
    resp_file = f"{EXP}_{START}_{STOP}_{s}_accept_all_load_{LOAD}_responsivness.json"
    runtime_file = f"{EXP}_{START}_{STOP}_{s}_accept_all_load_{LOAD}_run_time_stats.json"

    if not os.path.exists(job_file):
        print(f"SKIP {s}: {job_file} not found")
        continue

    with open(job_file) as f:
        d = json.load(f)
    jct_data[s] = {k: v[1] - v[0] for k, v in d.items()}

    if os.path.exists(resp_file):
        with open(resp_file) as f:
            d = json.load(f)
        resp_data[s] = {k: v[1] - v[0] for k, v in d.items()}

    if os.path.exists(runtime_file):
        with open(runtime_file) as f:
            runtime_data[s] = json.load(f)

available = [s for s in SCHEDS if s in jct_data]
if not available:
    print("No result files found. Run experiments first.")
    exit(1)

avail_labels = [LABELS[SCHEDS.index(s)] for s in available]
avail_colors = [COLORS[SCHEDS.index(s)] for s in available]

# ── 1. Average JCT Bar Chart ──
fig, ax = plt.subplots(figsize=(8, 5))
avgs = [np.mean(list(jct_data[s].values())) for s in available]
bars = ax.bar(avail_labels, [a / 3600 for a in avgs], color=avail_colors,
              width=0.6, edgecolor="white", linewidth=1.5)
ax.set_ylabel("Average JCT (hours)", fontsize=13)
ax.set_title("Average Job Completion Time\n(Alibaba GenAI Trace, Load=8, 128 GPUs)",
             fontsize=14, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar, avg in zip(bars, avgs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{avg/3600:.1f}h", ha="center", va="bottom", fontsize=11, fontweight="bold")
fifo_avg = avgs[0]
for i, (bar, avg) in enumerate(zip(bars, avgs)):
    if i > 0:
        pct = (avg - fifo_avg) / fifo_avg * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
                f"{pct:+.1f}%", ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/avg_jct_comparison.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/avg_jct_comparison.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("1. avg_jct_comparison")

# ── 2. JCT CDF ──
fig, ax = plt.subplots(figsize=(8, 5))
for s, label, color in zip(available, avail_labels, avail_colors):
    vals = sorted(jct_data[s].values())
    cdf_y = [float(i + 1) / len(vals) for i in range(len(vals))]
    ax.plot([v / 3600 for v in vals], cdf_y, label=label, color=color, linewidth=2)
ax.set_xlabel("Job Completion Time (hours)", fontsize=13)
ax.set_ylabel("CDF", fontsize=13)
ax.set_title("JCT Distribution (CDF)\n(Alibaba GenAI Trace, Load=8)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/jct_cdf.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/jct_cdf.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("2. jct_cdf")

# ── 3. JCT Percentiles ──
fig, ax = plt.subplots(figsize=(10, 5))
percentiles = ["Median", "P95", "P99"]
x = np.arange(len(percentiles))
width = 0.18
for i, (s, label, color) in enumerate(zip(available, avail_labels, avail_colors)):
    vals = sorted(jct_data[s].values())
    pcts = [np.median(vals)/3600, np.percentile(vals, 95)/3600, np.percentile(vals, 99)/3600]
    bars = ax.bar(x + i * width, pcts, width, label=label, color=color,
                  edgecolor="white", linewidth=1)
    for bar, val in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}h", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_ylabel("JCT (hours)", fontsize=13)
ax.set_title("JCT Percentile Comparison\n(Alibaba GenAI Trace, Load=8)", fontsize=14, fontweight="bold")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(percentiles, fontsize=12)
ax.legend(fontsize=10, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/jct_percentiles.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/jct_percentiles.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("3. jct_percentiles")

# ── 4. Responsiveness ──
if resp_data:
    fig, ax = plt.subplots(figsize=(8, 5))
    resp_scheds = [s for s in available if s in resp_data]
    resp_labels = [LABELS[SCHEDS.index(s)] for s in resp_scheds]
    resp_colors = [COLORS[SCHEDS.index(s)] for s in resp_scheds]
    resp_avgs = [np.mean(list(resp_data[s].values())) for s in resp_scheds]
    bars = ax.bar(resp_labels, resp_avgs, color=resp_colors, width=0.6,
                  edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Average Responsiveness (seconds)", fontsize=13)
    ax.set_title("Job Responsiveness (Time to First Execution)\n(Alibaba GenAI Trace, Load=8)",
                 fontsize=14, fontweight="bold")
    if max(resp_avgs) / max(min(resp_avgs), 1) > 10:
        ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, resp_avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
                f"{val:,.0f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/responsiveness.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{OUT}/responsiveness.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("4. responsiveness")

# ── 5. SLO Fulfillment Rate ──
# SLO = k × ideal_duration (from trace)
# A job meets SLO if JCT <= k × ideal_duration
SLO_MULTIPLIERS = [1.5, 2.0, 3.0, 5.0, 10.0]

def compute_slo_rate(sched, k):
    if sched not in jct_data or sched not in runtime_data:
        return None
    met = 0
    total = 0
    for jid_str, jct in jct_data[sched].items():
        if jid_str in runtime_data[sched]:
            job = runtime_data[sched][jid_str]
            ideal_dur = job.get("job_iteration_time", 1) * job.get("job_total_iteration", 1)
            if jct <= k * ideal_dur:
                met += 1
            total += 1
    return met / total * 100 if total > 0 else None

slo_scheds = [s for s in available if s in runtime_data]
if slo_scheds:
    fig, ax = plt.subplots(figsize=(10, 6))
    for s, color in zip(slo_scheds, [COLORS[SCHEDS.index(s)] for s in slo_scheds]):
        rates = []
        valid_ks = []
        for k in SLO_MULTIPLIERS:
            r = compute_slo_rate(s, k)
            if r is not None:
                rates.append(r)
                valid_ks.append(k)
        label = LABELS[SCHEDS.index(s)]
        ax.plot(valid_ks, rates, marker="o", linewidth=2.5, markersize=8,
                label=label, color=color)
        for kk, rr in zip(valid_ks, rates):
            ax.annotate(f"{rr:.1f}%", (kk, rr), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, fontweight="bold")

    ax.set_xlabel("SLO Multiplier (k)", fontsize=13)
    ax.set_ylabel("SLO Fulfillment Rate (%)", fontsize=13)
    ax.set_title("SLO Fulfillment Rate by Scheduler\n(SLO = k × ideal_duration, Alibaba GenAI Trace, Load=8)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_xticks(SLO_MULTIPLIERS)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/slo_fulfillment.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{OUT}/slo_fulfillment.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("5. slo_fulfillment")

    # ── 5b. SLO Fulfillment Bar (fixed k=2.0) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    k_fixed = 2.0
    slo_rates = [compute_slo_rate(s, k_fixed) or 0 for s in slo_scheds]
    slo_labels = [LABELS[SCHEDS.index(s)] for s in slo_scheds]
    slo_colors = [COLORS[SCHEDS.index(s)] for s in slo_scheds]
    bars = ax.bar(slo_labels, slo_rates, color=slo_colors, width=0.6,
                  edgecolor="white", linewidth=1.5)
    ax.set_ylabel("SLO Fulfillment Rate (%)", fontsize=13)
    ax.set_title(f"SLO Fulfillment Rate (k={k_fixed})\n(Alibaba GenAI Trace, Load=8)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, slo_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/slo_fulfillment_bar.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{OUT}/slo_fulfillment_bar.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("5b. slo_fulfillment_bar")

# ── 6. Summary Table ──
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")
col_labels = ["Scheduler", "Avg JCT (h)", "vs FIFO", "Median (h)",
              "P95 (h)", "P99 (h)", "Resp (s)", "SLO@2x (%)"]
fifo_jct_avg = np.mean(list(jct_data[available[0]].values()))
table_data = []
for s in available:
    vals = list(jct_data[s].values())
    avg = np.mean(vals)
    label = LABELS[SCHEDS.index(s)]
    pct = f"{(avg - fifo_jct_avg)/fifo_jct_avg*100:+.1f}%" if s != available[0] else "baseline"
    resp = f"{np.mean(list(resp_data[s].values())):,.0f}" if s in resp_data else "N/A"
    slo = compute_slo_rate(s, 2.0)
    slo_str = f"{slo:.1f}" if slo is not None else "N/A"
    table_data.append([
        label,
        f"{avg/3600:.1f}",
        pct,
        f"{np.median(vals)/3600:.1f}",
        f"{np.percentile(vals, 95)/3600:.1f}",
        f"{np.percentile(vals, 99)/3600:.1f}",
        resp,
        slo_str,
    ])
table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#0d1b2a")
    table[0, j].set_text_props(color="white", fontweight="bold")
slo_row = len(available)
for j in range(len(col_labels)):
    table[slo_row, j].set_facecolor("#f0e6c0")
    table[slo_row, j].set_text_props(fontweight="bold")
ax.set_title("Experiment Results Summary\n(Alibaba 2026 GenAI Trace, Load=8 jobs/hr, 128 GPUs)",
             fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{OUT}/results_table.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/results_table.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("6. results_table")

print(f"\nAll figures saved to {OUT}/")
