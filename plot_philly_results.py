"""Generate publication-quality charts from real Philly Trace experiment results."""
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

SCHEDS = ["Fifo", "Las", "Srtf", "SloScoring"]
LABELS = ["FIFO", "LAS", "SRTF", "SloScoring"]
COLORS = ["#6b7280", "#e67e22", "#3498db", "#0d1b2a"]

# Load data
jct_data = {}
resp_data = {}
for s in SCHEDS:
    with open(f"philly8_3000_3100_{s}_accept_all_load_8.0_job_stats.json") as f:
        d = json.load(f)
    jct_data[s] = sorted([v[1] - v[0] for v in d.values()])
    with open(f"philly8_3000_3100_{s}_accept_all_load_8.0_responsivness.json") as f:
        d = json.load(f)
    resp_data[s] = sorted([v[1] - v[0] for v in d.values()])

# ── 1. Average JCT Bar Chart ──
fig, ax = plt.subplots(figsize=(8, 5))
avgs = [np.mean(jct_data[s]) for s in SCHEDS]
bars = ax.bar(LABELS, [a / 3600 for a in avgs], color=COLORS, width=0.6, edgecolor="white", linewidth=1.5)
ax.set_ylabel("Average JCT (hours)", fontsize=13)
ax.set_title("Average Job Completion Time by Scheduler\n(Real Philly Trace, Load=8, 128 GPUs)", fontsize=14, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Add value labels
for bar, avg in zip(bars, avgs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{avg/3600:.1f}h", ha="center", va="bottom", fontsize=11, fontweight="bold")
# Add % improvement annotation
fifo_avg = avgs[0]
for i, (bar, avg) in enumerate(zip(bars, avgs)):
    if i > 0:
        pct = (avg - fifo_avg) / fifo_avg * 100
        color = "#1a7431" if pct < 0 else "#c0392b"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
                f"{pct:+.1f}%", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/avg_jct_comparison.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/avg_jct_comparison.pdf", dpi=300, bbox_inches="tight")
print("1. avg_jct_comparison")

# ── 2. JCT CDF ──
fig, ax = plt.subplots(figsize=(8, 5))
for s, label, color in zip(SCHEDS, LABELS, COLORS):
    vals = jct_data[s]
    cdf_y = [float(i) / len(vals) for i in range(len(vals))]
    ax.plot([v / 3600 for v in vals], cdf_y, label=label, color=color, linewidth=2)
ax.set_xlabel("Job Completion Time (hours)", fontsize=13)
ax.set_ylabel("CDF", fontsize=13)
ax.set_title("JCT Distribution (CDF)\n(Real Philly Trace, Load=8)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, framealpha=0.9)
ax.set_xscale("log")
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/jct_cdf.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/jct_cdf.pdf", dpi=300, bbox_inches="tight")
print("2. jct_cdf")

# ── 3. JCT Percentiles Grouped Bar ──
fig, ax = plt.subplots(figsize=(10, 5))
percentiles = ["Median", "P95", "P99"]
x = np.arange(len(percentiles))
width = 0.18
for i, (s, label, color) in enumerate(zip(SCHEDS, LABELS, COLORS)):
    vals = jct_data[s]
    pcts = [np.median(vals)/3600, np.percentile(vals, 95)/3600, np.percentile(vals, 99)/3600]
    bars = ax.bar(x + i * width, pcts, width, label=label, color=color, edgecolor="white", linewidth=1)
    for bar, val in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.0f}h", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_ylabel("JCT (hours)", fontsize=13)
ax.set_title("JCT Percentile Comparison\n(Real Philly Trace, Load=8)", fontsize=14, fontweight="bold")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(percentiles, fontsize=12)
ax.legend(fontsize=10, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/jct_percentiles.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/jct_percentiles.pdf", dpi=300, bbox_inches="tight")
print("3. jct_percentiles")

# ── 4. Responsiveness Bar Chart ──
fig, ax = plt.subplots(figsize=(8, 5))
resp_avgs = [np.mean(resp_data[s]) for s in SCHEDS]
bars = ax.bar(LABELS, resp_avgs, color=COLORS, width=0.6, edgecolor="white", linewidth=1.5)
ax.set_ylabel("Average Responsiveness (seconds)", fontsize=13)
ax.set_title("Job Responsiveness (Time to First Execution)\n(Real Philly Trace, Load=8)", fontsize=14, fontweight="bold")
ax.set_yscale("log")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar, val in zip(bars, resp_avgs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
            f"{val:,.0f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/responsiveness.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{OUT}/responsiveness.pdf", dpi=300, bbox_inches="tight")
print("4. responsiveness")

# ── 5. Radar / Summary Table Image ──
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis("off")
col_labels = ["Scheduler", "Avg JCT (h)", "vs FIFO", "Median (h)", "P95 (h)", "P99 (h)", "Resp (s)"]
fifo_jct = np.mean(jct_data["Fifo"])
table_data = []
for s, label in zip(SCHEDS, LABELS):
    vals = jct_data[s]
    avg = np.mean(vals)
    pct = f"{(avg - fifo_jct)/fifo_jct*100:+.1f}%" if s != "Fifo" else "baseline"
    resp = np.mean(resp_data[s])
    table_data.append([
        label,
        f"{avg/3600:.1f}",
        pct,
        f"{np.median(vals)/3600:.1f}",
        f"{np.percentile(vals,95)/3600:.1f}",
        f"{np.percentile(vals,99)/3600:.1f}",
        f"{resp:,.0f}"
    ])
table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)
# Style header
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#0d1b2a")
    table[0, j].set_text_props(color="white", fontweight="bold")
# Highlight SloScoring row
for j in range(len(col_labels)):
    table[4, j].set_facecolor("#f0e6c0")
    table[4, j].set_text_props(fontweight="bold")
ax.set_title("Experiment Results Summary (Real Philly Trace, Load=8 jobs/hr, 128 GPUs)",
             fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{OUT}/results_table.png", dpi=200, bbox_inches="tight")
print("5. results_table")

print("\nAll figures saved to docs/figures/")
