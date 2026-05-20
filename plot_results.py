"""
Generate baseline comparison graphs from Blox simulation results.
Outputs: bar chart (Avg JCT) + CDF (JCT distribution) + bar chart (responsiveness).
"""
import json
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 14

RESULTS_DIR = "/Users/ericpark/Desktop/Projects/blox"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect results from all available job_stats files
schedulers_data = {}

patterns = [
    ("Fifo", "exp_3000_4000_Fifo_accept_all_load_6.0_job_stats.json"),
    ("LAS", "exp_3000_4000_Las_accept_all_load_6.0_job_stats.json"),
    ("SRTF", "exp_3000_4000_Srtf_accept_all_load_6.0_job_stats.json"),
]

for name, fname in patterns:
    fpath = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            data = json.load(f)
        jcts = [v[1] - v[0] for v in data.values()]
        schedulers_data[name] = jcts
        print(f"{name}: {len(jcts)} jobs, Avg JCT = {np.mean(jcts):.1f}s, Median = {np.median(jcts):.1f}s")
    else:
        print(f"Not found: {fpath}")

if not schedulers_data:
    print("No data found!")
    exit(1)

colors = {"Fifo": "#4a90d9", "LAS": "#7cb342", "SRTF": "#e57373"}

# --- Figure 1: Average JCT Bar Chart ---
fig, ax = plt.subplots(figsize=(8, 5))
names = list(schedulers_data.keys())
avg_jcts = [np.mean(schedulers_data[n]) for n in names]
bars = ax.bar(names, avg_jcts, color=[colors.get(n, "#999") for n in names], width=0.5, edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, avg_jcts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{val:.0f}s", ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("Average JCT (seconds)")
ax.set_title("Baseline Scheduling Policy Comparison\n(Load = 6 jobs/hour, 32 nodes × 4 GPUs)")
ax.set_ylim(0, max(avg_jcts) * 1.15)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "avg_jct_comparison.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "avg_jct_comparison.pdf"), dpi=300, bbox_inches="tight")
print("Saved: avg_jct_comparison.png")

# --- Figure 2: JCT CDF ---
fig, ax = plt.subplots(figsize=(8, 5))
for name in names:
    jcts = sorted(schedulers_data[name])
    cdf = np.arange(1, len(jcts) + 1) / len(jcts)
    ax.plot(jcts, cdf, label=name, color=colors.get(name, "#999"), linewidth=2)
ax.set_xlabel("Job Completion Time (seconds)")
ax.set_ylabel("CDF")
ax.set_title("JCT Distribution (CDF)\n(Load = 6 jobs/hour)")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
ax.set_xscale("log")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "jct_cdf.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "jct_cdf.pdf"), dpi=300, bbox_inches="tight")
print("Saved: jct_cdf.png")

# --- Figure 3: JCT Percentile Comparison ---
fig, ax = plt.subplots(figsize=(8, 5))
percentiles = [50, 90, 95, 99]
x = np.arange(len(percentiles))
width = 0.25

for i, name in enumerate(names):
    jcts = schedulers_data[name]
    pvals = [np.percentile(jcts, p) for p in percentiles]
    ax.bar(x + i * width, pvals, width, label=name,
           color=colors.get(name, "#999"), edgecolor="black", linewidth=0.5)

ax.set_xlabel("Percentile")
ax.set_ylabel("JCT (seconds)")
ax.set_title("JCT Percentile Comparison\n(Load = 6 jobs/hour)")
ax.set_xticks(x + width)
ax.set_xticklabels([f"P{p}" for p in percentiles])
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "jct_percentiles.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "jct_percentiles.pdf"), dpi=300, bbox_inches="tight")
print("Saved: jct_percentiles.png")

print(f"\nAll figures saved to: {OUTPUT_DIR}")
