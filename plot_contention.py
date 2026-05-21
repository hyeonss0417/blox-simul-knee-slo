"""Plot the contention (2-GPU) experiment results."""
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]

OUT = "docs/figures_v2"
os.makedirs(OUT, exist_ok=True)

CONFIGS = [
    ("co_fifo",        "FIFO",                 "#6b7280"),
    ("co_las",         "LAS",                  "#e67e22"),
    ("co_srtf",        "SRTF (oracle)",        "#3498db"),
    ("co_sjf",         "SjfTotal (cat-mean)",  "#9b59b6"),
    ("co_hrrn",        "HRRN",                 "#8e44ad"),
    ("co_metasrtf",    "MetaSrtf (non-Oracle)","#27ae60"),
    ("co_lasslo60",    "LasSlo SLO=60",        "#34495e"),
    ("co_srtfslo60",   "SrtfSlo SLO=60",       "#16a085"),
    ("co_metaslo60",   "MetaLasSlo SLO=60",    "#0d3b66"),
    ("co_lasslo120",   "LasSlo SLO=120",       "#7d6608"),
    ("co_metaslo_m3",  "MetaLasSlo mult=3",    "#9c640c"),
]

data = {}
for prefix, label, color in CONFIGS:
    pat = f"{prefix}_*_*_accept_all_load_200.0_job_stats.json"
    m = glob.glob(pat)
    if not m:
        continue
    with open(m[0]) as f:
        d = json.load(f)
    j = [v[1] - v[0] for v in d.values() if isinstance(v, list) and len(v) >= 2]
    if j:
        data[prefix] = (label, color, j)

print(f"Loaded {len(data)} configs")

# Avg JCT bar (sorted)
fig, ax = plt.subplots(figsize=(11, 5))
sorted_data = sorted(data.values(), key=lambda x: np.mean(x[2]))
labels = [x[0] for x in sorted_data]
avgs = [np.mean(x[2]) for x in sorted_data]
colors = [x[1] for x in sorted_data]
bars = ax.bar(labels, avgs, color=colors, edgecolor="white", linewidth=1.5)
ax.set_ylabel("Avg JCT (s)")
ax.set_title("2-GPU contention test — Avg JCT (lower = better)")
for bar, avg in zip(bars, avgs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{avg:.1f}", ha="center", fontsize=9, fontweight="bold")
plt.xticks(rotation=30, ha="right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/contention_avg.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"saved {OUT}/contention_avg.png")

# CDF
fig, ax = plt.subplots(figsize=(10, 6))
for prefix, (label, color, jcts) in data.items():
    srt = sorted(jcts)
    cdf = [(i + 1) / len(srt) for i in range(len(srt))]
    ax.plot(srt, cdf, label=label, color=color, lw=2)
ax.set_xlabel("JCT (s)")
ax.set_ylabel("CDF")
ax.set_title("2-GPU contention test — JCT CDF")
ax.set_xscale("log")
ax.axvline(60, color="gray", linestyle=":", alpha=0.5, label="SLO=60s")
ax.axvline(120, color="gray", linestyle="--", alpha=0.5, label="SLO=120s")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/contention_cdf.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"saved {OUT}/contention_cdf.png")
