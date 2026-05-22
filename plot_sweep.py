"""Plot the contention sweep results (multiple setups × algorithms)."""
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]

OUT = "docs/report_v2/figures"
os.makedirs(OUT, exist_ok=True)

SETUPS = [
    ("m1g1_l100", "1 GPU, load=100\n(mild ~1.3× over)"),
    ("m1g1_l200", "1 GPU, load=200\n(HEAVY ~2.6× over)"),
    ("m1g2_l400", "2 GPU, load=400\n(HEAVY ~2.6× over)"),
]
ALGOS = [
    ("fifo",        "FIFO",         "#6b7280"),
    ("las",         "LAS",          "#e67e22"),
    ("srtf",        "SRTF (oracle)","#3498db"),
    ("metasrtf",    "MetaSrtf",     "#27ae60"),
    ("srtfslo",     "SrtfSlo",      "#16a085"),
    ("metasrtfslo", "MetaSrtfSlo",  "#0d3b66"),
    ("hrrnslo",     "HrrnSlo ★ours","#dc2626"),
]


def load_setup_algo(setup, algo):
    files = glob.glob(f"results/contention_sweep/sw_{setup}_*_{algo}_*_accept_all_*_job_stats.json")
    if not files:
        return None
    all_j = []
    for f in files:
        d = json.load(open(f))
        for v in d.values():
            if isinstance(v, list) and len(v) >= 2:
                all_j.append(v[1] - v[0])
    return all_j if all_j else None


# Plot 1: Avg JCT bar chart per setup (subplot)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for col, (setup, title) in enumerate(SETUPS):
    ax = axes[col]
    labels, vals, colors = [], [], []
    for short, label, color in ALGOS:
        j = load_setup_algo(setup, short)
        if j is None:
            labels.append(label)
            vals.append(0)
            colors.append("#888888")
        else:
            labels.append(label)
            vals.append(np.mean(j))
            colors.append(color)
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{v:.0f}", ha="center", fontsize=8, fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, max(vals) * 0.05,
                    "💀", ha="center", fontsize=14)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Avg JCT (s)" if col == 0 else "")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
plt.suptitle("Contention sweep — Avg JCT (💀 = thrashed, timeout)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/sweep_avg_by_setup.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"saved {OUT}/sweep_avg_by_setup.png")

# Plot 2: Pareto Avg vs P99 across setups
fig, ax = plt.subplots(figsize=(9, 6))
markers = {"m1g1_l100": "o", "m1g1_l200": "s", "m1g2_l400": "^"}
for setup, _ in SETUPS:
    for short, label, color in ALGOS:
        j = load_setup_algo(setup, short)
        if j is None: continue
        ax.scatter(np.mean(j), np.percentile(j, 99), s=120, color=color,
                   marker=markers[setup], edgecolor="white", linewidth=1.5)
        ax.annotate(f"{label[:5]}", (np.mean(j), np.percentile(j, 99)),
                    xytext=(6, 6), textcoords="offset points", fontsize=7)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Avg JCT (s)")
ax.set_ylabel("P99 JCT (s)")
ax.set_title("Pareto frontier — Contention sweep\n(○ mild  ▢ 1G heavy  △ 2G heavy)")
ax.grid(True, alpha=0.3)
# Legend for algos
for short, label, color in ALGOS:
    ax.scatter([], [], color=color, s=80, label=label)
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/sweep_pareto.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"saved {OUT}/sweep_pareto.png")
