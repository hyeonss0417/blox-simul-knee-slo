"""Plot the scale-up sweep — same ρ, varying cluster size."""
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = [
    "Apple SD Gothic Neo", "AppleGothic", "Helvetica", "Arial", "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/report_v2/figures"
os.makedirs(OUT, exist_ok=True)

SETUPS = [
    ("g1_l200",  1,  200,  "1 GPU\nload 200"),
    ("g4_l800",  4,  800,  "4 GPU\nload 800"),
    ("g8_l1600", 8,  1600, "8 GPU\nload 1600"),
]
ALGOS = [
    ("fifo",        "FIFO",         "#6b7280"),
    ("las",         "LAS",          "#e67e22"),
    ("srtf",        "SRTF",         "#3498db"),
    ("metasrtf",    "MetaSrtf",     "#27ae60"),
    ("srtfslo",     "SrtfSlo",      "#16a085"),
    ("metasrtfslo", "MetaSrtfSlo",  "#0d3b66"),
]


def load(setup_tag, algo):
    files = glob.glob(
        f"results/contention_sweep/sc_{setup_tag}_{algo}_*_accept_all_*_job_stats.json"
    )
    if not files:
        return None
    d = json.load(open(files[0]))
    j = [v[1] - v[0] for v in d.values() if isinstance(v, list) and len(v) >= 2]
    return j if j else None


# Figure 1: scaling — avg JCT vs cluster size for each algorithm
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: per-algorithm scaling
ax = axes[0]
gpus = [s[1] for s in SETUPS]
for short, label, color in ALGOS:
    avgs = []
    for setup_tag, _, _, _ in SETUPS:
        j = load(setup_tag, short)
        if j is None:
            avgs.append(None)
        else:
            avgs.append(np.mean(j))
    # Plot non-None
    xs = [g for g, a in zip(gpus, avgs) if a is not None]
    ys = [a for a in avgs if a is not None]
    if ys:
        ax.plot(xs, ys, "o-", label=label, color=color, lw=2, markersize=10)
    # Mark thrashed points
    for g, a in zip(gpus, avgs):
        if a is None:
            ax.scatter([g], [50], marker="x", s=120, color=color, lw=3)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(gpus)
ax.set_xticklabels([f"{g} GPU" for g in gpus])
ax.set_xlabel("Cluster size (load proportionally scaled to keep ρ=2.6×)")
ax.set_ylabel("Avg JCT (s)")
ax.set_title("Scaling — same ρ, more GPUs (✕ = thrashed)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which="both")

# Right: bar chart per setup
ax = axes[1]
width = 0.13
x = np.arange(len(SETUPS))
for i, (short, label, color) in enumerate(ALGOS):
    vals = []
    for setup_tag, _, _, _ in SETUPS:
        j = load(setup_tag, short)
        vals.append(np.mean(j) if j else 0)
    bars = ax.bar(x + i * width - 2.5 * width, vals, width,
                  label=label, color=color, edgecolor="white", linewidth=1)
    for xi, v in zip(x + i * width - 2.5 * width, vals):
        if v == 0:
            ax.text(xi, 30, "💀", ha="center", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels([s[3] for s in SETUPS])
ax.set_ylabel("Avg JCT (s)")
ax.set_yscale("log")
ax.set_title("Same ρ=2.6×, scaling cluster size")
ax.legend(fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/scale_sweep.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"saved {OUT}/scale_sweep.png")
