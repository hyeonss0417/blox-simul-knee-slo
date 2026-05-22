"""Generate headline figure for HrrnSlo vs HRRN vs FIFO in mixed workload."""
import json, glob, os, re, statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "docs/report_v2/figures")

# --- Collect ---
results = defaultdict(dict)  # results[(setup, load)][algo] = {n, mean, p95, p99, jcts}
for f in sorted(glob.glob(os.path.join(ROOT, "results/mixed/mx2_*_accept_all_load_*_job_stats.json"))):
    base = os.path.basename(f)
    m = re.match(r"mx2_(m\d+g\d+)_l(\d+)_(.+?)_200_350_(\w+)_accept_all_load_[\d.]+_job_stats\.json", base)
    if not m:
        continue
    setup, load, tag, _ = m.group(1), m.group(2), m.group(3), m.group(4)
    try:
        d = json.load(open(f))
    except Exception:
        continue
    jcts = []
    for v in d.values():
        if isinstance(v, list) and len(v) == 2:
            s, e = v
            if isinstance(s,(int,float)) and isinstance(e,(int,float)) and e > s:
                jcts.append(e - s)
    if not jcts:
        continue
    jcts_sorted = sorted(jcts)
    results[(setup, int(load))][tag] = dict(
        n=len(jcts),
        mean=statistics.mean(jcts),
        p95=jcts_sorted[min(len(jcts_sorted)-1, int(len(jcts_sorted)*0.95))],
        p99=jcts_sorted[min(len(jcts_sorted)-1, int(len(jcts_sorted)*0.99))],
        jcts=jcts_sorted,
    )

SETUP_ORDER = [("m1g2", 10), ("m1g4", 14), ("m1g4", 20), ("m1g4", 25)]
SETUP_LABELS = {
    ("m1g2", 10):  "2 GPU\nload=10",
    ("m1g4", 14):  "4 GPU\nload=14",
    ("m1g4", 20):  "4 GPU\nload=20",
    ("m1g4", 25):  "4 GPU\nload=25",
}

ALGOS = [
    ("fifo",                "FIFO",        "#888888"),
    ("hrrn",                "HRRN",        "#3b82f6"),
    ("srtfslo300",          "SRTF+SLO",    "#a16207"),
    ("metaslo300",          "MetaSRTF+SLO","#c026d3"),
    ("hrrnslo2_oracle1500", "HrrnSlo (ours)", "#dc2626"),
]

# --- Figure: 2-panel ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel 1: bar chart of mean JCT % vs FIFO
ax = axes[0]
x = np.arange(len(SETUP_ORDER))
W = 0.16
for i, (algo, label, color) in enumerate(ALGOS):
    deltas = []
    for key in SETUP_ORDER:
        algos = results.get(key, {})
        fifo = algos.get("fifo", {}).get("mean")
        v = algos.get(algo, {}).get("mean")
        deltas.append((v/fifo - 1)*100 if (fifo and v) else 0)
    offset = (i - len(ALGOS)/2 + 0.5) * W
    bars = ax.bar(x + offset, deltas, W, label=label, color=color, edgecolor="black", linewidth=0.5)
    if algo == "hrrnslo2_oracle1500":
        for j, d_ in enumerate(deltas):
            ax.text(x[j] + offset, d_ - 2, f"{d_:+.0f}%",
                    ha="center", va="top", fontsize=9, fontweight="bold", color="white")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([SETUP_LABELS[k] for k in SETUP_ORDER], fontsize=10)
ax.set_ylabel("Mean JCT change vs FIFO (%)")
ax.set_title("(a) Mean JCT improvement (lower = better)")
ax.legend(fontsize=9, loc="lower left", ncol=2)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(top=12)

# Panel 2: CDF for m1g4/l25 (most dramatic)
ax = axes[1]
key = ("m1g4", 25)
for algo, label, color in ALGOS:
    if algo not in results[key]:
        continue
    jcts = results[key][algo]["jcts"]
    y = np.linspace(0, 1, len(jcts))
    lw = 2.5 if algo == "hrrnslo2_oracle1500" else 1.5
    ax.plot(jcts, y, label=label, color=color, linewidth=lw)
ax.set_xlabel("JCT (seconds)")
ax.set_ylabel("CDF")
ax.set_title("(b) JCT distribution at 4 GPU, load=25 (heavy load)")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)
ax.set_xscale("log")

plt.suptitle("HrrnSlo headline: mixed inference + training workload (CoV ≈ 2.5)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join(FIG_DIR, "hrrnslo_headline.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved: {out}")

# --- Print summary table for the report ---
print("\n=== HrrnSlo headline summary ===")
print(f"{'Setup':<12} {'FIFO':>8} {'HRRN':>8} {'HrrnSlo':>10} {'HRRN%':>7} {'HrrnSlo%':>10}")
for key in SETUP_ORDER:
    a = results[key]
    fifo = a["fifo"]["mean"]
    hrrn = a["hrrn"]["mean"]
    ours = a["hrrnslo2_oracle1500"]["mean"]
    print(f"{key[0]+'/l'+str(key[1]):<12} {fifo:>8.1f} {hrrn:>8.1f} {ours:>10.1f}"
          f" {(hrrn/fifo-1)*100:>+6.1f}% {(ours/fifo-1)*100:>+9.1f}%")
