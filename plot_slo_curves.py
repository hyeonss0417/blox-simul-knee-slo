"""
Multi-target SLO miss-rate curve for every algorithm.

For each algorithm at default load, plot SLO miss% vs SLO target (hours).
Shows that SLO=6h kills everyone (saturation at 100%) while SLO=36h+
makes everyone look good.  Use this to defend the choice of "operating
point" in the report.
"""
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]

OUT = "docs/report_v2/figures"
os.makedirs(OUT, exist_ok=True)

# Curated list of representative prefixes — we don't want 30 lines on
# one plot.
PREFIXES = [
    ("v2_fifo",     "FIFO",                  "#6b7280"),
    ("v2_las",      "LAS",                   "#e67e22"),
    ("v2_srtf",     "SRTF (oracle)",         "#3498db"),
    ("v2_sjf",      "SJF-total (pred.)",     "#9b59b6"),
    ("v2_edf",      "EDF",                   "#16a085"),
    ("v2_llf",      "LLF",                   "#c0392b"),
    ("v2_hrrn",     "HRRN",                  "#8e44ad"),
    ("v2_slo_v1",   "v1 SloScoring",         "#f39c12"),
    ("v2_knee_t7_q","Knee θ=0.7 (SLO=6h)",   "#0d1b2a"),
    ("v2_knee24_t7","Knee θ=0.7 (SLO=24h)",  "#a04000"),
    ("v2_knee_np_t7","Knee NonPreempt",      "#34495e"),
]


def load_jcts(prefix):
    pat = f"results/v2_training/{prefix}_3000_3100_*_accept_all_load_8.0_job_stats.json"
    m = glob.glob(pat)
    if not m:
        return None
    with open(m[0]) as f:
        d = json.load(f)
    return [v[1] - v[0] for v in d.values() if isinstance(v, list) and len(v) >= 2]


def main():
    targets_h = np.linspace(2, 72, 36)
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for prefix, label, color in PREFIXES:
        jcts = load_jcts(prefix)
        if not jcts:
            continue
        miss = []
        for h in targets_h:
            miss.append(sum(1 for v in jcts if v > h * 3600) / len(jcts) * 100)
        ax.plot(targets_h, miss, label=label, color=color, linewidth=2)
        plotted += 1

    if plotted == 0:
        print("no data")
        return

    ax.set_xlabel("SLO target (hours)")
    ax.set_ylabel("SLO miss rate (%)")
    ax.set_title("SLO miss rate vs target — load=8 jobs/hr, 128 GPUs")
    ax.axvspan(20, 28, alpha=0.12, color="green",
               label="recommended operating range (~24h)")
    ax.set_ylim(0, 105)
    ax.set_xticks([6, 12, 18, 24, 36, 48, 72])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(f"{OUT}/slo_target_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT}/slo_target_curve.png  ({plotted} algorithms)")


if __name__ == "__main__":
    main()
