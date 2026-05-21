"""
Plot real-inference experiment results.

Generates:
  - inf_jct_cdf.png  — JCT CDF across schedulers
  - inf_summary.png  — bar chart of avg/p99/max JCT
  - inf_distribution.png — JCT histogram + density
  - inf_slo_curve.png — miss rate vs SLO target
"""
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]

OUT = "docs/figures_v2"
os.makedirs(OUT, exist_ok=True)
LOAD = 8000.0

# Curated representative configs.
CONFIGS = [
    ("inf_fifo",         "FIFO",                  "#6b7280"),
    ("inf_las",          "LAS",                   "#e67e22"),
    ("inf_srtf",         "SRTF (oracle)",         "#3498db"),
    ("inf_sjf",          "SJF-total",             "#9b59b6"),
    ("inf_hrrn",         "HRRN",                  "#8e44ad"),
    ("inf_edf_60s",      "EDF (SLO=60s)",         "#16a085"),
    ("inf_llf_60s",      "LLF (SLO=60s)",         "#c0392b"),
    ("inf_knee_t7_60s",  "Knee θ=0.7 SLO=60s",    "#0d3b66"),
    ("inf_knee_t5_60s",  "Knee θ=0.5 SLO=60s",    "#1c4e80"),
    ("inf_knee_t7_60s_lin",  "Knee linear SLO=60s",  "#a04000"),
    ("inf_knee_t7_5min", "Knee θ=0.7 SLO=5min",   "#34495e"),
    ("inf_knee_np_60s",  "Knee NonPreempt SLO=60s","#7e5109"),
]


def load_jcts(prefix):
    pat = f"{prefix}_*_*_accept_all_load_{LOAD}_job_stats.json"
    m = glob.glob(pat)
    if not m:
        return None
    with open(m[0]) as f:
        d = json.load(f)
    return [v[1] - v[0] for v in d.values() if isinstance(v, list) and len(v) >= 2]


def main():
    data = {}
    for prefix, label, color in CONFIGS:
        jcts = load_jcts(prefix)
        if jcts:
            data[prefix] = (label, color, jcts)

    if not data:
        print("no inference data found")
        return

    # ── Summary table ──
    print(f"\n{'Config':30s} {'N':>4s} {'Avg':>6s} {'P50':>6s} {'P95':>6s} {'P99':>6s} {'Max':>6s} "
          f"{'m30':>6s} {'m60':>6s} {'m120':>6s}")
    print("-" * 100)
    rows = []
    for prefix, (label, color, jcts) in data.items():
        miss_30 = sum(1 for v in jcts if v > 30) / len(jcts) * 100
        miss_60 = sum(1 for v in jcts if v > 60) / len(jcts) * 100
        miss_120 = sum(1 for v in jcts if v > 120) / len(jcts) * 100
        rows.append((prefix, label, color, jcts, miss_30, miss_60, miss_120))
        print(f"{label:30s} {len(jcts):4d} {np.mean(jcts):6.1f} "
              f"{np.median(jcts):6.1f} {np.percentile(jcts,95):6.1f} "
              f"{np.percentile(jcts,99):6.1f} {max(jcts):6.0f} "
              f"{miss_30:5.1f}% {miss_60:5.1f}% {miss_120:5.1f}%")

    # ── 1. JCT CDF ──
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for prefix, (label, color, jcts) in data.items():
        srt = sorted(jcts)
        cdf = [(i + 1) / len(srt) for i in range(len(srt))]
        ax.plot(srt, cdf, label=label, color=color, lw=2)
    ax.set_xlabel("JCT (seconds)")
    ax.set_ylabel("CDF")
    ax.set_title(f"JCT CDF — real inference (32 GPU, load={int(LOAD)} jobs/hr)")
    ax.axvline(30, color="gray", linestyle=":", alpha=0.5, label="SLO=30s")
    ax.axvline(60, color="gray", linestyle="--", alpha=0.5, label="SLO=60s")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/inf_jct_cdf.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT}/inf_jct_cdf.png")

    # ── 2. JCT summary bar chart (avg / p99 / max) ──
    fig, ax = plt.subplots(figsize=(11, 5))
    labels_plot = [r[1] for r in rows]
    colors_plot = [r[2] for r in rows]
    avgs = [np.mean(r[3]) for r in rows]
    p99s = [np.percentile(r[3], 99) for r in rows]
    maxs = [max(r[3]) for r in rows]
    x = np.arange(len(rows))
    w = 0.25
    ax.bar(x - w, avgs, w, label="Avg", color="#3498db")
    ax.bar(x, p99s, w, label="P99", color="#e67e22")
    ax.bar(x + w, maxs, w, label="Max", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("JCT (s)")
    ax.set_title(f"JCT Avg / P99 / Max — real inference (32 GPU, load={int(LOAD)})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/inf_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT}/inf_summary.png")

    # ── 3. Multi-target SLO miss curve ──
    fig, ax = plt.subplots(figsize=(9, 5.5))
    targets = np.arange(15, 200, 5)
    for prefix, (label, color, jcts) in data.items():
        miss = [sum(1 for v in jcts if v > t) / len(jcts) * 100 for t in targets]
        ax.plot(targets, miss, label=label, color=color, lw=2)
    ax.set_xlabel("SLO target (s)")
    ax.set_ylabel("SLO miss rate (%)")
    ax.set_title(f"SLO miss vs target — real inference (32 GPU, load={int(LOAD)})")
    ax.axvspan(25, 35, alpha=0.10, color="green", label="reasonable SLO 30s")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/inf_slo_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT}/inf_slo_curve.png")

    # ── 4. JCT distribution (histogram) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 160, 40)
    for prefix, (label, color, jcts) in list(data.items())[:6]:
        ax.hist(jcts, bins=bins, label=label, color=color, alpha=0.4,
                histtype="stepfilled", edgecolor=color, linewidth=1.5)
    ax.set_xlabel("JCT (s)")
    ax.set_ylabel("job count")
    ax.set_title("JCT distribution — top 6 schedulers")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/inf_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT}/inf_distribution.png")


if __name__ == "__main__":
    main()
