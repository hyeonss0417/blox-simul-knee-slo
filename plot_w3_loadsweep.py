"""
Wave 3 plotter: load sensitivity for FIFO / LAS / SRTF / KneeSlo across
different jobs-per-hour values.  Each load uses its own prefix `w3_l${L}_`
plus the scheduler name (lowercased).
"""
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]

OUT = "docs/figures_v2"
os.makedirs(OUT, exist_ok=True)
LOADS = [4, 8, 12, 16]
SCHEDS = ["Fifo", "Las", "Srtf", "KneeSlo"]
COLORS = {"Fifo": "#6b7280", "Las": "#e67e22", "Srtf": "#3498db", "KneeSlo": "#0d1b2a"}
SLO_TARGET = 21600.0


def load(load_jph, sched):
    # Load=8 uses v2 prefixes (already done in wave 1)
    if load_jph == 8:
        prefix = {
            "Fifo": "v2_fifo", "Las": "v2_las",
            "Srtf": "v2_srtf", "KneeSlo": "v2_knee_t7_q",
        }[sched]
    else:
        prefix = f"w3_l{load_jph}_{sched.lower()}"
    pat = f"{prefix}_3000_3100_{sched}_accept_all_load_{load_jph}.0_job_stats.json"
    matches = glob.glob(pat)
    if not matches:
        return None
    with open(matches[0]) as f:
        d = json.load(f)
    return [v[1] - v[0] for v in d.values() if isinstance(v, list) and len(v) >= 2]


def main():
    series = {s: {"avg": [], "p99": [], "miss": [], "loads": []} for s in SCHEDS}
    for L in LOADS:
        for S in SCHEDS:
            jcts = load(L, S)
            if jcts is None:
                continue
            series[S]["avg"].append(np.mean(jcts) / 3600)
            series[S]["p99"].append(np.percentile(jcts, 99) / 3600)
            series[S]["miss"].append(sum(1 for j in jcts if j > SLO_TARGET) / len(jcts) * 100)
            series[S]["loads"].append(L)

    for metric, ylabel, fname in [
        ("avg", "Avg JCT (h)", "loadsweep_avg.png"),
        ("p99", "P99 JCT (h)", "loadsweep_p99.png"),
        ("miss", "SLO miss rate (%)", "loadsweep_miss.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for S in SCHEDS:
            y = series[S][metric]
            x = series[S]["loads"]
            if y:
                ax.plot(x, y, "o-", label=S, color=COLORS[S], linewidth=2.2,
                        markersize=8)
        ax.set_xlabel("Load (jobs / hour)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Load sensitivity — {ylabel}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{OUT}/{fname}", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"saved {fname}")

    # JSON summary too
    with open(f"{OUT}/loadsweep_summary.json", "w") as f:
        json.dump(series, f, indent=2)


if __name__ == "__main__":
    main()
