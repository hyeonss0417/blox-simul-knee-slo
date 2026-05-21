"""
v2 evaluation: reads all v2_*_job_stats.json + matching responsiveness +
run_time_stats, computes the extended metric suite (SLO miss rate vs
absolute target, tardiness, normalized lateness), and generates plots.

Run as:
    python plot_v2_results.py
"""
import json
import os
import glob
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams["font.size"] = 11

OUT = "docs/figures_v2"
os.makedirs(OUT, exist_ok=True)

START, STOP = 3000, 3100
LOAD = 8.0
SLO_TARGET = 21600.0  # 6h, matches Knee-SLO default

# Display labels for known configs.
KNOWN = {
    "v2_fifo":      ("FIFO",                 "#6b7280"),
    "v2_las":       ("LAS",                  "#e67e22"),
    "v2_srtf":      ("SRTF",                 "#3498db"),
    "v2_slo_v1":    ("v1 SloScoring",        "#f39c12"),
    "v2_sjf":       ("SJF-total",            "#9b59b6"),
    "v2_edf":       ("EDF",                  "#16a085"),
    "v2_llf":       ("LLF",                  "#c0392b"),
    "v2_knee_t3_q":  ("Knee θ=0.3 q",        "#1f3a5f"),
    "v2_knee_t5_q":  ("Knee θ=0.5 q",        "#114b5f"),
    "v2_knee_t7_q":  ("Knee θ=0.7 q (default)", "#0d1b2a"),
    "v2_knee_t7_q3": ("Knee θ=0.7 γ=3",      "#1b263b"),
    "v2_knee_t5_q3": ("Knee θ=0.5 γ=3",      "#1c4e80"),
    "v2_knee_t7_lin":("Knee θ=0.7 linear",   "#7d3c98"),
    "v2_knee_t7_sig":("Knee θ=0.7 sigmoid",  "#c0392b"),
    "v2_knee_t5_sig":("Knee θ=0.5 sigmoid",  "#d35400"),
    # Wave 2 — algorithmic extensions
    "v2_hrrn":          ("HRRN",                  "#8e44ad"),
    "v2_knee_np_t7":    ("Knee NonPreempt θ=0.7", "#34495e"),
    "v2_knee_np_t5":    ("Knee NonPreempt θ=0.5", "#5d6d7e"),
    "v2_knee_cls_t7":   ("Knee class-aware θ=0.7","#2c3e50"),
    "v2_knee_cls_t5":   ("Knee class-aware θ=0.5","#34495e"),
    "v2_knee_age_hi":   ("Knee w_age=0.5",        "#138d75"),
    "v2_knee_age_lo":   ("Knee w_age=0.0",        "#48c9b0"),
    "v2_knee_urg_hi":   ("Knee urg-heavy",        "#a93226"),
    "v2_knee_size_hi":  ("Knee size-heavy",       "#1f618d"),
    "v2_knee_b3h":      ("Knee budget=3h",        "#7e5109"),
    "v2_knee_b12h":     ("Knee budget=12h",       "#b9770e"),
    "v2_knee_c3_lo":    ("Knee c3=5",             "#117a65"),
    "v2_knee_c3_hi":    ("Knee c3=100",           "#239b56"),
    "v2_knee_adapt":    ("Knee adaptive θ",       "#a04000"),
    # Wave 4 — extreme + combined variants
    "v2_knee_t9_q":     ("Knee θ=0.9 γ=2 quad",   "#566573"),
    "v2_knee_t1_q":     ("Knee θ=0.1 γ=2 quad",   "#5dade2"),
    "v2_knee_g4":       ("Knee θ=0.7 γ=4",        "#28b463"),
    "v2_knee_only_urg": ("Knee urg-only",          "#c0392b"),
    "v2_knee_only_size":("Knee size-only",         "#1abc9c"),
    "v2_knee_np_age":   ("NonPreempt + w_age=0.5", "#7e5109"),
    "v2_knee_cls_t3":   ("Knee class θ=0.3",      "#bdc3c7"),
    # Wave 5 — SLO recalibrated to 24h
    "v2_knee24_t3":     ("Knee[24h] θ=0.3",       "#85929e"),
    "v2_knee24_t5":     ("Knee[24h] θ=0.5",       "#5d6d7e"),
    "v2_knee24_t7":     ("Knee[24h] θ=0.7",       "#34495e"),
    "v2_knee24_t9":     ("Knee[24h] θ=0.9",       "#2c3e50"),
    "v2_knee24_t7_sig": ("Knee[24h] θ=0.7 sig",   "#922b21"),
    "v2_knee24_t7_lin": ("Knee[24h] θ=0.7 lin",   "#943126"),
    "v2_knee24_np":     ("Knee[24h] NonPreempt",  "#7d6608"),
    "v2_knee24_age":    ("Knee[24h] w_age=0.5",   "#9c640c"),
    "v2_knee24_cdur_m2":("Knee class-dur ×2",    "#0e6655"),
    "v2_knee24_cdur_m3":("Knee class-dur ×3",    "#117a65"),
    "v2_knee24_cdur_m5":("Knee class-dur ×5",    "#148f77"),
}


def _find(prefix, kind):
    pattern = f"{prefix}_{START}_{STOP}_*_accept_all_load_{LOAD}_{kind}.json"
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def load_all():
    jct_data, resp_data, runtime_data = {}, {}, {}
    found = []
    for prefix in KNOWN:
        jf = _find(prefix, "job_stats")
        if not jf:
            continue
        with open(jf) as f:
            d = json.load(f)
        jct_data[prefix] = {k: v[1] - v[0] for k, v in d.items() if isinstance(v, list) and len(v) >= 2}
        rf = _find(prefix, "responsivness")
        if rf and os.path.exists(rf):
            with open(rf) as f:
                d = json.load(f)
            resp_data[prefix] = {k: v[1] - v[0] for k, v in d.items() if isinstance(v, list) and len(v) >= 2}
        tf = _find(prefix, "run_time_stats")
        if tf and os.path.exists(tf):
            with open(tf) as f:
                runtime_data[prefix] = json.load(f)
        found.append(prefix)
    return jct_data, resp_data, runtime_data, found


def slo_miss_rate(jct_dict, target=SLO_TARGET):
    """SLO miss rate vs absolute target."""
    vals = list(jct_dict.values())
    if not vals:
        return None
    miss = sum(1 for v in vals if v > target)
    return miss / len(vals) * 100


def tardiness_stats(jct_dict, target=SLO_TARGET):
    vals = [max(0.0, v - target) for v in jct_dict.values()]
    if not vals:
        return None
    return {
        "mean": float(np.mean(vals)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
    }


def normalized_lateness(jct_dict, target=SLO_TARGET):
    return [(v - target) / target for v in jct_dict.values()]


def main():
    jct_data, resp_data, runtime_data, found = load_all()
    if not found:
        print("No v2_* result files found. Did the grid finish?")
        return

    rows = []
    fifo_avg = None
    for p in found:
        vals = list(jct_data[p].values())
        avg = float(np.mean(vals))
        if p == "v2_fifo":
            fifo_avg = avg
        rows.append({
            "prefix": p,
            "label": KNOWN[p][0],
            "color": KNOWN[p][1],
            "avg_jct_h": avg / 3600,
            "median_h": float(np.median(vals)) / 3600,
            "p95_h": float(np.percentile(vals, 95)) / 3600,
            "p99_h": float(np.percentile(vals, 99)) / 3600,
            "n": len(vals),
            "slo_miss_pct": slo_miss_rate(jct_data[p]),
            "tardiness": tardiness_stats(jct_data[p]),
            "resp_s": (float(np.mean(list(resp_data[p].values())))
                       if p in resp_data else None),
        })

    # Pareto candidates: minimise (avg_jct, slo_miss).
    # Print a console summary table first.
    print(f"\n{'config':25s}  {'AvgJCT(h)':>10s}  {'P95(h)':>7s}  {'P99(h)':>7s}  "
          f"{'SLOmiss%':>8s}  {'Tard_mean(h)':>12s}  {'Resp(s)':>9s}")
    print("-" * 90)
    for r in sorted(rows, key=lambda x: x["avg_jct_h"]):
        tard_mean = r["tardiness"]["mean"] / 3600 if r["tardiness"] else float("nan")
        resp_str = f"{r['resp_s']:.0f}" if r["resp_s"] is not None else "N/A"
        print(f"{r['label']:25s}  {r['avg_jct_h']:>10.2f}  {r['p95_h']:>7.2f}  "
              f"{r['p99_h']:>7.2f}  {r['slo_miss_pct']:>7.1f}%  "
              f"{tard_mean:>12.2f}  {resp_str:>9s}")

    # Save raw table for the report
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # --- Plot 1: Avg JCT bar ---
    fig, ax = plt.subplots(figsize=(12, 5))
    rs = sorted(rows, key=lambda x: x["avg_jct_h"])
    ax.bar([r["label"] for r in rs], [r["avg_jct_h"] for r in rs],
           color=[r["color"] for r in rs])
    ax.set_ylabel("Average JCT (hours)")
    ax.set_title(f"Avg JCT — v2 grid (Alibaba GenAI, Load={LOAD})")
    plt.xticks(rotation=35, ha="right")
    for i, r in enumerate(rs):
        ax.text(i, r["avg_jct_h"] + 0.4, f"{r['avg_jct_h']:.1f}",
                ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/avg_jct_v2.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Plot 2: SLO miss rate (absolute target = 6h) ---
    fig, ax = plt.subplots(figsize=(12, 5))
    rs = sorted(rows, key=lambda x: (x["slo_miss_pct"], x["avg_jct_h"]))
    ax.bar([r["label"] for r in rs], [r["slo_miss_pct"] for r in rs],
           color=[r["color"] for r in rs])
    ax.set_ylabel("SLO miss rate (%)")
    ax.set_title(f"SLO miss rate (target = {int(SLO_TARGET/3600)}h absolute)")
    plt.xticks(rotation=35, ha="right")
    for i, r in enumerate(rs):
        ax.text(i, r["slo_miss_pct"] + 0.6, f"{r['slo_miss_pct']:.1f}%",
                ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/slo_miss_v2.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Plot 3: Pareto scatter (JCT vs SLO miss) ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in rows:
        ax.scatter(r["avg_jct_h"], r["slo_miss_pct"], s=110,
                   color=r["color"], edgecolor="white", linewidth=1.5)
        ax.annotate(r["label"], (r["avg_jct_h"], r["slo_miss_pct"]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=9)
    ax.set_xlabel("Average JCT (hours) ← lower is better")
    ax.set_ylabel("SLO miss rate (%) ← lower is better")
    ax.set_title("Pareto: Avg JCT vs SLO miss rate")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/pareto_jct_vs_slo.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Plot 4: Tardiness P95 ---
    fig, ax = plt.subplots(figsize=(12, 5))
    rs = sorted(rows, key=lambda x: (x["tardiness"] or {"p95": 0})["p95"])
    p95s = [(r["tardiness"]["p95"] if r["tardiness"] else 0) / 3600 for r in rs]
    ax.bar([r["label"] for r in rs], p95s, color=[r["color"] for r in rs])
    ax.set_ylabel("P95 Tardiness (hours)")
    ax.set_title(f"P95 Tardiness (jobs missing {int(SLO_TARGET/3600)}h SLO)")
    plt.xticks(rotation=35, ha="right")
    for i, val in enumerate(p95s):
        ax.text(i, val + 0.4, f"{val:.1f}h", ha="center", fontsize=9,
                fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/p95_tardiness_v2.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Plot 5: Responsiveness (log scale, only if available) ---
    rs = [r for r in rows if r["resp_s"] is not None]
    if rs:
        fig, ax = plt.subplots(figsize=(12, 5))
        rs = sorted(rs, key=lambda x: x["resp_s"])
        ax.bar([r["label"] for r in rs], [r["resp_s"] for r in rs],
               color=[r["color"] for r in rs])
        ax.set_ylabel("Responsiveness (s, log)")
        ax.set_yscale("log")
        ax.set_title("Time to first execution (lower is better)")
        plt.xticks(rotation=35, ha="right")
        for i, r in enumerate(rs):
            ax.text(i, r["resp_s"] * 1.15, f"{r['resp_s']:.0f}s",
                    ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{OUT}/responsiveness_v2.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"\nFigures saved to {OUT}/")


if __name__ == "__main__":
    main()
