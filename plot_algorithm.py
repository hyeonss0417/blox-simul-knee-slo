"""
Algorithm visualisation for the Knee-SLO report.

Produces 4 PNGs in docs/figures_v2/:
  1. algo_urgency_curves.png  — knee vs linear vs sigmoid urgency
  2. algo_risk_zones.png      — annotated risk zones with example jobs
  3. algo_score_breakdown.png — how the final score is composed
  4. algo_timeline.png        — how risk evolves over a job's life
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams["font.size"] = 11

OUT = "docs/figures_v2"
os.makedirs(OUT, exist_ok=True)


# ────────────────────────────────────────────────────────────
# 1. urgency function comparison
# ────────────────────────────────────────────────────────────
def urgency_quadratic(risk, theta=0.7, gamma=2, c1=1, c2=6, c3=30):
    out = np.zeros_like(risk, dtype=float)
    for i, r in enumerate(risk):
        if r < theta:
            out[i] = c1 * r
        elif r < 1:
            x = (r - theta) / max(1e-9, 1 - theta)
            out[i] = c1 * theta + c2 * x ** gamma
        else:
            out[i] = c1 * theta + c2 + c3 * (r - 1)
    return out


def urgency_linear(risk, c1=1):
    return c1 * risk


def urgency_sigmoid(risk, theta=0.7, c1=1, c2=6, c3=30):
    base = c1 * risk
    sig = 1.0 / (1.0 + np.exp(-(risk - theta) * 6.0))
    late = np.maximum(0, risk - 1) * c3
    return base + c2 * sig + late


def plot_urgency_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # left: full range, log-y
    risks = np.linspace(0, 2.0, 400)
    axes[0].plot(risks, urgency_quadratic(risks, theta=0.7),
                 label="quadratic knee (θ=0.7, γ=2)", color="#0d3b66", lw=2.5)
    axes[0].plot(risks, urgency_linear(risks), label="linear",
                 color="#e07a5f", lw=2, linestyle="--")
    axes[0].plot(risks, urgency_sigmoid(risks, theta=0.7),
                 label="sigmoid (θ=0.7)", color="#81b29a", lw=2, linestyle=":")
    axes[0].axvspan(0, 0.7, alpha=0.10, color="green", label="safe zone")
    axes[0].axvspan(0.7, 1.0, alpha=0.10, color="orange", label="danger zone")
    axes[0].axvspan(1.0, 2.0, alpha=0.10, color="red", label="late zone")
    axes[0].axvline(0.7, color="black", lw=0.6, alpha=0.4)
    axes[0].axvline(1.0, color="black", lw=0.6, alpha=0.4)
    axes[0].set_xlabel("SLO risk = (now − submit + remaining) / B")
    axes[0].set_ylabel("urgency")
    axes[0].set_title("Knee-SLO urgency function — three variants")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(alpha=0.25)
    axes[0].set_xlim(0, 2.0)
    axes[0].set_ylim(0, 40)

    # right: theta sweep for quadratic
    for theta, col in [(0.3, "#114b5f"), (0.5, "#1c4e80"), (0.7, "#0d3b66"),
                       (0.9, "#34495e")]:
        axes[1].plot(risks, urgency_quadratic(risks, theta=theta),
                     label=f"θ = {theta}", color=col, lw=2)
    axes[1].axvline(1.0, color="black", lw=0.6, alpha=0.4)
    axes[1].set_xlabel("SLO risk")
    axes[1].set_ylabel("urgency")
    axes[1].set_title("knee_quadratic — θ sweep")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[1].set_xlim(0, 2.0)
    axes[1].set_ylim(0, 40)

    plt.tight_layout()
    plt.savefig(f"{OUT}/algo_urgency_curves.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT}/algo_urgency_curves.png")


# ────────────────────────────────────────────────────────────
# 2. risk zones with example jobs
# ────────────────────────────────────────────────────────────
def plot_risk_zones():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    # zones
    ax.axvspan(0, 0.7, alpha=0.15, color="#2ecc71", zorder=0)
    ax.axvspan(0.7, 1.0, alpha=0.20, color="#f39c12", zorder=0)
    ax.axvspan(1.0, 2.2, alpha=0.18, color="#e74c3c", zorder=0)

    # zone labels at top
    ax.text(0.35, 0.94, "SAFE\n(risk < θ)", ha="center", fontsize=13,
            fontweight="bold", color="#1e8449", transform=ax.get_xaxis_transform())
    ax.text(0.85, 0.94, "DANGER\n(θ ≤ risk < 1)", ha="center", fontsize=13,
            fontweight="bold", color="#b9770e", transform=ax.get_xaxis_transform())
    ax.text(1.55, 0.94, "LATE\n(risk ≥ 1)", ha="center", fontsize=13,
            fontweight="bold", color="#c0392b", transform=ax.get_xaxis_transform())

    # urgency curve
    risks = np.linspace(0, 2.2, 400)
    urg = urgency_quadratic(risks, theta=0.7)
    ax.plot(risks, urg, color="#0d3b66", lw=3, label="urgency = knee(risk)")

    # example jobs (small dots)
    examples = [
        (0.20, "Job A — new short job", "#1e8449"),
        (0.55, "Job B — medium progress", "#1e8449"),
        (0.82, "Job C — knee approaching", "#b9770e"),
        (1.40, "Job D — already missed", "#c0392b"),
    ]
    for r, label, col in examples:
        u = urgency_quadratic(np.array([r]), theta=0.7)[0]
        ax.scatter([r], [u], s=140, color=col, edgecolor="white",
                   linewidth=2, zorder=5)
        ax.annotate(label, (r, u),
                    xytext=(10, 12), textcoords="offset points",
                    fontsize=10, fontweight="bold")

    ax.axvline(0.7, color="black", lw=0.8, alpha=0.5)
    ax.axvline(1.0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("SLO risk = (now − submit + predicted_remaining) / SLO_budget")
    ax.set_ylabel("urgency boost (higher → execute sooner)")
    ax.set_title("Knee-SLO — 3 risk zones and what they mean")
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 40)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{OUT}/algo_risk_zones.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT}/algo_risk_zones.png")


# ────────────────────────────────────────────────────────────
# 3. score breakdown for a couple of example jobs
# ────────────────────────────────────────────────────────────
def plot_score_breakdown():
    """
    For four representative jobs (small/large × waiting/new), show the
    contributions of each term to the final score.
    """
    jobs = [
        # (label, norm_rem,  risk, age_norm, color)
        ("Small\nnew",     0.3,  0.15, 0.0,  "#27ae60"),
        ("Small\nwaiting", 0.3,  0.55, 1.2,  "#16a085"),
        ("Large\nnew",     2.5,  0.40, 0.0,  "#7d3c98"),
        ("Large\nold,\nover-risk", 2.5, 1.40, 2.0, "#c0392b"),
    ]
    w_size, w_urg, w_age = 1.0, 1.0, 0.1

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(jobs))
    bar_w = 0.6
    size_terms, urg_terms, age_terms, totals = [], [], [], []
    labels = []
    colors = []
    for lbl, nr, r, age, col in jobs:
        s = w_size * nr
        u = -w_urg * urgency_quadratic(np.array([r]), theta=0.7)[0]
        a = -w_age * age
        size_terms.append(s)
        urg_terms.append(u)
        age_terms.append(a)
        totals.append(s + u + a)
        labels.append(lbl)
        colors.append(col)

    # stacked bars: positive (size) above zero, negatives (urgency, age) below
    ax.bar(x, size_terms, bar_w, label="+ w_size × norm_remaining",
           color="#85c1e9", edgecolor="white")
    ax.bar(x, urg_terms, bar_w, label="− w_urg × urgency",
           color="#e59866", edgecolor="white")
    bottom_for_age = np.array(urg_terms)
    ax.bar(x, age_terms, bar_w, bottom=bottom_for_age,
           label="− w_age × age_bonus", color="#bdc3c7", edgecolor="white")

    # total score marker
    for xi, t, c in zip(x, totals, colors):
        ax.scatter([xi], [t], s=200, color=c, edgecolor="black",
                   linewidth=2, zorder=10)
        ax.annotate(f"score = {t:.2f}", (xi, t),
                    xytext=(0, 14 if t >= 0 else -28),
                    textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold",
                    color=c)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("score component (lower total ⇒ executed first)")
    ax.set_title("Knee-SLO — how each term contributes to the final score")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/algo_score_breakdown.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT}/algo_score_breakdown.png")


# ────────────────────────────────────────────────────────────
# 4. risk evolution timeline for one job
# ────────────────────────────────────────────────────────────
def plot_timeline():
    """
    Show how a job's risk changes over time while it waits and (eventually) runs.
    """
    B = 24 * 3600.0   # SLO budget 24h
    submit = 0
    actual_work = 12 * 3600.0
    # Timeline: 0..30h
    t = np.linspace(0, 30 * 3600.0, 600)
    # phase 1: queueing (0..18h, no progress)
    # phase 2: running (18h..30h) gaining 1 unit per sec
    wait_end = 18 * 3600.0
    attained = np.where(t < wait_end, 0.0, np.minimum(actual_work, t - wait_end))
    remaining = actual_work - attained
    risk = (t - submit + remaining) / B

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhspan(0, 0.7, alpha=0.10, color="#2ecc71")
    ax.axhspan(0.7, 1.0, alpha=0.16, color="#f39c12")
    ax.axhspan(1.0, 1.6, alpha=0.14, color="#e74c3c")
    ax.plot(t / 3600.0, risk, color="#0d3b66", lw=3, label="risk(t)")
    ax.axvline(wait_end / 3600.0, color="gray", linestyle="--", alpha=0.5,
               label="first scheduled")
    ax.axvline((wait_end + actual_work) / 3600.0, color="black",
               linestyle=":", alpha=0.5, label="job done")
    ax.text(2, 0.35, "SAFE", fontsize=12, fontweight="bold", color="#1e8449")
    ax.text(2, 0.85, "DANGER", fontsize=12, fontweight="bold", color="#b9770e")
    ax.text(2, 1.30, "LATE", fontsize=12, fontweight="bold", color="#c0392b")
    ax.set_xlabel("time (hours since submit)")
    ax.set_ylabel("risk")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.6)
    ax.set_title("Risk evolution — single job, SLO budget = 24h, "
                 "queues 18h then runs 12h")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{OUT}/algo_timeline.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT}/algo_timeline.png")


# ────────────────────────────────────────────────────────────
# 5. flow diagram (matplotlib drawn boxes)
# ────────────────────────────────────────────────────────────
def plot_flow():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, color="#0d3b66", tx="white", fs=10, ec="black"):
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.08",
                           facecolor=color, edgecolor=ec, linewidth=1.5)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, color=tx, fontweight="bold")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=18,
                                     lw=1.5, color="#34495e"))

    # row 1: inputs
    box(0.2, 4.5, 2.3, 0.9, "Job arrives:\n(submit, class_id, ...)",
        color="#5dade2")
    box(2.9, 4.5, 2.3, 0.9, "Profiled latency\n_cat_avg[class]",
        color="#5dade2")
    box(5.6, 4.5, 2.3, 0.9, "Wall-clock\nself.current_time",
        color="#5dade2")
    box(8.3, 4.5, 2.3, 0.9, "SLO budget\nB (absolute, per-class)",
        color="#5dade2")

    # row 2: derived
    box(1.0, 3.0, 3.0, 0.9,
        "completion_risk\n= (now−s+rem)/B",
        color="#85c1e9", tx="#1a1f2e")
    box(4.5, 3.0, 3.0, 0.9,
        "wait_risk\n= (now−s)/(B−w)",
        color="#85c1e9", tx="#1a1f2e")
    box(8.0, 3.0, 3.0, 0.9,
        "norm_remaining\n= rem / global_avg",
        color="#85c1e9", tx="#1a1f2e")

    arrow(2.0, 4.5, 2.5, 3.9)   # job → completion_risk
    arrow(4.0, 4.5, 4.5, 3.9)   # cat_avg → completion_risk
    arrow(6.7, 4.5, 6.0, 3.9)   # wall clock → wait_risk
    arrow(9.5, 4.5, 9.5, 3.9)   # SLO → norm_rem / wait_risk

    # row 3: knee + score
    box(3.0, 1.7, 3.5, 0.9,
        "urgency = knee(max(c_risk, w_risk),\n θ=0.7, γ=2, c1, c2, c3)",
        color="#f1948a")
    box(7.0, 1.7, 3.5, 0.9,
        "score = w_size·norm_rem\n  − w_urg·urgency − w_age·age",
        color="#f7dc6f", tx="#1a1f2e")

    arrow(3.5, 3.0, 4.7, 2.6)
    arrow(6.0, 3.0, 5.0, 2.6)
    arrow(9.5, 3.0, 8.7, 2.6)
    arrow(6.5, 2.1, 7.0, 2.1)

    # row 4: final
    box(4.0, 0.2, 5.5, 0.9,
        "sort job_dict by (priority, score) ASC\nplace top-k on free GPUs",
        color="#0d3b66")
    arrow(8.8, 1.7, 8.0, 1.1)
    arrow(7.0, 1.7, 6.5, 1.1)

    ax.set_title("Knee-SLO scheduling step (one simulator round)",
                 fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/algo_flow.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT}/algo_flow.png")


if __name__ == "__main__":
    plot_urgency_curves()
    plot_risk_zones()
    plot_score_breakdown()
    plot_timeline()
    plot_flow()
    print("\nAll algorithm visualizations done.")
