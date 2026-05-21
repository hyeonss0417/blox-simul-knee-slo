"""
Algorithm gallery — each algorithm visualized side-by-side.

Produces 3 figures:
  1. algo_feature_matrix.png  — which signals each algorithm uses
  2. algo_ordering_demo.png   — same queue, different orderings
  3. algo_pseudocode_grid.png — text card per algorithm
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
# Use Apple SD Gothic Neo for Korean glyphs on macOS, fall back gracefully
plt.rcParams["font.family"] = [
    "Apple SD Gothic Neo", "AppleGothic", "Nanum Gothic",
    "Helvetica", "Arial", "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

OUT = "docs/report_v2/figures"
os.makedirs(OUT, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Common example queue
# ─────────────────────────────────────────────────────────────────
# 5 jobs at "now=t". Each has: (submit, attained, remaining_oracle,
#                               remaining_meta, predicted_total)
JOBS = [
    # (name, color,   submit, attained, rem_oracle, rem_meta, pred_total)
    ("J1 short-new",       "#27ae60",  90,  0,   10,  12,  12),   # tiny job, just arrived
    ("J2 short-old",       "#16a085",  20,  0,   10,  12,  12),   # tiny but waiting
    ("J3 medium-progress", "#3498db",  50, 30,   20,  18,  50),   # mid-size, half done
    ("J4 long-new",        "#9b59b6",  85,  0,  100, 110, 110),   # huge, just arrived
    ("J5 long-overdue",    "#c0392b",  10,  5,   80,  85,  90),   # huge + long wait → OVERDUE
]
NOW = 100
SLO = 60         # seconds
THETA = 0.7

def wait(j): return NOW - j[2]
def remaining_oracle(j): return j[4]
def remaining_meta(j): return j[5]
def attained(j): return j[3]
def submit(j): return j[2]


# Algorithm sort-key functions — return a tuple (sort_key, optional_meta_dict)
# Lower sort_key = scheduled first.
def algo_fifo(j):
    return (submit(j),)

def algo_las(j):
    return (attained(j),)

def algo_srtf(j):
    return (remaining_oracle(j),)

def algo_sjftotal(j):
    # category-mean — pretend it's just the predicted total (rough)
    return (j[6],)

def algo_metasrtf(j):
    return (remaining_meta(j),)

def algo_edf(j):
    deadline = submit(j) + SLO
    return (deadline,)

def algo_llf(j):
    slack = (submit(j) + SLO) - NOW - remaining_oracle(j)
    return (slack,)

def algo_hrrn(j):
    w = wait(j); s = max(1, j[6])
    R = (w + s) / s
    return (-R,)  # highest R first

def algo_lasslo(j):
    w = wait(j)
    if w >= SLO:      return (0, -w)          # critical
    if w >= THETA*SLO:return (1, attained(j))  # warning
    return (2, attained(j))                    # safe

def algo_srtfslo(j):
    w = wait(j)
    if w >= SLO:      return (0, -w)
    if w >= THETA*SLO:return (1, remaining_oracle(j))
    return (2, remaining_oracle(j))

def algo_metasrtfslo(j):
    w = wait(j)
    if w >= SLO:      return (0, -w)
    if w >= THETA*SLO:return (1, remaining_meta(j))
    return (2, remaining_meta(j))


ALGORITHMS = [
    ("FIFO",          algo_fifo,        "사용: submit_time",                     "submit ↑"),
    ("LAS",           algo_las,         "사용: attained_service",                 "attained ↑"),
    ("SRTF (oracle)", algo_srtf,        "사용: 진짜 remaining (oracle)",          "remaining ↑"),
    ("SjfTotal",      algo_sjftotal,    "사용: category-mean total",              "pred_total ↑"),
    ("MetaSrtf",      algo_metasrtf,    "사용: metadata-predicted remaining",     "meta-rem ↑"),
    ("EDF",           algo_edf,         "사용: submit + SLO (절대 deadline)",     "deadline ↑"),
    ("LLF",           algo_llf,         "사용: deadline - now - remaining",       "slack ↑"),
    ("HRRN",          algo_hrrn,        "사용: (wait + service) / service",       "R ↓"),
    ("LasSlo",        algo_lasslo,      "bucket + LAS",                           "bucket→attained"),
    ("SrtfSlo",       algo_srtfslo,     "bucket + SRTF",                          "bucket→remaining"),
    ("MetaSrtfSlo",   algo_metasrtfslo, "bucket + MetaSrtf",                      "bucket→meta-rem"),
]


# ─────────────────────────────────────────────────────────────────
# Figure 1: Ordering demo — same queue, different orderings
# ─────────────────────────────────────────────────────────────────
def fig_ordering_demo():
    n_algos = len(ALGORITHMS)
    n_jobs = len(JOBS)

    fig, ax = plt.subplots(figsize=(13, 7.5))

    # For each algorithm compute the order
    orderings = {}
    for name, fn, _, _ in ALGORITHMS:
        keys = [(fn(j), idx) for idx, j in enumerate(JOBS)]
        keys.sort()
        # rank: original_idx → position
        rank = {orig: pos for pos, (_, orig) in enumerate(keys)}
        orderings[name] = rank

    # Draw a grid: rows = algorithms, cols = positions 1..N
    for row, (name, _, _, _) in enumerate(ALGORITHMS):
        for orig_idx, j in enumerate(JOBS):
            pos = orderings[name][orig_idx]
            color = j[1]
            rect = mpatches.FancyBboxPatch(
                (pos, n_algos - row - 1), 0.95, 0.85,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="white", linewidth=1.5,
                alpha=0.85,
            )
            ax.add_patch(rect)
            ax.text(pos + 0.475, n_algos - row - 1 + 0.42,
                    j[0].split()[0],   # "J1", "J2", etc.
                    ha="center", va="center",
                    color="white", fontweight="bold", fontsize=11)

    # Y-axis labels (algorithm names)
    ax.set_yticks([n_algos - row - 1 + 0.42 for row in range(n_algos)])
    ax.set_yticklabels([n for n, _, _, _ in ALGORITHMS], fontsize=10)
    ax.set_xticks([i + 0.475 for i in range(n_jobs)])
    ax.set_xticklabels([f"#{i+1}\n(먼저 실행)" if i == 0 else f"#{i+1}"
                        for i in range(n_jobs)])

    ax.set_xlim(-0.1, n_jobs + 0.1)
    ax.set_ylim(-0.1, n_algos + 0.1)
    ax.set_title("같은 큐, 다른 알고리즘 → 다른 순서\n"
                 "(J1 short-new, J2 short-old, J3 mid-progress, "
                 "J4 long-new, J5 long-overdue · SLO=60s · now=100)",
                 fontsize=12, pad=10)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    # Job legend (top)
    job_legend = [
        mpatches.Patch(color=j[1], label=f"{j[0]}  "
                       f"(submit={submit(j)}, wait={wait(j)}, "
                       f"attained={attained(j)}, rem={remaining_oracle(j)})")
        for j in JOBS
    ]
    ax.legend(handles=job_legend, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=9,
              frameon=False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/algo_ordering_demo.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT}/algo_ordering_demo.png")


# ─────────────────────────────────────────────────────────────────
# Figure 2: Feature usage matrix
# ─────────────────────────────────────────────────────────────────
def fig_feature_matrix():
    features = [
        "submit_time", "attained_service",
        "predicted_remaining\n(category-mean)",
        "predicted_remaining\n(metadata)",
        "remaining_oracle",
        "SLO_target",
        "wait_time",
        "deadline\n(submit + SLO)",
    ]
    # rows = algorithms, cols = features. 1 = uses, 0 = doesn't
    usage = {
        # FIFO
        "FIFO":         [1,0,0,0,0,0,0,0],
        # LAS
        "LAS":          [0,1,0,0,0,0,0,0],
        # SRTF (oracle)
        "SRTF (oracle)":[0,0,0,0,1,0,0,0],
        # SjfTotal
        "SjfTotal":     [0,0,1,0,0,0,0,0],
        # MetaSrtf
        "MetaSrtf":     [0,0,0,1,0,0,0,0],
        # EDF
        "EDF":          [1,0,0,0,0,1,0,1],
        # LLF
        "LLF":          [1,0,0,0,1,1,1,1],
        # HRRN
        "HRRN":         [1,0,1,0,0,0,1,0],
        # LasSlo
        "LasSlo":       [1,1,0,0,0,1,1,0],
        # SrtfSlo
        "SrtfSlo":      [1,0,0,0,1,1,1,0],
        # MetaSrtfSlo
        "MetaSrtfSlo":  [1,0,0,1,0,1,1,0],
    }
    rows = list(usage.keys())
    matrix = np.array([usage[r] for r in rows])

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1.5)

    # Annotate
    for i, r in enumerate(rows):
        for j, _ in enumerate(features):
            if matrix[i, j] == 1:
                ax.text(j, i, "✓", ha="center", va="center",
                        color="white", fontweight="bold", fontsize=14)

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=10)
    ax.set_title("어떤 신호를 사용하는가? — 알고리즘 × Feature matrix",
                 fontsize=12, pad=10)

    # Add scheduler-family color stripes on left
    families = {
        "FIFO": "#6b7280", "LAS": "#e67e22",
        "SRTF (oracle)": "#3498db", "SjfTotal": "#9b59b6", "MetaSrtf": "#27ae60",
        "EDF": "#16a085", "LLF": "#c0392b",
        "HRRN": "#8e44ad",
        "LasSlo": "#34495e", "SrtfSlo": "#16a085", "MetaSrtfSlo": "#0d3b66",
    }
    for i, r in enumerate(rows):
        ax.add_patch(mpatches.Rectangle((-0.7, i - 0.4), 0.3, 0.8,
                                         color=families[r], clip_on=False))

    ax.set_xlim(-0.5, len(features) - 0.5)
    plt.tight_layout()
    plt.savefig(f"{OUT}/algo_feature_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT}/algo_feature_matrix.png")


# ─────────────────────────────────────────────────────────────────
# Figure 3: One-liner pseudocode cards
# ─────────────────────────────────────────────────────────────────
def fig_pseudocode_cards():
    cards = [
        ("FIFO",         "#6b7280",
         "sort by submit_time ↑",
         "장점: 단순, fair  단점: short-job 우선순위 없음"),
        ("LAS",          "#e67e22",
         "sort by attained_service ↑",
         "장점: short-job 우선  단점: open-system에서 thrash"),
        ("SRTF",         "#3498db",
         "sort by remaining (oracle) ↑",
         "장점: theoretic best (closed)  단점: oracle 필요 + thrash"),
        ("SjfTotal",     "#9b59b6",
         "sort by category-mean total ↑",
         "장점: non-Oracle  단점: 예측 부정확 (R²=-0.06)"),
        ("MetaSrtf",     "#27ae60",
         "sort by metadata-predicted remaining ↑",
         "★ 장점: non-Oracle + 정확 (R²=0.39)  단점: thrash 가능"),
        ("EDF",          "#16a085",
         "sort by (submit + SLO) ↑",
         "장점: deadline-aware  단점: 잡 크기 무시"),
        ("LLF",          "#c0392b",
         "sort by (deadline - now - remaining) ↑",
         "장점: laxity-aware  단점: 단일 신호, oscillation"),
        ("HRRN",         "#8e44ad",
         "sort by (wait+service)/service ↓",
         "장점: aging 내장  단점: ≈ FIFO when uniform"),
        ("LasSlo",       "#34495e",
         "(bucket, attained) — overdue jobs absolute priority",
         "★ 장점: LAS + SLO 보호, 안정적"),
        ("SrtfSlo",      "#16a085",
         "(bucket, remaining_oracle) — overdue jobs absolute priority",
         "★ 장점: SRTF 효율 + tail 보호"),
        ("MetaSrtfSlo",  "#0d3b66",
         "(bucket, metadata-predicted remaining)",
         "★★ 본 연구 winning combo: non-Oracle + safe"),
    ]
    n = len(cards)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 2.6))
    axes = axes.flatten()
    for i, (name, color, pseudo, pros_cons) in enumerate(cards):
        ax = axes[i]
        ax.axis("off")
        # Card background
        box = mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                       boxstyle="round,pad=0.02",
                                       facecolor="#fdfdfd",
                                       edgecolor=color, linewidth=2.5,
                                       transform=ax.transAxes)
        ax.add_patch(box)
        # Title bar
        ax.add_patch(mpatches.Rectangle((0.02, 0.78), 0.96, 0.2,
                                         facecolor=color, edgecolor=color,
                                         transform=ax.transAxes))
        ax.text(0.05, 0.88, name, fontsize=13, fontweight="bold", color="white",
                transform=ax.transAxes, va="center")
        # Pseudocode
        ax.text(0.05, 0.60, "sort key:", fontsize=8.5, color="#666",
                transform=ax.transAxes, va="center")
        ax.text(0.05, 0.50, pseudo, fontsize=10, color="#1a1f2e",
                family="monospace", transform=ax.transAxes, va="center")
        # Pros/cons
        ax.text(0.05, 0.22, pros_cons, fontsize=9, color="#1a1f2e",
                transform=ax.transAxes, va="center", wrap=True)

    # Hide unused
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Scheduler 카드 — 한 줄 정리", fontsize=14, fontweight="bold",
                 y=1.005)
    plt.tight_layout()
    plt.savefig(f"{OUT}/algo_pseudocode_grid.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT}/algo_pseudocode_grid.png")


if __name__ == "__main__":
    fig_ordering_demo()
    fig_feature_matrix()
    fig_pseudocode_cards()
