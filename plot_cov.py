"""Plot: job duration distribution + CoV-driven SJF gain analysis."""
import json, datetime, os
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

def fmt(t): return datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")

data = json.load(open("cluster_job_log"))
durs = []
for j in data:
    for a in j.get("attempts", []):
        try:
            d = (fmt(a["end_time"]) - fmt(a["start_time"])).total_seconds()
            if d > 0:
                durs.append(d); break
        except: continue
d = np.array(durs)
cov = d.std() / d.mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: distribution
ax = axes[0]
ax.hist(d, bins=80, color="#3498db", alpha=0.7, edgecolor="white")
ax.axvline(d.mean(), color="#c0392b", linestyle="--", lw=2,
           label=f"mean = {d.mean():.1f}s")
ax.axvline(np.median(d), color="#16a085", linestyle="--", lw=2,
           label=f"median = {np.median(d):.0f}s")
ax.set_xlabel("Job duration (s)")
ax.set_ylabel("count")
ax.set_xlim(0, 100)
ax.set_title(f"추론 잡 duration 분포\n"
             f"mean={d.mean():.1f}s, std={d.std():.1f}s, **CoV = {cov:.3f}** (낮음)")
ax.legend()
ax.grid(alpha=0.3)

# Right: SJF gain vs CoV
ax = axes[1]
cov_range = np.linspace(0.1, 2.5, 100)
theory_gain = cov_range ** 2 / 2 * 100  # Pollaczek-Khinchine approx
ax.plot(cov_range, theory_gain, color="#0d3b66", lw=2,
        label="이론 (Pollaczek-Khinchine ~c²/2)")

# Annotate workloads
workloads = [
    (0.29, 5, "Uniform[0.5x,1.5x]", "#bdc3c7", "right"),
    (cov, 21, "★ 본 trace (추론)", "#27ae60", "right"),
    (1.0, 30, "Exponential", "#3498db", "left"),
    (1.7, 49, "★ 합성 training-like", "#9b59b6", "left"),
]
for c, gain, label, color, ha in workloads:
    ax.scatter([c], [gain], s=150, color=color, edgecolor="white", linewidth=2,
               zorder=5)
    if ha == "right":
        ax.annotate(label, (c, gain), xytext=(c + 0.1, gain + 3),
                    fontsize=10, fontweight="bold", color=color)
    else:
        ax.annotate(label, (c, gain), xytext=(c - 0.7, gain - 5),
                    fontsize=10, fontweight="bold", color=color)

ax.set_xlabel("Coefficient of Variation (CoV = std / mean)")
ax.set_ylabel("SJF / SRTF Avg JCT 개선 (%)")
ax.set_title("CoV 에 따른 SJF/SRTF gain 한계\n"
             "잡 분포가 균일할수록 (CoV 작을수록) 알고리즘 차이가 작아짐")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
ax.set_xlim(0, 2.5)
ax.set_ylim(0, 70)

plt.tight_layout()
plt.savefig(f"{OUT}/cov_analysis.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"saved {OUT}/cov_analysis.png")
print(f"CoV = {cov:.3f}, theoretical gain = {cov**2/2*100:.1f}%")
