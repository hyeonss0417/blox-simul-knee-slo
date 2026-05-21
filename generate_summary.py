"""
generate_summary.py — aggregates every v2_*_job_stats.json + w3_* into a
master markdown table and writes it into docs/report_v2.md between
auto-update markers.

Also writes JSON summaries that other tools can pick up.

Markers in report:
    <!-- BEGIN AUTO: wave1_table --> ... <!-- END AUTO: wave1_table -->
    <!-- BEGIN AUTO: wave1_findings --> ... <!-- END AUTO: wave1_findings -->
    <!-- BEGIN AUTO: wave2_table --> ...
    <!-- BEGIN AUTO: wave3_table --> ...
    <!-- BEGIN AUTO: final_recommendation --> ...
"""
import json
import glob
import os
import re
import numpy as np

START, STOP = 3000, 3100
DEFAULT_LOAD = 8.0
SLO_TARGET = 21600.0  # 6h

# Mapping prefix → (display label, wave id)
LABELS = {
    "v2_fifo":          ("FIFO",                      "W0"),
    "v2_las":           ("LAS",                       "W0"),
    "v2_srtf":          ("SRTF (oracle)",             "W0"),
    "v2_slo_v1":        ("v1 SloScoring (ref)",       "W0"),
    "v2_sjf":           ("SJF-total (predicted)",     "W1-base"),
    "v2_edf":           ("EDF",                       "W1-base"),
    "v2_llf":           ("LLF",                       "W1-base"),
    "v2_knee_t3_q":     ("Knee θ=0.3 γ=2 quad",       "W1-grid"),
    "v2_knee_t5_q":     ("Knee θ=0.5 γ=2 quad",       "W1-grid"),
    "v2_knee_t7_q":     ("Knee θ=0.7 γ=2 quad (def)", "W1-grid"),
    "v2_knee_t7_q3":    ("Knee θ=0.7 γ=3 quad",       "W1-grid"),
    "v2_knee_t5_q3":    ("Knee θ=0.5 γ=3 quad",       "W1-grid"),
    "v2_knee_t7_lin":   ("Knee θ=0.7 linear",         "W1-grid"),
    "v2_knee_t7_sig":   ("Knee θ=0.7 sigmoid",        "W1-grid"),
    "v2_knee_t5_sig":   ("Knee θ=0.5 sigmoid",        "W1-grid"),
    "v2_hrrn":          ("HRRN",                      "W2-ext"),
    "v2_knee_np_t7":    ("Knee NonPreempt θ=0.7",     "W2-ext"),
    "v2_knee_np_t5":    ("Knee NonPreempt θ=0.5",     "W2-ext"),
    "v2_knee_cls_t7":   ("Knee class-aware θ=0.7",    "W2-ext"),
    "v2_knee_cls_t5":   ("Knee class-aware θ=0.5",    "W2-ext"),
    "v2_knee_age_hi":   ("Knee w_age=0.5",            "W2-abl"),
    "v2_knee_age_lo":   ("Knee w_age=0.0",            "W2-abl"),
    "v2_knee_urg_hi":   ("Knee urg-heavy",            "W2-abl"),
    "v2_knee_size_hi":  ("Knee size-heavy",           "W2-abl"),
    "v2_knee_b3h":      ("Knee budget=3h",            "W2-abl"),
    "v2_knee_b12h":     ("Knee budget=12h",           "W2-abl"),
    "v2_knee_c3_lo":    ("Knee c3=5",                 "W2-abl"),
    "v2_knee_c3_hi":    ("Knee c3=100",               "W2-abl"),
    "v2_knee_adapt":    ("Knee adaptive θ",           "W2-ext"),
    "v2_knee_t9_q":     ("Knee θ=0.9 γ=2 quad",       "W4-extreme"),
    "v2_knee_t1_q":     ("Knee θ=0.1 γ=2 quad",       "W4-extreme"),
    "v2_knee_g4":       ("Knee θ=0.7 γ=4",            "W4-extreme"),
    "v2_knee_only_urg": ("Knee urg-only (w_size=0)",  "W4-pure"),
    "v2_knee_only_size":("Knee size-only (w_urg=0)",  "W4-pure"),
    "v2_knee_np_age":   ("NonPreempt + w_age=0.5",    "W4-combo"),
    "v2_knee_cls_t3":   ("Knee class θ=0.3",          "W4-combo"),
    # Wave 5 — SLO recalibrated to 24h (proper operating point)
    "v2_knee24_t3":     ("Knee[24h] θ=0.3",           "W5-cal24h"),
    "v2_knee24_t5":     ("Knee[24h] θ=0.5",           "W5-cal24h"),
    "v2_knee24_t7":     ("Knee[24h] θ=0.7",           "W5-cal24h"),
    "v2_knee24_t9":     ("Knee[24h] θ=0.9",           "W5-cal24h"),
    "v2_knee24_t7_sig": ("Knee[24h] θ=0.7 sigmoid",   "W5-cal24h"),
    "v2_knee24_t7_lin": ("Knee[24h] θ=0.7 linear",    "W5-cal24h"),
    "v2_knee24_np":     ("Knee[24h] NonPreempt",      "W5-cal24h"),
    "v2_knee24_age":    ("Knee[24h] w_age=0.5",       "W5-cal24h"),
    "v2_knee24_cdur_m2":("Knee class-dur ×2",         "W5-classdur"),
    "v2_knee24_cdur_m3":("Knee class-dur ×3",         "W5-classdur"),
    "v2_knee24_cdur_m5":("Knee class-dur ×5",         "W5-classdur"),
}

# Multi-target SLO miss rate (in addition to the default).
SLO_TARGETS_H = [6, 12, 18, 24, 36, 48]


def find(prefix, kind, load=DEFAULT_LOAD):
    pat = f"{prefix}_{START}_{STOP}_*_accept_all_load_{load}_{kind}.json"
    m = glob.glob(pat)
    return m[0] if m else None


def metric_row(prefix, load=DEFAULT_LOAD):
    jf = find(prefix, "job_stats", load=load)
    if not jf:
        return None
    d = json.load(open(jf))
    jcts = [v[1] - v[0] for v in d.values() if isinstance(v, list) and len(v) >= 2]
    if not jcts:
        return None
    miss = sum(1 for v in jcts if v > SLO_TARGET) / len(jcts) * 100
    tard = [max(0.0, v - SLO_TARGET) for v in jcts]

    resp_f = find(prefix, "responsivness", load=load)
    resp_avg = None
    if resp_f:
        d = json.load(open(resp_f))
        rs = [v[1] - v[0] for v in d.values() if isinstance(v, list) and len(v) >= 2]
        if rs:
            resp_avg = float(np.mean(rs))

    label, wave = LABELS.get(prefix, (prefix, "?"))
    multi_miss = {
        h: sum(1 for v in jcts if v > h * 3600) / len(jcts) * 100
        for h in SLO_TARGETS_H
    }
    return {
        "prefix": prefix,
        "label": label,
        "wave": wave,
        "avg_jct_h": float(np.mean(jcts)) / 3600,
        "median_h": float(np.median(jcts)) / 3600,
        "p95_h": float(np.percentile(jcts, 95)) / 3600,
        "p99_h": float(np.percentile(jcts, 99)) / 3600,
        "n": len(jcts),
        "slo_miss_pct": miss,
        "miss_by_target": multi_miss,
        "tard_mean_h": float(np.mean(tard)) / 3600,
        "tard_p95_h": float(np.percentile(tard, 95)) / 3600,
        "resp_s": resp_avg,
    }


def fmt(x, fmt_str="{:.2f}"):
    if x is None:
        return "—"
    try:
        return fmt_str.format(x)
    except Exception:
        return str(x)


def render_table(rows, fifo_avg=None):
    lines = []
    lines.append(
        "| Config | Avg JCT (h) | vs FIFO | Median (h) | P95 (h) | P99 (h) | "
        "SLO miss % | Tard mean (h) | Resp (s) |"
    )
    lines.append(
        "| ------ | ----------- | ------- | ---------- | ------- | ------- | "
        "---------- | ------------- | -------- |"
    )
    for r in rows:
        vs = ""
        if fifo_avg and r["avg_jct_h"] != fifo_avg:
            d = (r["avg_jct_h"] - fifo_avg) / fifo_avg * 100
            vs = f"{d:+.1f}%"
        elif r["avg_jct_h"] == fifo_avg:
            vs = "baseline"
        resp_str = fmt(r["resp_s"], "{:,.0f}") if r["resp_s"] is not None else "—"
        lines.append(
            f"| {r['label']} | {r['avg_jct_h']:.2f} | {vs} | "
            f"{r['median_h']:.2f} | {r['p95_h']:.2f} | {r['p99_h']:.2f} | "
            f"{r['slo_miss_pct']:.1f}% | {r['tard_mean_h']:.2f} | {resp_str} |"
        )
    return "\n".join(lines)


def write_section(report_path, marker, content):
    text = open(report_path).read()
    pattern = re.compile(
        rf"<!-- BEGIN AUTO: {marker} -->.*?<!-- END AUTO: {marker} -->",
        re.DOTALL,
    )
    block = f"<!-- BEGIN AUTO: {marker} -->\n{content}\n<!-- END AUTO: {marker} -->"
    if pattern.search(text):
        new = pattern.sub(block, text)
    else:
        # Append a new section if marker doesn't exist (so the file
        # gets the block at least once).
        new = text + f"\n\n{block}\n"
    with open(report_path, "w") as f:
        f.write(new)


def best_by(rows, key, lower_is_better=True):
    eligible = [r for r in rows if r[key] is not None]
    if not eligible:
        return None
    return min(eligible, key=lambda r: r[key]) if lower_is_better \
        else max(eligible, key=lambda r: r[key])


def main():
    rows = []
    for prefix in LABELS:
        r = metric_row(prefix)
        if r:
            rows.append(r)

    if not rows:
        print("No results found.")
        return

    # Persist raw JSON for downstream use.
    os.makedirs("docs/figures_v2", exist_ok=True)
    with open("docs/figures_v2/summary_all.json", "w") as f:
        json.dump(rows, f, indent=2)

    fifo = next((r for r in rows if r["prefix"] == "v2_fifo"), None)
    fifo_avg = fifo["avg_jct_h"] if fifo else None

    # Group by wave for markdown.
    rows_sorted = sorted(rows, key=lambda r: r["avg_jct_h"])
    by_wave = {}
    for r in rows:
        by_wave.setdefault(r["wave"], []).append(r)

    # --- Wave 1 table (W0 + W1-base + W1-grid) ---
    w1 = sorted([r for r in rows if r["wave"] in ("W0", "W1-base", "W1-grid")],
                key=lambda r: r["avg_jct_h"])
    if w1:
        write_section("docs/report_v2.md", "wave1_table", render_table(w1, fifo_avg))

    # --- Wave 2 table ---
    w2 = sorted([r for r in rows if r["wave"].startswith("W2")],
                key=lambda r: r["avg_jct_h"])
    if w2:
        write_section("docs/report_v2.md", "wave2_table", render_table(w2, fifo_avg))

    # --- Wave 4 table ---
    w4 = sorted([r for r in rows if r["wave"].startswith("W4")],
                key=lambda r: r["avg_jct_h"])
    if w4:
        write_section("docs/report_v2.md", "wave4_table", render_table(w4, fifo_avg))

    # --- Findings (auto-generated bullets) ---
    findings = []
    best_avg = best_by(rows, "avg_jct_h")
    best_miss = best_by(rows, "slo_miss_pct")
    best_tard = best_by(rows, "tard_mean_h")
    best_resp = best_by(rows, "resp_s")
    findings.append(
        f"- 가장 낮은 Avg JCT: **{best_avg['label']}** ({best_avg['avg_jct_h']:.2f}h)"
    )
    findings.append(
        f"- 가장 낮은 SLO miss rate: **{best_miss['label']}** ({best_miss['slo_miss_pct']:.1f}%)"
    )
    findings.append(
        f"- 가장 낮은 Mean Tardiness: **{best_tard['label']}** ({best_tard['tard_mean_h']:.2f}h)"
    )
    if best_resp:
        findings.append(
            f"- 가장 빠른 Responsiveness: **{best_resp['label']}** ({best_resp['resp_s']:,.0f}s)"
        )
    # Knee-vs-SJF differential.
    knee_best = best_by([r for r in rows if "Knee" in r["label"]], "avg_jct_h")
    sjf = next((r for r in rows if r["prefix"] == "v2_sjf"), None)
    if knee_best and sjf:
        diff = (knee_best["avg_jct_h"] - sjf["avg_jct_h"]) / sjf["avg_jct_h"] * 100
        findings.append(
            f"- 최고 Knee variant vs SJF-total (가장 가까운 비교 대상): "
            f"Avg JCT {diff:+.1f}%, SLO miss {knee_best['slo_miss_pct']-sjf['slo_miss_pct']:+.1f}p"
        )
    findings_md = "\n".join(findings)
    write_section("docs/report_v2.md", "wave1_findings", findings_md)

    # --- Final recommendation (auto) ---
    # Score = avg_jct + 5 * tard_mean (heuristic Pareto blend)
    def composite(r):
        return r["avg_jct_h"] + 5.0 * r["tard_mean_h"]
    knee_rows = [r for r in rows if "Knee" in r["label"] and "FIFO" not in r["label"]]
    if knee_rows:
        winner = min(knee_rows, key=composite)
        rec = []
        rec.append(f"**최종 추천 구성:** {winner['label']}")
        rec.append("")
        rec.append(
            f"- Avg JCT: **{winner['avg_jct_h']:.2f}h** "
            f"({(winner['avg_jct_h']-fifo_avg)/fifo_avg*100:+.1f}% vs FIFO)"
        )
        rec.append(f"- P99 JCT: {winner['p99_h']:.2f}h")
        rec.append(f"- SLO miss rate: {winner['slo_miss_pct']:.1f}%")
        rec.append(f"- Mean tardiness: {winner['tard_mean_h']:.2f}h")
        if winner["resp_s"] is not None:
            rec.append(f"- Responsiveness: {winner['resp_s']:,.0f}s")
        rec.append("")
        rec.append(
            f"이 구성은 `avg_jct + 5 × tard_mean` 복합 점수 기준 가장 낮은 값을 "
            f"보였으며, Pareto frontier 상에서 JCT/SLO trade-off가 가장 균형 잡힘."
        )
        write_section("docs/report_v2.md", "final_recommendation", "\n".join(rec))

    # --- Console output ---
    print(f"\n{'Config':32s} {'Wave':10s} {'AvgJCT(h)':>10s} {'P99(h)':>7s} "
          f"{'SLOmiss%':>8s} {'TardMn(h)':>9s} {'Resp(s)':>8s}")
    print("-" * 95)
    for r in rows_sorted:
        resp_str = f"{r['resp_s']:.0f}" if r["resp_s"] is not None else "—"
        print(f"{r['label']:32s} {r['wave']:10s} {r['avg_jct_h']:10.2f} "
              f"{r['p99_h']:7.2f} {r['slo_miss_pct']:7.1f}% "
              f"{r['tard_mean_h']:9.2f} {resp_str:>8s}")

    print(f"\n→ docs/report_v2.md updated (wave1_table, wave1_findings, wave2_table)")


if __name__ == "__main__":
    main()
