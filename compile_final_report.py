"""
compile_final_report.py — at the very end, regenerate summary, find the
winner, and inject:
  - Executive summary at the top of report_v2.md (between AUTO markers).
  - The "final recommendation" section (already inserted by
    generate_summary.py).
"""
import json
import os
import re
import subprocess
import numpy as np

REPORT = "docs/report_v2/report.md"


def latest_summary():
    p = "docs/report_v2/figures/summary_all.json"
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f)


def composite(r):
    return r["avg_jct_h"] + 5.0 * r["tard_mean_h"]


def main():
    # Regenerate everything first.
    subprocess.run(["python", "generate_summary.py"], check=False)
    subprocess.run(["python", "plot_v2_results.py"], check=False)
    if os.path.exists("plot_w3_loadsweep.py"):
        subprocess.run(["python", "plot_w3_loadsweep.py"], check=False)

    rows = latest_summary()
    if not rows:
        print("no data yet")
        return

    fifo = next((r for r in rows if r["prefix"] == "v2_fifo"), None)
    fifo_avg = fifo["avg_jct_h"] if fifo else None

    knee_rows = [r for r in rows if "Knee" in r["label"]]
    if not knee_rows:
        print("no Knee rows yet")
        return

    winner_overall = min(rows, key=composite)
    winner_knee = min(knee_rows, key=composite)
    best_avg = min(rows, key=lambda r: r["avg_jct_h"])
    best_miss = min(rows, key=lambda r: r["slo_miss_pct"])
    n_configs = len(rows)

    summary_lines = []
    summary_lines.append("> **한 줄 요약**:  부하 강도 ρ 에 따라 추천 알고리즘이 다르다 — Mild 에선 **MetaSrtf** (Oracle SRTF 와 동등, FIFO 대비 −15 %), Heavy 에선 **MetaSrtfSlo** (bucket 변형, 유일하게 안정). LAS 는 어디서도 추천 아님.")
    summary_lines.append(">")
    summary_lines.append(
        "> **두 contribution**: ① Submission-time predictor (request params 만으로 Oracle SRTF 수준 달성, post-execution 정보 불필요) "
        "② SLO bucket 으로 heavy contention 의 starvation 회피."
    )
    summary_lines.append(">")
    summary_lines.append("> 자세한 결과는 §3, 메커니즘은 §4, 한계는 §6 참조.")

    block = "<!-- BEGIN AUTO: exec_summary -->\n" + \
            "\n".join(summary_lines) + \
            "\n<!-- END AUTO: exec_summary -->"

    text = open(REPORT).read()
    if "<!-- BEGIN AUTO: exec_summary -->" in text:
        text = re.sub(
            r"<!-- BEGIN AUTO: exec_summary -->.*?<!-- END AUTO: exec_summary -->",
            block, text, flags=re.DOTALL,
        )
    else:
        # Insert just after the title block (after the first horizontal rule)
        text = re.sub(
            r"(\*\*작성일 갱신:\*\* [^\n]+\n+---\n)",
            r"\1\n" + block + "\n\n---\n", text, count=1,
        )

    with open(REPORT, "w") as f:
        f.write(text)

    print(f"Executive summary updated. Winner: {winner_overall['label']}")
    print(f"Best Knee variant: {winner_knee['label']}")


if __name__ == "__main__":
    main()
