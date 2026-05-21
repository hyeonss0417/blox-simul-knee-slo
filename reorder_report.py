"""
Reorder docs/report_v2.md so that:
  - High-level findings come FIRST (TL;DR + conclusion + algorithm intuition)
  - Detailed Wave results come MIDDLE
  - Debugging journal / trial-and-error come BOTTOM
"""
import re
import os

SRC = "docs/report_v2.md"
OUT = "docs/report_v2.md"

text = open(SRC).read()

# Split by top-level "## " headers, preserving the title block.
# Title block is everything before the first "## "
title_match = re.search(r"\n## ", text)
title_block = text[: title_match.start()]
rest = text[title_match.start() :]

# Split rest into sections.  Each section starts with "## ".
sections = re.split(r"(?=\n## )", rest)
sections = [s for s in sections if s.strip()]

# Index sections by their header line.
def header_of(s):
    m = re.search(r"^## (.+)", s, re.MULTILINE)
    return m.group(1).strip() if m else "?"


for i, s in enumerate(sections):
    print(f"{i:2d}. {header_of(s)[:80]}")

# Target order: keys are substring matches of section header.
TARGET_ORDER = [
    # === TOP: 핵심 ===
    "결론",                                    # §11 결론 → §1 TL;DR
    "문제 정의",                                # §1 → §2  (워크로드 A/B 표)
    "Knee-SLO 알고리즘",                       # §2 → §3 알고리즘 직관
    # === MIDDLE: 상세 결과 ===
    "최종 분석",                                # §10 → §4 워크로드 A 상세 분석
    "결과 — Wave 1",                          # §7 → §5
    "Wave 2",                                  # §8
    "Wave 3",                                  # §9
    "Wave 4",                                  # §9bis
    "추론 워크로드 재실험",                    # §12 → 워크로드 B
    # === BOTTOM: 시행착오 & 디테일 ===
    "사후 정정",                                # 워크로드 라벨 오류 일지
    "Baselines",                                # 디테일
    "평가 지표",
    "Grid Search 설계",
    "진행 상태 체크리스트",                    # 디버깅 일지 포함
    "변경 요약",                                # v1→v2 이력
    "부록 A",
    "부록 B",
]


def rank(s):
    h = header_of(s)
    for i, key in enumerate(TARGET_ORDER):
        if key in h:
            return i
    return 999  # unknown → end


# Sort
sections.sort(key=rank)

# Re-number top-level sections.  Skip the "부록" entries.
# Also rewrite ALL sub-section numbers consistently within each section.
new_sections = []
counter = 1
# Build mapping for cross-references: old top-level → new top-level
xref_map = {}
for s in sections:
    h = header_of(s)
    m = re.match(r"^(\d+)(bis|ter)?\.\s*", h)
    if m:
        old_num = m.group(1) + (m.group(2) or "")
        xref_map[old_num] = counter if "부록" not in h else None
    if "부록" in h:
        new_sections.append(s)
        continue

    # Rewrite top-level header
    s_new = re.sub(
        r"^## (?:\d+(?:bis|ter)?\.\s*|⚠️\s*)?(.+)$",
        lambda m, c=counter: f"## {c}. {m.group(1).strip()}",
        s,
        count=1,
        flags=re.MULTILINE,
    )
    # Rewrite sub-section headers ### N.X → ### counter.X
    def sub_rename(m, c=counter):
        suffix = m.group(2)  # e.g. ".1", ".2bis"
        return f"### {c}{suffix}"
    s_new = re.sub(
        r"^### (\d+)(\.\d+(?:bis|ter)?)",
        sub_rename,
        s_new,
        flags=re.MULTILINE,
    )
    # Handle the "9bis.1", "9bis.2" pattern -> just keep the .X under new counter
    s_new = re.sub(
        r"^### \d+bis(\.\d+)",
        lambda m, c=counter: f"### {c}{m.group(1)}",
        s_new,
        flags=re.MULTILINE,
    )
    new_sections.append(s_new)
    counter += 1

# Now fix cross-references in the body: "§7", "§7~§11", "§12" etc.
def rewrite_refs(text, xref_map):
    def fix_ref(m):
        n = m.group(1)
        if n in xref_map and xref_map[n] is not None:
            return f"§{xref_map[n]}"
        return m.group(0)
    # Single section refs: §N or §N.
    text = re.sub(r"§(\d+(?:bis|ter)?)", fix_ref, text)
    return text

new_sections = [rewrite_refs(s, xref_map) for s in new_sections]
title_block = rewrite_refs(title_block, xref_map)

result = title_block + "\n" + "\n".join(s.lstrip("\n") for s in new_sections)

# Also strip "(워크로드 A, hours)" suffix in section titles to make
# the top-of-document story flow better — we have a clear label table
# in the algorithm-overview section anyway.
# But keep it for the Wave-X sections in MIDDLE.  Skip this transform.

with open(OUT, "w") as f:
    f.write(result)

print(f"\nWrote {OUT}  ({len(result):,} bytes, {len(sections)} sections)")
print("\nNew structure:")
final = open(OUT).read()
for m in re.finditer(r"^## (.+)$", final, re.MULTILINE):
    print(f"  {m.group(0)[:90]}")
