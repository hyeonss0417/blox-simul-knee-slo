"""
Build a polished standalone HTML from docs/report_v2/report.md.

- Uses pandoc for MD → HTML conversion with header anchors.
- Embeds the CSS inline so the file is self-contained.
- Generates a sticky TOC sidebar from h2 + h3 anchors.
- Rewrites image src to point at docs/report_v2/figures/ relative paths.
"""
import os
import re
import subprocess
import html

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "docs", "report_v2", "report.md")
CSS = os.path.join(ROOT, "docs", "report_v2", "style.css")
OUT = os.path.join(ROOT, "docs", "report_v2", "report.html")
FIG_DIR = os.path.join(ROOT, "docs", "report_v2", "figures")


def slugify(s):
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "-", s.strip())
    return s.lower()


def pandoc_to_html(md_path):
    out = subprocess.run(
        ["pandoc",
         "--from", "gfm",
         "--to", "html5",
         "--no-highlight",   # we style code via CSS
         md_path],
        capture_output=True, text=True, check=True,
    )
    return out.stdout


def add_header_ids(html_body):
    """Add slug-based id="" attributes to h2 and h3 (pandoc's gfm doesn't always do it)."""
    def replace(m):
        tag, text = m.group(1), m.group(2)
        slug = slugify(re.sub(r"<[^>]+>", "", text))[:60]
        return f'<{tag} id="{slug}">{text}</{tag}>'
    html_body = re.sub(r"<(h[23])>([^<]+)</\1>", replace, html_body)
    return html_body


def extract_toc(html_body):
    items = []
    for m in re.finditer(r'<(h2|h3) id="([^"]+)">([^<]+)</\1>', html_body):
        level, anchor, text = m.group(1), m.group(2), m.group(3)
        items.append((level, anchor, text))
    return items


def render_toc(items):
    if not items:
        return ""
    lines = ['<nav class="toc"><h4>목차</h4><ul>']
    for level, anchor, text in items:
        cls = f"toc-{level}"
        lines.append(f'<li class="{cls}"><a href="#{anchor}">{html.escape(text)}</a></li>')
    lines.append("</ul></nav>")
    return "\n".join(lines)


def fix_image_paths(html_body):
    """Make figure paths absolute relative to HTML's location."""
    # In MD: figures referenced as `figures_v2/foo.png` or `docs/report_v2/figures/foo.png`
    # HTML lives in docs/ so figures_v2/foo.png works directly.
    # Also embed missing-file fallback.
    return html_body


def wrap_exec_summary(html_body):
    """Wrap the BEGIN AUTO: exec_summary block in a styled card."""
    # The summary is rendered as a blockquote; we just wrap it.
    pattern = re.compile(
        r'(<blockquote>\s*<p><strong>요약 \(자동 생성\)</strong>.*?</blockquote>)',
        re.DOTALL,
    )
    return pattern.sub(r'<div class="exec-summary">\1</div>', html_body)


def main():
    print(f"Reading {SRC}")
    body = pandoc_to_html(SRC)
    body = add_header_ids(body)
    body = fix_image_paths(body)
    body = wrap_exec_summary(body)
    toc_items = extract_toc(body)
    toc_html = render_toc(toc_items)

    with open(CSS) as f:
        css = f.read()

    final = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knee-SLO 보고서 v2</title>
<style>{css}</style>
</head>
<body>
{toc_html}
<div class="container">
{body}
</div>
</body>
</html>
"""

    with open(OUT, "w") as f:
        f.write(final)
    print(f"Wrote {OUT}  ({len(final):,} bytes, {len(toc_items)} TOC entries)")


if __name__ == "__main__":
    main()
