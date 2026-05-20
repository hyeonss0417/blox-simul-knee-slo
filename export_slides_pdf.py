"""Export each slide as a screenshot, then assemble into PDF.
Each web slide (100vh) becomes exactly one PDF page."""
from playwright.sync_api import sync_playwright
from PIL import Image
import os

HTML_PATH = os.path.abspath("docs/midterm_presentation.html")
OUT_DIR = "docs/slides_pdf"
os.makedirs(OUT_DIR, exist_ok=True)

SLIDES = [f"s{i}" for i in range(13)]
W, H = 1920, 1080  # 16:9 Full HD

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
    page.goto(f"file://{HTML_PATH}")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1000)

    png_paths = []
    for i, slide_id in enumerate(SLIDES):
        # Scroll to the slide
        page.evaluate(f"document.getElementById('{slide_id}').scrollIntoView()")
        page.wait_for_timeout(400)

        # Screenshot just the slide element
        el = page.query_selector(f"#{slide_id}")
        png_path = os.path.join(OUT_DIR, f"slide_{i:02d}.png")
        el.screenshot(path=png_path)
        png_paths.append(png_path)
        print(f"Screenshot: {png_path}")

    # Convert PNGs to individual PDFs and one merged PDF
    images = []
    for i, png_path in enumerate(png_paths):
        img = Image.open(png_path).convert("RGB")
        # Individual PDF
        pdf_path = os.path.join(OUT_DIR, f"slide_{i:02d}.pdf")
        img.save(pdf_path, "PDF", resolution=150)
        images.append(img)
        print(f"PDF: {pdf_path}")

    # Merged PDF
    merged_path = os.path.join(OUT_DIR, "midterm_presentation_all.pdf")
    images[0].save(merged_path, "PDF", resolution=150, save_all=True, append_images=images[1:])
    print(f"Merged: {merged_path}")

    browser.close()

print("Done!")
