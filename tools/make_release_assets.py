"""Build the static release assets: the before/after comparison, the social
card and the six-second hero loop.

Dev-only, exactly like tools/make_hero_gif.py: it needs selenium and Pillow,
it is never imported by the package, and CI does not run it. The committed
outputs under assets/ are the artefacts; this script explains where they came
from and regenerates them.

    python tools/make_release_assets.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = ROOT / "assets"
WORK = ROOT / "build" / "assets"

BG = (11, 14, 20)
INK = (230, 237, 243)
DIM = (154, 164, 178)
RED = (247, 118, 142)
GREEN = (158, 206, 106)
MONO = (192, 202, 245)

FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts"), Path("/Library/Fonts")]
BOLD_NAMES = ["segoeuib.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"]
REGULAR_NAMES = ["segoeui.ttf", "DejaVuSans.ttf", "arial.ttf"]
MONO_NAMES = ["consola.ttf", "DejaVuSansMono.ttf", "cour.ttf"]


def font(names, size):
    for directory in FONT_DIRS:
        for name in names:
            path = directory / name
            if path.is_file():
                return ImageFont.truetype(str(path), size)
    raise SystemExit("no usable font among %s" % (names,))


def capture(pages):
    """Screenshot each report and return the bounding boxes we need."""
    from selenium import webdriver
    from selenium.webdriver.edge.options import Options

    options = Options()
    for flag in (
        "--headless=new",
        "--window-size=1280,860",
        "--force-device-scale-factor=1",
        "--disable-gpu",
        "--hide-scrollbars",
        "--lang=en-US",
    ):
        options.add_argument(flag)

    rects = {}
    driver = webdriver.Edge(options=options)
    try:
        for name, page in pages:
            driver.get(page.resolve().as_uri())
            time.sleep(1.4)
            target = WORK / (name + ".png")
            driver.save_screenshot(str(target))
            rects[name] = driver.execute_script(
                """
                function box(el) {
                  const r = el.getBoundingClientRect();
                  return {x: r.x, y: r.y, w: r.width, h: r.height};
                }
                return {
                  assertions: box(document.querySelector('#assertions').closest('section')),
                  diff: box(document.querySelector('#diff-summary').closest('section')),
                  viewport: {w: window.innerWidth, h: window.innerHeight}
                };
                """
            )
            rects[name]["file"] = str(target)
    finally:
        driver.quit()
    return rects


def crop_panel(entry, pad=0):
    img = Image.open(entry["file"]).convert("RGB")
    box = entry["assertions"]
    scale = img.width / entry["viewport"]["w"]
    left = int((box["x"] - pad) * scale)
    top = int((box["y"] - pad) * scale)
    right = int((box["x"] + box["w"] + pad) * scale)
    bottom = int((box["y"] + box["h"] + pad) * scale)
    return img.crop((max(left, 0), max(top, 0), min(right, img.width), min(bottom, img.height)))


def fit(img, max_h):
    if img.height <= max_h:
        return img
    width = round(img.width * max_h / img.height)
    return img.resize((width, max_h), Image.LANCZOS)


def compose(width, height, title_px, panel_w, gap, panels, labels):
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    margin = round(width * 0.0375)

    draw.text((margin, round(height * 0.055)), "Same tape. One policy key. FAIL \u2192 PASS.",
              font=font(BOLD_NAMES, title_px), fill=INK)
    sub_px = round(title_px * 0.44)
    sub_y = round(height * 0.055) + title_px + 12
    draw.text((margin, sub_y), "one recorded run, replayed twice offline \u2014 no model call, no API key",
              font=font(REGULAR_NAMES, sub_px), fill=DIM)

    def scaled(img):
        target_h = round(img.height * panel_w / img.width)
        return img.resize((panel_w, target_h), Image.LANCZOS)

    left_panel = scaled(panels["before"])
    right_panel = scaled(panels["after"])

    total = panel_w * 2 + gap
    left_x = (width - total) // 2
    right_x = left_x + panel_w + gap

    label_px = round(title_px * 0.36)
    label_font = font(BOLD_NAMES, label_px)
    bottom_px = round(title_px * 0.40)
    bottom = font(REGULAR_NAMES, bottom_px)
    bottom_y = height - margin - bottom_px

    block_h = label_px + 14 + max(left_panel.height, right_panel.height)
    free_top = sub_y + sub_px + 24
    block_top = free_top + max(0, (bottom_y - 30 - free_top - block_h) // 2)

    panels_top = block_top + label_px + 14
    for key, x in (("before", left_x), ("after", right_x)):
        text, colour = labels[key]
        draw.text((x, block_top), text, font=label_font, fill=colour)
    canvas.paste(left_panel, (left_x, panels_top))
    canvas.paste(right_panel, (right_x, panels_top))

    arrow_font = font(BOLD_NAMES, round(title_px * 0.5))
    note_font = font(REGULAR_NAMES, round(title_px * 0.30))
    mid = (left_x + panel_w + right_x) // 2
    centre = panels_top + min(left_panel.height, right_panel.height) // 2
    draw.text((mid, centre - 46), "\u2192", font=arrow_font, fill=DIM, anchor="ma")
    draw.text((mid, centre + 6), "one policy key", font=note_font, fill=DIM, anchor="ma")
    draw.text((mid, centre + 6 + note_font.size + 4), "changed", font=note_font, fill=DIM, anchor="ma")

    draw.text((width // 2, bottom_y),
              "pip install agenttwin  \u00b7  no API key  \u00b7  no model call  \u00b7  no side effects",
              font=bottom, fill=DIM, anchor="ma")
    return canvas


def terminal_frame(lines, size):
    width, height = size
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    body = font(MONO_NAMES, 15)
    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=12, outline=(35, 42, 54), width=2)
    draw.text((44, 44), "$ python examples/hero.py", font=body, fill=DIM)

    y = 44 + body.size + 14
    for line in lines:
        text = line if len(line) <= 96 else line[:93] + "..."
        if "FAIL" in text:
            colour = RED
        elif "PASS" in text:
            colour = GREEN
        elif "exit 0" in text:
            colour = GREEN
        elif "exit 1" in text:
            colour = RED
        else:
            colour = MONO
        draw.text((44, y), text, font=body, fill=colour)
        y += body.size + 6
        if y > height - 60:
            break
    return canvas


def promise_frame(size):
    width, height = size
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    big = font(BOLD_NAMES, 40)
    mono = font(MONO_NAMES, 26)
    small = font(REGULAR_NAMES, 20)
    draw.text((width // 2, height // 2 - 90), "no model call", font=big, fill=INK, anchor="ma")
    draw.text((width // 2, height // 2 - 30), "no API key  \u00b7  no side effects",
              font=big, fill=INK, anchor="ma")
    draw.text((width // 2, height // 2 + 40), "pip install agenttwin", font=mono, fill=GREEN, anchor="ma")
    draw.text((width // 2, height // 2 + 90), "replay any recorded run offline, as a test you can commit",
              font=small, fill=DIM, anchor="ma")
    return canvas


def caption(img, text):
    bar = 46
    canvas = Image.new("RGB", (img.width, img.height + bar), BG)
    canvas.paste(img, (0, bar))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, img.width, bar), fill=(11, 14, 20))
    draw.rectangle((0, 0, 4, bar), fill=(122, 162, 247))
    draw.text((18, 12), text, font=font(BOLD_NAMES, 19), fill=INK)
    return canvas


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    if not (DOCS / "before.html").is_file() or not (DOCS / "after.html").is_file():
        raise SystemExit("run tools/gen_demo_pages.py first: docs/ is missing the reports")

    rects = capture([("before", DOCS / "before.html"), ("after", DOCS / "after.html")])
    for name, entry in rects.items():
        print("%s: viewport %s assertions %s" % (name, entry["viewport"], entry["assertions"]))

    panels = {name: crop_panel(entry) for name, entry in rects.items()}
    labels = {
        "before": ("BEFORE \u00b7 exit 1 \u00b7 2 FAIL", RED),
        "after": ("AFTER \u00b7 exit 0 \u00b7 3 PASS", GREEN),
    }

    wide = compose(1600, 900, 44, 560, 240, panels, labels)
    wide.save(ASSETS / "before-after.png")
    wide.save(DOCS / "before-after.png")

    card = compose(1280, 640, 40, 400, 170, panels, labels)
    card.save(ASSETS / "social-preview.png")

    stdout = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "hero.py")],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "AGENTTWIN_HERO_OUT": str(WORK / "hero"), "PYTHONPATH": str(ROOT / "src")},
    ).stdout.splitlines()
    lines = [line for line in stdout if line.strip()][:16]

    size = (1000, 616)
    frames = [
        caption(wide.resize((1000, 562), Image.LANCZOS), "1/3  one tape, two replays: exit 1 then exit 0"),
        caption(terminal_frame(lines, size), "2/3  the whole loop is one command"),
        caption(promise_frame(size), "3/3  offline, no key, no model call"),
    ]
    frames = [frame.quantize(colors=160) for frame in frames]
    frames[0].save(
        ASSETS / "hero-short.gif",
        save_all=True, append_images=frames[1:], duration=[2200, 2000, 1800],
        loop=0, optimize=True,
    )

    for name in ("before-after.png", "social-preview.png", "hero-short.gif"):
        path = ASSETS / name
        print("assets/%-20s %7d bytes" % (name, path.stat().st_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
