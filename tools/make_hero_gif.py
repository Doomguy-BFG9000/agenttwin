"""Build the hero GIF: two reports, four screenshots, one animated file.

Dev-only.  Needs Edge, selenium (which fetches msedgedriver itself) and Pillow;
neither is a package dependency and CI never runs this.  Regenerate the asset
whenever the replayer UI changes:

    PYTHONPATH=src python tools/make_hero_gif.py
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
HERO = ROOT / "build" / "hero"
FRAMES = HERO / "frames"
ASSET = ROOT / "assets" / "hero.gif"

EDGE = ("--headless=new", "--window-size=1280,860", "--force-device-scale-factor=1",
        "--disable-gpu", "--hide-scrollbars", "--lang=en-US")

STEPS = (
    ("01-before.png", "1/4  one recorded tape, replayed offline: exit 1, two assertions fail", "#f85149"),
    ("02-before-jump.png", "2/4  click a failing assertion: the viewer jumps to seq 12", "#f85149"),
    ("03-after.png", "3/4  same tape, one policy change: exit 0, every assertion passes", "#3fb950"),
    ("04-after-diff.png", "4/4  the trajectories diverge at seq 3, and the diff says where", "#3fb950"),
)
DURATIONS = (2600, 2600, 2800, 3200)
WIDTH = 1000
CAPTION_H = 42


def run_hero() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, str(ROOT / "examples" / "hero.py")],
                   cwd=ROOT, check=True, env=env)


def shoot() -> None:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.edge.options import Options

    FRAMES.mkdir(parents=True, exist_ok=True)
    opts = Options()
    for arg in EDGE:
        opts.add_argument(arg)
    driver = webdriver.Edge(options=opts)
    try:
        driver.set_window_size(1280, 860)
        page = HERO / "before.html"
        driver.get(page.as_uri())
        time.sleep(1.2)
        driver.save_screenshot(str(FRAMES / "01-before.png"))

        failing = driver.find_elements(By.CSS_SELECTOR, ".assert-row.is-jumpable")
        if not failing:
            raise SystemExit("no failing assertion row to click: is the tape still failing?")
        failing[0].click()
        time.sleep(0.8)
        driver.save_screenshot(str(FRAMES / "02-before-jump.png"))

        driver.get((HERO / "after.html").as_uri())
        time.sleep(1.2)
        driver.save_screenshot(str(FRAMES / "03-after.png"))

        heads = driver.find_elements(By.CSS_SELECTOR, ".diff-head")
        if heads:
            heads[0].click()
            time.sleep(0.6)
        driver.execute_script("window.scrollTo(0, 0)")
        time.sleep(0.3)
        driver.save_screenshot(str(FRAMES / "04-after-diff.png"))
    finally:
        driver.quit()


def compose() -> None:
    title_font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 19)
    frames = []
    for name, caption, accent in STEPS:
        shot = Image.open(FRAMES / name).convert("RGB")
        height = round(shot.height * WIDTH / shot.width)
        shot = shot.resize((WIDTH, height), Image.LANCZOS)
        canvas = Image.new("RGB", (WIDTH, height + CAPTION_H), "#0b0e14")
        canvas.paste(shot, (0, CAPTION_H))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, 4, CAPTION_H - 1], fill=accent)
        draw.text((16, CAPTION_H // 2 - 11), caption, font=title_font, fill="#e6edf3")
        frames.append(canvas.quantize(colors=160, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG))

    ASSET.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(ASSET, save_all=True, append_images=frames[1:], duration=DURATIONS,
                   loop=0, optimize=True, disposal=2)
    print(f"{ASSET.relative_to(ROOT)}  {ASSET.stat().st_size} bytes  {WIDTH}x{frames[0].height}")


def main() -> int:
    run_hero()
    shoot()
    compose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
