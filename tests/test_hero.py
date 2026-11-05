"""The hero demo has to stay true to the things that show it off.

The transcript in the README is the output of a real run, and the images it
sits next to are committed.  Both drift silently otherwise.
"""
from __future__ import annotations

import os
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "README.md"
ZH = ROOT / "README.zh-CN.md"
GIFS = ("assets/hero-short.gif", "assets/hero-full.gif")
CARDS = {"assets/before-after.png": (1600, 900), "assets/social-preview.png": (1280, 640)}
COMMAND = "PYTHONPATH=src python examples/hero.py"
TEXT_BLOCK = re.compile(r"```text\n(.*?)\n```", re.S)


def _hero_transcript(text: str) -> str:
    for block in TEXT_BLOCK.findall(text):
        if block.startswith("tape: deathloop.jsonl"):
            return block
    raise AssertionError("no hero transcript found: no text block starts with 'tape: deathloop.jsonl'")


def _png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    return struct.unpack(">II", header[16:24])


def test_the_readme_transcript_is_the_output_of_a_real_run() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    done = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "hero.py")],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip() == _hero_transcript(EN.read_text(encoding="utf-8")), (
        "README.md shows a hero transcript that is not what examples/hero.py prints today"
    )
    assert done.stdout.strip() == _hero_transcript(ZH.read_text(encoding="utf-8"))


def test_the_animations_are_committed_and_both_editions_embed_them() -> None:
    english = EN.read_text(encoding="utf-8")
    chinese = ZH.read_text(encoding="utf-8")
    for needle in GIFS:
        path = ROOT / needle
        assert path.is_file(), f"{needle} is missing"
        assert path.stat().st_size < 2 * 1024 * 1024, f"{needle} is {path.stat().st_size} bytes"
        assert path.read_bytes()[:6] in (b"GIF87a", b"GIF89a"), f"{needle} is not a GIF"
        assert needle in english, f"README.md never shows {needle}"
        assert needle in chinese, f"README.zh-CN.md never shows {needle}"


def test_the_static_cards_keep_the_sizes_the_hosts_ask_for() -> None:
    for name, size in CARDS.items():
        path = ROOT / name
        assert path.is_file(), f"{name} is missing"
        assert _png_size(path) == size, f"{name} is {_png_size(path)}, expected {size}"
        assert path.stat().st_size < 1024 * 1024, f"{name} is {path.stat().st_size} bytes"


def test_the_demo_script_is_the_one_the_readme_tells_you_to_run() -> None:
    assert (ROOT / "examples" / "hero.py").is_file()
    assert COMMAND in EN.read_text(encoding="utf-8")
    assert COMMAND in ZH.read_text(encoding="utf-8")
