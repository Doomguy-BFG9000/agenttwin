"""The hero demo has to stay true to the two things that show it off.

The transcript in the README is the output of a real run, and the animation it
sits next to is committed.  Both drift silently otherwise.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "README.md"
ZH = ROOT / "README.zh-CN.md"
GIF = ROOT / "assets" / "hero.gif"
COMMAND = "PYTHONPATH=src python examples/hero.py"
TEXT_BLOCK = re.compile(r"```text\n(.*?)\n```", re.S)


def _hero_transcript(text: str) -> str:
    for block in TEXT_BLOCK.findall(text):
        if block.startswith("tape: deathloop.jsonl"):
            return block
    raise AssertionError("no hero transcript found: no text block starts with 'tape: deathloop.jsonl'")


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


def test_the_animation_is_committed_and_both_editions_embed_it() -> None:
    assert GIF.is_file(), "assets/hero.gif is missing"
    assert GIF.stat().st_size < 2 * 1024 * 1024, f"hero.gif is {GIF.stat().st_size} bytes"
    assert GIF.read_bytes()[:6] in (b"GIF87a", b"GIF89a"), "assets/hero.gif is not a GIF"
    needle = "assets/hero.gif"
    assert needle in EN.read_text(encoding="utf-8")
    assert needle in ZH.read_text(encoding="utf-8")


def test_the_demo_script_is_the_one_the_readme_tells_you_to_run() -> None:
    assert (ROOT / "examples" / "hero.py").is_file()
    assert COMMAND in EN.read_text(encoding="utf-8")
    assert COMMAND in ZH.read_text(encoding="utf-8")
