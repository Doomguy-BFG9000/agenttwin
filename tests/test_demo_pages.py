"""docs/ is the GitHub Pages demo: generated, offline, and never stale.

Both report pages come from examples/hero.py through tools/gen_demo_pages.py,
so a hand-edited or out-of-date docs/ directory is a bug: the first test below
regenerates the three pages into a scratch directory and holds the committed
bytes against the result.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SCRIPT = ROOT / "tools" / "gen_demo_pages.py"
EN = ROOT / "README.md"
ZH = ROOT / "README.zh-CN.md"
DEMO_URL = "https://ravenkernel.github.io/agenttwin/"
PAGES = ("before.html", "after.html", "index.html")
# Values that legitimately differ between a developer machine and CI; the page
# has to be reproducible everywhere else.
HOST_FIELDS = re.compile(r'"(python|platform|agenttwin)":\s*"[^"]*"')
SRC_ATTRIBUTE = re.compile(r'src="([^"]*)"')


def normalise(text: str) -> str:
    return HOST_FIELDS.sub(lambda m: f'"{m.group(1)}": "<host>"', text)


def test_the_committed_pages_are_what_the_generator_prints_today(tape_dir) -> None:
    env = {**os.environ, "AGENTTWIN_HERO_OUT": str(tape_dir), "PYTHONPATH": str(ROOT / "src")}
    done = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    for name in PAGES:
        committed = (DOCS / name).read_text(encoding="utf-8")
        fresh = (tape_dir / name).read_text(encoding="utf-8")
        assert normalise(fresh) == normalise(committed), (
            f"docs/{name} is stale: run python tools/gen_demo_pages.py"
        )


def test_the_index_links_both_reports_and_the_static_card() -> None:
    index = (DOCS / "index.html").read_text(encoding="utf-8")
    for needle in ('href="before.html"', 'href="after.html"', 'src="before-after.png"'):
        assert needle in index, f"docs/index.html does not contain {needle}"
    assert (DOCS / "before-after.png").is_file(), "docs/before-after.png is missing"


def test_the_demo_pages_load_nothing_from_the_network() -> None:
    for name in PAGES:
        page = (DOCS / name).read_text(encoding="utf-8")
        assert "@import" not in page, f"docs/{name} imports a stylesheet"
        assert "<script src=" not in page, f"docs/{name} loads a script"
        assert "<link" not in page, f"docs/{name} loads a linked resource"
        for source in SRC_ATTRIBUTE.findall(page):
            assert not source.startswith(("http://", "https://", "//")), (
                f"docs/{name} pulls {source} from the network"
            )


def test_the_reports_are_filled_in_not_the_empty_template() -> None:
    for name in ("before.html", "after.html"):
        page = (DOCS / name).read_text(encoding="utf-8")
        assert "__AGENTTWIN_PAYLOAD__" not in page, f"docs/{name} is the unfilled template"


def test_both_readmes_point_at_the_published_demo() -> None:
    assert DEMO_URL in EN.read_text(encoding="utf-8"), "README.md never links the online demo"
    assert DEMO_URL in ZH.read_text(encoding="utf-8"), "README.zh-CN.md never links the online demo"
