"""The two README editions have to stay in lockstep.

The code blocks are byte-identical on purpose: they are transcripts of real
runs against this repository, so a translated transcript would be a lie.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "README.md"
ZH = ROOT / "README.zh-CN.md"
SWITCH = "[English](README.md) · [中文](README.zh-CN.md)"
EXPECTED_BLOCKS = 13
EXPECTED_SECTIONS = 8
FENCE = "```"
CJK = re.compile(r"[\u4e00-\u9fff]")


def _read(path: Path) -> str:
    assert path.is_file(), f"{path.name} is missing from the repository root"
    return path.read_text(encoding="utf-8")


def _blocks(text: str) -> list[str]:
    """Fenced code blocks in document order, fence lines included."""
    out: list[str] = []
    current: list[str] = []
    inside = False
    for line in text.split("\n"):
        if line.startswith(FENCE):
            current.append(line)
            if inside:
                out.append("\n".join(current))
                current, inside = [], False
            else:
                inside = True
            continue
        if inside:
            current.append(line)
    assert not inside, "unterminated fenced code block"
    return out


def _prose(text: str) -> str:
    """The text with fenced blocks and the shared switch line removed."""
    kept: list[str] = []
    inside = False
    for index, line in enumerate(text.split("\n")):
        if line.startswith(FENCE):
            inside = not inside
            continue
        if inside or index == 0:
            continue
        kept.append(line)
    return "\n".join(kept)


def test_both_editions_exist_and_link_to_each_other():
    english, chinese = _read(EN), _read(ZH)
    assert english.split("\n")[0] == SWITCH
    assert chinese.split("\n")[0] == SWITCH
    assert "](README.md)" in english and "](README.zh-CN.md)" in english
    assert "](README.md)" in chinese and "](README.zh-CN.md)" in chinese


def test_code_blocks_are_byte_identical():
    english, chinese = _blocks(_read(EN)), _blocks(_read(ZH))
    assert len(english) == EXPECTED_BLOCKS, f"expected {EXPECTED_BLOCKS} fenced blocks in README.md, found {len(english)}"
    assert len(chinese) == len(english), f"README.zh-CN.md has {len(chinese)} fenced blocks, README.md has {len(english)}"
    for number, (left, right) in enumerate(zip(english, chinese), 1):
        if left == right:
            continue
        left_lines, right_lines = left.split("\n"), right.split("\n")
        for offset, (a, b) in enumerate(zip(left_lines, right_lines), 1):
            if a != b:
                raise AssertionError(
                    f"block {number} differs at line {offset} of the block:\n"
                    f"  README.md:      {a!r}\n"
                    f"  README.zh-CN.md:{b!r}"
                )
        raise AssertionError(
            f"block {number} differs in length: {len(left_lines)} vs {len(right_lines)} lines"
        )


def test_english_prose_has_no_chinese():
    text = _prose(_read(EN))
    hits = sorted({match.group(0) for match in CJK.finditer(text)})
    assert not hits, f"README.md still contains Chinese characters outside code blocks: {''.join(hits[:20])}"


def test_chinese_edition_keeps_the_chinese():
    lines = [line for line in _prose(_read(ZH)).split("\n") if CJK.search(line)]
    assert len(lines) >= 50, f"README.zh-CN.md has only {len(lines)} lines of Chinese prose"


def test_both_editions_have_the_same_sections():
    english = [line for line in _read(EN).split("\n") if line.startswith("## ")]
    chinese = [line for line in _read(ZH).split("\n") if line.startswith("## ")]
    assert len(english) == EXPECTED_SECTIONS, f"README.md has {len(english)} level-2 sections"
    assert len(chinese) == len(english), f"README.zh-CN.md has {len(chinese)} level-2 sections, README.md has {len(english)}"


def test_pyproject_declares_the_english_readme():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if sys.version_info >= (3, 11):
        import tomllib

        project = tomllib.loads(text)["project"]
        assert project.get("readme") == "README.md", project.get("readme")
    else:  # pragma: no cover - exercised only on 3.10
        assert re.search(r'(?m)^readme\s*=\s*"README\.md"', text)
