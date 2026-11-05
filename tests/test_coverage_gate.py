"""The coverage floor is a published promise, so it is guarded like one.

Both READMEs and the design notes state 82% of statements.  The number has
exactly one home (pyproject.toml) and one enforcer (the coverage job in
ci.yml); these tests keep the three from drifting apart, and keep the READMEs
from going back to claiming the gate does not exist.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
READMES = (ROOT / "README.md", ROOT / "README.zh-CN.md")

FLOOR = 82


def _load_pyproject() -> dict:
    text = PYPROJECT.read_text(encoding="utf-8")
    try:
        import tomllib
    except ImportError:  # Python 3.10
        section = re.search(r"\[tool\.coverage\.report\](.*?)(?=\n\[|\Z)", text, re.S)
        assert section, "pyproject.toml has no [tool.coverage.report] section"
        floor = re.search(r"fail_under\s*=\s*(\d+)", section.group(1))
        assert floor, "no fail_under in [tool.coverage.report]"
        run = re.search(r"\[tool\.coverage\.run\](.*?)(?=\n\[|\Z)", text, re.S)
        source = re.findall(r'"([^"]+)"', run.group(1)) if run else []
        return {"tool": {"coverage": {"report": {"fail_under": int(floor.group(1))},
                                      "run": {"source": source}}}}
    return tomllib.loads(text)


def test_the_promise_is_one_number_in_one_place() -> None:
    coverage = _load_pyproject()["tool"]["coverage"]
    assert coverage["report"]["fail_under"] == FLOOR
    assert coverage["run"]["source"] == ["src"], (
        "source must stay a path list: the package is not installed when the "
        "tests run, so a package name would silently measure nothing"
    )


def test_coverage_is_a_dev_dependency() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    dev = re.search(r"^dev\s*=\s*\[(.*?)\]", text, re.M | re.S)
    assert dev, "no dev extra in pyproject.toml"
    assert any(item.strip().startswith('"coverage') for item in dev.group(1).split(",")), (
        "the CI coverage job installs the yaml and dev extras, so coverage has to be in dev"
    )


def test_ci_runs_the_report_without_restating_the_floor() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"^  coverage:$", workflow, re.M), "no coverage job in ci.yml"
    assert "python -m coverage run -m pytest" in workflow
    assert re.search(r"^\s+run: python -m coverage report$", workflow, re.M), (
        "the job must end on a bare 'coverage report': the exit code is the gate"
    )
    assert "--fail-under" not in workflow, (
        "a second copy of the number in ci.yml is exactly how the two drift apart"
    )


@pytest.mark.parametrize("readme", READMES, ids=lambda path: path.name)
def test_both_readmes_state_the_same_floor(readme: Path) -> None:
    text = readme.read_text(encoding="utf-8")
    assert f"{FLOOR}%" in text, f"{readme.name} does not mention the floor"
    assert "Not done yet: a coverage gate" not in text
    assert "还没做：覆盖率门槛" not in text
