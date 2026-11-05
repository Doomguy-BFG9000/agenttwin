"""Contract tests for the standalone HTML replayer (CONTRACTS.md section 8)."""

from __future__ import annotations

import importlib.util
import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "src" / "agenttwin" / "html"
TEMPLATE_PATH = PACKAGE_DIR / "replayer.html"
PLACEHOLDER = "__AGENTTWIN_PAYLOAD__"
PAYLOAD_TAG = '<script type="application/json" id="agenttwin-payload">'

REQUIRED_HOOKS = (
    'id="agenttwin-payload"',
    'id="shell"',
    'id="error-panel"',
    'id="error-message"',
    'id="trace-id"',
    'id="mode-chip"',
    'id="theme-toggle"',
    'id="btn-play"',
    'id="btn-pause"',
    'id="btn-prev"',
    'id="btn-next"',
    'id="speed"',
    'id="scrubber"',
    'id="status"',
    'id="filter-kinds"',
    'id="actor-filter"',
    'id="filter-reset"',
    'id="label-baseline"',
    'id="label-current"',
    'id="count-baseline"',
    'id="count-current"',
    'id="col-baseline"',
    'id="col-current"',
    'id="filter-note"',
    'id="assertions"',
    'id="assertion-summary"',
    'id="state-diff"',
    'id="diff-summary"',
    'id="header-fingerprint"',
    'id="fp-line"',
    'id="notice"',
)

REQUIRED_MARKERS = (
    'value="0.5"',
    'value="1"',
    'value="2"',
    'value="4"',
    'data-theme="dark"',
    'data-theme="light"',
    "chip-patched",
    "chip-added",
    "chip-removed",
    "chip-changed",
    "is-patched",
    "empty-slot",
    "is-flash",
    "Spacebar",
    "ArrowLeft",
    "ArrowRight",
    "first_offense_seq",
    "JSON.parse",
)

SAMPLE_PAYLOAD = {
    "trace_id": "t-1",
    "mode": "replay",
    "header": {"schema_version": "0.1"},
    "baseline": [],
    "current": [],
    "results": [],
    "diff": {"first_divergence_seq": None, "events": []},
    "labels": {"baseline": "recorded", "current": "replay"},
}


@pytest.fixture(scope="module")
def template_text() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def html_module():
    spec = importlib.util.spec_from_file_location(
        "agenttwin_html_template", PACKAGE_DIR / "__init__.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PayloadParser(HTMLParser):
    """Collect the text of the embedded JSON payload script."""

    def __init__(self) -> None:
        super().__init__()
        self.attributes: dict[str, str | None] = {}
        self.payload: str | None = None
        self._capture = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "script" and dict(attrs).get("id") == "agenttwin-payload":
            self.attributes = dict(attrs)
            self._capture = True

    def handle_data(self, data: str) -> None:
        if self._capture:
            self.payload = data
            self._capture = False


def test_template_is_utf8_with_lf_newlines() -> None:
    raw = TEMPLATE_PATH.read_bytes()
    assert raw, "replayer.html is empty"
    text = raw.decode("utf-8")
    assert "\r" not in text


def test_payload_placeholder_appears_exactly_once(template_text: str) -> None:
    assert template_text.count(PLACEHOLDER) == 1
    assert PAYLOAD_TAG + PLACEHOLDER + "</script>" in template_text


def test_no_external_resources(template_text: str) -> None:
    lowered = template_text.lower()
    for token in ("http://", "https://", "@import", "<link", "src=", "cdn."):
        assert token not in lowered, "external reference found: " + token


def test_required_hooks_and_markers_are_present(template_text: str) -> None:
    missing = [hook for hook in REQUIRED_HOOKS if hook not in template_text]
    missing += [marker for marker in REQUIRED_MARKERS if marker not in template_text]
    assert not missing, "missing from replayer.html: " + repr(missing)


def test_template_stays_small(template_text: str) -> None:
    assert TEMPLATE_PATH.stat().st_size < 200 * 1024
    assert len(template_text.splitlines()) < 1200


def test_load_template_returns_the_file(html_module, template_text: str) -> None:
    assert html_module.load_template() == template_text


def test_template_path_points_at_the_replayer(html_module) -> None:
    path = Path(html_module.TEMPLATE_PATH)
    assert path.is_file()
    assert path.resolve() == TEMPLATE_PATH.resolve()


def test_substituting_the_payload_yields_parseable_json(template_text: str) -> None:
    rendered = template_text.replace(PLACEHOLDER, json.dumps(SAMPLE_PAYLOAD), 1)
    parser = PayloadParser()
    parser.feed(rendered)
    assert parser.attributes.get("type") == "application/json"
    assert parser.payload is not None, "payload script has no text content"
    assert json.loads(parser.payload) == SAMPLE_PAYLOAD


def test_payload_failure_is_reported_without_a_generic_prefix(template_text: str) -> None:
    start = template_text.index("function boot()")
    boot_body = template_text[start : template_text.index("\n  }\n", start)]
    assert "data = normalize(readPayload());" in boot_body
    assert "} catch (error) {" in boot_body
    assert "showFatal(" in boot_body
    assert "Unexpected error" not in boot_body
    assert template_text.count("Unexpected error while rendering this trace: ") == 1
