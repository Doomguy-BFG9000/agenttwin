"""Contract tests for the standalone HTML replayer (CONTRACTS.md section 8)."""

from __future__ import annotations

import importlib.util
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "src" / "agenttwin" / "html"
TEMPLATE_PATH = PACKAGE_DIR / "replayer.html"
PLACEHOLDER = "__AGENTTWIN_PAYLOAD__"
DICT_IDS = ("agenttwin-i18n-en", "agenttwin-i18n-zh")
TRANSLATED_ATTRIBUTES = (
    "data-i18n",
    "data-i18n-title",
    "data-i18n-aria-label",
    "data-i18n-placeholder",
)
CJK = re.compile(r"[\u4e00-\u9fff]")
PLACEHOLDER_TOKEN = re.compile(r"\{[a-z]+\}")
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


class ScriptTextParser(HTMLParser):
    """Collect the text of the embedded JSON script elements asked for."""

    def __init__(self, wanted: tuple[str, ...]) -> None:
        super().__init__()
        self.wanted = set(wanted)
        self.scripts: dict[str, str] = {}
        self._current: str | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag != "script":
            return
        attributes = dict(attrs)
        if attributes.get("id") in self.wanted and attributes.get("type") == "application/json":
            self._current = str(attributes["id"])

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self.scripts[self._current] = data
            self._current = None


class AttributeParser(HTMLParser):
    """Collect element ids and every data-i18n* translation key used in markup."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.keys: set[str] = set()

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(str(attributes["id"]))
        for name, value in attributes.items():
            if name in TRANSLATED_ATTRIBUTES and value:
                self.keys.add(str(value))


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
    assert len(template_text.splitlines()) < 1600


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

@pytest.fixture(scope="module")
def dictionaries(template_text: str) -> dict[str, dict[str, str]]:
    parser = ScriptTextParser(DICT_IDS)
    parser.feed(template_text)
    assert set(parser.scripts) == set(DICT_IDS), "missing language dictionary script"
    parsed = {name: json.loads(body) for name, body in parser.scripts.items()}
    for name, table in parsed.items():
        assert isinstance(table, dict), name + " dictionary is not a JSON object"
    return parsed


@pytest.fixture(scope="module")
def translated_keys(template_text: str) -> set[str]:
    parser = AttributeParser()
    parser.feed(template_text)
    assert "lang-toggle" in parser.ids, "the replayer has no language toggle"
    return parser.keys


def test_language_dictionaries_share_one_key_set(dictionaries) -> None:
    english = set(dictionaries["agenttwin-i18n-en"])
    chinese = set(dictionaries["agenttwin-i18n-zh"])
    assert len(english) >= 80
    assert english == chinese, "dictionary keys differ: " + repr(sorted(english ^ chinese))
    for key in sorted(english):
        english_tokens = set(PLACEHOLDER_TOKEN.findall(dictionaries["agenttwin-i18n-en"][key]))
        chinese_tokens = set(PLACEHOLDER_TOKEN.findall(dictionaries["agenttwin-i18n-zh"][key]))
        assert english_tokens == chinese_tokens, "placeholder mismatch at " + key


def test_every_translated_attribute_has_a_dictionary_key(dictionaries, translated_keys) -> None:
    assert translated_keys, "no data-i18n attributes found in the markup"
    known = set(dictionaries["agenttwin-i18n-en"])
    missing = sorted(translated_keys - known)
    assert not missing, "translation keys missing from the dictionaries: " + repr(missing)


def test_english_dictionary_is_english(dictionaries) -> None:
    english = dictionaries["agenttwin-i18n-en"]
    chinese_values = sorted(key for key, value in english.items() if CJK.search(value))
    assert chinese_values == ["lang.other"], chinese_values
    assert english["lang.other"] == "中文"


def test_chinese_dictionary_is_chinese(dictionaries) -> None:
    chinese = dictionaries["agenttwin-i18n-zh"]
    translated = [key for key, value in chinese.items() if CJK.search(value)]
    assert len(translated) >= 40, "chinese dictionary looks untranslated: " + str(len(translated))


def test_language_switch_updates_the_document(template_text: str) -> None:
    assert 'document.documentElement.setAttribute("lang", LANG)' in template_text
    assert 'setLang(LANG === "zh" ? "en" : "zh")' in template_text
    assert "agenttwin.lang" in template_text
    assert "preferredLang()" in template_text
