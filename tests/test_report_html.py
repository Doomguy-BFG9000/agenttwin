"""Contract tests for Report.html() — the single-file offline replayer.

CONTRACTS.md §8: one placeholder, one file, no external references, and the
payload keyed exactly {trace_id, mode, header, baseline, current, results, diff,
labels}. The '</script>' case is the reason the payload escapes '<': a recorded
model response is attacker-controlled text and must never close the script block.
"""

from __future__ import annotations

import json

import pytest

import agenttwin.report as report_module
from agenttwin.report import Report
from agenttwin.trace import SCHEMA_VERSION, Header, Trace

PAYLOAD_TAG = '<script type="application/json" id="agenttwin-payload">'
CLOSE_TAG = "</script>"
PAYLOAD_KEYS = {"trace_id", "mode", "header", "baseline", "current", "results", "diff", "labels"}
MAX_BYTES = 400 * 1024


def make_header(trace_id: str = "checkout", redacted: int = 2) -> Header:
    return Header(
        schema_version=SCHEMA_VERSION,
        trace_id=trace_id,
        seed=0,
        env_fingerprint={
            "model": "gpt-4o-mini",
            "prompt_hash": "deadbeef",
            "python": "3.12",
            "agenttwin": "0.1.0",
            "platform": "win32",
            "deps": {"httpx": "0.27.0"},
        },
        redaction={"redacted": redacted, "policy": {}, "mode": "scrub"},
    )


def make_trace(trace_id: str = "checkout", *, response: str = "hello", count: int = 4) -> Trace:
    """count events: repeated llm_call/tool_call/tool_result triples plus a terminal."""
    trace = Trace(header=make_header(trace_id))
    for index in range(count - 1):
        phase = index % 3
        if phase == 0:
            trace.append("llm_call", "planner", {"prompt_hash": f"p{index}", "response": response})
        elif phase == 1:
            trace.append("tool_call", "shell", {"tool": "shell", "args_hash": f"a{index}"})
        else:
            trace.append("tool_result", "shell", {"tool": "shell", "result": "ok"})
    trace.append("terminal", "planner", {"state": {"done": True}, "state_hash": "z"})
    return trace


def make_report(*, response: str = "hello", count: int = 4, **kwargs) -> Report:
    baseline = make_trace(response=response, count=count)
    return Report(baseline=baseline, trace=baseline.copy(), **kwargs)


def read_payload(path) -> dict:
    text = path.read_text(encoding="utf-8")
    start = text.index(PAYLOAD_TAG) + len(PAYLOAD_TAG)
    end = text.index(CLOSE_TAG, start)
    return json.loads(text[start:end])


def test_html_writes_one_file_and_returns_its_path(tape_dir):
    report = make_report()
    path = tape_dir / "out" / "report.html"
    result = report.html(path)
    assert result == path
    assert path.is_file()
    assert path.stat().st_size > 0
    assert [item.name for item in path.parent.iterdir()] == ["report.html"]


def test_html_replaces_the_placeholder_exactly_once(tape_dir):
    path = make_report().html(tape_dir / "report.html")
    text = path.read_text(encoding="utf-8")
    assert report_module.PLACEHOLDER not in text
    assert text.count(PAYLOAD_TAG) == 1


def test_html_payload_has_the_frozen_key_set_and_event_lists(tape_dir):
    report = make_report().assert_()
    payload = read_payload(report.html(tape_dir / "report.html"))
    assert set(payload) == PAYLOAD_KEYS
    assert payload["trace_id"] == "checkout"
    assert payload["mode"] == "replay"
    assert payload["labels"] == {"baseline": "recorded", "current": "replay"}
    assert payload["header"] == report.trace.header.to_json()
    assert payload["baseline"] == [event.to_json() for event in report.baseline.events]
    assert payload["current"] == [event.to_json() for event in report.trace.events]


def test_html_payload_diff_is_seq_aligned(tape_dir):
    report = make_report()
    payload = read_payload(report.html(tape_dir / "report.html"))
    assert set(payload["diff"]) == {"first_divergence_seq", "events"}
    assert payload["diff"]["first_divergence_seq"] is None
    assert [item["seq"] for item in payload["diff"]["events"]] == [0, 1, 2, 3]


def test_html_payload_results_match_json(tape_dir):
    report = make_report().assert_()
    payload = read_payload(report.html(tape_dir / "report.html"))
    assert payload["results"] == report.json()["results"]


def test_html_contains_no_external_urls(tape_dir):
    text = make_report().html(tape_dir / "report.html").read_text(encoding="utf-8").lower()
    assert "http://" not in text
    assert "https://" not in text


def test_html_stays_under_400_kb_for_a_20_event_trace(tape_dir):
    report = make_report(count=20)
    assert len(report.trace.events) == 20
    path = report.html(tape_dir / "report.html")
    assert path.stat().st_size < MAX_BYTES


def test_html_escapes_a_script_breakout_in_a_response(tape_dir):
    breakout = "</script><script>alert(1)</script>"
    report = make_report(response=breakout)
    path = report.html(tape_dir / "report.html")
    text = path.read_text(encoding="utf-8")
    assert "</script><script>" not in text
    assert "\\u003c/script>" in text
    assert read_payload(path)["current"][0]["payload"]["response"] == breakout


def test_html_keeps_non_ascii_readable(tape_dir):
    path = make_report(response="こんにちは").html(tape_dir / "report.html")
    text = path.read_text(encoding="utf-8")
    assert "こんにちは" in text
    assert "\\u3053" not in text


def test_html_raises_when_the_placeholder_is_missing(tape_dir, monkeypatch):
    template = tape_dir / "no-placeholder.html"
    template.write_text("<html></html>", encoding="utf-8", newline="")
    monkeypatch.setattr(report_module, "TEMPLATE_PATH", template)
    with pytest.raises(RuntimeError) as excinfo:
        make_report().html(tape_dir / "report.html")
    assert "found 0" in str(excinfo.value)


def test_html_raises_when_the_placeholder_is_duplicated(tape_dir, monkeypatch):
    template = tape_dir / "duplicated.html"
    template.write_text(report_module.PLACEHOLDER * 2, encoding="utf-8", newline="")
    monkeypatch.setattr(report_module, "TEMPLATE_PATH", template)
    with pytest.raises(RuntimeError) as excinfo:
        make_report().html(tape_dir / "report.html")
    assert "found 2" in str(excinfo.value)
