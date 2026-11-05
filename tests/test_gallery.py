"""The gallery story end to end: one assertion set, FAIL before / PASS after.

Design doc section 8 makes the gallery the first flagship artefact, so these tests
are not smoke tests: they judge each committed tape with the very controller that
recorded it (CONTRACTS section 3 -- Record and Replay inject their own llm/tools,
so the same callable runs in both modes), and they pin the claims the gallery
makes: deathloop fails the shared assertions, repaired passes them, both replay
offline, and the committed bytes are exactly what the build script records.

CONTRACTS 9.17: pytest tmp_path is unusable here, every scratch path goes through
the tape_dir fixture from tests/conftest.py.
"""

from __future__ import annotations

import importlib.util
import json
import re
import socket
from pathlib import Path

import pytest

from agenttwin import Twin, gallery
from agenttwin.redact import DEFAULT_PATTERNS
from agenttwin.trace import Trace

REPO_ROOT = Path(__file__).resolve().parents[1]

# The assertion names the gallery ships; spelled out so a rename is a test failure
# rather than a silently different story.
EXPECTED_ASSERTIONS = (
    "max_repeats(shell, n=3)",
    "budget_within(calls<=8)",
    "never_calls(shell, args_contains='rm -rf')",
)


def judged(fixture: gallery.Fixture):
    """Load, replay and assert exactly as the CLI does: one call, one verdict."""
    report = Twin.load(fixture.tape).replay(fixture.agent)
    report.assert_(*fixture.asserts)
    return report


def build_script():
    """Import build/gen_fixtures.py, the single producer of the committed tapes."""
    path = REPO_ROOT / "build" / "gen_fixtures.py"
    spec = importlib.util.spec_from_file_location("gen_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pattern_sources():
    """DEFAULT_PATTERNS as (name, regex) pairs, whether or not they are precompiled."""
    for name, pattern in DEFAULT_PATTERNS:
        yield name, getattr(pattern, "pattern", pattern)


def scrubbed_payloads(path: Path) -> list:
    """Every parsed line with *_hash values replaced, so no sha256 counts as a leak.

    CONTRACTS 9.20 pins the hash as computed over the pre-redaction request and
    args, so a bare hex digest is contract machinery, not redactable content; the
    long_hex default pattern would otherwise match every tape ever written.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]

    def strip(value):
        if isinstance(value, list):
            return [strip(item) for item in value]
        if isinstance(value, dict):
            return {
                key: "<hash>" if key.endswith("_hash") else strip(item)
                for key, item in value.items()
            }
        if isinstance(value, str):
            return re.sub(r"\b[0-9a-f]{64}\b", "<hash>", value)
        return value

    return [strip(payload) for payload in payloads]


def test_ids_are_unique_and_every_filename_matches_its_id():
    fixtures = gallery.list_fixtures()
    ids = [fixture.id for fixture in fixtures]
    assert ids == ["deathloop", "repaired"]
    assert len(set(ids)) == len(ids)
    for fixture in fixtures:
        assert fixture.tape.name == fixture.id + ".jsonl"
        assert fixture.tape.parent == gallery.FIXTURE_DIR
        assert fixture.tape.is_absolute()
        assert fixture.tape.is_file()
        assert isinstance(fixture.asserts, tuple)
        assert isinstance(fixture.commands, tuple)
        assert callable(fixture.agent)
        assert fixture.title and fixture.description


def test_get_returns_the_gallery_entry_and_raises_keyerror_for_an_unknown_id():
    assert gallery.get("deathloop") is gallery.get("deathloop")
    assert {fixture.id for fixture in gallery.list_fixtures()} == {"deathloop", "repaired"}
    with pytest.raises(KeyError):
        gallery.get("nope")
    with pytest.raises(KeyError):
        gallery.get("")


def test_the_two_fixtures_share_one_assertion_set_and_two_controllers():
    deathloop = gallery.get("deathloop")
    repaired = gallery.get("repaired")
    assert deathloop.asserts == repaired.asserts
    assert tuple(spec.name for spec in deathloop.asserts) == EXPECTED_ASSERTIONS
    assert deathloop.agent is gallery.deathloop_agent
    assert repaired.agent is gallery.repaired_agent
    assert deathloop.agent is not repaired.agent


def test_deathloop_repeats_one_shell_call_and_fails_the_shared_assertions():
    fixture = gallery.get("deathloop")
    tape = Trace.load(fixture.tape)
    report = judged(fixture)

    assert report.exit_code == 1
    assert report.divergent is False
    assert report.misses == []

    shell_calls = [
        event
        for event in tape.events
        if event.kind == "tool_call" and event.payload["tool"] == "shell"
    ]
    assert len(shell_calls) == 6
    assert len({event.payload["args_hash"] for event in shell_calls}) == 1

    calls = [event.seq for event in tape.events if event.kind in ("llm_call", "tool_call")]
    rows = {result.name: result for result in report.results}
    assert rows["max_repeats(shell, n=3)"].passed is False
    assert rows["max_repeats(shell, n=3)"].first_offense_seq == shell_calls[3].seq
    assert rows["budget_within(calls<=8)"].passed is False
    assert rows["budget_within(calls<=8)"].first_offense_seq == calls[8]
    assert rows["never_calls(shell, args_contains='rm -rf')"].passed is True
    assert rows["never_calls(shell, args_contains='rm -rf')"].first_offense_seq is None
    assert [result.name for result in report.results] == list(EXPECTED_ASSERTIONS)


def test_repaired_passes_the_identical_assertions_without_touching_the_shell():
    fixture = gallery.get("repaired")
    report = judged(fixture)

    assert report.exit_code == 0
    assert report.divergent is False
    assert report.misses == []
    assert [result.passed for result in report.results] == [True, True, True]
    assert all(result.first_offense_seq is None for result in report.results)
    assert [
        event for event in report.trace.events if event.kind == "tool_call"
    ] == [
        event
        for event in report.trace.events
        if event.kind == "tool_call" and event.payload["tool"] == "repo_status"
    ]
    assert not any(
        event.payload["tool"] == "shell"
        for event in report.trace.events
        if event.kind == "tool_call"
    )


@pytest.mark.parametrize("fixture_id", ["deathloop", "repaired"])
def test_replay_reproduces_the_recorded_events_payload_for_payload(fixture_id):
    fixture = gallery.get(fixture_id)
    tape = Trace.load(fixture.tape)
    report = Twin.load(fixture.tape).replay(fixture.agent)

    assert report.divergent is False
    assert [(event.kind, event.actor, event.payload) for event in report.trace.events] == [
        (event.kind, event.actor, event.payload) for event in tape.events
    ]


@pytest.mark.parametrize("fixture_id", ["deathloop", "repaired"])
def test_replay_needs_no_socket_at_all(fixture_id, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("replay opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert judged(gallery.get(fixture_id)).divergent is False


@pytest.mark.parametrize("fixture_id", ["deathloop", "repaired"])
def test_committed_bytes_carry_no_default_redaction_match_and_no_path(fixture_id):
    path = gallery.get(fixture_id).tape
    text = path.read_text(encoding="utf-8")

    assert text.endswith("\n")
    assert "\r" not in text
    lines = text.splitlines()
    assert all(line.strip() for line in lines)
    assert json.loads(lines[0])["redaction"]["redacted"] == 0

    payloads = scrubbed_payloads(path)
    assert payloads[0]["schema_version"] == "0.1"
    scanned = json.dumps(payloads, sort_keys=True)
    for name, pattern in pattern_sources():
        assert re.search(pattern, scanned) is None, name

    # No absolute path of any flavour, independent of the default patterns.
    assert re.search(r"[A-Za-z]:[\\/]", scanned) is None
    assert re.search(r"/(?:Users|home)/", scanned) is None
    # Nothing that looks like wall-clock state (CONTRACTS section 1 has no such field).
    for payload in payloads:
        for key in payload:
            assert key.lower() not in {"timestamp", "time", "ts", "date", "created_at"}


@pytest.mark.parametrize("fixture_id", ["deathloop", "repaired"])
def test_the_build_script_regenerates_the_committed_bytes_exactly(fixture_id, tape_dir):
    fixture = gallery.get(fixture_id)
    regenerated = build_script().record(fixture, out_dir=tape_dir)

    assert regenerated.read_bytes() == fixture.tape.read_bytes()
    assert regenerated.stat().st_size < 8 * 1024
