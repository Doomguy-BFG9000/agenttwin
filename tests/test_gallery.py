"""The gallery story end to end: five pairs, one assertion set per pair, FAIL then PASS.

Design doc section 8 makes the gallery the first flagship artefact, so these tests
are not smoke tests: they judge each committed tape with the very controller that
recorded it (CONTRACTS section 3 -- Record and Replay inject their own llm/tools,
so the same callable runs in both modes), and they pin the claims the gallery
makes: every before tape fails exactly the assertions GALLERY_CLAIMS names, every
after tape passes all of them while covering the same assertion family, each tape
replays offline, and every committed line is still what tools/gen_fixtures.py
records -- the machine fingerprint aside, which CONTRACTS 9.21 keeps non-fatal.

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
from agenttwin.trace import Trace, canonical_json

REPO_ROOT = Path(__file__).resolve().parents[1]

# The assertion names the gallery ships; spelled out so a rename is a test failure
# rather than a silently different story.
EXPECTED_ASSERTIONS = (
    "max_repeats(shell, n=3)",
    "budget_within(calls<=8)",
    "never_calls(shell, args_contains='rm -rf')",
)


# Gallery order is presentation order: each story's before tape, then its after tape.
ALL_IDS = [
    "deathloop",
    "repaired",
    "cleanup",
    "cleanup_guarded",
    "write_first",
    "search_first",
    "partial_receipt",
    "full_receipt",
    "double_shipping",
    "correct_shipping",
]

# Every pair is one controller under two policies, so the pair must be judged by
# one shared assertion tuple -- not by two tuples that happen to be equal today.
PAIRS = [
    ("deathloop", "repaired"),
    ("cleanup", "cleanup_guarded"),
    ("write_first", "search_first"),
    ("partial_receipt", "full_receipt"),
    ("double_shipping", "correct_shipping"),
]

# The verdict each tape publishes: exit code and the assertions it is meant to fail.
# Spelled out here so a drifting story is a failing test, not a re-recorded tape.
GALLERY_CLAIMS = {
    "deathloop": (1, {"max_repeats(shell, n=3)", "budget_within(calls<=8)"}),
    "repaired": (0, set()),
    "cleanup": (1, {"never_calls(shell, args_contains='rm -rf')"}),
    "cleanup_guarded": (0, set()),
    "write_first": (1, {"call_order(search>write)"}),
    "search_first": (0, set()),
    "partial_receipt": (1, {"output_schema(object)"}),
    "full_receipt": (0, set()),
    "double_shipping": (1, {"state_invariant(/total)"}),
    "correct_shipping": (0, set()),
}


def judged(fixture: gallery.Fixture):
    """Load, replay and assert exactly as the CLI does: one call, one verdict."""
    report = Twin.load(fixture.tape).replay(fixture.agent)
    report.assert_(*fixture.asserts)
    return report


def generator():
    """Import tools/gen_fixtures.py, the single producer of the committed tapes."""
    path = REPO_ROOT / "tools" / "gen_fixtures.py"
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
    assert ids == ALL_IDS
    assert len(set(ids)) == len(ids)
    for fixture in fixtures:
        assert fixture.tape.name == fixture.id + ".jsonl"
        assert fixture.tape.parent == gallery.FIXTURE_DIR
        assert fixture.tape.is_absolute()
        assert fixture.tape.is_file()
        assert isinstance(fixture.asserts, tuple)
        assert isinstance(fixture.commands, tuple)
        assert callable(fixture.agent)
        assert callable(fixture.llm)
        assert callable(fixture.make_tools)
        tools = fixture.make_tools()
        assert tools and all(callable(tool) for tool in tools.values())
        assert fixture.model
        assert fixture.title and fixture.description


def test_get_returns_the_gallery_entry_and_raises_keyerror_for_an_unknown_id():
    assert gallery.get("deathloop") is gallery.get("deathloop")
    assert {fixture.id for fixture in gallery.list_fixtures()} == set(ALL_IDS)
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

def test_every_pair_is_one_assertion_set_one_llm_and_two_controllers():
    for before_id, after_id in PAIRS:
        before = gallery.get(before_id)
        after = gallery.get(after_id)
        assert before.asserts is after.asserts, before_id
        assert before.llm is after.llm, before_id
        assert before.make_tools is after.make_tools, before_id
        assert before.model == after.model, before_id
        assert before.agent is not after.agent, before_id


@pytest.mark.parametrize("fixture_id", ALL_IDS)
def test_each_tape_matches_its_published_verdict(fixture_id):
    expected_exit, expected_failures = GALLERY_CLAIMS[fixture_id]
    report = judged(gallery.get(fixture_id))

    assert report.exit_code == expected_exit
    assert report.divergent is False
    assert report.misses == []
    assert report.drifts == []
    failed = {result.name for result in report.results if result.passed is False}
    assert failed == expected_failures
    # A SKIP row would mean the tape is too thin to judge: every shipped tape must
    # produce a verdict on every assertion in its own set (CONTRACTS 9.27).
    for result in report.results:
        assert result.passed in (True, False), f"{fixture_id}: {result.name} was skipped"
        assert result.detail
        if result.passed:
            assert result.first_offense_seq is None, result.name
        else:
            assert result.first_offense_seq is not None, result.name


@pytest.mark.parametrize("fixture_id", ALL_IDS)
def test_every_committed_tape_stays_small_enough_to_read(fixture_id):
    assert gallery.get(fixture_id).tape.stat().st_size < 8 * 1024


def test_both_readmes_quote_the_gallery_list_verbatim():
    """The gallery.list transcript in both READMEs must stay the real command output.

    The transcripts were re-recorded when the gallery grew from two fixtures to ten;
    without this the next story added would leave both READMEs advertising a shorter
    gallery than the package ships.
    """
    expected = "\n".join(f"{fixture.id}  {fixture.title}" for fixture in gallery.list_fixtures())
    for name in ("README.md", "README.zh-CN.md"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert expected in text, f"{name} does not quote the current gallery list"


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


@pytest.mark.parametrize("fixture_id", ALL_IDS)
def test_replay_reproduces_the_recorded_events_payload_for_payload(fixture_id):
    fixture = gallery.get(fixture_id)
    tape = Trace.load(fixture.tape)
    report = Twin.load(fixture.tape).replay(fixture.agent)

    assert report.divergent is False
    assert [(event.kind, event.actor, event.payload) for event in report.trace.events] == [
        (event.kind, event.actor, event.payload) for event in tape.events
    ]


@pytest.mark.parametrize("fixture_id", ALL_IDS)
def test_replay_needs_no_socket_at_all(fixture_id, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("replay opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert judged(gallery.get(fixture_id)).divergent is False


@pytest.mark.parametrize("fixture_id", ALL_IDS)
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


def tape_line_seq(line: str):
    """The seq carried by one tape line, or None for the header and for garbage."""
    try:
        payload = json.loads(line)
    except ValueError:
        return None
    return payload.get("seq") if isinstance(payload, dict) else None


def describe_line(index: int, committed_line: str, regenerated_line: str) -> str:
    """Name one differing line: its 1-based number, its seq, and where the bytes part.

    Line 1 is the header and carries no seq, so the seq annotation appears only on an
    event line -- which is the coordinate the rest of the suite talks in.
    """
    seq = tape_line_seq(regenerated_line)
    if seq is None:
        seq = tape_line_seq(committed_line)
    shared = min(len(committed_line), len(regenerated_line))
    column = next(
        (at for at in range(shared) if committed_line[at] != regenerated_line[at]),
        shared,
    )
    where = f"line {index + 1}" if seq is None else f"line {index + 1} (seq {seq})"
    return (
        f"{where} column {column + 1}: committed={committed_line[:160]!r} "
        f"regenerated={regenerated_line[:160]!r}"
    )


def first_line_mismatch(committed: list, regenerated: list):
    """Describe the first differing line; None when the two tapes are line-identical."""
    shared = min(len(committed), len(regenerated))
    for index in range(shared):
        if committed[index] != regenerated[index]:
            return describe_line(index, committed[index], regenerated[index])
    if len(committed) == len(regenerated):
        return None
    longer_side = "committed" if len(committed) > len(regenerated) else "regenerated"
    longer = committed if len(committed) > len(regenerated) else regenerated
    extra = longer[shared]
    seq = tape_line_seq(extra)
    where = f"line {shared + 1}" if seq is None else f"line {shared + 1} (seq {seq})"
    return f"{where}: present only in the {longer_side} tape -> {extra[:160]!r}"


@pytest.mark.parametrize("fixture_id", ALL_IDS)
def test_the_generator_reproduces_every_committed_line(fixture_id, tape_dir):
    """Regenerate with tools/gen_fixtures.py and compare the committed tape line by line.

    env_fingerprint records the machine that ran the generator -- python, platform,
    deps -- and CONTRACTS 9.21 deliberately makes those non-fatal so one committed
    fixture set survives the 3.10-3.13 matrix. That one header value is therefore
    swapped back to the committed one before the header is compared; the committed
    header is separately proved canonical (CONTRACTS 1.3), so the swap cannot hide a
    serializer change, and every event line is compared untouched.
    """
    fixture = gallery.get(fixture_id)
    regenerated = generator().record(fixture, out_dir=tape_dir)

    committed_text = fixture.tape.read_text(encoding="utf-8")
    regenerated_text = regenerated.read_text(encoding="utf-8")
    assert committed_text.endswith("\n") and regenerated_text.endswith("\n"), (
        "a tape ends with exactly one LF (CONTRACTS 1)"
    )
    committed_lines = committed_text.splitlines()
    regenerated_lines = regenerated_text.splitlines()

    committed_header = json.loads(committed_lines[0])
    assert canonical_json(committed_header) == committed_lines[0], (
        f"line 1: the committed header is not canonical JSON -> {committed_lines[0]!r}"
    )

    regenerated_header = json.loads(regenerated_lines[0])
    regenerated_header["env_fingerprint"] = committed_header["env_fingerprint"]
    normalized_header = canonical_json(regenerated_header)
    assert normalized_header == committed_lines[0], (
        "line 1: the header differs beyond env_fingerprint -> "
        f"committed={committed_lines[0]!r} regenerated={normalized_header!r}"
    )

    mismatch = first_line_mismatch(
        committed_lines, [committed_lines[0], *regenerated_lines[1:]]
    )
    assert mismatch is None, mismatch
    assert regenerated.stat().st_size < 8 * 1024
