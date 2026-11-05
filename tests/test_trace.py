"""Contract tests for the tape layer: CONTRACTS §1 (JSONL schema 0.1) and §2 (seq).

Every test is named for the clause it guards; the docstring names the section.
"""

from __future__ import annotations

import json

import pytest

from agenttwin.trace import (
    CLOCK_STEP,
    EVENT_KINDS,
    SCHEMA_VERSION,
    Event,
    Header,
    SchemaError,
    Trace,
    canonical_json,
    sha256_hex,
)

FIXTURE = {"tool": "shell", "args": {"cmd": "ls -la", "cwd": "C:/tmp"}, "args_hash": "9f2c"}
FIXTURE_CANONICAL = '{"args":{"cmd":"ls -la","cwd":"C:/tmp"},"args_hash":"9f2c","tool":"shell"}'
FIXTURE_SHA256 = "3d5c18ed69ad81c8928899ea7d7e5b88ce2241d822ead5080d3418f2958eefcb"


def make_header(**overrides) -> Header:
    """A §1.5-shaped header; the overrides let a test break exactly one field."""
    fields = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": "checkout",
        "seed": 7,
        "env_fingerprint": {
            "model": "gpt-4o-mini",
            "prompt_hash": None,
            "python": "3.12.7",
            "agenttwin": "0.1.0",
            "platform": "win32",
            "deps": {},
        },
        "redaction": {"redacted": 0, "policy": {}, "mode": "scrub"},
    }
    fields.update(overrides)
    return Header(**fields)


def make_trace() -> Trace:
    """A short but complete tape: one event of most kinds, ending in terminal."""
    trace = Trace(header=make_header())
    trace.append("human_input", "user", {"input": "list the files"})
    trace.append("llm_call", "planner", {"prompt_hash": "ph-1", "response": {"text": "call shell"}})
    trace.append("tool_call", "shell", {"tool": "shell", "args_hash": "ah-1", "args_preview": "ls"})
    trace.append("tool_result", "shell", {"tool": "shell", "result": ["a.txt"], "ok": True})
    trace.append("state_patch", "planner", {"patch": {"/files": ["a.txt"]}})
    trace.append("decision", "planner", {"note": "done", "why": "every file listed"})
    trace.append(
        "terminal",
        "planner",
        {"state": {"files": ["a.txt"], "done": True}, "state_hash": "sh-1", "outcome": "ok"},
    )
    return trace


def decision(seq: int, **overrides) -> dict:
    """A minimal valid decision event dict, with room to break exactly one field."""
    event = {
        "seq": seq,
        "t": round(seq * CLOCK_STEP, 4),
        "kind": "decision",
        "actor": "agent",
        "payload": {"note": "n"},
    }
    event.update(overrides)
    return event


def tape(*events: dict) -> str:
    """A header line plus the given event dicts, joined into JSONL text."""
    lines = [canonical_json(make_header().to_json())]
    lines.extend(canonical_json(event) for event in events)
    return "\n".join(lines) + "\n"


# --- §1.3 canonical form and hashing ---


def test_canonical_json_is_independent_of_key_order():
    """§1.3 — sort_keys=True: insertion order must not change the bytes."""
    assert canonical_json({"b": 1, "a": {"d": 2, "c": 3}}) == canonical_json({"a": {"c": 3, "d": 2}, "b": 1})
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_canonical_json_nests_compactly_without_whitespace():
    """§1.3 — separators=(",", ":") applies at every nesting depth."""
    assert canonical_json({"a": [1, {"b": 2}], "c": None}) == '{"a":[1,{"b":2}],"c":null}'


def test_canonical_json_keeps_unicode_unescaped():
    """§1.3 — ensure_ascii=False keeps a tape readable as UTF-8."""
    assert canonical_json({"note": "下单"}) == '{"note":"下单"}'


def test_canonical_json_falls_back_to_repr_for_unserialisable_values():
    """§1.3 — default=repr means a stray object degrades to its repr instead of raising."""

    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    assert canonical_json({"obj": Opaque()}) == '{"obj":"<opaque>"}'


def test_sha256_hex_matches_the_frozen_fixture():
    """§1.3 — sha256 over the UTF-8 bytes of the canonical JSON, 64 lowercase hex."""
    assert canonical_json(FIXTURE) == FIXTURE_CANONICAL
    assert sha256_hex(FIXTURE_CANONICAL) == FIXTURE_SHA256
    assert len(FIXTURE_SHA256) == 64
    assert FIXTURE_SHA256 == FIXTURE_SHA256.lower()


# --- §1.2 the event line ---


def test_event_to_json_omits_patched_when_false():
    """§1.2 — the patched key exists on the wire only when it is true."""
    event = Event(seq=2, t=0.2, kind="tool_call", actor="shell", payload={"tool": "shell", "args_hash": "h"})
    wire = event.to_json()
    assert "patched" not in wire
    assert set(wire) == {"seq", "t", "kind", "actor", "payload"}


def test_event_to_json_includes_patched_when_true():
    """§1.2 / §2 R-PATCH — an injected counterfactual is serialised explicitly."""
    event = Event(
        seq=4,
        t=0.4,
        kind="llm_call",
        actor="planner",
        payload={"prompt_hash": "p", "response": 1},
        patched=True,
    )
    assert event.to_json()["patched"] is True


def test_event_json_roundtrips_through_from_json():
    """§1.2 — to_json/from_json are inverse for both patched values."""
    for patched in (False, True):
        event = Event(seq=3, t=0.3, kind="decision", actor="planner", payload={"note": "n"}, patched=patched)
        assert Event.from_json(event.to_json()) == event


def test_event_kinds_are_the_eight_from_the_table_in_order():
    """§1.4 — exactly eight kinds, in table order (len(EVENT_KINDS) == 8)."""
    assert EVENT_KINDS == (
        "llm_call",
        "tool_call",
        "tool_result",
        "state_patch",
        "decision",
        "human_input",
        "error",
        "terminal",
    )


def test_append_assigns_dense_seq_and_deterministic_t():
    """§1.2 + §9.4 — t is a pure function of seq: round(seq * CLOCK_STEP, 4)."""
    trace = make_trace()
    assert len(trace) == 7
    assert [event.seq for event in trace] == list(range(len(trace)))
    assert [event.t for event in trace] == [round(seq * CLOCK_STEP, 4) for seq in range(len(trace))]
    assert trace.events[0].t == 0.0
    assert trace.events[6].t == 0.6


# --- §1 the file format ---


def test_save_and_load_are_byte_identical(tape_dir):
    """§1 — UTF-8 / LF roundtrip: reloading and re-saving must not change a byte."""
    first = tape_dir / "tape.jsonl"
    second = tape_dir / "tape-copy.jsonl"
    original = make_trace()
    original.save(first)
    reloaded = Trace.load(first)
    assert reloaded == original
    reloaded.save(second)
    assert first.read_bytes() == second.read_bytes()
    raw = first.read_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b"\r" not in raw


def test_load_accepts_a_tape_without_trailing_newline(tape_dir):
    """§1 — our writer always ends with LF; a truncated tail must still parse."""
    path = tape_dir / "truncated.jsonl"
    path.write_bytes(make_trace().to_jsonl().rstrip("\n").encode("utf-8"))
    assert Trace.load(path) == make_trace()


def test_to_jsonl_is_one_object_per_line_with_trailing_lf():
    """§1 — header first, then one event per line, no blank lines, LF terminated."""
    trace = make_trace()
    text = trace.to_jsonl()
    assert text.endswith("\n")
    assert "\n\n" not in text
    lines = text.split("\n")[:-1]
    assert len(lines) == len(trace) + 1
    assert json.loads(lines[0])["schema_version"] == SCHEMA_VERSION
    assert [json.loads(line)["seq"] for line in lines[1:]] == list(range(len(trace)))


def test_header_only_tape_roundtrips():
    """§1 — a tape with zero events is still valid JSONL: one header line."""
    trace = Trace(header=make_header())
    text = trace.to_jsonl()
    assert text.count("\n") == 1
    assert Trace.from_jsonl(text) == trace
    assert trace.state() == {}


# --- §1 / §1.1 / §1.2 rejection cases ---


def test_from_jsonl_rejects_malformed_json_line():
    """§1 — every line is JSON; the error names the line and the JSON problem."""
    lines = make_trace().to_jsonl().split("\n")
    lines[2] = "{not json"
    with pytest.raises(SchemaError) as excinfo:
        Trace.from_jsonl("\n".join(lines))
    assert "line 3: malformed JSON:" in str(excinfo.value)


def test_from_jsonl_rejects_first_line_without_schema_version():
    """§1.1 — the first line carrying schema_version is the header; an event is not."""
    with pytest.raises(SchemaError) as excinfo:
        Trace.from_jsonl(canonical_json(decision(0)) + "\n")
    assert str(excinfo.value) == "line 1: header is missing the key 'schema_version'"


def test_from_jsonl_rejects_foreign_schema_version():
    """§1.1 — 0.1 is pinned; any other version is drift, not a warning."""
    header = dict(make_header().to_json(), schema_version="0.2")
    with pytest.raises(SchemaError) as excinfo:
        Trace.from_jsonl(canonical_json(header) + "\n")
    assert str(excinfo.value) == "line 1: unsupported schema_version '0.2' (expected '0.1')"


def test_from_jsonl_rejects_unknown_event_key():
    """§1 — unknown top-level event keys are refused, so drift cannot be dropped silently."""
    with pytest.raises(SchemaError) as excinfo:
        Trace.from_jsonl(tape(decision(0, extra=1)))
    assert str(excinfo.value) == "line 2: unknown event key 'extra' (allowed: seq, t, kind, actor, payload, patched)"


def test_from_jsonl_rejects_unknown_header_key():
    """§1 — the header is a closed set too."""
    header = dict(make_header().to_json(), future_field=True)
    with pytest.raises(SchemaError) as excinfo:
        Trace.from_jsonl(canonical_json(header) + "\n")
    assert str(excinfo.value) == (
        "line 1: unknown header key 'future_field' "
        "(allowed: schema_version, trace_id, seed, env_fingerprint, redaction)"
    )


def test_from_jsonl_rejects_missing_required_event_key():
    """§1.2 — seq, t, kind, actor and payload are all required."""
    broken = decision(0)
    del broken["kind"]
    with pytest.raises(SchemaError) as excinfo:
        Trace.from_jsonl(tape(broken))
    assert str(excinfo.value) == "line 2: missing required event key 'kind'"


def test_from_jsonl_rejects_unknown_event_kind():
    """§1.4 — exactly eight kinds; a typo must not degrade into a generic event."""
    with pytest.raises(SchemaError) as excinfo:
        Trace.from_jsonl(tape(decision(0, kind="llm_cal")))
    message = str(excinfo.value)
    assert message.startswith("line 2: unknown event kind 'llm_cal'")
    assert "llm_call, tool_call, tool_result, state_patch, decision, human_input, error, terminal" in message


def test_from_jsonl_rejects_seq_not_starting_at_zero():
    """§1.2 — seq starts at 0."""
    with pytest.raises(SchemaError) as excinfo:
        Trace.from_jsonl(tape(decision(1)))
    assert str(excinfo.value).startswith("line 2: expected seq 0 but found 1")


def test_from_jsonl_rejects_seq_gap():
    """§1.2 — dense seq: a gap breaks the cursor, diff alignment and patch targeting."""
    with pytest.raises(SchemaError) as excinfo:
        Trace.from_jsonl(tape(decision(0), decision(2)))
    assert str(excinfo.value) == (
        "line 3: expected seq 1 but found 2 (seq must start at 0, be dense and strictly increasing)"
    )


def test_from_jsonl_rejects_repeated_seq():
    """§1.2 — strictly increasing: a duplicate seq is not a valid cursor step."""
    with pytest.raises(SchemaError) as excinfo:
        Trace.from_jsonl(tape(decision(0), decision(0)))
    assert str(excinfo.value).startswith("line 3: expected seq 1 but found 0")


def test_from_jsonl_rejects_t_that_contradicts_seq():
    """§1.2 — t = round(seq * 0.1, 4) within 1e-6; a hand-edited clock is a schema error."""
    with pytest.raises(SchemaError) as excinfo:
        Trace.from_jsonl(tape(decision(0), decision(1, t=0.25)))
    assert str(excinfo.value) == "line 3: t=0.25 does not match seq 1 (expected t=0.1, tolerance 1e-06)"


def test_from_jsonl_tolerates_float_noise_below_one_microsecond():
    """§1.2 — the t check carries a 1e-6 tolerance, so float repr noise still loads."""
    loaded = Trace.from_jsonl(tape(decision(0), decision(1, t=0.1 + 5e-7)))
    assert len(loaded) == 2


def test_from_jsonl_rejects_missing_required_payload_key():
    """§1.4 — payload shape is fixed per kind."""
    with pytest.raises(SchemaError) as excinfo:
        Trace.from_jsonl(tape(decision(0, kind="tool_call", payload={"tool": "shell"})))
    assert str(excinfo.value) == "line 2: payload for kind 'tool_call' is missing required key 'args_hash'"


def test_from_jsonl_rejects_state_patch_key_without_leading_slash():
    """§1.4 — patch keys are JSON Pointers and must start with '/'."""
    with pytest.raises(SchemaError) as excinfo:
        Trace.from_jsonl(tape(decision(0, kind="state_patch", payload={"patch": {"cart/items": []}})))
    assert str(excinfo.value) == "line 2: state_patch key 'cart/items' must start with '/'"


def test_from_jsonl_rejects_blank_line():
    """§1 — no blank lines and no comments between events."""
    text = tape(decision(0)).replace("\n", "\n\n", 1)
    with pytest.raises(SchemaError) as excinfo:
        Trace.from_jsonl(text)
    assert str(excinfo.value) == "line 2: blank line is not allowed"


def test_decision_payload_allows_arbitrary_extra_keys():
    """§1.4 — decision's optional keys are open, unlike its required ones."""
    loaded = Trace.from_jsonl(tape(decision(0, payload={"note": "n", "score": 0.5, "why": ["because"]})))
    assert loaded[0].payload["score"] == 0.5


# --- §1.4 state folding ---


def test_state_folds_patches_in_seq_order_with_nested_pointers():
    """§1.4 — state() folds state_patch by seq; later patches win and pointers nest."""
    trace = Trace(header=make_header())
    trace.append("state_patch", "agent", {"patch": {"/cart/items": [], "/cart/currency": "CNY"}})
    trace.append("state_patch", "agent", {"patch": {"/cart/items/0": {"sku": "a"}, "/meta/attempts": 1}})
    trace.append("state_patch", "agent", {"patch": {"/cart/items/1": {"sku": "b"}, "/meta/attempts": 2}})
    assert trace.state() == {
        "cart": {"items": [{"sku": "a"}, {"sku": "b"}], "currency": "CNY"},
        "meta": {"attempts": 2},
    }


def test_state_terminal_payload_is_authoritative():
    """§1.4 — terminal.payload.state wins over every folded patch."""
    trace = Trace(header=make_header())
    trace.append("state_patch", "agent", {"patch": {"/done": False, "/files": ["stale.txt"]}})
    trace.append("terminal", "agent", {"state": {"done": True, "files": ["a.txt"]}, "state_hash": "sh-1"})
    trace.append("state_patch", "agent", {"patch": {"/done": False}})
    assert trace.state() == {"done": True, "files": ["a.txt"]}


def test_state_is_empty_dict_when_nothing_applies():
    """§1.4 — no patches and no terminal means empty state, not None."""
    trace = Trace(header=make_header())
    trace.append("decision", "agent", {"note": "thinking"})
    assert trace.state() == {}


def test_state_does_not_mutate_event_payloads():
    """§1.4 — folding is a read-only projection; the tape itself must stay untouched."""
    trace = Trace(header=make_header())
    trace.append("state_patch", "agent", {"patch": {"/items": [1]}})
    before = json.dumps(trace.events[0].payload, sort_keys=True)
    snapshot = trace.state()
    snapshot["items"].append(2)
    snapshot["injected"] = True
    assert json.dumps(trace.events[0].payload, sort_keys=True) == before


# --- §2 seq semantics, R-TERM and R-PATCH ---


def test_copy_is_deep_for_the_events_list():
    """§2 — replay builds a new Trace; mutating the copy must not touch the baseline."""
    original = make_trace()
    clone = original.copy()
    assert clone == original
    assert clone.events is not original.events
    assert all(left is not right for left, right in zip(clone.events, original.events))
    clone.events[0].payload["input"] = "mutated"
    clone.append("decision", "clone", {"note": "extra"})
    assert original.events[0].payload["input"] == "list the files"
    assert len(original) == 7


def test_trace_sequence_protocol_and_event_lookup():
    """§2 — the cursor is an index into events; Trace exposes len/iter/getitem/event."""
    trace = make_trace()
    assert len(trace) == len(trace.events)
    assert [event.seq for event in trace] == [event.seq for event in trace.events]
    assert trace[2].seq == 2
    assert trace.event(2) is trace.events[2]
    with pytest.raises(IndexError) as excinfo:
        trace.event(len(trace))
    assert "is outside the tape" in str(excinfo.value)


def test_terminal_frame_survives_the_roundtrip_verbatim_r_term():
    """§2 R-TERM — the terminal frame is copied field for field (t, payload, patched)."""
    trace = Trace(header=make_header())
    trace.append("decision", "agent", {"note": "wrap up"})
    trace.append("terminal", "agent", {"state": {"done": True}, "state_hash": "sh-1", "outcome": "ok"}, patched=True)
    reloaded = Trace.from_jsonl(trace.to_jsonl())
    assert reloaded.events[-1] == trace.events[-1]
    assert reloaded.events[-1].patched is True
    assert reloaded.events[-1].t == 0.1
    assert json.loads(trace.to_jsonl().split("\n")[-2])["patched"] is True


def test_patched_replacement_keeps_seq_and_t_r_patch():
    """§2 R-PATCH — patch(seq=...) rewrites payload in place and only flips patched."""
    trace = make_trace()
    target = trace.events[3]
    assert target.kind == "tool_result"
    rewritten = Event(
        seq=target.seq,
        t=target.t,
        kind=target.kind,
        actor=target.actor,
        payload={"tool": "shell", "result": [], "ok": False},
        patched=True,
    )
    rebuilt = Trace(header=trace.header, events=list(trace.events[:3]) + [rewritten] + list(trace.events[4:]))
    assert rebuilt[target.seq].t == target.t
    assert rebuilt[target.seq].patched is True
    assert rebuilt.to_jsonl().split("\n")[target.seq + 1] == canonical_json(rewritten.to_json())
    assert Trace.from_jsonl(rebuilt.to_jsonl()) == rebuilt
