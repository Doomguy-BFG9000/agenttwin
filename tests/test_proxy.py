"""The recording and replay proxies (CONTRACTS sections 1.4, 2, 9.19, 9.20, 9.22)."""

from __future__ import annotations

import pytest

from agenttwin.drift import ArgsDrift, PromptDrift
from agenttwin.proxy import (
    LLM_ACTOR,
    PREVIEW_LIMIT,
    Cursor,
    RecordedToolError,
    RecordingLLM,
    RecordingToolSet,
    ReplayLLM,
    ReplayToolSet,
    args_hash,
    args_preview,
    patch_args_hash,
)
from agenttwin.redact import Redactor
from agenttwin.trace import Header, Trace, canonical_json, sha256_hex


def make_trace(trace_id="t"):
    return Trace(
        header=Header(
            schema_version="0.1",
            trace_id=trace_id,
            seed=0,
            env_fingerprint={},
            redaction={"redacted": 0, "policy": {}, "mode": "scrub"},
        ),
        events=[],
    )


def test_args_hash_matches_canonical_json_sha256():
    assert args_hash((1, "x"), {"b": 2}) == sha256_hex(
        canonical_json({"args": [1, "x"], "kwargs": {"b": 2}})
    )


def test_patch_args_hash_wraps_the_value_as_kwargs():
    assert patch_args_hash({"cmd": "ls"}) == sha256_hex(
        canonical_json({"args": [], "kwargs": {"cmd": "ls"}})
    )


def test_args_preview_is_canonical_json():
    assert args_preview(("echo hi",), {}) == canonical_json({"args": ["echo hi"], "kwargs": {}})


def test_cursor_take_returns_the_event_and_advances():
    tape = make_trace()
    tape.append("llm_call", LLM_ACTOR, {"prompt_hash": "h", "response": "r"})
    cursor = Cursor(tape)
    event = cursor.take("llm_call", LLM_ACTOR)
    assert event is not None and event.seq == 0
    assert cursor.index == 1
    assert cursor.misses == []


def test_cursor_miss_is_recorded_and_holds_the_cursor():
    tape = make_trace()
    tape.append("llm_call", LLM_ACTOR, {"prompt_hash": "h", "response": "r"})
    cursor = Cursor(tape)
    assert cursor.take("tool_call", "shell") is None
    assert cursor.index == 0
    assert [miss.to_dict()["expected_kind"] for miss in cursor.misses] == ["tool_call"]


def test_cursor_copies_transparent_events_into_the_produced_trace():
    tape = make_trace()
    tape.append("state_patch", "agent", {"patch": {"/x": 1}})
    tape.append("llm_call", LLM_ACTOR, {"prompt_hash": "h", "response": "r"})
    produced = make_trace("p")
    cursor = Cursor(tape, produced=produced)
    cursor.take("llm_call", LLM_ACTOR)
    assert [event.kind for event in produced.events] == ["state_patch"]
    assert produced.events[0].seq == 0
    assert produced.events[0].payload == {"patch": {"/x": 1}}


def test_recording_llm_appends_one_llm_call_and_returns_the_response():
    trace = make_trace()
    llm = RecordingLLM(trace, lambda request: {"text": "hi"})
    assert llm({"prompt": "p"}) == {"text": "hi"}
    assert len(trace.events) == 1
    event = trace.events[0]
    assert (event.kind, event.actor) == ("llm_call", LLM_ACTOR)
    assert event.payload["response"] == {"text": "hi"}
    assert event.payload["prompt_hash"] == sha256_hex(canonical_json({"prompt": "p"}))


def test_recording_llm_records_the_tokens_the_real_llm_reports():
    trace = make_trace()
    llm = RecordingLLM(
        trace, lambda request: {"text": "hi", "usage": {"prompt_tokens": 3, "completion_tokens": 4}}
    )
    llm({"prompt": "p"})
    assert trace.events[0].payload["tokens"] == {"prompt": 3, "completion": 4}


def test_recording_llm_hashes_before_redaction_and_scrubs_the_preview():
    trace = make_trace()
    llm = RecordingLLM(trace, lambda request: "ok", redactor=Redactor())
    llm({"prompt": "ping bob@example.com"})
    payload = trace.events[0].payload
    assert payload["prompt_hash"] == sha256_hex(canonical_json({"prompt": "ping bob@example.com"}))
    assert payload["prompt_preview"] == "<redacted:email>"


def test_recording_llm_truncates_a_long_preview():
    trace = make_trace()
    llm = RecordingLLM(trace, lambda request: "ok")
    llm({"prompt": "x" * (PREVIEW_LIMIT * 2)})
    assert len(trace.events[0].payload["prompt_preview"]) == PREVIEW_LIMIT


def test_recording_tools_pair_one_call_with_one_result():
    trace = make_trace()
    tools = RecordingToolSet(trace, {"shell": lambda cmd: "out"})
    assert tools.tools()["shell"]("echo hi") == "out"
    assert [event.kind for event in trace.events] == ["tool_call", "tool_result"]
    call, result = trace.events
    assert call.payload["tool"] == "shell"
    assert call.payload["args_hash"] == args_hash(("echo hi",), {})
    assert result.payload == {"tool": "shell", "result": "out", "ok": True}


def test_recording_tool_failure_records_ok_false_and_reraises():
    def explode(cmd):
        raise ValueError("boom")

    trace = make_trace()
    tools = RecordingToolSet(trace, {"shell": explode})
    with pytest.raises(ValueError):
        tools.tools()["shell"]("bad")
    assert [event.kind for event in trace.events] == ["tool_call", "tool_result"]
    assert trace.events[1].payload == {
        "tool": "shell",
        "result": {"type": "ValueError", "message": "boom"},
        "ok": False,
    }


def test_recording_tool_redacts_the_result_but_keeps_the_args_hash():
    trace = make_trace()
    tools = RecordingToolSet(trace, {"shell": lambda cmd: "mail bob@example.com"}, redactor=Redactor())
    tools.tools()["shell"]("echo hi")
    call, result = trace.events
    assert call.payload["args_hash"] == args_hash(("echo hi",), {})
    assert result.payload["result"] == "<redacted:email>"


def test_replay_llm_returns_the_recorded_response():
    request = {"prompt": "p"}
    tape = make_trace()
    tape.append(
        "llm_call", LLM_ACTOR, {"prompt_hash": sha256_hex(canonical_json(request)), "response": {"text": "r"}}
    )
    produced = make_trace("p")
    llm = ReplayLLM(Cursor(tape, produced=produced))
    assert llm(request) == {"text": "r"}
    assert produced.events[0].kind == "llm_call"


def test_replay_llm_raises_prompt_drift_and_lands_it_on_the_cursor():
    tape = make_trace()
    tape.append("llm_call", LLM_ACTOR, {"prompt_hash": "0" * 64, "response": "r"})
    cursor = Cursor(tape, produced=make_trace("p"))
    llm = ReplayLLM(cursor)
    with pytest.raises(PromptDrift):
        llm({"prompt": "different"})
    assert [drift.type for drift in cursor.drifts] == ["PromptDrift"]
    assert cursor.drifts[0].seq == 0


def test_replay_tools_returns_the_recorded_result():
    tape = make_trace()
    tape.append("tool_call", "shell", {"tool": "shell", "args_hash": args_hash(("echo hi",), {})})
    tape.append("tool_result", "shell", {"tool": "shell", "result": "out", "ok": True})
    tools = ReplayToolSet(Cursor(tape, produced=make_trace("p"))).tools()
    assert tools["shell"]("echo hi") == "out"


def test_replay_tools_raises_args_drift_on_a_different_call():
    tape = make_trace()
    tape.append("tool_call", "shell", {"tool": "shell", "args_hash": args_hash(("echo hi",), {})})
    cursor = Cursor(tape, produced=make_trace("p"))
    tools = ReplayToolSet(cursor).tools()
    with pytest.raises(ArgsDrift):
        tools["shell"]("rm -rf /")
    assert [drift.type for drift in cursor.drifts] == ["ArgsDrift"]


def test_replay_tools_reraises_a_recorded_failure():
    tape = make_trace()
    tape.append("tool_call", "shell", {"tool": "shell", "args_hash": args_hash(("bad",), {})})
    tape.append(
        "tool_result",
        "shell",
        {"tool": "shell", "result": {"type": "ValueError", "message": "boom"}, "ok": False},
    )
    tools = ReplayToolSet(Cursor(tape, produced=make_trace("p"))).tools()
    with pytest.raises(RecordedToolError):
        tools["shell"]("bad")
