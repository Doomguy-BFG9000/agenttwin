"""The L1 instrumentation layer and the env-var handshake (CONTRACTS section 3)."""

from __future__ import annotations

from agenttwin.env import TapeRecorder, Twin
from agenttwin.record import (
    RECORD_OUT_ENV,
    RECORD_TRACE_ID_ENV,
    TAPE_ID_ENV,
    TAPE_OUT_ENV,
    bind_record_env,
    record_from_env,
    twin_llm,
    twin_tool,
)
from agenttwin.trace import Trace


def test_twin_llm_records_one_llm_call(tape_dir):
    path = tape_dir / "l1-llm.jsonl"
    with Twin.record("l1-llm", out=path, llm=lambda request: "answer") as recorder:
        llm = twin_llm(recorder)
        assert llm({"prompt": "hi"}) == "answer"
    assert [event.kind for event in Trace.load(path).events] == ["llm_call", "terminal"]


def test_twin_tool_registers_and_records(tape_dir):
    path = tape_dir / "l1-tool.jsonl"
    with Twin.record("l1-tool", out=path) as recorder:

        @twin_tool(recorder, "shell")
        def shell(command):
            return f"ran:{command}"

        assert shell("echo hi") == "ran:echo hi"
    assert [event.kind for event in Trace.load(path).events] == [
        "tool_call",
        "tool_result",
        "terminal",
    ]


def test_bind_record_env_publishes_the_tape(tape_dir, monkeypatch):
    monkeypatch.delenv(TAPE_OUT_ENV, raising=False)
    monkeypatch.delenv(TAPE_ID_ENV, raising=False)
    monkeypatch.delenv(RECORD_OUT_ENV, raising=False)
    monkeypatch.delenv(RECORD_TRACE_ID_ENV, raising=False)
    path = tape_dir / "bound.jsonl"
    with Twin.record("bound", out=path) as recorder:
        published = bind_record_env(recorder)
    # CONTRACTS section 3: both spellings of the handshake are published, with the
    # same values, so a child that reads either pair attaches to this tape.
    assert published == {
        TAPE_OUT_ENV: str(path),
        TAPE_ID_ENV: "bound",
        RECORD_OUT_ENV: str(path),
        RECORD_TRACE_ID_ENV: "bound",
    }
    assert record_from_env(environ=published, llm=lambda request: "answer").out == path


def test_record_from_env_attaches_to_the_bound_tape(tape_dir, monkeypatch):
    path = tape_dir / "attached.jsonl"
    monkeypatch.setenv(TAPE_OUT_ENV, str(path))
    monkeypatch.setenv(TAPE_ID_ENV, "attached")
    recorder = record_from_env(llm=lambda request: "answer")
    assert isinstance(recorder, TapeRecorder)
    assert recorder.out == path
    assert recorder.trace.header.trace_id == "attached"
    with recorder:
        recorder.llm({"prompt": "hi"})
    assert Trace.load(path).header.trace_id == "attached"


def test_record_from_env_attaches_from_the_cli_pair_alone(tape_dir, monkeypatch):
    """CONTRACTS section 3: AGENTTWIN_RECORD_OUT is what 'agenttwin record' sets."""
    path = tape_dir / "cli-pair.jsonl"
    monkeypatch.delenv(TAPE_OUT_ENV, raising=False)
    monkeypatch.delenv(TAPE_ID_ENV, raising=False)
    monkeypatch.setenv(RECORD_OUT_ENV, str(path))
    monkeypatch.setenv(RECORD_TRACE_ID_ENV, "cli-pair")
    recorder = record_from_env(llm=lambda request: "answer")
    assert isinstance(recorder, TapeRecorder)
    assert recorder.out == path
    assert recorder.trace.header.trace_id == "cli-pair"
    with recorder:
        recorder.llm({"prompt": "hi"})
    assert [event.kind for event in Trace.load(path).events] == ["llm_call", "terminal"]


def test_the_tape_pair_wins_when_both_pairs_are_present(tape_dir, monkeypatch):
    """CONTRACTS section 3: TAPE_OUT/TAPE_ID take precedence over the CLI pair."""
    tape_pair = tape_dir / "tape-pair.jsonl"
    cli_pair = tape_dir / "cli-pair.jsonl"
    monkeypatch.setenv(TAPE_OUT_ENV, str(tape_pair))
    monkeypatch.setenv(TAPE_ID_ENV, "from-tape")
    monkeypatch.setenv(RECORD_OUT_ENV, str(cli_pair))
    monkeypatch.setenv(RECORD_TRACE_ID_ENV, "from-cli")
    recorder = record_from_env(llm=lambda request: "answer")
    assert recorder.out == tape_pair
    assert recorder.trace.header.trace_id == "from-tape"
    with recorder:
        recorder.llm({"prompt": "hi"})
    assert Trace.load(tape_pair).header.trace_id == "from-tape"
    assert not cli_pair.exists()


def test_record_from_env_returns_none_without_the_variables(monkeypatch):
    monkeypatch.delenv(TAPE_OUT_ENV, raising=False)
    monkeypatch.delenv(TAPE_ID_ENV, raising=False)
    monkeypatch.delenv(RECORD_OUT_ENV, raising=False)
    monkeypatch.delenv(RECORD_TRACE_ID_ENV, raising=False)
    assert record_from_env(llm=lambda request: "answer") is None
