"""The Twin facade (CONTRACTS sections 3, 9.1, 9.8, 9.19, 9.21, 9.23)."""

from __future__ import annotations

import types

import pytest

from agenttwin.adapters.openai_compatible import RECORD_OUT_ENV, RECORD_TRACE_ID_ENV
from agenttwin.drift import FingerprintDrift
from agenttwin.env import ReplaySession, TapeRecorder, Twin
from agenttwin.fingerprint import capture
from agenttwin.redact import Redactor
from agenttwin.trace import Header, Trace, canonical_json, sha256_hex


def looping_agent(rounds):
    def agent(llm, tools):
        for index in range(rounds):
            llm({"prompt": f"step {index}"})
            tools["shell"](f"echo {index}")

    return agent


def fake_llm(request):
    return {"text": f"ok:{request['prompt']}"}


class FakeCompletions:
    """The one SDK surface the L2 adapter models: create(**kwargs)."""

    def create(self, **kwargs):
        return {"id": "x", "model": kwargs.get("model")}


class FakeClient:
    """An openai-compatible client, duck-typed: nothing here imports the SDK."""

    def __init__(self):
        self.chat = types.SimpleNamespace(completions=FakeCompletions())


def write_tape(path, fingerprint, *, with_terminal=True):
    trace = Trace(
        header=Header(
            schema_version="0.1",
            trace_id="fp",
            seed=0,
            env_fingerprint=fingerprint,
            redaction={"redacted": 0, "policy": {}, "mode": "scrub"},
        ),
        events=[],
    )
    if with_terminal:
        trace.append(
            "terminal",
            "agent",
            {"state": {}, "state_hash": sha256_hex(canonical_json({})), "outcome": "ok"},
        )
    trace.save(path)
    return trace


def test_record_returns_a_recorder_exposing_llm_and_tools(tape_dir):
    recorder = Twin.record("facade", out=tape_dir / "facade.jsonl", tools={"shell": lambda cmd: "out"})
    assert isinstance(recorder, TapeRecorder)
    assert callable(recorder.llm)
    assert set(recorder.tools()) == {"shell"}
    assert isinstance(recorder.trace, Trace)


def test_from_openai_compatible_is_the_zero_config_record_path(tape_dir, monkeypatch):
    """CONTRACTS section 3 / design doc 4.2: one call opens a tape and returns a proxy."""
    path = tape_dir / "l2.jsonl"
    monkeypatch.setenv(RECORD_OUT_ENV, str(path))
    monkeypatch.setenv(RECORD_TRACE_ID_ENV, "l2")
    proxy = Twin.from_openai_compatible(FakeClient())
    assert proxy.chat.completions.create(model="gpt-4o") == {"id": "x", "model": "gpt-4o"}
    proxy.flush()
    tape = Trace.load(path)
    assert tape.header.trace_id == "l2"
    assert [event.kind for event in tape.events] == ["llm_call", "terminal"]


def test_recording_writes_one_terminal_event_on_a_clean_exit(tape_dir):
    path = tape_dir / "clean.jsonl"
    with Twin.record("clean", out=path, llm=fake_llm) as recorder:
        recorder.llm({"prompt": "hi"})
    tape = Trace.load(path)
    assert [event.kind for event in tape.events] == ["llm_call", "terminal"]
    terminal = tape.events[-1].payload
    assert terminal["outcome"] == "ok"
    assert terminal["state"] == {}
    assert terminal["state_hash"] == sha256_hex(canonical_json({}))


def test_patch_state_folds_into_the_terminal_state(tape_dir):
    path = tape_dir / "state.jsonl"
    with Twin.record("state", out=path) as recorder:
        recorder.patch_state("/count", 1)
        recorder.patch_state("/name", "x")
    assert Trace.load(path).state() == {"count": 1, "name": "x"}
    terminal = Trace.load(path).events[-1].payload
    assert terminal["state_hash"] == sha256_hex(canonical_json({"count": 1, "name": "x"}))


def test_agent_returned_dict_overrides_the_folded_state(tape_dir):
    path = tape_dir / "returned.jsonl"
    with Twin.record("returned", out=path) as recorder:
        recorder.patch_state("/count", 1)
        recorder.returned({"count": 2, "done": True})
    assert Trace.load(path).state() == {"count": 2, "done": True}


def test_note_redactions_accumulates_into_the_header(tape_dir):
    """CONTRACTS 9.28: zero, 0 and several counts are tolerated; they add up on exit."""
    path = tape_dir / "noted.jsonl"
    with Twin.record("noted", out=path, llm=fake_llm) as recorder:
        recorder.note_redactions(0)
        recorder.note_redactions(2)
        recorder.note_redactions(1)
        recorder.llm({"prompt": "hi"})
    assert Trace.load(path).header.redaction["redacted"] == 3


def test_noted_redactions_add_to_the_proxy_counts(tape_dir):
    """CONTRACTS 9.28: the header is proxy substitutions plus the adapter's own."""
    path = tape_dir / "noted-plus-proxy.jsonl"
    with Twin.record(
        "noted-plus-proxy", out=path, llm=lambda request: {"text": "ok"}, redactor=Redactor()
    ) as recorder:
        recorder.note_redactions(2)
        recorder.llm({"prompt": "ping bob@x.com now"})
    assert Trace.load(path).header.redaction["redacted"] == 3


def test_exception_writes_error_then_terminal_and_reraises(tape_dir):
    path = tape_dir / "error.jsonl"
    with pytest.raises(RuntimeError, match="agent exploded"):
        with Twin.record("error", out=path, llm=fake_llm) as recorder:
            recorder.llm({"prompt": "hi"})
            raise RuntimeError("agent exploded")
    tape = Trace.load(path)
    assert [event.kind for event in tape.events] == ["llm_call", "error", "terminal"]
    assert tape.events[1].payload == {"type": "RuntimeError", "message": "agent exploded"}
    assert tape.events[2].payload["outcome"] == "error"


def test_replaying_the_same_agent_is_byte_identical(tape_dir):
    path = tape_dir / "loop.jsonl"
    with Twin.record(
        "loop", out=path, llm=fake_llm, tools={"shell": lambda cmd: f"ran:{cmd}"}
    ) as recorder:
        looping_agent(6)(recorder.llm, recorder.tools())
    tape = Trace.load(path)
    assert len(tape.events) == 19  # 6 x (llm_call + tool_call + tool_result) + terminal

    session = Twin.load(path)
    report = session.replay(looping_agent(6))
    assert report.divergent is False
    assert report.drifts == [] and report.misses == []
    assert report.trace.to_jsonl() == path.read_text(encoding="utf-8")


def test_a_patched_response_diverges_and_the_trace_still_loads(tape_dir):
    path = tape_dir / "counterfactual.jsonl"
    with Twin.record(
        "counterfactual", out=path, llm=fake_llm, tools={"shell": lambda cmd: f"ran:{cmd}"}
    ) as recorder:
        looping_agent(6)(recorder.llm, recorder.tools())

    def stopping_agent(llm, tools):
        answer = llm({"prompt": "step 0"})
        if answer.get("text") == "stop":
            return
        tools["shell"]("echo 0")

    session = Twin.load(path).patch(seq=0, response={"text": "stop"})
    report = session.replay(stopping_agent)
    assert report.divergent is True
    assert [event.kind for event in report.trace.events] == ["llm_call", "terminal"]
    assert report.trace.events[0].patched is True
    assert report.trace.events[0].seq == 0
    assert report.trace.events[0].t == 0.0
    out = tape_dir / "counterfactual-produced.jsonl"
    report.trace.save(out)
    reloaded = Trace.load(out)
    assert reloaded.events[0].payload["response"] == {"text": "stop"}


def test_patch_returns_a_copy_and_leaves_the_parent_untouched(tape_dir):
    path = tape_dir / "patch.jsonl"
    with Twin.record("patch", out=path, llm=fake_llm) as recorder:
        recorder.llm({"prompt": "hi"})
    session = Twin.load(path)
    patched = session.patch(seq=0, response={"text": "other"})
    assert isinstance(patched, ReplaySession)
    assert patched is not session
    assert session._patches == {}
    assert patched._patches == {0: {"response": {"text": "other"}}}


def test_load_rejects_real_tools(tape_dir):
    path = tape_dir / "real.jsonl"
    with Twin.record("real", out=path):
        pass
    with pytest.raises(ValueError, match="real tools exist only in record mode"):
        Twin.load(path, allow_real=True)


def test_a_model_recorded_tape_loads_by_default_with_the_model_row_in_the_diff(tape_dir):
    path = tape_dir / "model-default.jsonl"
    with Twin.record("model-default", out=path, llm=fake_llm, model="gpt-4o") as recorder:
        recorder.llm({"prompt": "hi"})

    def agent(llm, tools):
        llm({"prompt": "hi"})

    session = Twin.load(path)
    rows = {item["field"]: item for item in session.fingerprint_diff}
    assert rows["model"] == {"field": "model", "expected": "gpt-4o", "actual": None}

    report = session.replay(agent)
    assert report.divergent is False
    assert report.exit_code == 0
    assert report.trace.to_jsonl() == path.read_text(encoding="utf-8")


def test_a_known_matching_model_is_not_a_drift(tape_dir):
    path = tape_dir / "model-known.jsonl"
    with Twin.record("model-known", out=path, llm=fake_llm, model="gpt-4o") as recorder:
        recorder.llm({"prompt": "hi"})
    assert Twin.load(path, model="gpt-4o").fingerprint_diff == []


def test_a_known_different_model_raises_fingerprint_drift(tape_dir):
    path = tape_dir / "model-other.jsonl"
    with Twin.record("model-other", out=path, llm=fake_llm, model="gpt-4o") as recorder:
        recorder.llm({"prompt": "hi"})
    with pytest.raises(FingerprintDrift) as caught:
        Twin.load(path, model="claude-3")
    assert "model tape='gpt-4o' now='claude-3'" in str(caught.value)


def test_python_platform_and_deps_drift_is_recorded_not_raised(tape_dir):
    path = tape_dir / "soft.jsonl"
    fingerprint = dict(capture())
    # A value no runner can actually report: "linux" here would silently match
    # the real sys.platform on the ubuntu jobs and drop the platform difference.
    fingerprint.update(
        {"python": "3.9", "platform": "fixture-platform", "deps": {"vendor-sdk": "1.0"}}
    )
    write_tape(path, fingerprint)
    session = Twin.load(path)
    fields = {item["field"] for item in session.fingerprint_diff}
    assert {"python", "platform", "deps"} <= fields
    assert session.drifts == []


def test_a_tape_miss_returns_none_and_holds_the_cursor(tape_dir):
    path = tape_dir / "miss.jsonl"
    with Twin.record("miss", out=path, llm=fake_llm) as recorder:
        recorder.llm({"prompt": "one"})

    answers = []

    def greedy_agent(llm, tools):
        answers.append(llm({"prompt": "one"}))
        answers.append(llm({"prompt": "two"}))

    session = Twin.load(path)
    report = session.replay(greedy_agent)
    assert answers == [{"text": "ok:one"}, None]
    assert len(report.misses) == 1
    assert report.misses[0].to_dict()["expected_kind"] == "llm_call"
    assert report.misses[0].seq == 1
    assert report.divergent is True


def test_a_header_only_tape_replays_without_crashing(tape_dir):
    path = tape_dir / "header-only.jsonl"
    write_tape(path, capture(), with_terminal=False)
    session = Twin.load(path)
    report = session.replay(lambda llm, tools: None)
    assert report.trace.events == []
    produced = tape_dir / "header-only-produced.jsonl"
    report.trace.save(produced)
    assert Trace.load(produced).events == []


def test_an_agent_object_with_a_run_method_is_supported(tape_dir):
    path = tape_dir / "object.jsonl"
    with Twin.record("object", out=path, llm=fake_llm) as recorder:
        recorder.llm({"prompt": "hi"})

    class Agent:
        def __init__(self):
            self.answer = None

        def run(self, llm, tools):
            self.answer = llm({"prompt": "hi"})

    agent = Agent()
    Twin.load(path).replay(agent)
    assert agent.answer == {"text": "ok:hi"}


def test_policy_reaches_an_agent_that_declares_it(tape_dir):
    path = tape_dir / "policy.jsonl"
    seen = []

    def agent(llm, tools, policy=None):
        seen.append(policy)
        llm({"prompt": "hi"})

    with Twin.record("policy", out=path, llm=fake_llm) as recorder:
        agent(recorder.llm, recorder.tools())
    assert seen == [None]
    Twin.load(path).replay(agent, policy={"halt": True})
    assert seen[-1] == {"halt": True}


def test_a_required_third_positional_replays_through_the_public_path(tape_dir):
    """CONTRACTS section 3: the policy is passed whenever the signature accepts it."""
    path = tape_dir / "required-policy.jsonl"
    with Twin.record("required-policy", out=path, llm=fake_llm) as recorder:
        recorder.llm({"prompt": "hi"})

    class Agent:
        def __init__(self):
            self.answer = None

        def run(self, llm, tools, policy):
            self.answer = llm({"prompt": "hi"})
            return (self.answer, policy)

    agent = Agent()
    report = Twin.load(path).replay(agent)
    assert agent.answer == {"text": "ok:hi"}
    assert report.divergent is False
    assert report.misses == [] and report.drifts == []
