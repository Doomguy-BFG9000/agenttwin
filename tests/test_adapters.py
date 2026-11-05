"""The openai-compatible adapter layer (design doc 4.2 L1+L2; CONTRACTS sections 1.4, 3, 9.20)."""

from __future__ import annotations

import json
import threading
import types

import pytest

from agenttwin.adapters import install, invoke, wrap
from agenttwin.adapters.base import AgentCallable, AgentRunner, is_agent
from agenttwin.adapters.openai_compatible import RECORD_OUT_ENV, RECORD_TRACE_ID_ENV
from agenttwin.env import Twin
from agenttwin.proxy import LLM_ACTOR, PREVIEW_LIMIT
from agenttwin.redact import Redactor
from agenttwin.trace import Trace, canonical_json, sha256_hex

# --- fakes -----------------------------------------------------------------


class FakeCompletions:
    def create(self, **kwargs):
        return {
            "id": "x",
            "choices": [{"message": {"content": "hi"}}],
            "model": kwargs.get("model"),
        }


class FakeClient:
    def __init__(self, completions=None, **extra):
        self.chat = types.SimpleNamespace(completions=completions or FakeCompletions())
        for name, value in extra.items():
            setattr(self, name, value)


class StubCompletions:
    """A completions namespace that returns one canned value and keeps its calls."""

    def __init__(self, value):
        self.value = value
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.value


def recorded(path, value, *, wrap_model=None, request=None):
    """Record exactly one create() returning value; return that llm_call's payload."""
    with Twin.record("adapter", out=path) as recorder:
        proxy = wrap(FakeClient(completions=StubCompletions(value)), recorder, model=wrap_model)
        proxy.chat.completions.create(**(request if request is not None else {"model": "gpt-4o"}))
    calls = [event for event in Trace.load(path).events if event.kind == "llm_call"]
    assert len(calls) == 1
    return calls[0].payload


# --- base.invoke (CONTRACTS section 3) -------------------------------------


def test_invoke_calls_the_plain_two_argument_form():
    tools = {"shell": object()}
    seen = []

    def agent(llm, tools):
        seen.append((llm, tools))
        return "done"

    assert invoke(agent, "LLM", tools, {"halt": True}) == "done"
    assert seen == [("LLM", tools)]


def test_invoke_passes_the_policy_to_a_third_positional_parameter():
    seen = []

    def agent(llm, tools, policy):
        seen.append(policy)
        return "ok"

    assert invoke(agent, "LLM", {}, {"halt": True}) == "ok"
    assert seen == [{"halt": True}]


def test_invoke_passes_the_policy_to_a_keyword_only_parameter():
    seen = []

    def agent(llm, tools, *, policy=None):
        seen.append(policy)

    invoke(agent, "LLM", {}, None)
    invoke(agent, "LLM", {}, {"halt": True})
    assert seen == [None, {"halt": True}]


def test_invoke_uses_run_and_prefers_it_over_being_callable():
    seen = []

    class Control:
        def run(self, llm, tools, policy=None):
            seen.append(("run", policy))
            return "from-run"

        def __call__(self, llm, tools):
            seen.append(("call", None))
            return "from-call"

    assert invoke(Control(), "LLM", {}, {"halt": True}) == "from-run"
    assert seen == [("run", {"halt": True})]


def test_invoke_passes_the_policy_as_a_third_positional_parameter_whatever_its_name():
    seen = []

    def agent(llm, tools, context):
        seen.append(context)

    invoke(agent, "LLM", {}, {"halt": True})
    assert seen == [{"halt": True}]


def test_invoke_passes_the_policy_through_var_positional():
    seen = []

    def agent(llm, tools, *rest):
        seen.append(rest)

    invoke(agent, "LLM", {}, {"halt": True})
    assert seen == [({"halt": True},)]


def test_invoke_rejects_an_object_that_is_neither_form():
    class Nothing:
        pass

    for agent in (Nothing(), types.SimpleNamespace(run="not callable"), "agent"):
        with pytest.raises(TypeError) as caught:
            invoke(agent, "LLM", {}, None)
        message = str(caught.value)
        assert "callable(llm, tools)" in message
        assert "run(llm, tools)" in message


def test_is_agent_accepts_both_forms_and_rejects_the_rest():
    class Runner:
        def run(self, llm, tools):
            return None

    assert is_agent(lambda llm, tools: None)
    assert is_agent(Runner())
    assert not is_agent(types.SimpleNamespace(run="not callable"))
    assert not is_agent("agent")


def test_the_protocols_describe_each_form():
    class Runner:
        def run(self, llm, tools):
            return None

    assert isinstance(Runner(), AgentRunner)
    assert isinstance(lambda llm, tools: None, AgentCallable)


def test_invoke_matches_the_private_helper_env_py_uses():
    """env.py is frozen; this pins that invoke() calls each signature shape the same way."""
    from agenttwin.env import _invoke_agent

    def plain(llm, tools):
        return ("plain", llm, tools)

    def positional(llm, tools, policy):
        return ("positional", policy)

    def keyword(llm, tools, *, policy=None):
        return ("keyword", policy)

    class Runner:
        def run(self, llm, tools, policy=None):
            return ("runner", policy)

    for agent in (plain, positional, keyword, Runner()):
        expected = _invoke_agent(agent, "LLM", {}, {"halt": True})
        assert invoke(agent, "LLM", {}, {"halt": True}) == expected


def test_a_required_third_positional_stays_callable_when_no_policy_was_supplied():
    """CONTRACTS section 3 -- the policy is passed whenever the signature accepts it."""
    from agenttwin.env import _invoke_agent

    def agent(llm, tools, policy):
        return policy

    assert _invoke_agent(agent, "LLM", {}, None) is None
    assert invoke(agent, "LLM", {}, None) is None


# --- wrap (CONTRACTS sections 1.4, 3, 9.20) --------------------------------


def test_wrap_records_one_llm_call_per_create_and_returns_the_response(tape_dir):
    path = tape_dir / "wrapped.jsonl"
    request = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    with Twin.record("wrapped", out=path) as recorder:
        proxy = wrap(FakeClient(), recorder)
        first = proxy.chat.completions.create(**request)
        assert first == {
            "id": "x",
            "choices": [{"message": {"content": "hi"}}],
            "model": "gpt-4o",
        }
        proxy.chat.completions.create(model="gpt-4o-mini", messages=[])

    calls = [event for event in Trace.load(path).events if event.kind == "llm_call"]
    assert len(calls) == 2
    payload = calls[0].payload
    assert calls[0].actor == LLM_ACTOR
    assert payload["prompt_hash"] == sha256_hex(canonical_json(request))
    assert payload["prompt_preview"] == canonical_json(request)
    assert payload["model"] == "gpt-4o"
    assert payload["response"]["id"] == "x"
    assert calls[1].payload["model"] == "gpt-4o-mini"


def test_wrap_without_a_tape_is_transparent():
    client = FakeClient()
    proxy = wrap(client)
    assert proxy.chat.completions.create(model="m") == client.chat.completions.create(model="m")


def test_wrap_delegates_every_other_attribute_to_the_client():
    completions = StubCompletions({"id": "x"})
    completions.retrieve = lambda completion_id: {"id": completion_id}
    client = FakeClient(completions=completions, api_key="sk-test", region="eu")
    proxy = wrap(client)
    assert proxy.api_key == "sk-test"
    assert proxy.region == "eu"
    assert proxy.chat.completions.retrieve("c1") == {"id": "c1"}
    with pytest.raises(AttributeError):
        getattr(proxy, "missing")


def test_wrap_rejects_a_client_without_chat_completions_create():
    with pytest.raises(TypeError) as caught:
        wrap(types.SimpleNamespace(chat=types.SimpleNamespace()))
    assert "chat.completions" in str(caught.value)


def test_wrap_rejects_a_tape_that_is_neither_a_recorder_nor_a_trace():
    with pytest.raises(TypeError) as caught:
        wrap(FakeClient(), tape=object())
    assert "TapeRecorder" in str(caught.value)


def test_wrap_reuses_the_recorders_redaction_policy(tape_dir):
    path = tape_dir / "policy.jsonl"
    redactor = Redactor(patterns=[("ticket", r"TICKET-\d+")])
    with Twin.record("policy", out=path, redactor=redactor) as recorder:
        wrap(FakeClient(), recorder).chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "TICKET-42"}]
        )
    tape = Trace.load(path)
    payload = tape.events[0].payload
    assert payload["prompt_preview"] == "<redacted:ticket>"
    assert "TICKET-42" not in json.dumps(payload)
    assert "ticket" in tape.header.redaction["policy"]["patterns"]


def test_the_preview_is_redacted_while_the_hash_is_not(tape_dir):
    request = {"model": "gpt-4o", "messages": [{"role": "user", "content": "ping bob@x.com now"}]}
    payload = recorded(tape_dir / "redacted.jsonl", {"id": "x"}, request=request)
    assert payload["prompt_preview"] == "<redacted:email>"
    assert payload["prompt_hash"] == sha256_hex(canonical_json(request))


def test_a_secret_in_the_call_kwargs_never_reaches_the_tape(tape_dir):
    request = {"model": "gpt-4o", "messages": [], "api_key": "sk-abcdefghijklmnop1234"}
    payload = recorded(tape_dir / "secret.jsonl", {"id": "x"}, request=request)
    assert "sk-abcdefghijklmnop1234" not in json.dumps(payload)
    assert payload["prompt_preview"].startswith("<redacted:")
    assert payload["prompt_hash"] == sha256_hex(canonical_json(request))


def test_the_preview_is_truncated_at_the_contract_limit(tape_dir):
    assert PREVIEW_LIMIT == 4096
    request = {"model": "gpt-4o", "messages": [{"role": "user", "content": "x" * 5000}]}
    payload = recorded(tape_dir / "long.jsonl", {"id": "x"}, request=request)
    assert len(payload["prompt_preview"]) == PREVIEW_LIMIT
    assert payload["prompt_hash"] == sha256_hex(canonical_json(request))


def test_the_model_argument_fills_in_when_the_call_omits_it(tape_dir):
    named = recorded(
        tape_dir / "model.jsonl", {"id": "x"}, wrap_model="gpt-4o-2024", request={"messages": []}
    )
    assert named["model"] == "gpt-4o-2024"
    anonymous = recorded(tape_dir / "anon.jsonl", {"id": "x"}, request={"messages": []})
    assert "model" not in anonymous


# --- the response fallback chain (CONTRACTS section 1.4) -------------------


def test_a_create_that_raises_records_nothing(tape_dir):
    """The event is appended after the real call returns, like proxy.RecordingLLM."""

    class Boom:
        def create(self, **kwargs):
            raise RuntimeError("upstream 500")

    path = tape_dir / "boom.jsonl"
    with Twin.record("boom", out=path) as recorder:
        proxy = wrap(FakeClient(completions=Boom()), recorder)
        with pytest.raises(RuntimeError, match="upstream 500"):
            proxy.chat.completions.create(model="gpt-4o")

    assert [event.kind for event in Trace.load(path).events] == ["terminal"]


def test_the_response_model_dump_is_recorded(tape_dir):
    class Dumped:
        def model_dump(self):
            return {"id": "dumped", "choices": []}

    assert recorded(tape_dir / "dumped.jsonl", Dumped())["response"] == {
        "id": "dumped",
        "choices": [],
    }


def test_the_response_to_dict_is_recorded(tape_dir):
    class Dicted:
        def to_dict(self):
            return {"id": "dicted"}

    assert recorded(tape_dir / "dicted.jsonl", Dicted())["response"] == {"id": "dicted"}


def test_a_plain_mapping_response_is_recorded_as_is(tape_dir):
    assert recorded(tape_dir / "dict.jsonl", {"id": "plain"})["response"] == {"id": "plain"}


def test_a_response_without_dump_hooks_keeps_only_its_json_safe_attributes(tape_dir):
    class Opaque:
        def __init__(self):
            self.id = "opaque"
            self.lock = threading.Lock()

    assert recorded(tape_dir / "opaque.jsonl", Opaque())["response"] == {"id": "opaque"}


def test_a_response_with_no_json_representation_is_recorded_as_repr(tape_dir):
    class Unserialisable:
        __slots__ = ()

        def __repr__(self):
            return "ChatCompletion(id='locked')"

    payload = recorded(tape_dir / "repr.jsonl", Unserialisable())
    assert payload["response"] == {"repr": "ChatCompletion(id='locked')"}


def test_a_dump_hook_that_fails_falls_through_to_the_next_step(tape_dir):
    class BrokenDump:
        def __init__(self):
            self.id = "broken"

        def model_dump(self):
            raise RuntimeError("not implemented for this version")

    assert recorded(tape_dir / "broken.jsonl", BrokenDump())["response"] == {"id": "broken"}


# --- install (design doc 4.2 L2) ------------------------------------------


def test_install_writes_a_tape_that_loads_and_hashes_the_request(tape_dir):
    path = tape_dir / "a.jsonl"
    request = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    proxy = install(FakeClient(), out=path, trace_id="installed")
    assert proxy.chat.completions.create(**request)["id"] == "x"
    proxy.flush()

    tape = Trace.load(path)
    assert tape.header.trace_id == "installed"
    assert "email" in tape.header.redaction["policy"]["patterns"]
    calls = [event for event in tape.events if event.kind == "llm_call"]
    assert len(calls) == 1
    assert calls[0].payload["prompt_hash"] == sha256_hex(canonical_json(request))
    assert [event.kind for event in tape.events][-1] == "terminal"


def test_the_installed_tape_counts_the_previews_the_adapter_redacts(tape_dir):
    """CONTRACTS 9.28: an adapter-recorded tape reports the substitutions it made."""
    path = tape_dir / "counted.jsonl"
    request = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "my key is sk-ABCDEFG0123456789"}],
    }
    proxy = install(FakeClient(), out=path, trace_id="counted")
    proxy.chat.completions.create(**request)
    proxy.flush()

    tape = Trace.load(path)
    assert tape.header.redaction["redacted"] >= 1
    payload = [event for event in tape.events if event.kind == "llm_call"][0].payload
    assert payload["prompt_preview"] == "<redacted:openai_sk>"
    assert "sk-" not in payload["prompt_preview"]
    assert payload["prompt_hash"] == sha256_hex(canonical_json(request))


def test_a_clean_installed_request_reports_no_redactions(tape_dir):
    """CONTRACTS 9.28: counting adapter substitutions must not invent any."""
    path = tape_dir / "clean-count.jsonl"
    proxy = install(FakeClient(), out=path, trace_id="clean-count")
    proxy.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    proxy.flush()
    assert Trace.load(path).header.redaction["redacted"] == 0


def test_the_installed_tape_replays_without_prompt_drift(tape_dir):
    path = tape_dir / "replayable.jsonl"
    request = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    proxy = install(FakeClient(), out=path, trace_id="replayable")
    proxy.chat.completions.create(**request)
    proxy.flush()

    expected = {"id": "x", "choices": [{"message": {"content": "hi"}}], "model": "gpt-4o"}

    def agent(llm, tools):
        assert llm(request) == expected

    report = Twin.load(path).replay(agent)
    assert report.misses == []
    assert report.drifts == []


def test_install_reads_the_env_vars_and_creates_parent_directories(tape_dir, monkeypatch):
    path = tape_dir / "nested" / "deeper" / "env.jsonl"
    monkeypatch.setenv(RECORD_OUT_ENV, str(path))
    monkeypatch.setenv(RECORD_TRACE_ID_ENV, "from-env")
    proxy = install(FakeClient())
    proxy.chat.completions.create(model="gpt-4o", messages=[])
    proxy.flush()

    assert path.exists()
    assert Trace.load(path).header.trace_id == "from-env"


def test_install_without_a_trace_id_names_the_tape_after_the_file(tape_dir, monkeypatch):
    monkeypatch.delenv(RECORD_TRACE_ID_ENV, raising=False)
    path = tape_dir / "named.jsonl"
    proxy = install(FakeClient(), out=path)
    proxy.chat.completions.create(model="gpt-4o", messages=[])
    proxy.flush()
    assert Trace.load(path).header.trace_id == "named"


def test_install_records_its_model_argument_and_the_fingerprint(tape_dir):
    path = tape_dir / "installed-model.jsonl"
    proxy = install(FakeClient(), out=path, model="gpt-4o-mini")
    proxy.chat.completions.create(messages=[])
    proxy.flush()

    tape = Trace.load(path)
    assert tape.header.env_fingerprint["model"] == "gpt-4o-mini"
    assert tape.events[0].payload["model"] == "gpt-4o-mini"


def test_install_refuses_to_run_unrecorded(monkeypatch):
    monkeypatch.delenv(RECORD_OUT_ENV, raising=False)
    monkeypatch.delenv(RECORD_TRACE_ID_ENV, raising=False)
    with pytest.raises(RuntimeError) as caught:
        install(FakeClient())
    message = str(caught.value)
    assert RECORD_OUT_ENV in message
    assert RECORD_TRACE_ID_ENV in message


def test_flush_can_run_twice_and_leaves_exactly_one_terminal(tape_dir):
    path = tape_dir / "twice.jsonl"
    proxy = install(FakeClient(), out=path)
    proxy.chat.completions.create(model="gpt-4o", messages=[])
    proxy.flush()
    proxy.flush()
    assert [event.kind for event in Trace.load(path).events] == ["llm_call", "terminal"]
