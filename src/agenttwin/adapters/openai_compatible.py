"""The openai-compatible adapter (design doc section 4.2, L1 + L2).

Duck-typed on purpose: nothing here imports the SDK, so importing the package
never pulls one in.  ``wrap`` records into a tape the caller already opened;
``install`` opens its own tape from the environment and writes it when the
interpreter exits.

Recording follows CONTRACTS section 1.4 and 9.20, which replay depends on: the
request the caller passed is hashed *before* redaction, the preview is the
redacted and truncated request, and the response is carried as JSON.  Nothing
beyond ``chat.completions.create(**kwargs)`` is modelled: the adapter is not an
SDK shim.
"""

from __future__ import annotations

import atexit
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..env import TapeRecorder, Twin
from ..proxy import LLM_ACTOR, PREVIEW_LIMIT
from ..redact import Redactor
from ..trace import Trace, canonical_json, sha256_hex

__all__ = ["RECORD_OUT_ENV", "RECORD_TRACE_ID_ENV", "install", "wrap"]

# The L2 zero-config pair.  An unset AGENTTWIN_RECORD_OUT is a configuration
# error, never a silent no-op: recording nowhere is how a tape goes missing.
RECORD_OUT_ENV: str = "AGENTTWIN_RECORD_OUT"
RECORD_TRACE_ID_ENV: str = "AGENTTWIN_TRACE_ID"

_UNSAFE: Any = object()


def wrap(client: Any, tape: Any = None, *, model: str | None = None) -> Any:
    """CONTRACTS section 3 -- route client.chat.completions.create through a tape.

    tape is a TapeRecorder from Twin.record() or a Trace; None keeps the proxy
    transparent.  model names the model for calls whose kwargs omit it.

    A TapeRecorder is told the substitutions this proxy makes, so its header
    counts them (CONTRACTS 9.28).  A bare Trace's header is already final and is
    never retroactively recounted: with no recorder there is nothing to rebuild it.
    """
    _require_client(client)
    return _RecordingClient(
        client, _trace_of(tape), _redactor_of(tape), model, _note_of(tape)
    )


def install(
    client: Any,
    *,
    out: str | Path | None = None,
    trace_id: str | None = None,
    model: str | None = None,
) -> Any:
    """Design doc section 4.2 L2 -- zero-config recording for an existing client.

    out falls back to AGENTTWIN_RECORD_OUT and trace_id to AGENTTWIN_TRACE_ID
    (then to the tape filename).  The tape is written when the interpreter
    exits; call flush() on the returned proxy to write it sooner.
    """
    _require_client(client)
    target = _target(out)
    trace_id = trace_id or _env(RECORD_TRACE_ID_ENV) or target.stem or "trace"
    # The adapter scrubs every preview, so the tape must say so: a header policy of
    # "no patterns" printed next to a scrubbed preview would be a false statement.
    recorder = Twin.record(trace_id, out=target, model=model, redactor=Redactor())
    proxy = _InstalledClient(client, recorder, model)
    atexit.register(proxy.flush)
    return proxy


# --- client shape ----------------------------------------------------------


def _require_client(client: Any) -> Any:
    """Fail at wrap time on a shape no create() call could ever reach."""
    try:
        create = client.chat.completions.create
    except AttributeError:
        raise TypeError(
            "client must look like the OpenAI SDK -- a callable create(**kwargs) on "
            f"client.chat.completions; got {type(client).__name__}"
        ) from None
    if not callable(create):
        raise TypeError(f"client.chat.completions.create must be callable; got {type(create).__name__}")
    return client


def _target(out: str | Path | None) -> Path:
    """The tape path from out or the environment; refuse to record nowhere."""
    raw = out if out is not None else _env(RECORD_OUT_ENV)
    if not raw:
        raise RuntimeError(
            "install() has no tape to write: pass out=... or set "
            f"{RECORD_OUT_ENV} (optionally {RECORD_TRACE_ID_ENV} for the trace id); "
            "refusing to run unrecorded"
        )
    target = Path(raw)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _env(name: str) -> str | None:
    """An environment variable, with the empty string treated as unset."""
    return os.environ.get(name) or None


def _trace_of(tape: Any) -> Trace | None:
    """The Trace behind a TapeRecorder (or a Trace itself); None records nothing."""
    if tape is None:
        return None
    trace = getattr(tape, "trace", tape)
    if not callable(getattr(trace, "append", None)):
        raise TypeError(f"tape must be a TapeRecorder or a Trace; got {type(tape).__name__}")
    return trace


def _redactor_of(tape: Any) -> Any:
    """Reuse the recorder's policy; a bare Trace or wrap(tape=None) scrubs by default.

    The default is the full pattern set rather than "no redaction": the preview is
    the only place a real request is stored, so it must never carry a secret
    (CONTRACTS 9.20).
    """
    redactor = getattr(tape, "redactor", None)
    if callable(getattr(redactor, "payload", None)):
        return redactor
    return Redactor()


def _note_of(tape: Any) -> Any:
    """The recorder's redaction counter, or None for a bare Trace (CONTRACTS 9.28)."""
    note = getattr(tape, "note_redactions", None)
    return note if callable(note) else None


# --- the proxies -----------------------------------------------------------


class _RecordingClient:
    """A proxy over an openai-compatible client: create() records, the rest delegates."""

    def __init__(
        self,
        client: Any,
        trace: Trace | None,
        redactor: Any,
        model: str | None,
        note: Any = None,
    ) -> None:
        self._client = client
        self._trace = trace
        self._redactor = redactor
        self._model = model
        self._note = note
        self._chat = _ChatProxy(self)

    @property
    def chat(self) -> "_ChatProxy":
        """The client's chat namespace, whose completions.create records the call."""
        return self._chat

    def __getattr__(self, name: str) -> Any:
        """CONTRACTS section 3 -- everything that is not create() is the real client's."""
        try:
            client = object.__getattribute__(self, "_client")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(client, name)

    def __repr__(self) -> str:
        return f"<agenttwin adapter for {type(self._client).__name__}>"

    def _create(self, kwargs: dict) -> Any:
        """One create() in, one llm_call event out; the real response is returned.

        The event is appended only after the real call returns, so a call that
        raises records nothing -- the same rule proxy.RecordingLLM follows.
        """
        response = self._client.chat.completions.create(**kwargs)
        if self._trace is not None:
            self._trace.append("llm_call", LLM_ACTOR, self._payload(kwargs, response))
        return response

    def _payload(self, request: dict, response: Any) -> dict:
        """CONTRACTS section 1.4 -- the llm_call payload, hashed before redaction."""
        payload: dict[str, Any] = {
            "prompt_hash": sha256_hex(canonical_json(request)),
            "response": _json_safe(response),
            "prompt_preview": canonical_json(request),
        }
        model = request.get("model")
        if model is None:
            model = self._model
        if model is not None:
            payload["model"] = model
        return _scrub("llm_call", payload, self._redactor, self._note)


class _ChatProxy:
    """client.chat: completions records, every other attribute is the real namespace's."""

    def __init__(self, owner: _RecordingClient) -> None:
        self._owner = owner
        self._completions = _CompletionsProxy(owner)

    @property
    def completions(self) -> "_CompletionsProxy":
        """The recording completions namespace."""
        return self._completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._owner._client.chat, name)


class _CompletionsProxy:
    """client.chat.completions: create() records, every other attribute delegates."""

    def __init__(self, owner: _RecordingClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        """CONTRACTS section 1.4 -- one call in, one llm_call event out."""
        return self._owner._create(kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._owner._client.chat.completions, name)


class _InstalledClient(_RecordingClient):
    """install()'s proxy: it owns the tape, so it decides when the tape is written."""

    def __init__(self, client: Any, recorder: TapeRecorder, model: str | None) -> None:
        self._recorder = recorder
        super().__init__(
            client, recorder.trace, _redactor_of(recorder), model, _note_of(recorder)
        )

    def flush(self) -> None:
        """Write the tape now, terminal event included; calling it twice is harmless."""
        self._recorder.__exit__(None, None, None)


# --- payload helpers -------------------------------------------------------


def _scrub(kind: str, payload: dict, redactor: Any, note: Any = None) -> dict:
    """CONTRACTS 9.20 -- hashes stay verbatim, previews are scrubbed then truncated.

    A sha256 is itself a 32+ hex run, so redaction would otherwise rewrite the
    very value replay compares; proxy.py splits the two for the same reason.

    note, when given, receives the substitution count so the tape header can
    report it (CONTRACTS 9.28); a bare Trace has no recorder to tell.
    """
    hashes = {key: value for key, value in payload.items() if key.endswith("_hash")}
    scrub = {key: value for key, value in payload.items() if not key.endswith("_hash")}
    scrubbed, count = redactor.payload(kind, scrub)
    if note is not None:
        note(count)
    scrubbed.update(hashes)
    for key, value in scrubbed.items():
        if key.endswith("_preview") and isinstance(value, str) and len(value) > PREVIEW_LIMIT:
            scrubbed[key] = value[:PREVIEW_LIMIT]
    return scrubbed


def _json_safe(response: Any) -> Any:
    """CONTRACTS section 1.4 -- the response as something the tape can carry.

    model_dump(), to_dict(), a mapping, the JSON-safe entries of __dict__, then
    {"repr": ...}: a tape line always names what came back instead of failing
    the recording on an SDK object.
    """
    for name in ("model_dump", "to_dict"):
        value = _called_json(response, name)
        if value is not _UNSAFE:
            return value
    if isinstance(response, Mapping):
        value = _round_trip(dict(response))
        if value is not _UNSAFE:
            return value
    value = _attributes_json(response)
    if value is not _UNSAFE:
        return value
    return {"repr": repr(response)}


def _called_json(response: Any, name: str) -> Any:
    """response.name() round-tripped, or _UNSAFE when absent, raising or not JSON."""
    try:
        method = getattr(response, name, None)
    except Exception:
        return _UNSAFE
    if not callable(method):
        return _UNSAFE
    try:
        return _round_trip(method())
    except Exception:
        return _UNSAFE


def _attributes_json(response: Any) -> Any:
    """The __dict__ entries JSON can represent, or _UNSAFE when there are none."""
    try:
        data = vars(response)
    except TypeError:
        return _UNSAFE
    picked = {
        key: value
        for key, value in data.items()
        if isinstance(key, str) and _round_trip(value) is not _UNSAFE
    }
    return _round_trip(picked) if picked else _UNSAFE


def _round_trip(value: Any) -> Any:
    """value through json and back, or _UNSAFE when json cannot represent it."""
    try:
        return json.loads(json.dumps(value, sort_keys=True))
    except (TypeError, ValueError):
        return _UNSAFE
