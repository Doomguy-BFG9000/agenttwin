"""Recording and replay proxies for the two effectful capabilities (CONTRACTS §2, §4).

Recording proxies forward to the real callable and append faithful events to the
tape.  Replay proxies consume the tape through a forward-only Cursor, never touch
the network or the filesystem, and raise PromptDrift / ArgsDrift when the
controller asked for something the tape does not hold.

Two rules live here because both sides must agree on them:

* CONTRACTS §9.20 - every *_hash is computed over the *pre-redaction* request or
  arguments, while the recorded _preview / response / result values are scrubbed.
  A sha256 is itself a 32+ hex run, so redaction is applied to the non-hash fields
  only; otherwise the redactor would rewrite the very value replay compares.
* CONTRACTS §9.22 - patch(args=X) means "the tape expects {"args": [], "kwargs": X}",
  so an args patch changes what the drift check compares against, not just display.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

from .drift import ArgsDrift, DriftError, PromptDrift, TapeMiss
from .redact import Redactor
from .trace import Event, Trace, canonical_json, sha256_hex

__all__ = [
    "AGENT_ACTOR",
    "LLM_ACTOR",
    "PREVIEW_LIMIT",
    "TRANSPARENT_KINDS",
    "RecordedToolError",
    "Cursor",
    "RecordingLLM",
    "RecordingToolSet",
    "ReplayLLM",
    "ReplayToolSet",
    "args_hash",
    "args_preview",
    "patch_args_hash",
]

# Actor names the recorder and the replayer both use, so the cursor can match them.
LLM_ACTOR: str = "llm"
AGENT_ACTOR: str = "agent"

# CONTRACTS §9.20: previews are truncated at 4096 characters.
PREVIEW_LIMIT: int = 4096

# Kinds no controller call can consume: they are copied forward verbatim when the
# cursor walks past them (see Cursor._skip_passthrough).
TRANSPARENT_KINDS: tuple[str, ...] = ("state_patch", "decision", "human_input", "error")


class RecordedToolError(RuntimeError):
    """A recorded tool call raised; replay re-raises the recorded error (payload ok=False)."""

    def __init__(self, tool: str, detail: Any) -> None:
        super().__init__(f"recorded tool {tool!r} failed: {detail}")
        self.tool = tool
        self.detail = detail


# --- canonical argument forms (shared by recorder and replayer) --------------


def _arguments(args: tuple, kwargs: Mapping[str, Any]) -> dict:
    return {"args": list(args), "kwargs": dict(kwargs)}


def args_hash(args: tuple, kwargs: Mapping[str, Any]) -> str:
    """CONTRACTS §1.4 - sha256 over canonical JSON of {"args": [...], "kwargs": {...}}."""
    return sha256_hex(canonical_json(_arguments(args, kwargs)))


def patch_args_hash(value: Any) -> str:
    """CONTRACTS §9.22 - a patched argument payload is {"args": [], "kwargs": value}."""
    return sha256_hex(canonical_json({"args": [], "kwargs": value}))


def args_preview(args: tuple, kwargs: Mapping[str, Any]) -> str:
    """The readable, pre-truncation copy of the arguments (redacted later)."""
    return canonical_json(_arguments(args, kwargs))


# --- redaction helpers -------------------------------------------------------


def _redact_payload(
    kind: str, payload: dict, redactor: Redactor | None
) -> tuple[dict, int]:
    """Scrub every non-hash field; hashes are carried through untouched (§9.20)."""
    if redactor is None:
        return payload, 0
    hashes = {key: value for key, value in payload.items() if key.endswith("_hash")}
    scrub = {key: value for key, value in payload.items() if not key.endswith("_hash")}
    scrubbed, count = redactor.payload(kind, scrub)
    scrubbed.update(hashes)
    return scrubbed, count


def _truncate_previews(payload: dict) -> dict:
    for key, value in payload.items():
        if key.endswith("_preview") and isinstance(value, str) and len(value) > PREVIEW_LIMIT:
            payload[key] = value[:PREVIEW_LIMIT]
    return payload


def _tokens(response: Any) -> dict | None:
    """Token counts recorded when the model reports them (§1.4 optional 'tokens')."""
    if not isinstance(response, Mapping):
        return None
    raw = response.get("tokens")
    if isinstance(raw, Mapping):
        picked = {
            name: raw[name]
            for name in ("prompt", "completion")
            if isinstance(raw.get(name), int) and not isinstance(raw.get(name), bool)
        }
        if picked:
            return picked
    usage = response.get("usage")
    if isinstance(usage, Mapping):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if isinstance(prompt, int) and isinstance(completion, int):
            return {"prompt": prompt, "completion": completion}
    return None


def _error_result(exc: BaseException) -> dict:
    return {"type": type(exc).__name__, "message": str(exc)}


# --- the forward-only cursor (CONTRACTS §2) -------------------------------


class Cursor:
    """A forward-only index into a tape's events (CONTRACTS §2).

    'produced' is the Trace the replay is writing: pass it and the cursor will
    carry controller-independent events (state_patch, decision, human_input,
    error) across verbatim instead of blocking on them.  Without it the cursor is
    a plain reader, which is what the proxy unit tests want.
    """

    def __init__(self, tape: Trace, *, produced: Trace | None = None) -> None:
        self.tape = tape
        self.produced = produced
        self.index = 0
        self.misses: list[TapeMiss] = []
        self.drifts: list[DriftError] = []

    def __len__(self) -> int:
        return len(self.tape.events)

    def at_end(self) -> bool:
        return self.index >= len(self.tape.events)

    def take(self, kind: str, actor: str) -> Event | None:
        """Consume the event at the cursor iff it is (kind, actor); else record a miss.

        A miss never moves the cursor (CONTRACTS §9.19) and never raises (§9.6).
        """
        self._skip_passthrough()
        if self.index >= len(self.tape.events):
            return self._miss(kind, actor)
        event = self.tape.events[self.index]
        if event.kind != kind or event.actor != actor:
            return self._miss(kind, actor)
        self.index += 1
        return event

    def fail(self, drift: DriftError) -> DriftError:
        """Record a hard drift on this cursor and hand it back for raising."""
        self.drifts.append(drift)
        return drift

    def _skip_passthrough(self) -> None:
        while self.index < len(self.tape.events):
            event = self.tape.events[self.index]
            if event.kind not in TRANSPARENT_KINDS:
                return
            if self.produced is not None:
                self.produced.events.append(copy.deepcopy(event))
            self.index += 1

    def _miss(self, kind: str, actor: str) -> None:
        self.misses.append(TapeMiss(self.index, kind, actor))
        return None


# --- recording proxies -------------------------------------------------------


class RecordingLLM:
    """Callable that forwards to the real llm and records one llm_call per call."""

    def __init__(
        self,
        trace: Trace,
        llm: Callable[[dict], Any],
        *,
        redactor: Redactor | None = None,
        actor: str = LLM_ACTOR,
        model: str | None = None,
    ) -> None:
        self.trace = trace
        self.actor = actor
        self.model = model
        self.redacted = 0
        self._llm = llm
        self._redactor = redactor

    def __call__(self, request: dict) -> Any:
        prompt_hash = sha256_hex(canonical_json(request))
        response = self._llm(request)
        payload: dict[str, Any] = {
            "prompt_hash": prompt_hash,
            "response": response,
            "prompt_preview": canonical_json(request),
        }
        if self.model is not None:
            payload["model"] = self.model
        tokens = _tokens(response)
        if tokens is not None:
            payload["tokens"] = tokens
        scrubbed, count = _redact_payload("llm_call", payload, self._redactor)
        self.redacted += count
        self.trace.append("llm_call", self.actor, _truncate_previews(scrubbed))
        return response


class RecordingToolSet:
    """Wraps a real tool mapping so each call records tool_call + tool_result."""

    def __init__(
        self,
        trace: Trace,
        tools: Mapping[str, Callable[..., Any]],
        *,
        redactor: Redactor | None = None,
    ) -> None:
        self.trace = trace
        self.redacted = 0
        self._tools = dict(tools)
        self._redactor = redactor

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Add or replace one tool (the instrumentation layer wraps tools after start)."""
        self._tools[str(name)] = fn

    def tools(self) -> Mapping[str, Callable[..., Any]]:
        return {name: self._wrap(name, fn) for name, fn in self._tools.items()}

    def _wrap(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            payload: dict[str, Any] = {
                "tool": name,
                "args_hash": args_hash(args, kwargs),
                "args_preview": args_preview(args, kwargs),
            }
            scrubbed, count = _redact_payload("tool_call", payload, self._redactor)
            self.redacted += count
            self.trace.append("tool_call", name, _truncate_previews(scrubbed))
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                failed: dict[str, Any] = {
                    "tool": name,
                    "result": _error_result(exc),
                    "ok": False,
                }
                scrubbed, count = _redact_payload("tool_result", failed, self._redactor)
                self.redacted += count
                self.trace.append("tool_result", name, scrubbed)
                raise
            done: dict[str, Any] = {"tool": name, "result": result, "ok": True}
            scrubbed, count = _redact_payload("tool_result", done, self._redactor)
            self.redacted += count
            self.trace.append("tool_result", name, scrubbed)
            return result

        call.__name__ = name
        return call


# --- replay proxies ----------------------------------------------------------


def _emit(produced: Trace | None, event: Event, payload: dict, *, patched: bool) -> None:
    """Mirror one consumed tape event into the produced trace (CONTRACTS §2)."""
    if produced is None:
        return
    produced.append(event.kind, event.actor, payload, patched=patched)


class ReplayLLM:
    """Callable that answers from the tape, raising PromptDrift when the call changed."""

    def __init__(
        self,
        cursor: Cursor,
        *,
        patches: Mapping[int, Mapping[str, Any]] | None = None,
        actor: str = LLM_ACTOR,
    ) -> None:
        self._cursor = cursor
        self._patches = patches or {}
        self._actor = actor

    def __call__(self, request: dict) -> Any:
        event = self._cursor.take("llm_call", self._actor)
        if event is None:
            return None
        actual = sha256_hex(canonical_json(request))
        expected = event.payload["prompt_hash"]
        if actual != expected:
            raise self._cursor.fail(
                PromptDrift(
                    f"prompt_hash mismatch at seq {event.seq}: "
                    f"tape expects {expected}, controller sent {actual}",
                    seq=event.seq,
                    expected=expected,
                    actual=actual,
                )
            )
        payload = copy.deepcopy(event.payload)
        patch = self._patches.get(event.seq) or {}
        patched = "response" in patch
        if patched:
            payload["response"] = copy.deepcopy(patch["response"])
        _emit(self._cursor.produced, event, payload, patched=patched)
        return payload["response"]


class _ToolMapping(Mapping[str, Callable[..., Any]]):
    """Read-only tool mapping: the tape's tools plus a lazy wrapper for any other name.

    An unknown tool name therefore lands as a TapeMiss rather than a KeyError,
    which keeps CONTRACTS §2's "type/actor mismatch is a miss, not a crash" true.
    """

    def __init__(self, names: list[str], factory: Callable[[str], Callable[..., Any]]) -> None:
        self._names = list(names)
        self._factory = factory
        self._cache: dict[str, Callable[..., Any]] = {}

    def __getitem__(self, name: str) -> Callable[..., Any]:
        if name not in self._cache:
            self._cache[name] = self._factory(name)
        return self._cache[name]

    def __iter__(self) -> Any:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


class ReplayToolSet:
    """Serves tool calls from the tape: one tool_call plus its paired tool_result."""

    def __init__(
        self,
        cursor: Cursor,
        *,
        patches: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> None:
        self._cursor = cursor
        self._patches = patches or {}

    def tools(self) -> Mapping[str, Callable[..., Any]]:
        names = list(
            dict.fromkeys(
                event.actor for event in self._cursor.tape.events if event.kind == "tool_call"
            )
        )
        return _ToolMapping(names, self._wrap)

    def _wrap(self, name: str) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            return self._call(name, args, kwargs)

        call.__name__ = name
        return call

    def _call(self, name: str, args: tuple, kwargs: Mapping[str, Any]) -> Any:
        event = self._cursor.take("tool_call", name)
        if event is None:
            return None
        patch = self._patches.get(event.seq) or {}
        expected = event.payload["args_hash"]
        if "args" in patch:
            expected = patch_args_hash(patch["args"])
        actual = args_hash(args, kwargs)
        if actual != expected:
            raise self._cursor.fail(
                ArgsDrift(
                    f"args_hash mismatch at seq {event.seq}: "
                    f"tape expects {expected}, controller sent {actual}",
                    seq=event.seq,
                    expected=expected,
                    actual=actual,
                )
            )
        payload = copy.deepcopy(event.payload)
        patched = "args" in patch
        if patched:
            payload["args_hash"] = expected
            payload["args_preview"] = args_preview((), patch["args"])[:PREVIEW_LIMIT]
        _emit(self._cursor.produced, event, payload, patched=patched)

        result_event = self._cursor.take("tool_result", name)
        if result_event is None:
            return None
        result_payload = copy.deepcopy(result_event.payload)
        result_patch = self._patches.get(result_event.seq) or {}
        result_patched = "result" in result_patch
        if result_patched:
            result_payload["result"] = copy.deepcopy(result_patch["result"])
        _emit(
            self._cursor.produced,
            result_event,
            result_payload,
            patched=result_patched,
        )
        if result_payload.get("ok", True) is False:
            raise RecordedToolError(name, result_payload.get("result"))
        return result_payload["result"]
