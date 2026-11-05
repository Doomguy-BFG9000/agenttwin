"""JSONL tape — schema 0.1 (CONTRACTS §1) and the seq rules replay depends on (§2).

Trace is the only durable artefact AgentTwin produces; proxy, diff, asserts and
report all read Trace/Event from here, so the surface below is frozen: adding a
key or a kind means bumping SCHEMA_VERSION. Nothing here touches the network,
the wall clock, or a third-party package.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "CLOCK_STEP",
    "SCHEMA_VERSION",
    "EVENT_KINDS",
    "SchemaError",
    "canonical_json",
    "sha256_hex",
    "Event",
    "Header",
    "Trace",
]

CLOCK_STEP: float = 0.1
SCHEMA_VERSION: str = "0.1"

# CONTRACTS §1.4, in table order. Exactly these eight, no aliases.
EVENT_KINDS: tuple[str, ...] = (
    "llm_call",
    "tool_call",
    "tool_result",
    "state_patch",
    "decision",
    "human_input",
    "error",
    "terminal",
)

# CONTRACTS §1.4 required payload keys per kind. Optional keys are not a closed
# set ("decision" allows arbitrary extra keys), so extra payload keys are never
# rejected — only §1's top-level keys are a closed set.
REQUIRED_PAYLOAD: dict[str, tuple[str, ...]] = {
    "llm_call": ("prompt_hash", "response"),
    "tool_call": ("tool", "args_hash"),
    "tool_result": ("tool", "result"),
    "state_patch": ("patch",),
    "decision": ("note",),
    "human_input": ("input",),
    "error": ("type", "message"),
    "terminal": ("state", "state_hash"),
}

# Top-level keys are closed sets on both header and event: schema drift must be
# loud rather than silently dropped (CONTRACTS §1, §9.5).
_HEADER_KEYS: tuple[str, ...] = (
    "schema_version",
    "trace_id",
    "seed",
    "env_fingerprint",
    "redaction",
)
_EVENT_KEYS: tuple[str, ...] = ("seq", "t", "kind", "actor", "payload")
_EVENT_OPTIONAL_KEYS: tuple[str, ...] = ("patched",)

_TOLERANCE: float = 1e-6


class SchemaError(Exception):
    """A tape line breaks CONTRACTS §1; the message names the line and the field."""


def canonical_json(obj: Any) -> str:
    """CONTRACTS §1.3 — the single canonical form behind every *_hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=repr)


def sha256_hex(text: str) -> str:
    """CONTRACTS §1.3 — 64 lowercase hex chars over the UTF-8 bytes of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Event:
    """One tape line (CONTRACTS §1.2)."""

    seq: int
    t: float
    kind: str
    actor: str
    payload: dict
    patched: bool = False

    def to_json(self) -> dict:
        """Serialise for the tape; patched is written only when true (CONTRACTS §1.2)."""
        obj: dict[str, Any] = {
            "seq": self.seq,
            "t": self.t,
            "kind": self.kind,
            "actor": self.actor,
            "payload": self.payload,
        }
        if self.patched:
            obj["patched"] = True
        return obj

    @classmethod
    def from_json(cls, obj: dict) -> "Event":
        return _validate_event(obj, "event")


@dataclass(frozen=True)
class Header:
    """The tape's first line (CONTRACTS §1.1)."""

    schema_version: str
    trace_id: str
    seed: int
    env_fingerprint: dict
    redaction: dict

    def to_json(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "seed": self.seed,
            "env_fingerprint": self.env_fingerprint,
            "redaction": self.redaction,
        }

    @classmethod
    def from_json(cls, obj: dict) -> "Header":
        return _validate_header(obj, "header")


@dataclass
class Trace:
    """A whole tape: one header plus the events in seq order (CONTRACTS §1, §2)."""

    header: Header
    events: list[Event] = field(default_factory=list)

    def append(self, kind: str, actor: str, payload: dict, *, patched: bool = False) -> Event:
        """Append the next event; seq and t come from the tape, never from the caller (§9.4)."""
        seq = len(self.events)
        event = Event(
            seq=seq,
            t=round(seq * CLOCK_STEP, 4),
            kind=kind,
            actor=actor,
            payload=payload,
            patched=patched,
        )
        self.events.append(event)
        return event

    def event(self, seq: int) -> Event:
        """The event at cursor position seq (§2: the cursor is an index into events)."""
        if not _is_int(seq) or not 0 <= seq < len(self.events):
            raise IndexError(f"seq {seq!r} is outside the tape (0..{len(self.events) - 1})")
        return self.events[seq]

    def state(self) -> dict:
        """CONTRACTS §1.4 — fold state_patch by seq, then let terminal payload win."""
        state: dict[str, Any] = {}
        terminal_state: dict | None = None
        for event in self.events:
            if event.kind == "state_patch":
                for pointer, value in event.payload["patch"].items():
                    _fold_patch(state, pointer, value, f"state_patch seq {event.seq}")
            elif event.kind == "terminal":
                terminal_state = event.payload["state"]
        if terminal_state is not None:
            return copy.deepcopy(terminal_state)
        return state

    def copy(self) -> "Trace":
        """Deep copy: replay hands one Trace to the diff and keeps the baseline intact (§2)."""
        return Trace(header=copy.deepcopy(self.header), events=copy.deepcopy(self.events))

    def to_jsonl(self) -> str:
        """CONTRACTS §1 — header line, one event per line, LF, trailing newline."""
        lines = [canonical_json(self.header.to_json())]
        lines.extend(canonical_json(event.to_json()) for event in self.events)
        return "\n".join(lines) + "\n"

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_jsonl(), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        return cls.from_jsonl(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_jsonl(cls, text: str) -> "Trace":
        """Parse and validate a whole tape (CONTRACTS §1). Raises SchemaError, never returns junk."""
        if not text:
            raise SchemaError("line 1: tape is empty, expected a header")
        lines = text.split("\n")
        if lines[-1] == "":
            lines.pop()
        header: Header | None = None
        events: list[Event] = []
        for number, line in enumerate(lines, start=1):
            where = f"line {number}"
            if not line.strip():
                raise SchemaError(f"{where}: blank line is not allowed")
            obj = _loads_line(line, where)
            if number == 1:
                header = _validate_header(obj, where)
                continue
            event = _validate_event(obj, where)
            expected = len(events)
            if event.seq != expected:
                raise SchemaError(
                    f"{where}: expected seq {expected} but found {event.seq} "
                    "(seq must start at 0, be dense and strictly increasing)"
                )
            events.append(event)
        assert header is not None
        return cls(header=header, events=events)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def __getitem__(self, index: int) -> Event:
        return self.events[index]


# --- validation and folding helpers (private: the public surface is above) ---


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _loads_line(line: str, where: str) -> Any:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{where}: malformed JSON: {exc.msg} (column {exc.colno})") from None


def _validate_header(obj: Any, where: str) -> Header:
    if not isinstance(obj, dict):
        raise SchemaError(f"{where}: header must be a JSON object, got {type(obj).__name__}")
    if "schema_version" not in obj:
        raise SchemaError(f"{where}: header is missing the key 'schema_version'")
    unknown = sorted(set(obj) - set(_HEADER_KEYS))
    if unknown:
        raise SchemaError(f"{where}: unknown header key {unknown[0]!r} (allowed: {', '.join(_HEADER_KEYS)})")
    for key in _HEADER_KEYS:
        if key not in obj:
            raise SchemaError(f"{where}: missing required header key {key!r}")
    version = obj["schema_version"]
    if version != SCHEMA_VERSION:
        raise SchemaError(f"{where}: unsupported schema_version {version!r} (expected {SCHEMA_VERSION!r})")
    trace_id = obj["trace_id"]
    if not isinstance(trace_id, str):
        raise SchemaError(f"{where}: header 'trace_id' must be a string, got {trace_id!r}")
    seed = obj["seed"]
    if not _is_int(seed):
        raise SchemaError(f"{where}: header 'seed' must be an integer, got {seed!r}")
    env_fingerprint = obj["env_fingerprint"]
    if not isinstance(env_fingerprint, dict):
        raise SchemaError(f"{where}: header 'env_fingerprint' must be an object, got {env_fingerprint!r}")
    redaction = obj["redaction"]
    if not isinstance(redaction, dict):
        raise SchemaError(f"{where}: header 'redaction' must be an object, got {redaction!r}")
    return Header(
        schema_version=version,
        trace_id=trace_id,
        seed=seed,
        env_fingerprint=env_fingerprint,
        redaction=redaction,
    )


def _validate_event(obj: Any, where: str) -> Event:
    if not isinstance(obj, dict):
        raise SchemaError(f"{where}: event must be a JSON object, got {type(obj).__name__}")
    unknown = sorted(set(obj) - set(_EVENT_KEYS) - set(_EVENT_OPTIONAL_KEYS))
    if unknown:
        raise SchemaError(
            f"{where}: unknown event key {unknown[0]!r} "
            f"(allowed: {', '.join(_EVENT_KEYS + _EVENT_OPTIONAL_KEYS)})"
        )
    for key in _EVENT_KEYS:
        if key not in obj:
            raise SchemaError(f"{where}: missing required event key {key!r}")

    seq = obj["seq"]
    if not _is_int(seq) or seq < 0:
        raise SchemaError(f"{where}: event 'seq' must be a non-negative integer, got {seq!r}")
    t = obj["t"]
    if not _is_number(t):
        raise SchemaError(f"{where}: event 't' must be a number, got {t!r}")
    kind = obj["kind"]
    if not isinstance(kind, str):
        raise SchemaError(f"{where}: event 'kind' must be a string, got {kind!r}")
    if kind not in EVENT_KINDS:
        raise SchemaError(f"{where}: unknown event kind {kind!r} (expected one of: {', '.join(EVENT_KINDS)})")
    actor = obj["actor"]
    if not isinstance(actor, str):
        raise SchemaError(f"{where}: event 'actor' must be a string, got {actor!r}")
    payload = obj["payload"]
    if not isinstance(payload, dict):
        raise SchemaError(f"{where}: event 'payload' must be an object, got {payload!r}")
    patched = obj.get("patched", False)
    if not isinstance(patched, bool):
        raise SchemaError(f"{where}: event 'patched' must be a boolean, got {patched!r}")

    expected_t = round(seq * CLOCK_STEP, 4)
    if abs(t - expected_t) > _TOLERANCE:
        raise SchemaError(
            f"{where}: t={t!r} does not match seq {seq} "
            f"(expected t={expected_t!r}, tolerance 1e-06)"
        )

    for key in REQUIRED_PAYLOAD[kind]:
        if key not in payload:
            raise SchemaError(f"{where}: payload for kind {kind!r} is missing required key {key!r}")
    if kind == "state_patch":
        patch = payload["patch"]
        if not isinstance(patch, dict):
            raise SchemaError(f"{where}: state_patch payload 'patch' must be an object, got {patch!r}")
        for pointer in patch:
            if not isinstance(pointer, str) or not pointer.startswith("/"):
                raise SchemaError(f"{where}: state_patch key {pointer!r} must start with '/'")
    if kind == "terminal" and not isinstance(payload["state"], dict):
        raise SchemaError(f"{where}: terminal payload 'state' must be an object, got {payload['state']!r}")

    return Event(seq=seq, t=float(t), kind=kind, actor=actor, payload=payload, patched=patched)


def _pointer_tokens(pointer: Any, where: str) -> list[str]:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise SchemaError(f"{where}: state_patch key {pointer!r} must start with '/'")
    return [token.replace("~1", "/").replace("~0", "~") for token in pointer.split("/")[1:]]


def _list_index(token: str, where: str) -> int:
    if not token.isdigit():
        raise SchemaError(f"{where}: state_patch token {token!r} is not a list index")
    return int(token)


def _child_container(node: Any, token: str, next_token: str, where: str) -> Any:
    if isinstance(node, list):
        position = _list_index(token, where)
        while len(node) <= position:
            node.append(None)
        child = node[position]
        if not isinstance(child, (dict, list)):
            child = [] if next_token.isdigit() else {}
            node[position] = child
        return child
    child = node.get(token)
    if not isinstance(child, (dict, list)):
        child = [] if next_token.isdigit() else {}
        node[token] = child
    return child


def _fold_patch(state: dict, pointer: str, value: Any, where: str) -> None:
    tokens = _pointer_tokens(pointer, where)
    node: Any = state
    for index, token in enumerate(tokens[:-1]):
        node = _child_container(node, token, tokens[index + 1], where)
    leaf = tokens[-1]
    if isinstance(node, list):
        position = _list_index(leaf, where)
        while len(node) <= position:
            node.append(None)
        node[position] = copy.deepcopy(value)
    else:
        node[leaf] = copy.deepcopy(value)
