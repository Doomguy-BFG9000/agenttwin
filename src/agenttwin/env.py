"""The record/replay facade (CONTRACTS section 3): Twin, TapeRecorder, ReplaySession.

Twin.record() wraps the real llm and the real tools so that one call becomes one
tape event; Twin.load() replays that tape through forward-only proxies.  The only
IO here is reading and writing the tape file: recording never depends on the
replayer, and the replayer never touches a real llm or a real tool.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .drift import DriftError, FingerprintDrift
from .fingerprint import capture, compare
from .proxy import (
    AGENT_ACTOR,
    Cursor,
    RecordingLLM,
    RecordingToolSet,
    ReplayLLM,
    ReplayToolSet,
)
from .redact import Redactor
from .trace import SCHEMA_VERSION, Header, Trace, canonical_json, sha256_hex

__all__ = ["Twin", "TapeRecorder", "ReplaySession"]

_UNSET: Any = object()

# CONTRACTS 9.21: agenttwin is always a hard failure; model only when the caller
# states the model it replays with, because replay itself never loads one, and an
# unknown current value is not the same as a different one. python / platform /
# deps / prompt_hash differences are informational (ReplaySession.fingerprint_diff).


class Twin:
    """CONTRACTS 9.2 -- the entry points of the whole library (section 3)."""

    @classmethod
    def record(
        cls,
        trace_id: str,
        *,
        out: str | Path,
        seed: int = 0,
        llm: Callable[[dict], Any] | None = None,
        tools: Mapping[str, Callable[..., Any]] | None = None,
        model: str | None = None,
        redactor: Redactor | None = None,
    ) -> "TapeRecorder":
        """Start a recording session whose tape is written to out on exit."""
        return TapeRecorder(
            trace_id,
            out=out,
            seed=seed,
            llm=llm,
            tools=tools,
            model=model,
            redactor=redactor,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        model: str | None = None,
        allow_fingerprint_drift: bool = False,
        allow_real: bool = False,
    ) -> "ReplaySession":
        """Open a recorded tape for replay; never performs a real call.

        model names the llm the caller is about to replay against. Replay never
        loads one, so an omitted model is unknown rather than different: a model
        row the current side cannot fill stays informational in fingerprint_diff.
        """
        if allow_real:
            raise ValueError(
                "allow_real=True is not supported: real tools exist only in record mode"
                " (Twin.record), never during replay"
            )
        tape = Trace.load(path)
        return ReplaySession(
            tape,
            source=path,
            model=model,
            allow_fingerprint_drift=allow_fingerprint_drift,
        )

    @classmethod
    def from_openai_compatible(cls, client: Any, **kwargs: Any) -> Any:
        """CONTRACTS section 3 / design document section 4.2 -- the zero-config record path.

        This is the L2 record path and nothing more: it opens a tape (out=, or
        AGENTTWIN_RECORD_OUT), returns a client proxy whose
        chat.completions.create writes one llm_call per call and hands the real
        response back, and writes that tape on interpreter exit -- call flush() on
        the proxy to write it sooner.  It is not usable during replay: replay has
        no client and performs no real call, so a tape is replayable only through
        Twin.load().

        The import is local because adapters.openai_compatible imports this module
        for Twin.record; a module-level import here would be a cycle.
        """
        from .adapters.openai_compatible import install

        return install(client, **kwargs)


class TapeRecorder:
    """CONTRACTS section 3 -- one recording session; use it as a context manager."""

    def __init__(
        self,
        trace_id: str,
        *,
        out: str | Path,
        seed: int = 0,
        llm: Callable[[dict], Any] | None = None,
        tools: Mapping[str, Callable[..., Any]] | None = None,
        model: str | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self.trace = Trace(
            header=Header(
                schema_version=SCHEMA_VERSION,
                trace_id=str(trace_id),
                seed=int(seed),
                env_fingerprint=capture(model=model),
                redaction=_redaction_header(redactor, 0),
            )
        )
        self.out = Path(out)
        self.redactor = redactor
        self._llm = RecordingLLM(self.trace, llm or _no_llm, redactor=redactor, model=model)
        self._tools = RecordingToolSet(self.trace, dict(tools or {}), redactor=redactor)
        self._returned: Any = None
        self._closed = False

    @property
    def llm(self) -> RecordingLLM:
        """The recording llm callable: one call in, one llm_call event out."""
        return self._llm

    def tools(self) -> Mapping[str, Callable[..., Any]]:
        """The recording tool mapping: one call in, a tool_call + tool_result pair."""
        return self._tools.tools()

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Add a tool to the mapping returned by tools()."""
        self._tools.register(name, fn)

    def patch_state(self, pointer: str, value: Any) -> None:
        """Record one state_patch event (CONTRACTS 1.4)."""
        self.trace.append("state_patch", AGENT_ACTOR, {"patch": {str(pointer): copy.deepcopy(value)}})

    def returned(self, value: Any) -> None:
        """Hand the agent return value to the recorder; a dict overrides the folded state."""
        self._returned = value

    def __enter__(self) -> "TapeRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._finish(exc)
        return False

    def _finish(self, exc: BaseException | None) -> None:
        if self._closed:
            return
        self._closed = True
        state = self.trace.state()
        if exc is not None:
            self.trace.append("error", AGENT_ACTOR, {"type": type(exc).__name__, "message": str(exc)})
            outcome = "error"
        else:
            outcome = "ok"
            if isinstance(self._returned, Mapping):
                state.update(copy.deepcopy(dict(self._returned)))
        self.trace.append(
            "terminal",
            AGENT_ACTOR,
            {
                "state": state,
                "state_hash": sha256_hex(canonical_json(state)),
                "outcome": outcome,
            },
        )
        header = self.trace.header
        self.trace.header = Header(
            schema_version=header.schema_version,
            trace_id=header.trace_id,
            seed=header.seed,
            env_fingerprint=header.env_fingerprint,
            redaction=_redaction_header(
                self.redactor, self._llm.redacted + self._tools.redacted
            ),
        )
        self.trace.save(self.out)


class ReplaySession:
    """CONTRACTS section 3 -- one loaded tape, ready to replay a controller against it."""

    def __init__(
        self,
        tape: Trace,
        *,
        source: str | Path | None = None,
        model: str | None = None,
        allow_fingerprint_drift: bool = False,
    ) -> None:
        self.tape = tape
        self.source = None if source is None else Path(source)
        self.model = model
        self.allow_fingerprint_drift = bool(allow_fingerprint_drift)
        self.misses: list = []
        self.drifts: list[DriftError] = []
        self._patches: dict[int, dict[str, Any]] = {}
        self.fingerprint_diff = compare(tape.header.env_fingerprint, capture(model=model))
        fatal = [
            item
            for item in self.fingerprint_diff
            if item.get("field") == "agenttwin"
            or (item.get("field") == "model" and model is not None)
        ]
        if fatal and not self.allow_fingerprint_drift:
            raise FingerprintDrift(
                "env_fingerprint mismatch: "
                + ", ".join(
                    f"{item['field']} tape={item['expected']!r} now={item['actual']!r}"
                    for item in fatal
                ),
                expected=fatal[0]["expected"],
                actual=fatal[0]["actual"],
            )

    def patch(
        self,
        *,
        seq: int,
        response: Any = _UNSET,
        result: Any = _UNSET,
        args: Any = _UNSET,
    ) -> "ReplaySession":
        """CONTRACTS 9.22 -- the original session is unchanged: use the return value.

        Returns a clone carrying the patch. Replaying the original replays the
        unpatched tape and reports green, so a dropped return value silently
        ships an unpatched run.
        """
        clone = copy.copy(self)
        clone._patches = {key: dict(value) for key, value in self._patches.items()}
        entry = clone._patches.setdefault(int(seq), {})
        if response is not _UNSET:
            entry["response"] = response
        if result is not _UNSET:
            entry["result"] = result
        if args is not _UNSET:
            entry["args"] = args
        clone.misses = []
        clone.drifts = []
        return clone

    def replay(self, agent: Any, *, policy: Any = None) -> Any:
        """CONTRACTS 9.8 -- run agent(llm, tools) against the tape; return a Report."""
        produced = Trace(header=copy.deepcopy(self.tape.header))
        cursor = Cursor(self.tape, produced=produced)
        llm = ReplayLLM(cursor, patches=self._patches)
        tools = ReplayToolSet(cursor, patches=self._patches).tools()
        try:
            _invoke_agent(agent, llm, tools, policy)
        finally:
            self.misses = list(cursor.misses)
            self.drifts = list(cursor.drifts)

        terminal = _last_terminal(self.tape)
        consumed = cursor.index
        clean = not cursor.misses and not cursor.drifts
        if (
            terminal is not None
            and clean
            and consumed == terminal.seq
            and len(produced.events) == terminal.seq
        ):
            produced.events.append(copy.deepcopy(terminal))
            divergent = False
        else:
            if terminal is not None:
                produced.append("terminal", terminal.actor, copy.deepcopy(terminal.payload))
            divergent = True
        return _build_report(self.tape, produced, self.misses, self.drifts, divergent)


# --- helpers ---------------------------------------------------------------


def _redaction_header(redactor: Redactor | None, redacted: int) -> dict:
    if redactor is None:
        return {
            "redacted": int(redacted),
            "policy": {"mode": "scrub", "drop_keys": [], "patterns": [], "whitelist": {}},
            "mode": "scrub",
        }
    return {"redacted": int(redacted), "policy": redactor.policy(), "mode": redactor.mode}


def _no_llm(request: dict) -> Any:
    raise RuntimeError("Twin.record() was given no llm callable")


def _last_terminal(tape: Trace):
    for event in reversed(tape.events):
        if event.kind == "terminal":
            return event
    return None


def _invoke_agent(agent: Any, llm: Any, tools: Mapping[str, Callable[..., Any]], policy: Any) -> Any:
    """Call agent the way adapters.base.invoke does (CONTRACTS section 3).

    The rule is not restated here: one implementation decides between the three
    call shapes, and it passes the policy whenever the signature accepts it --
    including None, so a controller with a required third positional stays
    callable under replay(agent) with no policy.  The import is local because
    adapters/__init__ reaches back into this package's facade.
    """
    from .adapters.base import invoke

    return invoke(agent, llm, tools, policy)


def _build_report(baseline: Trace, trace: Trace, misses: list, drifts: list, divergent: bool) -> Any:
    from .report import Report

    return Report(
        baseline=baseline,
        trace=trace,
        results=[],
        misses=misses,
        drifts=drifts,
        mode="replay",
        divergent=divergent,
    )
