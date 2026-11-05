"""AgentTwin -- record one agent run, replay it forever (CONTRACTS section 3).

The tape layer (Trace, Event, Header, canonical_json, sha256_hex) is imported
eagerly.  Everything that reaches the replayer is resolved by name on first
access, so importing agenttwin never depends on the replay side of the library
being importable -- in particular not on agenttwin.diff.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .drift import ArgsDrift, DriftError, FingerprintDrift, PromptDrift, TapeMiss
from .redact import Redactor
from .trace import (
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

__version__: str = "0.1.0"

_LAZY: dict[str, str] = {
    "Twin": ".env",
    "TapeRecorder": ".env",
    "ReplaySession": ".env",
    "Report": ".report",
    "TraceDiff": ".diff",
}

__all__ = [
    "ArgsDrift",
    "CLOCK_STEP",
    "DriftError",
    "EVENT_KINDS",
    "Event",
    "FingerprintDrift",
    "Header",
    "PromptDrift",
    "Redactor",
    "Report",
    "ReplaySession",
    "SCHEMA_VERSION",
    "SchemaError",
    "TapeMiss",
    "TapeRecorder",
    "Trace",
    "TraceDiff",
    "Twin",
    "canonical_json",
    "sha256_hex",
]


def __getattr__(name: str) -> Any:
    """PEP 562: resolve the replay-side names on first access, then cache them."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
