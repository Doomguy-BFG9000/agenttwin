"""AgentTwin — record an LLM agent run once, replay it forever.

Only the tape layer exists so far: env, proxy, asserts, diff and report land on
top of the Trace/Event API re-exported here (CONTRACTS §1, §2).
"""

from __future__ import annotations

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

__version__ = "0.1.0"

__all__ = [
    "CLOCK_STEP",
    "EVENT_KINDS",
    "SCHEMA_VERSION",
    "Event",
    "Header",
    "SchemaError",
    "Trace",
    "canonical_json",
    "sha256_hex",
]
