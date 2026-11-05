"""Drift and miss types shared by the replay proxy (CONTRACTS §4).

A ``DriftError`` is a hard failure: replay stops and the error is recorded in
``Report.drifts``.  A ``TapeMiss`` is a structured result, not a crash: the
controller keeps running with a ``None`` observation (CONTRACTS §9.6, §9.19).

Pure: no IO, no clock, no third-party package.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DriftError",
    "PromptDrift",
    "ArgsDrift",
    "FingerprintDrift",
    "TapeMiss",
]


class DriftError(Exception):
    """Base class for the replay failures that abort a run (CONTRACTS §4)."""

    type: str = "DriftError"

    def __init__(
        self,
        message: str,
        *,
        seq: int | None = None,
        expected: Any = None,
        actual: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        self.seq = seq
        self.expected = expected
        self.actual = actual

    def to_dict(self) -> dict:
        """The ``Report.json()`` shape for one drift (CONTRACTS §4, §8)."""
        return {
            "type": self.type,
            "seq": self.seq,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
        }


class PromptDrift(DriftError):
    """The live llm request hashes differently from the recorded one (CONTRACTS §1.4)."""

    type = "PromptDrift"


class ArgsDrift(DriftError):
    """The live tool arguments hash differently from the recorded ones (CONTRACTS §1.4)."""

    type = "ArgsDrift"


class FingerprintDrift(DriftError):
    """The tape's env_fingerprint no longer matches this machine (CONTRACTS §1.5, §9.21)."""

    type = "FingerprintDrift"


class TapeMiss(Exception):
    """The controller asked for an event the tape does not hold at the cursor (§4, §9.19).

    Not fatal by design: the proxy records the miss in the report and returns
    ``None`` without moving the cursor, because "skip ahead to the next matching
    event" would align the replay against a different point in time.
    """

    def __init__(self, seq: int, expected_kind: str, actor: str) -> None:
        super().__init__(
            f"tape miss at seq {seq}: expected {expected_kind} by {actor!r}"
        )
        self.seq = seq
        self.expected_kind = expected_kind
        self.actor = actor

    def to_dict(self) -> dict:
        """CONTRACTS §4 — exactly these four keys."""
        return {
            "type": "TapeMiss",
            "seq": self.seq,
            "expected_kind": self.expected_kind,
            "actor": self.actor,
        }
