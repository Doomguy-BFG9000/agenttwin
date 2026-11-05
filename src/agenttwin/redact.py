"""Write-time redaction for AgentTwin tapes (CONTRACTS §1.1 and §1.4).

Pure: no IO, no randomness, stdlib only.  `Redactor.payload` returns a
scrubbed copy of an event payload plus the number of substitutions it made,
which the recorder writes as `header["redaction"]["redacted"]`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "MANDATORY_PAYLOAD_KEYS",
    "DEFAULT_DROP_KEYS",
    "DEFAULT_PATTERNS",
    "MODES",
    "Redactor",
]

MODES: tuple[str, ...] = ("scrub", "whitelist")

# CONTRACTS §1.4, "必需 payload 键" column.
MANDATORY_PAYLOAD_KEYS: dict[str, tuple[str, ...]] = {
    "llm_call": ("prompt_hash", "response"),
    "tool_call": ("tool", "args_hash"),
    "tool_result": ("tool", "result"),
    "state_patch": ("patch",),
    "decision": ("note",),
    "human_input": ("input",),
    "error": ("type", "message"),
    "terminal": ("state", "state_hash"),
}

# Values under these keys are replaced wholesale, matched case-insensitively.
DEFAULT_DROP_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "cookie",
        "set-cookie",
        "session_id",
        "private_key",
    }
)

# (name, regex) applied to every string value; the first matching name wins.
DEFAULT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("openai_sk", r"sk-[A-Za-z0-9_-]{16,}"),
    ("bearer", r"Bearer\s+\S+"),
    ("email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ("long_hex", r"\b[0-9a-f]{32,}\b"),
    ("win_user_path", r"C:\\Users\\[^\\\s]+"),
    ("posix_home_path", r"/(?:Users|home)/[^/\s]+"),
    ("aws_key", r"AKIA[0-9A-Z]{16}"),
)


class Redactor:
    """Write-time redaction policy for one recording session.

    ``mode="scrub"`` keeps every payload key and rewrites matching values.
    ``mode="whitelist"`` additionally drops every payload key that is neither
    mandatory for that kind (see ``MANDATORY_PAYLOAD_KEYS``) nor listed in
    ``whitelist[kind]``.
    """

    def __init__(
        self,
        *,
        drop_keys: Iterable[str] | None = None,
        patterns: Iterable[tuple[str, str]] | None = None,
        whitelist: Mapping[str, Iterable[str]] | None = None,
        mode: str = "scrub",
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES!r}, got {mode!r}")
        self._mode = mode
        self._drop_keys = (
            DEFAULT_DROP_KEYS
            if drop_keys is None
            else frozenset(str(key).lower() for key in drop_keys)
        )
        source = DEFAULT_PATTERNS if patterns is None else patterns
        self._patterns: tuple[tuple[str, Any], ...] = tuple(
            (name, pattern if hasattr(pattern, "search") else re.compile(pattern))
            for name, pattern in source
        )
        self._whitelist: dict[str, frozenset[str]] = {
            kind: frozenset(keys) for kind, keys in (whitelist or {}).items()
        }

    @property
    def mode(self) -> str:
        """The active mode: ``"scrub"`` or ``"whitelist"``."""
        return self._mode

    def payload(self, kind: str, payload: dict) -> tuple[dict, int]:
        """Return ``(scrubbed_copy, redaction_count)``; ``payload`` is untouched."""
        if not isinstance(payload, Mapping):
            raise TypeError(f"payload must be a mapping, got {type(payload).__name__}")
        out: dict[Any, Any] = {}
        count = 0
        for key, value in payload.items():
            if isinstance(key, str) and key.lower() in self._drop_keys:
                out[key] = f"<redacted:{key}>"
                count += 1
                continue
            if self._mode == "whitelist" and not self._is_allowed(kind, key):
                count += 1
                continue
            scrubbed, nested = self._scrub(value)
            out[key] = scrubbed
            count += nested
        return out, count

    def policy(self) -> dict:
        """The policy object stored under ``header["redaction"]["policy"]``."""
        return {
            "mode": self._mode,
            "drop_keys": sorted(self._drop_keys),
            "patterns": [name for name, _ in self._patterns],
            "whitelist": {
                kind: sorted(keys) for kind, keys in sorted(self._whitelist.items())
            },
        }

    def _is_allowed(self, kind: str, key: Any) -> bool:
        if key in MANDATORY_PAYLOAD_KEYS.get(kind, ()):
            return True
        return key in self._whitelist.get(kind, frozenset())

    def _scrub(self, value: Any) -> tuple[Any, int]:
        if isinstance(value, str):
            for name, pattern in self._patterns:
                if pattern.search(value):
                    return f"<redacted:{name}>", 1
            return value, 0
        if isinstance(value, Mapping):
            out: dict[Any, Any] = {}
            count = 0
            for key, item in value.items():
                if isinstance(key, str) and key.lower() in self._drop_keys:
                    out[key] = f"<redacted:{key}>"
                    count += 1
                    continue
                scrubbed, nested = self._scrub(item)
                out[key] = scrubbed
                count += nested
            return out, count
        if isinstance(value, (list, tuple)):
            items = []
            count = 0
            for item in value:
                scrubbed, nested = self._scrub(item)
                items.append(scrubbed)
                count += nested
            return (items if isinstance(value, list) else tuple(items)), count
        return value, 0
