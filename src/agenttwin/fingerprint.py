"""Environment fingerprint capture and comparison (CONTRACTS §1.5).

Pure: no IO, no randomness, stdlib only.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["AGENTTWIN_VERSION", "FATAL_FIELDS", "capture", "compare", "is_fatal"]

# Only used if the package cannot import itself; the real value is __version__,
# so there is exactly one place to bump.
_DEFAULT_VERSION = "0.0.0"

AGENTTWIN_VERSION: str = _DEFAULT_VERSION
try:
    import agenttwin as _agenttwin
except Exception:
    _agenttwin = None
else:
    _version = getattr(_agenttwin, "__version__", None)
    if isinstance(_version, str) and _version:
        AGENTTWIN_VERSION = _version

# Differences on these fields are a hard FingerprintDrift (CONTRACTS §1.5).
FATAL_FIELDS: tuple[str, ...] = ("model", "prompt_hash", "python", "agenttwin")

_FIELD_ORDER: tuple[str, ...] = (
    "model",
    "prompt_hash",
    "python",
    "agenttwin",
    "platform",
    "deps",
)
_MAJOR_MINOR_FIELDS: frozenset[str] = frozenset({"python", "agenttwin"})
_BOTH_SIDES_FIELDS: frozenset[str] = frozenset({"platform", "deps"})
_MAJOR_MINOR_RE = re.compile(r"(\d+)\.(\d+)")


def capture(
    *,
    model: str | None = None,
    prompt_hash: str | None = None,
    deps: Mapping[str, str] | None = None,
) -> dict:
    """Build the ``env_fingerprint`` object with exactly the CONTRACTS §1.5 keys."""
    info = sys.version_info
    return {
        "model": model,
        "prompt_hash": prompt_hash,
        "python": f"{info.major}.{info.minor}",
        "agenttwin": AGENTTWIN_VERSION,
        "platform": sys.platform,
        "deps": {str(name): str(version) for name, version in (deps or {}).items()},
    }


def compare(expected: dict, actual: dict) -> list[dict]:
    """Return ``[{"field", "expected", "actual"}]`` for every differing field.

    ``python`` and ``agenttwin`` are compared on major.minor only; ``platform``
    and ``deps`` are compared only when both sides carry them.
    """
    differences: list[dict] = []
    for field in _FIELD_ORDER:
        if field in _BOTH_SIDES_FIELDS and not (
            _carries(expected, field) and _carries(actual, field)
        ):
            continue
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if field in _MAJOR_MINOR_FIELDS:
            expected_value = _major_minor(expected_value)
            actual_value = _major_minor(actual_value)
        elif field == "deps":
            expected_value = _normalize_deps(expected_value)
            actual_value = _normalize_deps(actual_value)
        if expected_value != actual_value:
            differences.append(
                {"field": field, "expected": expected_value, "actual": actual_value}
            )
    return differences


def is_fatal(diff: Sequence[dict]) -> bool:
    """True when any reported difference touches ``FATAL_FIELDS``."""
    for item in diff:
        if isinstance(item, Mapping) and item.get("field") in FATAL_FIELDS:
            return True
    return False


def _carries(fingerprint: Mapping[str, Any], field: str) -> bool:
    return field in fingerprint and fingerprint[field] is not None


def _major_minor(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    match = _MAJOR_MINOR_RE.search(value)
    if match is None:
        return value
    return f"{match.group(1)}.{match.group(2)}"


def _normalize_deps(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {str(name): str(version) for name, version in value.items()}
