"""Diff layer: seq-aligned event diff plus a JSON-Pointer state diff (CONTRACTS §7).

v0.1 aligns on `seq` alone. §2 makes seq the one ordering key every other layer already
agrees on — the replay cursor, the patch anchor, and every assertion's
first_offense_seq — so a second alignment rule here would let the diff disagree
with the cursor that produced it. Aligning by content is a v0.2 question.

The row shape is frozen by the HTML replayer, which was committed first:
replayer.html:993 calls changedKeys(row.baseline, row.current) and
replayer.html:624-628 derives the change chip from row.change. Renaming
"baseline"/"current"/"change"/"seq" would blank the diff panel silently rather
than fail, so the key names below are part of the public contract.

TraceDiff is a pure view: it reads both tapes, optionally evaluates the caller's
assertion specs against each of them, and writes to neither.
"""

from __future__ import annotations

import copy
from typing import Any, Sequence

from .trace import Event, Trace, canonical_json

__all__ = ["TraceDiff"]


class TraceDiff:
    """CONTRACTS §7 — one baseline tape measured against one current tape."""

    def __init__(self, baseline: Trace, current: Trace, *, specs: Sequence[Any] = ()) -> None:
        """Align both tapes by seq, diff their folded state, and score the specs.

        specs is keyword-only and defaults to empty because Report builds its diff
        from the two tapes alone (§8): only a caller that actually wants the
        assertion matrix pays for evaluating every spec twice.
        """
        self.baseline = baseline
        self.current = current
        self.events = _align_events(baseline.events, current.events)
        self.first_divergence_seq = _first_divergence(self.events)
        self.state_diff = _diff_states(baseline.state(), current.state())
        self.assertion_matrix = _build_matrix(specs, baseline, current)
        self.summary = _summarise(self.assertion_matrix)

    def to_dict(self) -> dict:
        """Every public attribute as one JSON-safe dict (CONTRACTS §7)."""
        return {
            "first_divergence_seq": self.first_divergence_seq,
            "events": copy.deepcopy(self.events),
            "state_diff": copy.deepcopy(self.state_diff),
            "assertion_matrix": copy.deepcopy(self.assertion_matrix),
            "summary": copy.deepcopy(self.summary),
        }


# --- event alignment --------------------------------------------------------


def _align_events(baseline: Sequence[Event], current: Sequence[Event]) -> list[dict]:
    """One row per seq in the union of both tapes, ascending, each side copied out.

    Event.to_json() returns its payload by reference (trace.py:99-110), so every
    row is deep-copied: a caller editing a row must never reach back into a tape.
    """
    by_base = {event.seq: event for event in baseline}
    by_current = {event.seq: event for event in current}
    rows: list[dict] = []
    for seq in sorted(set(by_base) | set(by_current)):
        base = by_base.get(seq)
        cur = by_current.get(seq)
        rows.append(
            {
                "seq": seq,
                "change": _change_of(base, cur),
                "baseline": None if base is None else copy.deepcopy(base.to_json()),
                "current": None if cur is None else copy.deepcopy(cur.to_json()),
            }
        )
    return rows


def _change_of(base: Event | None, current: Event | None) -> str:
    """§7 names kind/actor/payload as the change test, so only those three are compared.

    `t` is a function of seq (§9.4) and `patched` is a provenance marker; a
    patch that rewrote a payload to its own value leaves the seq `same` while still
    showing the marker in the row.
    """
    if base is None:
        return "added"
    if current is None:
        return "removed"
    if base.kind != current.kind or base.actor != current.actor:
        return "changed"
    return "same" if _same_json(base.payload, current.payload) else "changed"


def _first_divergence(rows: Sequence[dict]) -> int | None:
    """The lowest seq that is not `same`; None when the tapes agree (rows are ascending)."""
    for row in rows:
        if row["change"] != "same":
            return row["seq"]
    return None


# --- state diff -------------------------------------------------------------


def _diff_states(before: dict, after: dict) -> list[dict]:
    """§7 — a JSON-Pointer path per difference, sorted so the output is stable."""
    rows: list[dict] = []
    _walk_state(before, after, "", rows)
    rows.sort(key=lambda row: row["path"])
    return rows


def _walk_state(before: dict, after: dict, path: str, rows: list[dict]) -> None:
    """Walk one level of two state dicts; the caller handles the leaf comparison."""
    for key, value in before.items():
        child = f"{path}/{_escape(key)}"
        if key in after:
            _compare_state(value, after[key], child, rows)
        else:
            rows.append({"path": child, "before": copy.deepcopy(value), "after": None})
    for key, value in after.items():
        if key not in before:
            rows.append({"path": f"{path}/{_escape(key)}", "before": None, "after": copy.deepcopy(value)})


def _compare_state(before: Any, after: Any, path: str, rows: list[dict]) -> None:
    """Recurse into dicts; every other JSON value is one leaf row when it differs.

    Lists are deliberately leaves: a list is one value to a state_patch pointer, so
    replacing it is one row rather than an index-by-index diff the replayer never
    renders.
    """
    if isinstance(before, dict) and isinstance(after, dict):
        _walk_state(before, after, path, rows)
    elif not _same_json(before, after):
        rows.append({"path": path, "before": copy.deepcopy(before), "after": copy.deepcopy(after)})


def _escape(token: Any) -> str:
    """RFC 6901 pointer token — '~' first, then '/', or the escaping eats its own output."""
    return str(token).replace("~", "~0").replace("/", "~1")


def _same_json(left: Any, right: Any) -> bool:
    """JSON equality via the canonical form §1.3 already defines.

    Two values are the same here exactly when canonical_json agrees, which is the
    same judgement state_hash makes, so the diff and the hash never contradict
    each other (key order is irrelevant, and 1 is not true).
    """
    return canonical_json(left) == canonical_json(right)


# --- assertion matrix -------------------------------------------------------


def _build_matrix(specs: Sequence[Any], baseline: Trace, current: Trace) -> dict[str, dict]:
    """§7 — every spec runs on both tapes, baseline first.

    A name reported on one side only keeps None on the other, which §5 already
    reads as SKIP. A name produced twice on one side takes its last value: the
    matrix is keyed by name and the contract offers nowhere else to put a second
    row for the same assertion.
    """
    matrix: dict[str, dict] = {}
    for spec in specs:
        for trace, side in ((baseline, "baseline"), (current, "current")):
            for result in _results_of(spec, trace):
                matrix.setdefault(result.name, {"baseline": None, "current": None})[side] = result.passed
    return matrix


def _results_of(spec: Any, trace: Trace) -> list[Any]:
    """Accept both shapes §5 defines: evaluate(trace) -> several, __call__(trace) -> one.

    .evaluate wins when an object has both, matching Report's own spec handling
    (report.py:254-259) so one spec cannot mean two things to two layers.
    """
    evaluate = getattr(spec, "evaluate", None)
    if callable(evaluate):
        produced = evaluate(trace)
    elif callable(spec):
        produced = spec(trace)
    else:
        raise TypeError(f"spec {spec!r} has neither evaluate(trace) nor __call__(trace)")
    results = list(produced) if isinstance(produced, (list, tuple)) else [produced]
    for result in results:
        if not isinstance(getattr(result, "name", None), str) or not hasattr(result, "passed"):
            raise TypeError(
                f"spec {spec!r} returned {result!r}, expected an AssertionResult-like (name, passed)"
            )
    return results


def _summarise(matrix: dict[str, dict]) -> dict:
    """§7 — [passed, failed, skipped] per side, plus the change in the passing count."""
    before = _tally(matrix, "baseline")
    after = _tally(matrix, "current")
    return {"before": before, "after": after, "delta": after[0] - before[0]}


def _tally(matrix: dict[str, dict], side: str) -> list[int]:
    """Statuses read the way Report._status reads them (report.py:262-265): None is SKIP."""
    counts = [0, 0, 0]
    for row in matrix.values():
        passed = row[side]
        if passed is None:
            counts[2] += 1
        elif passed:
            counts[0] += 1
        else:
            counts[1] += 1
    return counts
