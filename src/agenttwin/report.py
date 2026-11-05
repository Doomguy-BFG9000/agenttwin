"""Report layer: JSON, Markdown and the single-file HTML replayer (CONTRACTS §8).

A Report is a read-only view of one replay: the baseline tape, the tape the replay
produced, the assertions evaluated against it, and the exit code those facts earn.
Nothing here touches the network, the wall clock or a third-party package, and the
HTML it writes opens straight from file:// with no side resources, so a CI artefact
never phones home.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .diff import TraceDiff
from .trace import SCHEMA_VERSION, Trace

if TYPE_CHECKING:  # typing only: report depends on the shapes, not the modules
    from .asserts import AssertionResult
    from .drift import DriftError, TapeMiss

__all__ = ["Report", "TEMPLATE_PATH", "PLACEHOLDER", "LABELS"]

TEMPLATE_PATH: Path = Path(__file__).with_name("html") / "replayer.html"
PLACEHOLDER: str = "__AGENTTWIN_PAYLOAD__"
LABELS: dict[str, str] = {"baseline": "recorded", "current": "replay"}


class Report:
    """CONTRACTS §8 — the replay artefact: JSON, Markdown, and one HTML file."""

    def __init__(
        self,
        *,
        baseline: Trace,
        trace: Trace,
        results: list[AssertionResult] | None = None,
        misses: list[TapeMiss] | None = None,
        drifts: list[DriftError] | None = None,
        mode: str = "replay",
        strict: bool = False,
        divergent: bool | None = None,
    ) -> None:
        self.baseline = baseline
        self.trace = trace
        self.mode = mode
        self.strict = strict
        # §9.18: divergent is the R-TERM fork flag, not "the diff differs". A
        # patched replay legitimately differs from its baseline and is not divergent.
        self.divergent: bool = bool(divergent) if divergent is not None else False
        self.results: list[AssertionResult] = list(results) if results else []
        self.misses: list[TapeMiss] = list(misses) if misses else []
        self.drifts: list[DriftError] = list(drifts) if drifts else []
        self.diff = TraceDiff(baseline, trace)
        self._soft: list[bool] = [False] * len(self.results)

    # --- assertions ---------------------------------------------------------

    def assert_(self, *specs: Any) -> "Report":
        """Evaluate hard assertions now (CONTRACTS §5); a failure earns exit code 1."""
        return self._extend(specs, soft=False)

    def warn(self, *specs: Any) -> "Report":
        """Evaluate soft assertions now: reported, but never part of the exit code."""
        return self._extend(specs, soft=True)

    def _extend(self, specs: Sequence[Any], *, soft: bool) -> "Report":
        for spec in specs:
            # §9.25: a warn group is soft wherever it is evaluated, so
            # assert_(warn(x)) cannot contradict its own "[warn]" detail.
            spec_soft = soft or getattr(spec, "mode", None) == "warn"
            for result in _evaluate(spec, self.trace):
                self.results.append(result)
                self._soft.append(spec_soft)
        return self

    @property
    def exit_code(self) -> int:
        """CONTRACTS §6 — 2 beats 1 beats 0, and a soft result never moves it."""
        if self.drifts or self.misses or self.divergent:
            return 2
        for index, result in enumerate(self.results):
            if index < len(self._soft) and self._soft[index]:
                continue
            if result.passed is False:
                return 1
            if result.passed is None and self.strict:
                return 1
        return 0

    # --- serialisation ------------------------------------------------------

    def json(self) -> dict:
        """CONTRACTS §8 — the fixed top-level key set, JSON-safe all the way down."""
        return {
            "schema_version": SCHEMA_VERSION,
            "trace_id": self.trace.header.trace_id,
            "mode": self.mode,
            "divergent": self.divergent,
            "exit_code": self.exit_code,
            "results": self._result_rows(),
            "misses": [miss.to_dict() for miss in self.misses],
            "drifts": [drift.to_dict() for drift in self.drifts],
            "diff": self._diff_dict(),
            "header": self.trace.header.to_json(),
        }

    def _diff_dict(self) -> dict:
        return {
            "first_divergence_seq": self.diff.first_divergence_seq,
            "events": list(self.diff.events),
            "state_diff": list(self.diff.state_diff),
            "assertion_matrix": dict(self.diff.assertion_matrix),
        }

    def _result_rows(self) -> list[dict]:
        return [_result_dict(result) for result in self.results]

    def markdown(self) -> str:
        """A compact CI report: verdict, assertions, diff, fingerprint, redaction."""
        lines: list[str] = [
            f"# AgentTwin report: {self.trace.header.trace_id}",
            "",
            f"verdict: {self._verdict()} (exit code {self.exit_code})",
            f"mode: {self.mode}",
            f"divergent: {_yes_no(self.divergent)}",
            f"strict: {_yes_no(self.strict)}",
            "",
            "## Assertions",
            "",
            "| name | status | detail | first_offense_seq |",
            "| --- | --- | --- | --- |",
        ]
        for row in self._result_rows():
            lines.append(
                f"| {_cell(row['name'])} | {row['status']} | {_cell(row['detail'])} "
                f"| {_offense(row['first_offense_seq'])} |"
            )
        lines.extend(["", self._counts_line(), "", "## Diff", ""])
        lines.extend(self._diff_lines())
        lines.extend(["", "## Fingerprint", ""])
        lines.extend(self._fingerprint_lines())
        lines.extend(["", "## Redaction", ""])
        lines.append(self._redaction_line())
        failures = self._failure_lines()
        if failures:
            lines.extend(["", "## Failures", ""])
            lines.extend(failures)
        return "\n".join(lines) + "\n"

    def _verdict(self) -> str:
        if self.drifts:
            return "DRIFT"
        if self.misses:
            return "TAPE MISS"
        if self.divergent:
            return "DIVERGENT"
        return "FAIL" if self.exit_code else "PASS"

    def _counts_line(self) -> str:
        statuses = [_status(result.passed) for result in self.results]
        line = (
            f"{statuses.count('PASS')} passed, {statuses.count('FAIL')} failed, "
            f"{statuses.count('SKIP')} skipped"
        )
        soft = sum(1 for flag in self._soft if flag)
        if soft:
            line += f" ({soft} soft)"
        return line

    def _diff_lines(self) -> list[str]:
        events = list(self.diff.events)
        changes = {"same": 0, "changed": 0, "added": 0, "removed": 0}
        for event in events:
            change = event.get("change")
            if change in changes:
                changes[change] += 1
        divergence = self.diff.first_divergence_seq
        return [
            f"first divergence seq: {'none' if divergence is None else divergence}",
            f"events: {len(events)} total | same {changes['same']}, changed {changes['changed']}, "
            f"added {changes['added']}, removed {changes['removed']}",
            f"state paths changed: {len(self.diff.state_diff)}",
        ]

    def _fingerprint_lines(self) -> list[str]:
        fingerprint = self.trace.header.env_fingerprint
        lines = [
            f"{field}: {_scalar(fingerprint.get(field))}"
            for field in ("model", "prompt_hash", "python", "agenttwin", "platform")
        ]
        deps = fingerprint.get("deps") or {}
        lines.append(
            "deps: " + (", ".join(f"{name}={deps[name]}" for name in sorted(deps)) if deps else "none")
        )
        return lines

    def _redaction_line(self) -> str:
        redaction = self.trace.header.redaction
        return (
            f"this tape redacted {redaction.get('redacted', 0)} secrets "
            f"(mode: {redaction.get('mode', 'unknown')})"
        )

    def _failure_lines(self) -> list[str]:
        lines: list[str] = []
        for miss in self.misses:
            obj = miss.to_dict()
            lines.append(
                f"- TapeMiss seq={obj.get('seq')} expected_kind={obj.get('expected_kind')} "
                f"actor={obj.get('actor')}"
            )
        for drift in self.drifts:
            obj = drift.to_dict()
            head = f"- {obj.get('type', 'DriftError')} seq={obj.get('seq')}"
            message = obj.get("message")
            lines.append(f"{head}: {message}" if message else head)
        return lines

    # --- HTML ---------------------------------------------------------------

    def html(self, path: str | Path) -> Path:
        """CONTRACTS §8 — write the single-file offline replayer and return its path."""
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        count = template.count(PLACEHOLDER)
        if count != 1:
            raise RuntimeError(
                f"{TEMPLATE_PATH} must contain {PLACEHOLDER} exactly once, found {count}"
            )
        payload = json.dumps(self._payload(), ensure_ascii=False).replace("<", "\\u003c")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(template.replace(PLACEHOLDER, payload), encoding="utf-8", newline="")
        return target

    def _payload(self) -> dict:
        """CONTRACTS §8 — the exact payload shape the replayer reads."""
        return {
            "trace_id": self.trace.header.trace_id,
            "mode": self.mode,
            "header": self.trace.header.to_json(),
            "baseline": [event.to_json() for event in self.baseline.events],
            "current": [event.to_json() for event in self.trace.events],
            "results": self._result_rows(),
            "diff": {
                "first_divergence_seq": self.diff.first_divergence_seq,
                "events": list(self.diff.events),
            },
            "labels": dict(LABELS),
        }


# --- helpers (private: the public surface is the class above) ---------------


def _evaluate(spec: Any, trace: Trace) -> list[AssertionResult]:
    """An AssertionSpec yields one result, an AssertionGroup several (CONTRACTS §5)."""
    evaluate = getattr(spec, "evaluate", None)
    if callable(evaluate):
        return list(evaluate(trace))
    return [spec(trace)]


def _status(passed: bool | None) -> str:
    if passed is None:
        return "SKIP"
    return "PASS" if passed else "FAIL"


def _result_dict(result: AssertionResult) -> dict:
    """CONTRACTS §8 result row: exactly the four frozen keys, always JSON-safe."""
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        row = to_dict()
    else:
        row = {
            "name": result.name,
            "status": _status(result.passed),
            "detail": result.detail,
            "first_offense_seq": result.first_offense_seq,
        }
    return {
        "name": row["name"],
        "status": row["status"],
        "detail": row["detail"],
        "first_offense_seq": row.get("first_offense_seq"),
    }


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _offense(value: Any) -> str:
    return "-" if value is None else str(value)


def _scalar(value: Any) -> str:
    return "-" if value is None else str(value)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
