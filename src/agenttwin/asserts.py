"""Assertion DSL (CONTRACTS section 5): pure functions Trace -> AssertionResult.

Every assertion is a pure function of the tape: no IO, no randomness, no model, no
wall clock. That is what lets CI re-evaluate a year-old tape and reach the same
verdict. The only impure entry point is load_yaml(), which reads a file and
imports PyYAML lazily -- assertions themselves never do either.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .trace import Event, Trace, canonical_json

__all__ = [
    "MODES",
    "AssertionGroup",
    "AssertionResult",
    "AssertionSpec",
    "assert_",
    "assert_all",
    "assert_any",
    "budget_within",
    "call_after",
    "call_order",
    "load_yaml",
    "max_repeats",
    "never_calls",
    "no_deadlock",
    "output_schema",
    "state_invariant",
    "terminates_within",
    "warn",
]

# CONTRACTS section 5: the three group modes. "all" and "warn" evaluate every
# spec; "any" folds the specs into a single verdict.
MODES: tuple[str, ...] = ("all", "any", "warn")

_MISSING = object()


# --- the three frozen shapes (CONTRACTS section 5) --------------------------


@dataclass(frozen=True)
class AssertionResult:
    """One verdict. passed=None means SKIP: not enough tape to judge."""

    name: str
    passed: bool | None
    detail: str
    first_offense_seq: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """CONTRACTS section 8 result row: exactly the four frozen keys."""
        return {
            "name": self.name,
            "status": _status(self.passed),
            "detail": self.detail,
            "first_offense_seq": self.first_offense_seq,
        }


@dataclass(frozen=True)
class AssertionSpec:
    """A named, callable assertion: spec(trace) -> AssertionResult."""

    name: str
    fn: Callable[[Trace], AssertionResult]

    def __call__(self, trace: Trace) -> AssertionResult:
        return self.fn(trace)


@dataclass(frozen=True)
class AssertionGroup:
    """A bundle of specs plus the mode that decides how their verdicts combine."""

    name: str
    specs: tuple[AssertionSpec, ...]
    mode: str = "all"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"assertion group {self.name!r}: mode must be one of {MODES}, got {self.mode!r}")
        specs = tuple(self.specs)
        if not specs:
            raise ValueError(f"assertion group {self.name!r} holds no specs")
        object.__setattr__(self, "specs", specs)

    def evaluate(self, trace: Trace) -> list[AssertionResult]:
        """One row per spec for all/warn; one folded row for any (see _evaluate_any)."""
        if self.mode == "any":
            return [self._evaluate_any(trace)]
        return [_tag_mode(spec(trace), self.mode) for spec in self.specs]

    def _evaluate_any(self, trace: Trace) -> AssertionResult:
        """Fold: PASS if any spec passed, SKIP if none passed but one is undetermined."""
        results = [spec(trace) for spec in self.specs]
        summary = ", ".join(f"{result.name}={_status(result.passed)}" for result in results)
        if any(result.passed is True for result in results):
            return AssertionResult(self.name, True, f"[any] {summary}", None)
        seqs = [result.first_offense_seq for result in results if result.first_offense_seq is not None]
        if any(result.passed is None for result in results):
            return AssertionResult(self.name, None, f"[any] none passed, one undetermined: {summary}", None)
        return AssertionResult(
            self.name,
            False,
            f"[any] every spec failed: {summary}",
            min(seqs) if seqs else None,
        )


# --- combinators (CONTRACTS section 5) --------------------------------------


def assert_(*specs: AssertionSpec) -> AssertionGroup:
    """Every spec is hard: any FAIL moves the exit code."""
    return _group("all", "assert", specs)


def assert_all(*specs: AssertionSpec) -> AssertionGroup:
    """Alias of assert_ with its own label, kept because CONTRACTS names it."""
    return _group("all", "assert_all", specs)


def warn(*specs: AssertionSpec) -> AssertionGroup:
    """Soft: reported, tagged [warn], never part of the exit code."""
    return _group("warn", "warn", specs)


def assert_any(*specs: AssertionSpec) -> AssertionGroup:
    """Hard, folded: passes if at least one spec passes."""
    return _group("any", "assert_any", specs)


def _group(mode: str, label: str, specs: Sequence[AssertionSpec]) -> AssertionGroup:
    rendered = ", ".join(getattr(spec, "name", type(spec).__name__) for spec in specs)
    return AssertionGroup(name=f"{label}({rendered})", specs=tuple(specs), mode=mode)


# --- repetition family (CONTRACTS section 5) -------------------------------


def max_repeats(tool: str, n: int = 3) -> AssertionSpec:
    """FAIL when the tape holds more than n tool_call events whose actor is tool."""
    _require_name("tool", tool)
    _require_int("n", n, 0)
    name = f"max_repeats({tool}, n={n})"

    def check(trace: Trace) -> AssertionResult:
        calls = [event for event in trace.events if event.kind == "tool_call" and event.actor == tool]
        if len(calls) <= n:
            return AssertionResult(name, True, f"{len(calls)} call(s) of {tool} (limit {n})")
        offender = calls[n]  # the (n+1)-th call: the first one over the limit
        return AssertionResult(
            name,
            False,
            f"{len(calls)} calls of {tool} exceeds the limit of {n}; first excess at seq {offender.seq}",
            offender.seq,
        )

    return AssertionSpec(name, check)


def no_deadlock(limit: int = 3) -> AssertionSpec:
    """FAIL when one (actor, args_hash) pair runs more than limit times in a row."""
    _require_int("limit", limit, 0)
    name = f"no_deadlock(limit={limit})"

    def check(trace: Trace) -> AssertionResult:
        # tool_result events sit between calls by design, so "consecutive" is read
        # over the tool_call sub-sequence: a run is broken by a *different* call.
        calls = [event for event in trace.events if event.kind == "tool_call"]
        if not calls:
            return AssertionResult(name, None, "the tape holds no tool_call event, so a deadlock cannot be judged")
        previous_key: Any = _MISSING
        run_length = 0
        offence: tuple[int, str] | None = None
        longest = 0
        for event in calls:
            key = _call_key(event)
            if key == previous_key:
                run_length += 1
            else:
                previous_key, run_length = key, 1
            longest = max(longest, run_length)
            if run_length == limit + 1 and offence is None:
                offence = (event.seq, event.actor)
        if offence is not None:
            return AssertionResult(
                name,
                False,
                f"{offence[1]} repeats {limit + 1}+ times consecutively "
                f"(longest run {longest}, limit {limit}); first excess at seq {offence[0]}",
                offence[0],
            )
        return AssertionResult(
            name,
            True,
            f"{len(calls)} tool_call events, longest identical run {longest} (limit {limit})",
        )

    return AssertionSpec(name, check)


def budget_within(
    *,
    max_calls: float | None = None,
    max_seconds: float | None = None,
    max_tokens: float | None = None,
    price_table: Mapping[str, Mapping[str, float]] | None = None,
) -> AssertionSpec:
    """FAIL when any requested budget dimension is exceeded.

    A dimension the tape cannot measure (no tokens recorded, for instance) is
    skipped rather than failed; if nothing at all is measurable the spec SKIPs.
    price_table only adds an estimated cost to the detail: there is no max_cost
    parameter, so a price table never fails a tape on its own.
    """
    for label, bound in (("max_calls", max_calls), ("max_seconds", max_seconds), ("max_tokens", max_tokens)):
        if bound is None:
            continue
        if not _is_number(bound) or bound < 0:
            raise ValueError(f"budget_within: {label} must be a non-negative number, got {bound!r}")
    if price_table is not None and not isinstance(price_table, Mapping):
        raise ValueError(
            "budget_within: price_table must map model -> {'prompt': float, 'completion': float} per 1M tokens"
        )
    if isinstance(price_table, Mapping):
        for model, entry in price_table.items():
            if not isinstance(entry, Mapping) or not all(
                _is_number(entry.get(key)) for key in ("prompt", "completion")
            ):
                raise ValueError(
                    f"budget_within: price_table[{model!r}] must be {{'prompt': float, 'completion': float}}"
                )
    parts: list[str] = []
    if max_calls is not None:
        parts.append(f"calls<={_num(max_calls)}")
    if max_seconds is not None:
        parts.append(f"seconds<={_num(max_seconds)}")
    if max_tokens is not None:
        parts.append(f"tokens<={_num(max_tokens)}")
    if price_table:
        parts.append("priced")
    name = "budget_within(" + ", ".join(parts) + ")" if parts else "budget_within(none)"

    def check(trace: Trace) -> AssertionResult:
        exceeded: list[tuple[int, str]] = []
        measured: list[str] = []
        skipped: list[str] = []
        measured_count = 0

        # A "call" is an outward step: one llm_call or one tool_call. A
        # tool_result is an answer to a call, never a call of its own.
        call_seqs = [event.seq for event in trace.events if event.kind in ("llm_call", "tool_call")]
        if max_calls is not None:
            measured_count += 1
            if len(call_seqs) > max_calls:
                exceeded.append((call_seqs[int(max_calls)], f"calls {len(call_seqs)} > {_num(max_calls)}"))
            else:
                measured.append(f"calls {len(call_seqs)}/{_num(max_calls)}")

        if max_seconds is not None:
            if not trace.events:
                skipped.append("max_seconds (the tape holds no event)")
            else:
                measured_count += 1
                elapsed = trace.events[-1].t
                first_over = next((event for event in trace.events if event.t > max_seconds), None)
                if first_over is None:
                    measured.append(f"seconds {elapsed:g}/{_num(max_seconds)}")
                else:
                    exceeded.append((first_over.seq, f"seconds {elapsed:g} > {_num(max_seconds)}"))

        usage = _token_usage(trace)
        if max_tokens is not None:
            if not usage:
                skipped.append("max_tokens (no llm_call payload carries 'tokens')")
            else:
                measured_count += 1
                total = 0.0
                crossing: int | None = None
                for seq, prompt, completion, _model in usage:
                    total += prompt + completion
                    if crossing is None and total > max_tokens:
                        crossing = seq
                if crossing is None:
                    measured.append(f"tokens {_num(total)}/{_num(max_tokens)}")
                else:
                    exceeded.append((crossing, f"tokens {_num(total)} > {_num(max_tokens)}"))

        cost: float | None = None
        if price_table:
            priced = False
            cost = 0.0
            for _seq, prompt, completion, model in usage:
                entry = price_table.get(model) if isinstance(price_table, Mapping) else None
                if entry is None:
                    continue
                priced = True
                cost += prompt / 1_000_000 * entry["prompt"] + completion / 1_000_000 * entry["completion"]
            if not priced:
                cost = None

        if exceeded:
            exceeded.sort(key=lambda item: item[0])
            seq = exceeded[0][0]
            detail = "; ".join(text for _seq, text in exceeded) + f" (first excess at seq {seq})"
            return AssertionResult(name, False, detail, seq)
        if measured_count == 0:
            why = "; ".join(skipped) if skipped else "no budget dimension was requested"
            return AssertionResult(name, None, f"nothing measurable: {why}")
        detail = ", ".join(measured)
        if skipped:
            detail += "; not measured: " + "; ".join(skipped)
        if cost is not None:
            detail += f"; estimated cost {cost:.6f} USD"
        return AssertionResult(name, True, detail)

    return AssertionSpec(name, check)


# --- topical family (CONTRACTS section 5) ----------------------------------


def never_calls(tool: str, args_contains: str | None = None) -> AssertionSpec:
    """FAIL when any call of tool carries args_contains (case-insensitive)."""
    _require_name("tool", tool)
    if args_contains is not None and not isinstance(args_contains, str):
        raise ValueError(f"never_calls: args_contains must be a string, got {args_contains!r}")
    suffix = "" if args_contains is None else f", args_contains={args_contains!r}"
    name = f"never_calls({tool}{suffix})"
    needle = None if args_contains is None else args_contains.lower()

    def check(trace: Trace) -> AssertionResult:
        for event in trace.events:
            if event.kind != "tool_call" or event.actor != tool:
                continue
            if needle is None:
                return AssertionResult(
                    name, False, f"{tool} was called at seq {event.seq} and must never be called", event.seq
                )
            if needle in _args_text(event).lower():
                return AssertionResult(
                    name,
                    False,
                    f"{tool} was called at seq {event.seq} with args matching {args_contains!r}",
                    event.seq,
                )
        return AssertionResult(name, True, f"no call of {tool} matches")

    return AssertionSpec(name, check)


# --- ordering family (CONTRACTS section 5) ---------------------------------


def call_order(names: Sequence[str]) -> AssertionSpec:
    """FAIL unless the tools appear as tool_call actors in this relative order."""
    ordered = tuple(names)
    if len(ordered) < 2:
        raise ValueError(f"call_order: needs at least two names, got {ordered!r}")
    for item in ordered:
        _require_name("name", item)
    name = "call_order(" + ">".join(ordered) + ")"

    def check(trace: Trace) -> AssertionResult:
        seen = {event.actor for event in trace.events if event.kind == "tool_call"}
        present = [item for item in dict.fromkeys(ordered) if item in seen]
        if len(present) < 2:
            return AssertionResult(
                name,
                None,
                f"only {len(present)} of {len(set(ordered))} requested tools appear "
                f"({', '.join(present) or 'none'}), so the order cannot be judged",
            )
        position = 0
        matched: list[int] = []
        violation: tuple[int, str, str] | None = None
        for event in trace.events:
            if event.kind != "tool_call" or event.actor not in ordered:
                continue
            if event.actor == ordered[position]:
                matched.append(event.seq)
                position += 1
                if position == len(ordered):
                    break
            elif event.actor in ordered[position + 1:] and violation is None:
                violation = (event.seq, event.actor, ordered[position])
        if position == len(ordered):
            places = ", ".join(str(seq) for seq in matched)
            return AssertionResult(name, True, f"{' > '.join(ordered)} in order at seqs {places}")
        if violation is not None:
            seq, actor, owed = violation
            return AssertionResult(
                name,
                False,
                f"{actor} at seq {seq} arrived before {owed}, breaking {' > '.join(ordered)}",
                seq,
            )
        offence = matched[-1] if matched else max(len(trace.events) - 1, 0)
        return AssertionResult(
            name,
            False,
            f"{ordered[position]} never appears after seq {offence}, breaking {' > '.join(ordered)}",
            offence,
        )

    return AssertionSpec(name, check)


def call_after(a: str, b: str) -> AssertionSpec:
    """FAIL unless every call of b is preceded by at least one call of a."""
    _require_name("a", a)
    _require_name("b", b)
    name = f"call_after({a}, {b})"

    def check(trace: Trace) -> AssertionResult:
        seen_a = False
        occurrences = 0
        for event in trace.events:
            if event.kind != "tool_call":
                continue
            if event.actor == b and not seen_a:
                return AssertionResult(
                    name, False, f"{b} was called at seq {event.seq} before any call of {a}", event.seq
                )
            if event.actor == a:
                seen_a = True
            if event.actor == b:
                occurrences += 1
        if occurrences == 0:
            return AssertionResult(name, None, f"{b} is never called, so the ordering constraint does not apply")
        return AssertionResult(name, True, f"{occurrences} call(s) of {b}, each preceded by {a}")

    return AssertionSpec(name, check)


# --- state family (CONTRACTS section 5) ------------------------------------


def state_invariant(path: str, predicate: Callable[[Any], Any]) -> AssertionSpec:
    """FAIL when predicate(value at the JSON Pointer in Trace.state()) is falsy."""
    if not isinstance(path, str) or (path and not path.startswith("/")):
        raise ValueError(f"state_invariant: path must be a JSON Pointer starting with '/', got {path!r}")
    if not callable(predicate):
        raise ValueError("state_invariant: predicate must be callable")
    name = f"state_invariant({path or '/'})"

    def check(trace: Trace) -> AssertionResult:
        found, value = _resolve_pointer(trace.state(), path)
        if not found:
            return AssertionResult(name, None, f"the folded state has no value at {path or '/'}")
        try:
            verdict = predicate(value)
        except Exception as exc:  # a predicate that blows up is a failed invariant
            return AssertionResult(
                name,
                False,
                f"predicate raised {type(exc).__name__} at {path or '/'}: {exc}",
                _state_source_seq(trace, path),
            )
        if verdict:
            return AssertionResult(name, True, f"{path or '/'} = {_short(value)} satisfies the predicate")
        return AssertionResult(
            name,
            False,
            f"{path or '/'} = {_short(value)} violates the predicate",
            _state_source_seq(trace, path),
        )

    return AssertionSpec(name, check)


def output_schema(schema: Mapping[str, Any]) -> AssertionSpec:
    """FAIL when Trace.state() does not match the pure-Python JSON-Schema subset.

    Supported keywords: type, required, properties, enum, minimum, maximum,
    items, pattern. Anything else is a ValueError at construction time, because
    a keyword this module cannot enforce would otherwise be a silent lie.
    """
    if not isinstance(schema, Mapping):
        raise ValueError(f"output_schema: schema must be a mapping, got {type(schema).__name__}")
    _check_schema(schema, "$")
    frozen = dict(schema)
    declared = frozen.get("type")
    if isinstance(declared, str):
        name = f"output_schema({declared})"
    elif declared is not None:
        name = "output_schema(" + "|".join(str(item) for item in declared) + ")"
    else:
        name = "output_schema"

    def check(trace: Trace) -> AssertionResult:
        records_state = any(event.kind in ("state_patch", "terminal") for event in trace.events)
        if not records_state:
            return AssertionResult(
                name, None, "no terminal and no state_patch: the tape carries no output state to validate"
            )
        state = trace.state()
        try:
            _validate_value(state, frozen, "")
        except _SchemaViolation as exc:
            return AssertionResult(name, False, str(exc), _state_source_seq(trace, ""))
        return AssertionResult(name, True, f"{_short(state, 80)} matches the schema")

    return AssertionSpec(name, check)


# --- termination (CONTRACTS section 5) -------------------------------------


def terminates_within(n: int) -> AssertionSpec:
    """FAIL when the run terminates at seq >= n, or overruns n events with no terminal."""
    _require_int("n", n, 1)
    name = f"terminates_within(n={n})"

    def check(trace: Trace) -> AssertionResult:
        terminals = [event for event in trace.events if event.kind == "terminal"]
        late = next((event for event in terminals if event.seq >= n), None)
        if late is not None:
            return AssertionResult(name, False, f"terminal at seq {late.seq} is not within n={n}", late.seq)
        if terminals:
            return AssertionResult(name, True, f"terminal at seq {terminals[0].seq} < {n}")
        if len(trace.events) >= n:
            return AssertionResult(
                name,
                False,
                f"no terminal event after {len(trace.events)} events (n={n}), so the run overran its bound",
                n - 1,
            )
        return AssertionResult(
            name,
            None,
            f"no terminal event and only {len(trace.events)} < {n} events, so termination is not proven",
        )

    return AssertionSpec(name, check)


# --- YAML front end (CONTRACTS section 5) ----------------------------------

_SPEC_FACTORIES: dict[str, Callable[..., AssertionSpec]] = {
    "max_repeats": max_repeats,
    "no_deadlock": no_deadlock,
    "budget_within": budget_within,
    "never_calls": never_calls,
    "call_order": call_order,
    "call_after": call_after,
    "state_invariant": state_invariant,
    "output_schema": output_schema,
    "terminates_within": terminates_within,
}

_ENTRY_KEYS = ("spec", "args", "warn", "mode")


def load_yaml(path: str | Path) -> list[AssertionGroup]:
    """Parse an assertion file into groups.

        - {spec: max_repeats, args: {tool: shell, n: 3}, warn: false}

    PyYAML is imported here, not at module import time: the assertion core stays
    dependency-free and only a run that actually reads YAML needs the extra.
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "agenttwin.asserts.load_yaml() needs PyYAML, which is not installed; "
            "run 'pip install agenttwin[yaml]' (or 'pip install pyyaml')."
        ) from exc
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{source}: malformed YAML: {exc}") from None
    if document is None:
        return []
    if not isinstance(document, list):
        raise ValueError(f"{source}: expected a YAML list of assertion entries, got {type(document).__name__}")
    groups: list[AssertionGroup] = []
    for index, entry in enumerate(document, start=1):
        groups.append(_load_entry(source, index, entry))
    return groups


def _load_entry(source: Path, index: int, entry: Any) -> AssertionGroup:
    where = f"{source}: entry {index}"
    if not isinstance(entry, Mapping):
        raise ValueError(f"{where}: must be a mapping with a 'spec' key, got {type(entry).__name__}")
    unknown = sorted(set(entry) - set(_ENTRY_KEYS))
    if unknown:
        raise ValueError(f"{where}: unknown key {unknown[0]!r} (allowed: {', '.join(_ENTRY_KEYS)})")
    spec_name = entry.get("spec")
    if spec_name not in _SPEC_FACTORIES:
        raise ValueError(
            f"{where}: unknown assertion {spec_name!r} (known: {', '.join(sorted(_SPEC_FACTORIES))})"
        )
    args = entry.get("args")
    if args is None:
        args = {}
    if not isinstance(args, Mapping):
        raise ValueError(f"{where}: 'args' must be a mapping, got {type(args).__name__}")
    if "mode" in entry and "warn" in entry:
        raise ValueError(f"{where}: sets both 'mode' and 'warn'; use one of them")
    mode = entry.get("mode")
    if mode is None:
        mode = "warn" if entry.get("warn") else "all"
    if mode not in MODES:
        raise ValueError(f"{where}: mode must be one of {MODES}, got {mode!r}")
    try:
        spec = _SPEC_FACTORIES[spec_name](**dict(args))
    except TypeError as exc:
        raise ValueError(f"{where}: {spec_name} rejected args {sorted(args)}: {exc}") from None
    except ValueError as exc:
        raise ValueError(f"{where}: {spec_name}: {exc}") from None
    return AssertionGroup(name=spec.name, specs=(spec,), mode=mode)


# --- helpers (private: the surface above is what report.py and the CLI use) --


def _status(passed: bool | None) -> str:
    if passed is None:
        return "SKIP"
    return "PASS" if passed else "FAIL"


def _tag_mode(result: AssertionResult, mode: str) -> AssertionResult:
    """Surface the group mode in the row, so a soft result is not read as a hard one."""
    if mode == "all":
        return result
    return AssertionResult(
        name=result.name,
        passed=result.passed,
        detail=f"[{mode}] {result.detail}",
        first_offense_seq=result.first_offense_seq,
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _num(value: float) -> str:
    return f"{value:g}"


def _short(value: Any, limit: int = 120) -> str:
    text = canonical_json(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _require_name(label: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty tool name, got {value!r}")


def _require_int(label: str, value: Any, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")


def _call_key(event: Event) -> tuple[str, str]:
    """The identity a deadlock repeats on: the actor plus what it was asked to do."""
    args_hash = event.payload.get("args_hash")
    if isinstance(args_hash, str):
        return (event.actor, args_hash)
    return (event.actor, "preview:" + canonical_json(event.payload.get("args_preview")))


def _args_text(event: Event) -> str:
    """What never_calls searches: args_preview when recorded, else the canonical args."""
    payload = event.payload
    preview = payload.get("args_preview")
    if isinstance(preview, str):
        return canonical_json(preview)
    if "args" in payload:
        return canonical_json(payload["args"])
    return canonical_json({key: value for key, value in payload.items() if key != "args_hash"})


def _token_usage(trace: Trace) -> list[tuple[int, float, float, str | None]]:
    usage: list[tuple[int, float, float, str | None]] = []
    for event in trace.events:
        if event.kind != "llm_call":
            continue
        tokens = event.payload.get("tokens")
        if not isinstance(tokens, Mapping):
            continue
        prompt = tokens.get("prompt")
        completion = tokens.get("completion")
        if not _is_number(prompt) or not _is_number(completion):
            continue
        usage.append((event.seq, prompt, completion, event.payload.get("model")))
    return usage


def _resolve_pointer(state: Any, pointer: str) -> tuple[bool, Any]:
    """JSON Pointer lookup in the folded state; (found, value)."""
    if pointer == "":
        return True, state
    node = state
    for raw in pointer.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, Mapping) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        else:
            return False, None
    return True, node


def _state_source_seq(trace: Trace, path: str) -> int | None:
    """Where the value at path was written, so the replayer can jump to it."""
    terminal_seq: int | None = None
    for event in trace.events:
        if event.kind == "terminal":
            terminal_seq = event.seq  # state() returns the terminal state wholesale
    if terminal_seq is not None:
        return terminal_seq
    written: int | None = None
    for event in trace.events:
        if event.kind != "state_patch":
            continue
        for pointer in event.payload.get("patch", {}):
            if path == "" or pointer == path or path.startswith(pointer + "/"):
                written = event.seq
    if written is not None:
        return written
    return len(trace.events) - 1 if trace.events else None


class _SchemaViolation(Exception):
    """Internal: one concrete schema mismatch, already rendered for the report."""


_SCHEMA_KEYWORDS = frozenset(
    {"type", "required", "properties", "enum", "minimum", "maximum", "items", "pattern"}
)
_JSON_TYPES = ("object", "array", "string", "number", "integer", "boolean", "null")


def _check_schema(schema: Any, where: str) -> None:
    if not isinstance(schema, Mapping):
        raise ValueError(f"output_schema: {where} must be a mapping, got {type(schema).__name__}")
    unknown = sorted(set(schema) - _SCHEMA_KEYWORDS)
    if unknown:
        raise ValueError(
            f"output_schema: unsupported keyword {unknown[0]!r} at {where} "
            f"(supported: {', '.join(sorted(_SCHEMA_KEYWORDS))})"
        )
    if "type" in schema:
        declared = schema["type"]
        names = declared if isinstance(declared, (list, tuple)) else [declared]
        if not names:
            raise ValueError(f"output_schema: 'type' at {where} must not be empty")
        for item in names:
            if item not in _JSON_TYPES:
                raise ValueError(
                    f"output_schema: unknown type {item!r} at {where} (expected one of {', '.join(_JSON_TYPES)})"
                )
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, (list, tuple)) or not all(isinstance(key, str) for key in required):
            raise ValueError(f"output_schema: 'required' at {where} must be a list of property names")
    if "enum" in schema:
        if not isinstance(schema["enum"], (list, tuple)) or not schema["enum"]:
            raise ValueError(f"output_schema: 'enum' at {where} must be a non-empty list")
    for keyword in ("minimum", "maximum"):
        if keyword in schema and not _is_number(schema[keyword]):
            raise ValueError(f"output_schema: {keyword!r} at {where} must be a number")
    if "pattern" in schema:
        pattern = schema["pattern"]
        if not isinstance(pattern, str):
            raise ValueError(f"output_schema: 'pattern' at {where} must be a string")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"output_schema: invalid pattern at {where}: {exc}") from None
    if "properties" in schema:
        properties = schema["properties"]
        if not isinstance(properties, Mapping):
            raise ValueError(f"output_schema: 'properties' at {where} must be a mapping")
        for key, subschema in properties.items():
            _check_schema(subschema, f"{where}/properties/{key}")
    if "items" in schema:
        _check_schema(schema["items"], f"{where}/items")


def _is_type(value: Any, name: str) -> bool:
    if name == "object":
        return isinstance(value, Mapping)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "number":
        return _is_number(value)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    return value is None


def _type_name(value: Any) -> str:
    for name in _JSON_TYPES:
        if _is_type(value, name):
            return name
    return type(value).__name__


def _json_equal(left: Any, right: Any) -> bool:
    """JSON equality: true is not 1, and 1 is not 1.0's type partner."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if _is_number(left) and _is_number(right):
        return left == right
    return type(left) is type(right) and left == right


def _render_path(path: str) -> str:
    return path or "/"


def _validate_value(value: Any, schema: Mapping[str, Any], path: str) -> None:
    if "type" in schema:
        declared = schema["type"]
        names = declared if isinstance(declared, (list, tuple)) else [declared]
        if not any(_is_type(value, name) for name in names):
            raise _SchemaViolation(
                f"{_render_path(path)}: expected type {'|'.join(str(name) for name in names)}, "
                f"got {_type_name(value)}"
            )
    if "enum" in schema and not any(_json_equal(value, allowed) for allowed in schema["enum"]):
        raise _SchemaViolation(
            f"{_render_path(path)}: {_short(value, 60)} is not one of {_short(list(schema['enum']), 60)}"
        )
    if _is_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            raise _SchemaViolation(
                f"{_render_path(path)}: {value:g} is below the minimum of {schema['minimum']:g}"
            )
        if "maximum" in schema and value > schema["maximum"]:
            raise _SchemaViolation(
                f"{_render_path(path)}: {value:g} is above the maximum of {schema['maximum']:g}"
            )
    if "pattern" in schema and isinstance(value, str) and re.search(schema["pattern"], value) is None:
        raise _SchemaViolation(
            f"{_render_path(path)}: {_short(value, 60)} does not match pattern {schema['pattern']!r}"
        )
    if isinstance(value, Mapping):
        for key in schema.get("required", ()):
            if key not in value:
                raise _SchemaViolation(f"{_render_path(path)}: required key {key!r} is missing")
        for key, subschema in schema.get("properties", {}).items():
            if key in value:
                _validate_value(value[key], subschema, f"{path}/{key}")
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _validate_value(item, schema["items"], f"{path}/{index}")
