"""The command line surface (CONTRACTS section 6 and section 9.26).

Five commands, four exit codes, and one rule: every handler returns an int,
main() returns it, and no failure reaches the user as a traceback.  The codes
are CONTRACTS section 6's -- 0 pass, 1 assertion failure, 2 drift / TapeMiss /
SchemaError / unloadable tape, 3 configuration, usage or environment -- so
argparse's own status 2 (a mistyped flag) is remapped to 3: a usage error must
never read as a drifted tape (section 6, priority 3 > 2 > 1 > 0).

Stdlib only.  PyYAML is reached only through agenttwin.asserts.load_yaml, which
imports it itself, and agenttwin.gallery is imported inside the gallery handler
so a missing module is an exit code rather than an ImportError.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .diff import TraceDiff
from .drift import DriftError
from .redact import DEFAULT_PATTERNS
from .report import Report
from .trace import CLOCK_STEP, SchemaError, Trace

__all__ = ["main", "EXIT_OK", "EXIT_FAIL", "EXIT_DRIFT", "EXIT_USAGE"]

# CONTRACTS section 6: the four statuses, and nothing else.
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_DRIFT = 2
EXIT_USAGE = 3

# The two variables a recording adapter reads (record.py keeps the L1 pair).
RECORD_OUT_ENV = "AGENTTWIN_RECORD_OUT"
TRACE_ID_ENV = "AGENTTWIN_TRACE_ID"

_CLOCK_TOLERANCE = 1e-6


class _CliError(Exception):
    """One command failure already reduced to an exit code and a one-line reason."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = int(code)
        self.message = str(message)


class _Parser(argparse.ArgumentParser):
    """CONTRACTS section 6: a usage error is a 3, never argparse's own 2."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        raise _CliError(EXIT_USAGE, f"{self.prog}: error: {message}")


# --- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """CONTRACTS section 6 -- the five commands, each wired to its handler."""
    parser = _Parser(
        prog="agenttwin",
        description="Record one agent run, replay it forever (CONTRACTS v0.1).",
    )
    commands = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    record = commands.add_parser("record", help="run a command and check the tape it produced")
    record.add_argument(
        "--out",
        required=True,
        metavar="RUNS/run.jsonl",
        help="tape path the child must write (made absolute before the child sees it)",
    )
    record.add_argument(
        "child",
        nargs=argparse.REMAINDER,
        metavar="-- <cmd> [args...]",
        help="the command to run, after a -- separator",
    )
    record.set_defaults(handler=_cmd_record)

    replay = commands.add_parser("replay", help="replay a tape through a controller")
    replay.add_argument("tape", metavar="TAPE", help="the recorded JSONL tape")
    replay.add_argument(
        "--assert",
        dest="assert_file",
        metavar="FILE",
        help="YAML assertion file (agenttwin.asserts.load_yaml)",
    )
    replay.add_argument("--html", metavar="PATH", help="write the single-file replayer here")
    replay.add_argument(
        "--strict",
        action="store_true",
        help="treat SKIP as failure (CONTRACTS 9.26)",
    )
    replay.add_argument("--json", action="store_true", help="print Report.json() instead of markdown")
    replay.add_argument(
        "--agent",
        metavar="MODULE:CALLABLE",
        help="the controller to replay; a tape alone has none",
    )
    replay.set_defaults(handler=_cmd_replay)

    diff = commands.add_parser("diff", help="align two tapes by seq and report the divergence")
    diff.add_argument("baseline", metavar="A", help="the recorded tape")
    diff.add_argument("current", metavar="B", help="the tape to compare against A")
    diff.add_argument("--html", metavar="PATH", help="write the difference into the replayer")
    diff.set_defaults(handler=_cmd_diff)

    doctor = commands.add_parser("doctor", help="diagnose one tape")
    doctor.add_argument("tape", metavar="TAPE", help="the JSONL tape to diagnose")
    doctor.set_defaults(handler=_cmd_doctor)

    gallery = commands.add_parser("gallery", help="list, inspect and run bundled fixtures")
    actions = gallery.add_subparsers(dest="gallery_command", metavar="<action>", required=True)
    listing = actions.add_parser("list", help="one line per fixture")
    listing.set_defaults(handler=_cmd_gallery_list)
    show = actions.add_parser("show", help="show one fixture")
    show.add_argument("fixture_id", metavar="ID")
    show.set_defaults(handler=_cmd_gallery_show)
    run = actions.add_parser("run", help="replay one fixture and print its report")
    run.add_argument("fixture_id", metavar="ID")
    run.set_defaults(handler=_cmd_gallery_run)
    return parser


def _parse(argv: Sequence[str]) -> tuple[argparse.Namespace | None, int]:
    """Parse argv, turning both argparse exits into a status this module owns."""
    try:
        return build_parser().parse_args(list(argv)), EXIT_OK
    except _CliError as exc:
        print(exc.message, file=sys.stderr)
        return None, exc.code
    except SystemExit as exc:  # --help exits 0 here; anything else is a usage error
        return None, EXIT_OK if not exc.code else EXIT_USAGE


# --- entry point ------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """CONTRACTS section 6 -- parse, dispatch, and return the handler's int.

    This is the only place a status leaves the package: handlers raise
    _CliError instead of calling sys.exit, so the console script, the module
    entry point and a test calling main([...]) all see the same number.
    """
    namespace, code = _parse(sys.argv[1:] if argv is None else argv)
    if namespace is None:
        return code
    handler = getattr(namespace, "handler", None)
    if handler is None:  # pragma: no cover - every parser above sets a handler
        print("agenttwin: no command given; try 'agenttwin --help'", file=sys.stderr)
        return EXIT_USAGE
    try:
        return int(handler(namespace))
    except _CliError as exc:
        print(exc.message, file=sys.stderr)
        return exc.code
    except DriftError as exc:
        print(f"agenttwin: {exc.type} at seq {exc.seq}: {exc.message}", file=sys.stderr)
        return EXIT_DRIFT
    except SchemaError as exc:
        print(f"agenttwin: schema error: {exc}", file=sys.stderr)
        return EXIT_DRIFT
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("agenttwin: interrupted", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # a raw traceback is never an acceptable output
        print(f"agenttwin: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_USAGE


# --- record -----------------------------------------------------------------


def _cmd_record(args: argparse.Namespace) -> int:
    """CONTRACTS section 6 -- the child owns the tape; the CLI only checks it.

    The child exits 0 with a loadable tape => 0.  A non-zero child, or a child
    that wrote nothing, is a recording failure (3): an environment problem, not
    an assertion about the user's run.  A tape that exists but does not load is
    a schema failure (2).
    """
    child = list(args.child)
    if child and child[0] == "--":
        child = child[1:]
    if not child:
        raise _CliError(
            EXIT_USAGE,
            "agenttwin record: error: no command given"
            " (usage: agenttwin record --out RUNS/run.jsonl -- <cmd> [args...])",
        )
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env[RECORD_OUT_ENV] = str(out)
    env[TRACE_ID_ENV] = out.stem
    try:
        completed = subprocess.run(child, env=env)
    except OSError as exc:
        raise _CliError(EXIT_USAGE, f"agenttwin record: cannot start {child[0]!r}: {exc}") from None
    if completed.returncode != 0:
        raise _CliError(
            EXIT_USAGE,
            f"agenttwin record: child exited {completed.returncode};"
            " recording failed, so no tape is trusted",
        )
    if not out.is_file():
        raise _CliError(
            EXIT_USAGE,
            f"agenttwin record: child exited 0 but wrote no tape at {out}"
            f" (did it read {RECORD_OUT_ENV}?)",
        )
    try:
        Trace.load(out)
    except (SchemaError, UnicodeDecodeError) as exc:
        raise _CliError(EXIT_DRIFT, f"agenttwin record: {out} is not a loadable tape: {exc}") from None
    except OSError as exc:
        raise _CliError(EXIT_USAGE, f"agenttwin record: cannot read back {out}: {exc}") from None
    print(f"recorded {out}")
    return EXIT_OK


# --- replay -----------------------------------------------------------------


def _cmd_replay(args: argparse.Namespace) -> int:
    """CONTRACTS section 6 and 9.26 -- replay, assert, render, return Report's code."""
    specs = _load_specs(args.assert_file)
    agent = _load_agent(args.agent)
    session = _open_session(Path(args.tape))
    report = _run(session, agent)
    if args.strict:
        # CONTRACTS 9.26: strict lives on Report, and it is set before anything
        # reads exit_code -- markdown() and json() both read it.
        report.strict = True
    if specs:
        report.assert_(*specs)
    if args.html:
        _write_html(report, args.html)
    if args.json:
        print(json.dumps(report.json(), ensure_ascii=False, indent=2))
    else:
        print(report.markdown(), end="")
    return int(report.exit_code)


def _load_specs(path: str | None) -> list[Any]:
    """CONTRACTS section 5 -- assertions live in YAML; a broken file is a 3."""
    if not path:
        return []
    from .asserts import load_yaml  # PyYAML is imported inside load_yaml, not here

    try:
        return list(load_yaml(path))
    except FileNotFoundError:
        raise _CliError(EXIT_USAGE, f"agenttwin replay: no such assertion file: {path}") from None
    except (ValueError, RuntimeError) as exc:
        raise _CliError(EXIT_USAGE, f"agenttwin replay: {exc}") from None
    except OSError as exc:
        raise _CliError(EXIT_USAGE, f"agenttwin replay: cannot read {path}: {exc}") from None


def _load_agent(spec: str | None) -> Any:
    """CONTRACTS section 3 -- a tape holds no controller, so the caller names one."""
    if not spec:
        raise _CliError(
            EXIT_USAGE,
            "agenttwin replay: a tape has no controller of its own and the CLI cannot"
            " invent one; pass --agent module:callable",
        )
    module_name, _, attribute = spec.rpartition(":")
    if not module_name or not attribute:
        raise _CliError(
            EXIT_USAGE, f"agenttwin replay: --agent must be module:callable, got {spec!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # ImportError first, but a broken module must not traceback
        raise _CliError(
            EXIT_USAGE,
            f"agenttwin replay: cannot import {module_name!r}: {type(exc).__name__}: {exc}",
        ) from None
    agent = getattr(module, attribute, None)
    if agent is None:
        raise _CliError(
            EXIT_USAGE, f"agenttwin replay: {module_name!r} has no attribute {attribute!r}"
        )
    if not callable(agent):
        raise _CliError(EXIT_USAGE, f"agenttwin replay: {spec!r} is not callable")
    return agent


def _open_session(path: Path) -> Any:
    """CONTRACTS section 3 -- Twin.load, with its failures mapped onto section 6."""
    from .env import Twin

    if not path.is_file():
        raise _CliError(EXIT_USAGE, f"agenttwin: no such tape: {path}")
    try:
        return Twin.load(path)
    except UnicodeDecodeError as exc:
        raise _CliError(EXIT_DRIFT, f"agenttwin: {path} is not a UTF-8 tape: {exc}") from None
    except SchemaError as exc:
        raise _CliError(EXIT_DRIFT, f"agenttwin: {path} is not a tape: {exc}") from None
    except OSError as exc:
        raise _CliError(EXIT_USAGE, f"agenttwin: cannot read {path}: {exc}") from None


def _run(session: Any, agent: Any) -> Any:
    """CONTRACTS section 4 -- a drift aborts replay, so it is mapped, never echoed raw."""
    try:
        return session.replay(agent)
    except DriftError as exc:
        raise _CliError(
            EXIT_DRIFT, f"agenttwin: {exc.type} at seq {exc.seq}: {exc.message}"
        ) from None


def _write_html(report: Report, path: str | Path) -> Path:
    """CONTRACTS section 8 -- one self-contained file, or a 3 explaining why not."""
    try:
        return report.html(path)
    except OSError as exc:
        raise _CliError(EXIT_USAGE, f"agenttwin: cannot write HTML to {path}: {exc}") from None
    except RuntimeError as exc:
        raise _CliError(EXIT_USAGE, f"agenttwin: HTML template is broken: {exc}") from None


# --- diff -------------------------------------------------------------------


def _cmd_diff(args: argparse.Namespace) -> int:
    """CONTRACTS section 7 -- 0 when the tapes agree, 1 when they diverge, 2 if unloadable."""
    baseline = _load_tape(args.baseline)
    current = _load_tape(args.current)
    difference = TraceDiff(baseline, current)
    divergence = difference.first_divergence_seq
    counts = {"same": 0, "changed": 0, "added": 0, "removed": 0}
    for row in difference.events:
        counts[row["change"]] = counts.get(row["change"], 0) + 1
    print(f"first divergence seq: {'none' if divergence is None else divergence}")
    print(
        f"events: {len(difference.events)} total | same {counts['same']},"
        f" changed {counts['changed']}, added {counts['added']}, removed {counts['removed']}"
    )
    for row in difference.events:
        if row["change"] == "same":
            continue
        fields = _changed_fields(row)
        suffix = f": {', '.join(fields)}" if fields else ""
        print(f"  seq {row['seq']} {row['change']}{suffix}")
    if args.html:
        _write_html(Report(baseline=baseline, trace=current, results=[], mode="diff"), args.html)
    return EXIT_OK if divergence is None else EXIT_FAIL


def _changed_fields(row: dict) -> list[str]:
    """Which frozen keys differ on one changed row (kind / actor / payload, section 7)."""
    before = row.get("baseline")
    after = row.get("current")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return []
    return [key for key in ("kind", "actor", "payload") if before.get(key) != after.get(key)]


# --- doctor -----------------------------------------------------------------


def _cmd_doctor(args: argparse.Namespace) -> int:
    """CONTRACTS 9.7 -- diagnose one tape; an invalid tape is a 2, an unreadable file a 3."""
    target = Path(args.tape)
    text = _read_text(target, EXIT_DRIFT)
    try:
        trace = Trace.from_jsonl(text)
    except SchemaError as exc:
        raise _CliError(EXIT_DRIFT, f"agenttwin doctor: {target} is not a tape: {exc}") from None

    findings: list[str] = []
    print(f"tape: {target} (trace_id={trace.header.trace_id}, seed={trace.header.seed})")
    _doctor_schema(trace)
    print(f"events: {len(trace.events)}")
    print("kind counts: " + _kind_counts(trace))
    _doctor_seq(trace)
    _doctor_clock(trace, findings)
    _doctor_terminal(trace, findings)
    _doctor_fingerprint(trace)
    _doctor_redaction(trace)
    _doctor_secrets(text, trace, findings)
    if findings:
        print(f"findings: {len(findings)}")
        for finding in findings:
            print(f"  - {finding}")
        return EXIT_DRIFT
    print("findings: none")
    return EXIT_OK


def _doctor_schema(trace: Trace) -> None:
    """Section 1: the loader accepts only the frozen version and the closed key sets."""
    header_keys = ", ".join(sorted(trace.header.to_json()))
    print(f"schema: schema_version {trace.header.schema_version} ok; header keys ok ({header_keys})")
    event_keys = sorted({key for event in trace.events for key in event.to_json()})
    print(
        "key sets: closed, unknown header/event keys are rejected at load (section 1)"
        + (f"; observed event keys: {', '.join(event_keys)}" if event_keys else "; no events")
    )


def _kind_counts(trace: Trace) -> str:
    counts: dict[str, int] = {}
    for event in trace.events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    if not counts:
        return "none"
    return ", ".join(f"{kind}={counts[kind]}" for kind in sorted(counts))


def _doctor_seq(trace: Trace) -> None:
    """Section 1.2: seq starts at 0, is dense, and strictly increases.

    trace.py:221 rejects a tape that breaks the invariant while loading, so by
    the time doctor runs, density is a fact to report, not a finding to raise.
    """
    last = len(trace.events) - 1
    print(f"seq: dense 0..{last} and strictly increasing")


def _doctor_clock(trace: Trace, findings: list[str]) -> None:
    """Section 1.2: t is a function of seq, tolerance 1e-06."""
    offline = [
        event.seq
        for event in trace.events
        if abs(event.t - round(event.seq * CLOCK_STEP, 4)) > _CLOCK_TOLERANCE
    ]
    if offline:
        print(f"clock: {len(offline)} event(s) off; first at seq {offline[0]}")
        findings.append(
            f"clock: t != round(seq*{CLOCK_STEP}, 4) at seq {offline[0]} (tolerance 1e-06)"
        )
    else:
        print(
            f"clock: t == round(seq*{CLOCK_STEP}, 4) within 1e-06"
            f" for all {len(trace.events)} event(s)"
        )


def _doctor_terminal(trace: Trace, findings: list[str]) -> None:
    """Section 9.7: a tape without a terminal is incomplete, not unreadable."""
    terminal = next((event for event in reversed(trace.events) if event.kind == "terminal"), None)
    if terminal is None:
        print("terminal: MISSING (incomplete tape)")
        findings.append("terminal: no terminal event (incomplete tape, section 9.7)")
        return
    print(f"terminal: present at seq {terminal.seq} (outcome={terminal.payload.get('outcome', '-')})")


def _doctor_fingerprint(trace: Trace) -> None:
    """Section 1.5: one line for the recorded environment."""
    fingerprint = trace.header.env_fingerprint
    row = " ".join(
        f"{field}={_scalar(fingerprint.get(field))}"
        for field in ("model", "prompt_hash", "python", "agenttwin", "platform")
    )
    deps = fingerprint.get("deps") or {}
    print(f"env_fingerprint: {row} deps={len(deps) if isinstance(deps, dict) else 0}")


def _doctor_redaction(trace: Trace) -> None:
    """Section 1.1: how much the recorder already scrubbed."""
    redaction = trace.header.redaction
    print(f"redaction: redacted={redaction.get('redacted', 0)} mode={redaction.get('mode', 'unknown')}")


def _doctor_secrets(text: str, trace: Trace, findings: list[str]) -> None:
    """Sections 1.1 and 9.13 -- scan for what the recorder should have scrubbed.

    Two haystacks, because they differ: the raw file escapes backslashes, so a
    JSON-encoded Windows user path never matches the win_user_path rule on the
    raw text alone, while the decoded values are exactly what the Redactor saw.
    Structural hash values are excluded: sha256 is 64 lowercase hex characters
    and would otherwise make every tape with a terminal report long_hex.
    """
    structural = _structural_hashes(trace)
    haystacks = [text, *_decoded_strings(trace)]
    hits: list[str] = []
    for name, pattern in DEFAULT_PATTERNS:
        try:
            count = sum(
                1
                for haystack in haystacks
                for match in re.finditer(pattern, haystack)
                if match.group(0) not in structural
            )
        except re.error as exc:  # a broken pattern must never crash doctor
            findings.append(f"secret scan: pattern {name!r} is not a valid regex ({exc})")
            continue
        if count:
            hits.append(f"{name} x{count}")
    if hits:
        print("secret scan: hit " + ", ".join(hits))
        findings.append("secret scan: " + ", ".join(hits))
    else:
        print(
            f"secret scan: clean ({len(DEFAULT_PATTERNS)} patterns over raw text and decoded"
            f" values; {len(structural)} structural hash value(s) excluded)"
        )


def _structural_hashes(trace: Trace) -> set[str]:
    """Every value recorded under a hash key: schema data, not a leaked secret."""
    found: set[str] = set()

    def walk(node: Any, key: str | None = None) -> None:
        if isinstance(node, dict):
            for name, value in node.items():
                walk(value, str(name))
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item, key)
        elif isinstance(node, str) and key is not None and key.endswith("_hash"):
            found.add(node)

    walk(trace.header.to_json())
    for event in trace.events:
        walk(event.payload)
    return found


def _decoded_strings(trace: Trace) -> list[str]:
    """Every string the tape holds once JSON escapes are resolved."""
    values: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            values.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(trace.header.to_json())
    for event in trace.events:
        walk(event.payload)
        values.append(event.actor)
    return values


# --- gallery ----------------------------------------------------------------


def _gallery_module() -> Any:
    """Import agenttwin.gallery on first use: a missing module is a 3, not a traceback."""
    try:
        return importlib.import_module("agenttwin.gallery")
    except ImportError as exc:
        raise _CliError(
            EXIT_USAGE, f"agenttwin gallery: agenttwin.gallery is unavailable: {exc}"
        ) from None


def _fixture(module: Any, fixture_id: str) -> Any:
    """The frozen gallery interface: get(id) raises KeyError on an unknown id."""
    try:
        return module.get(fixture_id)
    except KeyError:
        raise _CliError(EXIT_USAGE, f"agenttwin gallery: unknown fixture {fixture_id!r}") from None


def _cmd_gallery_list(args: argparse.Namespace) -> int:
    """Gallery list -- one id and title line per fixture."""
    module = _gallery_module()
    for fixture in module.list_fixtures():
        print(f"{fixture.id}  {fixture.title}")
    return EXIT_OK


def _cmd_gallery_show(args: argparse.Namespace) -> int:
    """Gallery show -- identity, tape path and the fixture's own commands."""
    module = _gallery_module()
    fixture = _fixture(module, args.fixture_id)
    print(f"id: {fixture.id}")
    print(f"title: {fixture.title}")
    print(f"description: {fixture.description}")
    print(f"tape: {fixture.tape}")
    commands = tuple(getattr(fixture, "commands", ()) or ())
    print(f"commands: {len(commands)}")
    for command in commands:
        print(f"  {command}")
    return EXIT_OK


def _cmd_gallery_run(args: argparse.Namespace) -> int:
    """Gallery run -- replay the fixture, assert it, print the report's markdown."""
    module = _gallery_module()
    fixture = _fixture(module, args.fixture_id)
    session = _open_session(Path(fixture.tape))
    report = _run(session, fixture.agent)
    specs = tuple(getattr(fixture, "asserts", ()) or ())
    if specs:
        report.assert_(*specs)
    print(report.markdown(), end="")
    return int(report.exit_code)


# --- shared readers ---------------------------------------------------------


def _read_text(target: Path, not_a_tape_code: int) -> str:
    """Read a tape as UTF-8; a missing file is always an environment error (3)."""
    try:
        return target.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise _CliError(EXIT_USAGE, f"agenttwin: no such tape: {target}") from None
    except UnicodeDecodeError as exc:
        raise _CliError(not_a_tape_code, f"agenttwin: {target} is not a UTF-8 tape: {exc}") from None
    except OSError as exc:
        raise _CliError(EXIT_USAGE, f"agenttwin: cannot read {target}: {exc}") from None


def _load_tape(path: str | Path) -> Trace:
    """Section 1 -- a present but unloadable tape is a drift-class 2."""
    target = Path(path)
    text = _read_text(target, EXIT_DRIFT)
    try:
        return Trace.from_jsonl(text)
    except SchemaError as exc:
        raise _CliError(EXIT_DRIFT, f"agenttwin: {target} is not a tape: {exc}") from None


def _scalar(value: Any) -> str:
    return "-" if value is None else str(value)


if __name__ == "__main__":  # pragma: no cover - exercised via python -m agenttwin
    raise SystemExit(main())
