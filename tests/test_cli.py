"""The CLI surface (CONTRACTS section 6 and section 9.26).

Two properties matter here and both are user-visible: the five commands are
reachable, and the four statuses mean exactly what section 6 says they mean. So
every test drives main([...]) in-process, reads the captured streams and
compares the returned int -- nothing reaches into a private helper. The single
subprocess is the record smoke test, which is allowed to leave no tape behind
because that missing tape is the exit 3 path it pins.

Scratch paths come from the tape_dir fixture: CONTRACTS 9.17 makes pytest
tmp_path unusable on this platform.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import agenttwin
from agenttwin import Twin, cli
from agenttwin.diff import TraceDiff
from agenttwin.trace import Event, Trace

# --- fixtures and controllers ----------------------------------------------


def fake_llm(request: dict) -> dict:
    return {"text": "ok:" + request["prompt"]}


def shell_tool(command: str) -> str:
    return "ran:" + command


def record_tape(path: Path, *, with_tool: bool = False) -> Path:
    """One real tape: an llm_call, optionally a tool_call, then the terminal."""
    with Twin.record(path.stem, out=path, llm=fake_llm, tools={"shell": shell_tool}) as recorder:
        recorder.llm({"prompt": "hi"})
        if with_tool:
            recorder.tools()["shell"]("echo 0")
    return path


def agent_simple(llm, tools):
    llm({"prompt": "hi"})


def agent_with_tool(llm, tools):
    llm({"prompt": "hi"})
    tools["shell"]("echo 0")


def agent_drifting(llm, tools):
    llm({"prompt": "a request the tape never saw"})


@pytest.fixture
def agents(monkeypatch) -> str:
    """A module importable as module:callable, so --agent is exercised honestly."""
    module = types.ModuleType("cli_test_agents")
    module.simple = agent_simple
    module.with_tool = agent_with_tool
    module.drifting = agent_drifting
    module.not_callable = "not a callable"
    monkeypatch.setitem(sys.modules, "cli_test_agents", module)
    return "cli_test_agents"


def write_asserts(path: Path, entry: str) -> Path:
    path.write_text(entry, encoding="utf-8")
    return path


class FakeRun:
    """Stand-in for subprocess.run: records the child and may write a tape."""

    def __init__(self, returncode: int = 0, writer=None) -> None:
        self.returncode = returncode
        self.writer = writer
        self.calls: list[dict] = []

    def __call__(self, child, env=None, **kwargs):
        self.calls.append({"child": list(child), "env": dict(env or {})})
        if self.writer is not None:
            self.writer(self.calls[-1]["env"])
        return types.SimpleNamespace(returncode=self.returncode)


# --- parser and usage errors ------------------------------------------------


def test_help_exits_zero(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for command in ("record", "replay", "diff", "doctor", "gallery"):
        assert command in out


def test_version_prints_the_package_version(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"agenttwin {agenttwin.__version__}"


def test_gallery_help_exits_zero(capsys):
    assert cli.main(["gallery", "--help"]) == 0
    out = capsys.readouterr().out
    assert "list" in out and "show" in out and "run" in out


def test_an_unknown_command_is_a_three(capsys):
    assert cli.main(["frobnicate"]) == 3
    err = capsys.readouterr().err
    assert "frobnicate" in err and "Traceback" not in err


def test_no_command_at_all_is_a_three(capsys):
    assert cli.main([]) == 3
    assert "Traceback" not in capsys.readouterr().err


def test_a_missing_required_flag_is_a_three_not_an_argparse_two(capsys):
    """CONTRACTS section 6: usage errors are 3; argparse's own status 2 is remapped."""
    assert cli.main(["record", "--", "some-command"]) == 3
    err = capsys.readouterr().err
    assert "--out" in err and "Traceback" not in err


def test_the_module_entry_point_delegates_to_main():
    from agenttwin import __main__ as module_entry

    assert module_entry.main is cli.main


# --- record -----------------------------------------------------------------


def test_record_injects_an_absolute_out_and_the_out_stem(tape_dir, monkeypatch, capsys):
    out = tape_dir / "run.jsonl"
    fake = FakeRun(writer=lambda env: record_tape(Path(env[cli.RECORD_OUT_ENV])))
    monkeypatch.setattr(cli.subprocess, "run", fake)

    code = cli.main(["record", "--out", str(out), "--", "agent-cli", "--flag", "x"])

    assert code == 0
    call = fake.calls[0]
    assert call["child"] == ["agent-cli", "--flag", "x"]
    absolute = Path(call["env"][cli.RECORD_OUT_ENV])
    assert absolute.is_absolute() and absolute == out.resolve()
    assert call["env"][cli.TRACE_ID_ENV] == "run"
    assert "recorded" in capsys.readouterr().out


def test_record_a_nonzero_child_is_a_three(tape_dir, monkeypatch, capsys):
    fake = FakeRun(returncode=2)
    monkeypatch.setattr(cli.subprocess, "run", fake)

    assert cli.main(["record", "--out", str(tape_dir / "nope.jsonl"), "--", "agent-cli"]) == 3
    err = capsys.readouterr().err
    assert "child exited 2" in err and "Traceback" not in err


def test_record_without_a_tape_is_a_three(tape_dir, monkeypatch, capsys):
    monkeypatch.setattr(cli.subprocess, "run", FakeRun(returncode=0))

    assert cli.main(["record", "--out", str(tape_dir / "missing.jsonl"), "--", "agent-cli"]) == 3
    err = capsys.readouterr().err
    assert "wrote no tape" in err and "Traceback" not in err


def test_record_an_unloadable_tape_is_a_two(tape_dir, monkeypatch, capsys):
    junk = tape_dir / "junk.jsonl"

    def writer(env):
        Path(env[cli.RECORD_OUT_ENV]).write_text("this is not a tape", encoding="utf-8")

    monkeypatch.setattr(cli.subprocess, "run", FakeRun(returncode=0, writer=writer))

    assert cli.main(["record", "--out", str(junk), "--", "agent-cli"]) == 2
    err = capsys.readouterr().err
    assert "not a loadable tape" in err and "Traceback" not in err


def test_record_without_a_command_is_a_three(tape_dir, capsys):
    assert cli.main(["record", "--out", str(tape_dir / "empty.jsonl"), "--"]) == 3
    assert "no command given" in capsys.readouterr().err


def test_record_spawns_one_real_child(tape_dir, capsys):
    """The only real subprocess: a child that writes no tape leaves exit 3."""
    out = tape_dir / "real.jsonl"
    code = cli.main(["record", "--out", str(out), "--", sys.executable, "-c", "print(1)"])

    assert code == 3
    assert "wrote no tape" in capsys.readouterr().err


# --- replay -----------------------------------------------------------------


def test_replay_without_an_agent_explains_itself_and_exits_three(tape_dir, capsys):
    tape = record_tape(tape_dir / "noagent.jsonl")

    assert cli.main(["replay", str(tape)]) == 3
    err = capsys.readouterr().err
    assert "--agent" in err and "cannot invent one" in err


@pytest.mark.parametrize("spec", ["no-colon-at-all", "module:", ":attribute"])
def test_replay_rejects_a_malformed_agent_spec(tape_dir, capsys, spec):
    tape = record_tape(tape_dir / "badspec.jsonl")

    assert cli.main(["replay", str(tape), "--agent", spec]) == 3
    err = capsys.readouterr().err
    assert "module:callable" in err and "Traceback" not in err


def test_replay_an_unimportable_agent_module_is_a_three(tape_dir, capsys):
    tape = record_tape(tape_dir / "nomod.jsonl")

    assert cli.main(["replay", str(tape), "--agent", "definitely_not_installed:agent"]) == 3
    err = capsys.readouterr().err
    assert "cannot import" in err and "Traceback" not in err


def test_replay_a_missing_attribute_is_a_three(tape_dir, capsys):
    tape = record_tape(tape_dir / "noattr.jsonl")

    assert cli.main(["replay", str(tape), "--agent", "json:no_such_attribute"]) == 3
    assert "has no attribute" in capsys.readouterr().err


def test_replay_a_non_tape_file_is_a_two_without_a_traceback(tape_dir, agents, capsys):
    junk = tape_dir / "not-a-tape.jsonl"
    junk.write_text("plain text, no tape here", encoding="utf-8")

    assert cli.main(["replay", str(junk), "--agent", agents + ":simple"]) == 2
    err = capsys.readouterr().err
    assert "not a tape" in err and "Traceback" not in err


def test_replay_a_missing_tape_is_a_three(tape_dir, agents, capsys):
    assert cli.main(["replay", str(tape_dir / "absent.jsonl"), "--agent", agents + ":simple"]) == 3
    assert "no such tape" in capsys.readouterr().err


def test_replay_a_clean_tape_exits_zero_and_prints_markdown(tape_dir, agents, capsys):
    tape = record_tape(tape_dir / "clean.jsonl")

    assert cli.main(["replay", str(tape), "--agent", agents + ":simple"]) == 0
    out = capsys.readouterr().out
    assert "verdict:" in out and "exit code 0" in out and "strict: no" in out


def test_replay_json_prints_the_report_mapping(tape_dir, agents, capsys):
    tape = record_tape(tape_dir / "json.jsonl")

    assert cli.main(["replay", str(tape), "--agent", agents + ":simple", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "0.1"
    assert payload["exit_code"] == 0


def test_replay_html_writes_the_single_file_replayer(tape_dir, agents):
    tape = record_tape(tape_dir / "html.jsonl")
    target = tape_dir / "replayer.html"

    assert cli.main(["replay", str(tape), "--agent", agents + ":simple", "--html", str(target)]) == 0
    assert target.is_file()
    html = target.read_text(encoding="utf-8")
    assert "__AGENTTWIN_PAYLOAD__" not in html
    assert len(html) > 1000


def test_replay_a_failed_assertion_exits_one(tape_dir, agents, capsys):
    tape = record_tape(tape_dir / "fail.jsonl", with_tool=True)
    spec = write_asserts(tape_dir / "fail.yaml", "- {spec: max_repeats, args: {tool: shell, n: 0}}")

    code = cli.main(["replay", str(tape), "--assert", str(spec), "--agent", agents + ":with_tool"])

    assert code == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "max_repeats" in out


def test_replay_a_broken_assert_file_is_a_three(tape_dir, agents, capsys):
    tape = record_tape(tape_dir / "badyaml.jsonl")
    spec = write_asserts(tape_dir / "bad.yaml", "- {spec: no_such_assertion}")

    assert cli.main(["replay", str(tape), "--assert", str(spec), "--agent", agents + ":simple"]) == 3
    err = capsys.readouterr().err
    assert "unknown assertion" in err and "Traceback" not in err


def test_strict_turns_an_all_skip_tape_into_exit_one(tape_dir, agents, capsys):
    """CONTRACTS 9.26: strict lives on Report, and a SKIP-only tape fails under it."""
    tape = record_tape(tape_dir / "skip.jsonl")
    spec = write_asserts(tape_dir / "skip.yaml", "- {spec: budget_within}")
    argv = ["replay", str(tape), "--assert", str(spec), "--agent", agents + ":simple"]

    assert cli.main(argv) == 0
    lax = capsys.readouterr().out
    assert "SKIP" in lax and "strict: no" in lax

    assert cli.main(argv + ["--strict"]) == 1
    strict = capsys.readouterr().out
    assert "SKIP" in strict and "strict: yes" in strict and "exit code 1" in strict


def test_replay_maps_a_drift_to_two(tape_dir, agents, capsys):
    tape = record_tape(tape_dir / "drift.jsonl")

    code = cli.main(["replay", str(tape), "--agent", agents + ":drifting"])

    err = capsys.readouterr().err
    assert code == 2
    assert "Traceback" not in err


# --- diff -------------------------------------------------------------------

@pytest.mark.parametrize("destination", ["file", "stdout"])
@pytest.mark.parametrize("shorter", [False, True])
def test_diff_json_matches_library_and_retains_exit_code(tape_dir, capsys, destination, shorter):
    baseline = record_tape(tape_dir / "base.jsonl")
    trace = Trace.load(baseline)
    current = tape_dir / "current.jsonl"
    Trace(header=trace.header, events=trace.events[:1] if shorter else trace.events).save(current)
    expected = TraceDiff(trace, Trace.load(current)).to_dict()
    expected["counts"] = {"same": 1 if shorter else 2, "changed": 0, "added": 0, "removed": int(shorter)}
    output = tape_dir / "diff.json"
    args = ["diff", str(baseline), str(current), "--json"]
    if destination == "file":
        args.append(str(output))
    assert cli.main(args) == int(shorter)
    captured = capsys.readouterr()
    assert captured.err == ""
    text = output.read_text(encoding="utf-8") if destination == "file" else captured.out
    assert json.loads(text) == expected
    if destination == "stdout":
        assert not output.exists()


def test_diff_json_keeps_invalid_tape_exit_code(tape_dir, capsys):
    good = record_tape(tape_dir / "valid.jsonl")
    bad = tape_dir / "invalid.jsonl"
    bad.write_text("not JSONL", encoding="utf-8")
    output = tape_dir / "diff.json"
    assert cli.main(["diff", str(good), str(bad), "--json", str(output)]) == 2
    assert not output.exists()
    assert "not a tape" in capsys.readouterr().err


def test_diff_json_reports_output_errors(tape_dir, capsys):
    tape = record_tape(tape_dir / "valid.jsonl")
    target = tape_dir / "missing" / "diff.json"
    assert cli.main(["diff", str(tape), str(tape), "--json", str(target)]) == 3
    assert "cannot write JSON" in capsys.readouterr().err


def test_diff_of_a_tape_with_itself_exits_zero(tape_dir, capsys):
    tape = record_tape(tape_dir / "same.jsonl")

    assert cli.main(["diff", str(tape), str(tape)]) == 0
    out = capsys.readouterr().out
    assert "first divergence seq: none" in out and "same 2" in out


def test_diff_detects_a_changed_event(tape_dir, capsys):
    tape = record_tape(tape_dir / "base.jsonl", with_tool=True)
    trace = Trace.load(tape)
    first = trace.events[0]
    payload = dict(first.payload)
    payload["response"] = {"text": "a different answer"}
    changed = Trace(
        header=trace.header,
        events=[Event(first.seq, first.t, first.kind, first.actor, payload)] + trace.events[1:],
    )
    other = tape_dir / "changed.jsonl"
    changed.save(other)

    assert cli.main(["diff", str(tape), str(other)]) == 1
    out = capsys.readouterr().out
    assert "first divergence seq: 0" in out
    assert "seq 0 changed" in out and "changed 1" in out


def test_diff_reports_a_removed_event(tape_dir, capsys):
    tape = record_tape(tape_dir / "full.jsonl")
    trace = Trace.load(tape)
    shorter = Trace(header=trace.header, events=trace.events[:1])
    other = tape_dir / "shorter.jsonl"
    shorter.save(other)

    assert cli.main(["diff", str(tape), str(other)]) == 1
    out = capsys.readouterr().out
    assert "removed 1" in out and "seq 1 removed" in out


def test_diff_a_non_tape_is_a_two(tape_dir, capsys):
    good = record_tape(tape_dir / "ok.jsonl")
    junk = tape_dir / "junk-file.jsonl"
    junk.write_text("{not json", encoding="utf-8")

    assert cli.main(["diff", str(good), str(junk)]) == 2
    err = capsys.readouterr().err
    assert "not a tape" in err and "Traceback" not in err


def test_diff_html_renders_both_tapes(tape_dir):
    good = record_tape(tape_dir / "html-a.jsonl")
    junk_tape = Trace.load(good)
    shorter = Trace(header=junk_tape.header, events=junk_tape.events[:1])
    other = tape_dir / "html-b.jsonl"
    shorter.save(other)
    target = tape_dir / "diff.html"

    assert cli.main(["diff", str(good), str(other), "--html", str(target)]) == 1
    html = target.read_text(encoding="utf-8")
    assert "__AGENTTWIN_PAYLOAD__" not in html and len(html) > 1000


# --- doctor -----------------------------------------------------------------


def test_doctor_on_a_recorded_tape_is_clean(tape_dir, capsys):
    tape = record_tape(tape_dir / "doctor.jsonl")

    assert cli.main(["doctor", str(tape)]) == 0
    out = capsys.readouterr().out
    assert "schema: schema_version 0.1 ok" in out
    assert "kind counts: llm_call=1, terminal=1" in out
    assert "seq: dense 0..1" in out
    assert "clock: t == round(seq*0.1, 4) within 1e-06" in out
    assert "terminal: present at seq 1" in out
    assert "env_fingerprint:" in out
    assert "redaction: redacted=0" in out
    assert "secret scan: clean" in out
    assert "findings: none" in out


def test_doctor_flags_a_leaked_secret(tape_dir, capsys):
    tape = record_tape(tape_dir / "leak.jsonl")
    trace = Trace.load(tape)
    first = trace.events[0]
    payload = dict(first.payload)
    payload["request"] = {"prompt": "sk-" + "A" * 24}
    leaked = Trace(
        header=trace.header,
        events=[Event(first.seq, first.t, first.kind, first.actor, payload)] + trace.events[1:],
    )
    target = tape_dir / "leaked.jsonl"
    leaked.save(target)

    assert cli.main(["doctor", str(target)]) == 2
    out = capsys.readouterr().out
    assert "secret scan: hit" in out and "openai_sk" in out
    assert "findings: 1" in out


def test_doctor_flags_a_missing_terminal(tape_dir, capsys):
    tape = record_tape(tape_dir / "incomplete.jsonl")
    trace = Trace.load(tape)
    truncated = Trace(header=trace.header, events=trace.events[:-1])
    target = tape_dir / "no-terminal.jsonl"
    truncated.save(target)

    assert cli.main(["doctor", str(target)]) == 2
    out = capsys.readouterr().out
    assert "terminal: MISSING" in out and "section 9.7" in out


def test_doctor_rejects_a_non_tape_as_a_two(tape_dir, capsys):
    """CONTRACTS section 6: a present but unloadable tape is a 2, not a 3."""
    junk = tape_dir / "doctor-junk.jsonl"
    junk.write_text("nope", encoding="utf-8")

    assert cli.main(["doctor", str(junk)]) == 2
    err = capsys.readouterr().err
    assert "not a tape" in err and "Traceback" not in err


def test_doctor_reports_a_broken_seq_as_a_two(tape_dir, capsys):
    """The loader is where density is enforced; doctor must report it as a 2."""
    tape = record_tape(tape_dir / "dense.jsonl")
    trace = Trace.load(tape)
    first = trace.events[0]
    target = tape_dir / "sparse.jsonl"
    Trace(
        header=trace.header,
        events=[Event(3, 0.3, first.kind, first.actor, dict(first.payload))],
    ).save(target)

    assert cli.main(["doctor", str(target)]) == 2
    err = capsys.readouterr().err
    assert "is not a tape" in err
    assert "expected seq 0 but found 3" in err and "Traceback" not in err


def test_doctor_a_missing_file_is_a_three(tape_dir, capsys):
    assert cli.main(["doctor", str(tape_dir / "nowhere.jsonl")]) == 3
    assert "no such tape" in capsys.readouterr().err


# --- gallery ----------------------------------------------------------------


def test_gallery_list_prints_one_line_per_fixture(capsys):
    from agenttwin import gallery

    assert cli.main(["gallery", "list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [fixture.id + "  " + fixture.title for fixture in gallery.list_fixtures()]


def test_gallery_show_prints_the_fixture_fields(capsys):
    from agenttwin import gallery

    fixture = gallery.list_fixtures()[0]
    assert cli.main(["gallery", "show", fixture.id]) == 0
    out = capsys.readouterr().out
    for expected in (fixture.id, fixture.title, fixture.description, str(fixture.tape)):
        assert expected in out
    assert "commands: " + str(len(fixture.commands)) in out


@pytest.mark.parametrize("fixture_id, expected", [("deathloop", 1), ("repaired", 0)])
def test_gallery_run_returns_the_verdict_exit_code(fixture_id, expected, capsys):
    assert cli.main(["gallery", "run", fixture_id]) == expected
    out = capsys.readouterr().out
    assert "exit code " + str(expected) in out


def test_gallery_an_unknown_id_is_a_three(capsys):
    assert cli.main(["gallery", "show", "no-such-fixture"]) == 3
    err = capsys.readouterr().err
    assert "no-such-fixture" in err and "Traceback" not in err


def test_gallery_a_missing_module_is_a_three_not_a_traceback(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "agenttwin.gallery", None)

    assert cli.main(["gallery", "list"]) == 3
    err = capsys.readouterr().err
    assert "agenttwin.gallery" in err and "Traceback" not in err
