"""Tests for the assertion DSL (CONTRACTS section 5).

Every assertion is a pure function of the tape, so each test builds a tape by
hand and reads the verdict back. No tmp_path (CONTRACTS section 9.17): files go
into the tape_dir fixture from tests/conftest.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import agenttwin.asserts as asserts
from agenttwin.asserts import (
    MODES,
    AssertionGroup,
    AssertionResult,
    AssertionSpec,
    assert_,
    assert_all,
    assert_any,
    budget_within,
    call_after,
    call_order,
    load_yaml,
    max_repeats,
    never_calls,
    no_deadlock,
    output_schema,
    state_invariant,
    terminates_within,
    warn,
)
from agenttwin.trace import Header, Trace

ROW_KEYS = {"name", "status", "detail", "first_offense_seq"}


# --- tape builders ---------------------------------------------------------


def make_header(**overrides) -> Header:
    fields = {
        "schema_version": "0.1",
        "trace_id": "checkout",
        "seed": 7,
        "env_fingerprint": {"python": "3.12.7", "platform": "win32"},
        "redaction": {"rules": []},
    }
    fields.update(overrides)
    return Header(**fields)


def make_trace(*steps) -> Trace:
    trace = Trace(header=make_header())
    for kind, actor, payload in steps:
        trace.append(kind, actor, payload)
    return trace


def call(actor: str, args_hash: str = "ah-1", preview=None, **extra):
    payload = {"tool": actor, "args_hash": args_hash}
    if preview is not None:
        payload["args_preview"] = preview
    payload.update(extra)
    return ("tool_call", actor, payload)


def result_of(actor: str = "shell"):
    return ("tool_result", actor, {"tool": actor, "result": ["ok"], "ok": True})


def done(state=None):
    final = {"status": "done"} if state is None else state
    return ("terminal", "system", {"state": final, "state_hash": "sh-1"})


def patch(points, actor: str = "agent"):
    return ("state_patch", actor, {"patch": points})


def llm(model: str = "m1", prompt: int = 0, completion: int = 0, tokens: bool = True):
    payload = {"prompt_hash": "ph-1", "response": {"text": "ok"}}
    if tokens:
        payload["tokens"] = {"prompt": prompt, "completion": completion}
        payload["model"] = model
    return ("llm_call", "planner", payload)


def death_loop(calls: int = 6) -> Trace:
    """Six identical shell calls: the canonical runaway tape."""
    steps = [("human_input", "user", {"input": "tidy the tree"})]
    steps += [call("shell", "ah-dead", "ls -la") for _ in range(calls)]
    steps.append(done({"status": "stuck"}))
    return make_trace(*steps)


# --- the frozen result shape ------------------------------------------------


def test_to_dict_has_exactly_the_four_contract_keys():
    row = AssertionResult("max_repeats(shell, n=3)", True, "3 call(s)", None).to_dict()
    assert set(row) == ROW_KEYS
    assert row["name"] == "max_repeats(shell, n=3)"
    assert row["status"] == "PASS"
    assert row["first_offense_seq"] is None


def test_to_dict_maps_passed_to_status_pass_fail_skip():
    assert AssertionResult("x", True, "d").to_dict()["status"] == "PASS"
    assert AssertionResult("x", False, "d", 4).to_dict()["status"] == "FAIL"
    assert AssertionResult("x", None, "d").to_dict()["status"] == "SKIP"
    assert AssertionResult("x", False, "d", 4).to_dict()["first_offense_seq"] == 4


def test_spec_is_callable_and_keeps_its_name():
    spec = max_repeats("shell", 3)
    verdict = spec(death_loop())
    assert isinstance(spec, AssertionSpec)
    assert isinstance(verdict, AssertionResult)
    assert verdict.name == spec.name == "max_repeats(shell, n=3)"


# --- group modes -----------------------------------------------------------


def test_group_rejects_an_unknown_mode_and_an_empty_bundle():
    spec = never_calls("shell")
    with pytest.raises(ValueError, match="mode must be one of"):
        AssertionGroup("g", (spec,), mode="sometimes")
    with pytest.raises(ValueError, match="holds no specs"):
        AssertionGroup("g", ())


def test_all_mode_returns_one_untagged_row_per_spec():
    trace = death_loop()
    hard = assert_(max_repeats("shell", 3), never_calls("fs"))
    rows = hard.evaluate(trace)
    assert hard.mode == "all"
    assert [row.passed for row in rows] == [False, True]
    assert [row.name for row in rows] == [spec.name for spec in hard.specs]
    assert not rows[0].detail.startswith("[")


def test_warn_mode_tags_every_row_and_keeps_it_per_spec():
    trace = death_loop()
    soft = warn(max_repeats("shell", 3), never_calls("fs"))
    rows = soft.evaluate(trace)
    assert soft.mode == "warn"
    assert len(rows) == 2
    assert all(row.detail.startswith("[warn] ") for row in rows)
    assert [row.to_dict()["status"] for row in rows] == ["FAIL", "PASS"]


def test_assert_all_matches_assert_and_carries_its_own_label():
    trace = death_loop()
    left = assert_(max_repeats("shell", 3)).evaluate(trace)
    right = assert_all(max_repeats("shell", 3)).evaluate(trace)
    assert [row.to_dict() for row in left] == [row.to_dict() for row in right]
    assert assert_all(max_repeats("shell", 3)).name.startswith("assert_all(")


def test_any_mode_folds_to_one_row_and_passes_on_one_success():
    trace = death_loop()
    folded = assert_any(max_repeats("shell", 3), never_calls("fs")).evaluate(trace)
    assert len(folded) == 1
    assert folded[0].passed is True
    assert folded[0].detail.startswith("[any] ")
    assert folded[0].name == "assert_any(max_repeats(shell, n=3), never_calls(fs))"


def test_any_mode_fails_with_the_earliest_offence_when_every_spec_fails():
    trace = death_loop()
    folded = assert_any(max_repeats("shell", 0), max_repeats("shell", 3)).evaluate(trace)
    assert folded[0].passed is False
    assert folded[0].first_offense_seq == 1  # the n=0 spec trips on the first shell call
    assert "every spec failed" in folded[0].detail


def test_any_mode_skips_when_nothing_passed_but_one_spec_is_undetermined():
    trace = make_trace(("human_input", "user", {"input": "go"}), done())
    folded = assert_any(no_deadlock(3), terminates_within(1)).evaluate(trace)
    assert folded[0].passed is None
    assert folded[0].first_offense_seq is None
    assert "one undetermined" in folded[0].detail


# --- max_repeats -----------------------------------------------------------


def test_max_repeats_passes_at_the_limit_and_fails_on_the_next_call():
    at_limit = make_trace(*[call("shell", f"ah-{i}") for i in range(3)])
    assert max_repeats("shell", 3)(at_limit).passed is True
    over = make_trace(("human_input", "user", {"input": "go"}), *[call("shell", f"ah-{i}") for i in range(6)])
    verdict = max_repeats("shell", 3)(over)
    assert verdict.passed is False
    assert verdict.first_offense_seq == 4  # seqs 1..6; the 4th call is seq 4


def test_max_repeats_counts_only_the_named_actor_and_passes_when_absent():
    trace = make_trace(call("grep", "ah-1"), call("shell", "ah-2"), call("shell", "ah-3"))
    assert max_repeats("shell", 3)(trace).passed is True  # 2 shell calls stay within n=3
    absent = max_repeats("deploy", 3)(trace)
    assert absent.passed is True
    assert "0 call(s)" in absent.detail


def test_max_repeats_zero_fails_on_the_first_call():
    verdict = max_repeats("shell", 0)(make_trace(("human_input", "user", {"input": "go"}), call("shell")))
    assert verdict.passed is False
    assert verdict.first_offense_seq == 1


def test_max_repeats_rejects_a_negative_limit():
    with pytest.raises(ValueError, match="integer >= 0"):
        max_repeats("shell", -1)


# --- no_deadlock -----------------------------------------------------------


def test_no_deadlock_fails_on_the_run_that_exceeds_the_limit():
    stuck = make_trace(*[call("shell", "ah-same") for _ in range(4)])
    verdict = no_deadlock(3)(stuck)
    assert verdict.passed is False
    assert verdict.first_offense_seq == 3  # the fourth identical call, seqs 0..3


def test_no_deadlock_passes_when_the_run_is_broken_by_different_args():
    trace = make_trace(call("shell", "ah-1"), call("shell", "ah-2"), call("shell", "ah-1"), call("shell", "ah-2"))
    verdict = no_deadlock(1)(trace)
    assert verdict.passed is True
    assert "longest identical run 1" in verdict.detail


def test_no_deadlock_skips_a_tape_without_tool_calls():
    trace = make_trace(("human_input", "user", {"input": "go"}), done())
    verdict = no_deadlock(3)(trace)
    assert verdict.passed is None
    assert verdict.first_offense_seq is None
    assert "no tool_call" in verdict.detail


# --- budget_within ---------------------------------------------------------


def test_budget_within_counts_calls_and_names_the_first_excess():
    trace = make_trace(("human_input", "user", {"input": "go"}), *[call(f"t{i}", f"ah-{i}") for i in range(3)])
    verdict = budget_within(max_calls=2)(trace)
    assert verdict.passed is False
    assert verdict.first_offense_seq == 3  # call seqs 1,2,3 -> the third one breaks <=2
    under = budget_within(max_calls=5)(trace)
    assert under.passed is True
    assert "calls 3/5" in under.detail


def test_budget_within_seconds_compares_the_last_event_time():
    trace = make_trace(("human_input", "user", {"input": "go"}), call("a"), call("b", "ah-2"))
    assert trace.events[-1].t == 0.2
    verdict = budget_within(max_seconds=0.1)(trace)
    assert verdict.passed is False
    assert verdict.first_offense_seq == 2  # first event whose t exceeds 0.1
    assert budget_within(max_seconds=0.5)(trace).passed is True


def test_budget_within_skips_an_unmeasurable_dimension_instead_of_failing():
    no_usage = make_trace(("human_input", "user", {"input": "go"}), call("a"))
    verdict = budget_within(max_tokens=10)(no_usage)
    assert verdict.passed is None
    assert verdict.first_offense_seq is None
    assert "no llm_call payload carries 'tokens'" in verdict.detail

    mixed = budget_within(max_calls=10, max_tokens=10)(no_usage)
    assert mixed.passed is True
    assert "not measured: max_tokens" in mixed.detail


def test_budget_within_tokens_fails_at_the_llm_call_that_crosses_the_line():
    trace = make_trace(("human_input", "user", {"input": "go"}), llm(prompt=600), llm(prompt=600))
    verdict = budget_within(max_tokens=1000)(trace)
    assert verdict.passed is False
    assert verdict.first_offense_seq == 2
    assert budget_within(max_tokens=5000)(trace).passed is True


def test_budget_within_price_table_only_adds_an_estimate_to_the_detail():
    trace = make_trace(
        ("human_input", "user", {"input": "go"}),
        llm(model="m1", prompt=1_000_000, completion=500_000),
    )
    priced = budget_within(max_calls=10, price_table={"m1": {"prompt": 1.0, "completion": 2.0}})(trace)
    assert priced.passed is True
    assert "estimated cost 2.000000 USD" in priced.detail
    unpriced = budget_within(max_calls=10, price_table={"m9": {"prompt": 1.0, "completion": 2.0}})(trace)
    assert unpriced.passed is True
    assert "estimated cost" not in unpriced.detail


def test_budget_within_skips_when_no_dimension_was_requested():
    verdict = budget_within()(death_loop())
    assert verdict.passed is None
    assert "no budget dimension was requested" in verdict.detail


def test_budget_within_rejects_bad_bounds_and_a_malformed_price_table():
    with pytest.raises(ValueError, match="non-negative number"):
        budget_within(max_calls=-1)
    with pytest.raises(ValueError, match="non-negative number"):
        budget_within(max_seconds=True)
    with pytest.raises(ValueError, match="price_table"):
        budget_within(max_calls=1, price_table={"m1": {"prompt": 1.0}})


# --- never_calls -----------------------------------------------------------


def test_never_calls_fails_on_a_matching_call_and_names_its_seq():
    trace = make_trace(("human_input", "user", {"input": "go"}), call("shell", "ah-1", "rm -rf /tmp"))
    verdict = never_calls("shell", args_contains="rm -rf")(trace)
    assert verdict.passed is False
    assert verdict.first_offense_seq == 1
    assert "args matching" in verdict.detail


def test_never_calls_is_case_insensitive_and_falls_back_to_the_args_payload():
    upper = make_trace(call("shell", "ah-1", "RM -RF /"))
    assert never_calls("shell", args_contains="rm -rf")(upper).passed is False
    plain = make_trace(("tool_call", "fs", {"tool": "fs", "args_hash": "ah-9", "args": {"path": "/etc/shadow"}}))
    verdict = never_calls("fs", args_contains="/etc")(plain)
    assert verdict.passed is False
    assert verdict.first_offense_seq == 0


def test_never_calls_passes_when_the_tool_never_matches():
    trace = make_trace(call("shell", "ah-1", "ls -la"))
    assert never_calls("shell", args_contains="rm -rf")(trace).passed is True
    assert never_calls("shell")(trace).passed is False  # a bare tool name matches any call


# --- call_order ------------------------------------------------------------


def test_call_order_accepts_a_subsequence_with_other_calls_in_between():
    trace = make_trace(call("read", "ah-1"), call("grep", "ah-2"), call("edit", "ah-3"))
    verdict = call_order(("read", "edit"))(trace)
    assert verdict.passed is True
    assert "in order at seqs 0, 2" in verdict.detail


def test_call_order_fails_when_the_later_name_arrives_first():
    trace = make_trace(call("edit", "ah-1"), call("read", "ah-2"))
    verdict = call_order(("read", "edit"))(trace)
    assert verdict.passed is False
    assert verdict.first_offense_seq == 0
    assert "arrived before read" in verdict.detail


def test_call_order_fails_when_the_tail_of_a_three_step_chain_never_arrives():
    trace = make_trace(call("read", "ah-1"), call("edit", "ah-2"))
    verdict = call_order(("read", "edit", "test"))(trace)
    assert verdict.passed is False
    assert verdict.first_offense_seq == 1  # last matched call
    assert "test never appears" in verdict.detail


def test_call_order_skips_when_fewer_than_two_names_appear():
    trace = make_trace(call("read", "ah-1"), call("grep", "ah-2"))
    verdict = call_order(("read", "edit"))(trace)
    assert verdict.passed is None
    assert verdict.first_offense_seq is None
    assert "cannot be judged" in verdict.detail


def test_call_order_needs_two_names():
    with pytest.raises(ValueError, match="at least two names"):
        call_order(("read",))


# --- call_after ------------------------------------------------------------


def test_call_after_passes_when_every_b_call_follows_an_a_call():
    trace = make_trace(call("read", "ah-1"), call("edit", "ah-2"), call("edit", "ah-3"))
    verdict = call_after("read", "edit")(trace)
    assert verdict.passed is True
    assert "2 call(s) of edit" in verdict.detail


def test_call_after_fails_on_a_b_call_with_no_preceding_a():
    trace = make_trace(("human_input", "user", {"input": "go"}), call("edit", "ah-1"))
    verdict = call_after("read", "edit")(trace)
    assert verdict.passed is False
    assert verdict.first_offense_seq == 1


def test_call_after_skips_when_b_never_occurs():
    verdict = call_after("read", "edit")(make_trace(call("read", "ah-1")))
    assert verdict.passed is None
    assert "never called" in verdict.detail


def test_call_after_with_the_same_tool_fails_at_its_first_call():
    verdict = call_after("ls", "ls")(make_trace(call("ls", "ah-1"), call("ls", "ah-2")))
    assert verdict.passed is False
    assert verdict.first_offense_seq == 0


# --- state_invariant -------------------------------------------------------


def test_state_invariant_reads_the_folded_state_and_passes():
    trace = make_trace(("human_input", "user", {"input": "go"}), patch({"/count": 3}))
    verdict = state_invariant("/count", lambda value: value == 3)(trace)
    assert verdict.passed is True
    assert verdict.first_offense_seq is None


def test_state_invariant_fails_and_points_at_the_patch_that_wrote_it():
    trace = make_trace(("human_input", "user", {"input": "go"}), patch({"/count": 3}))
    verdict = state_invariant("/count", lambda value: value > 5)(trace)
    assert verdict.passed is False
    assert verdict.first_offense_seq == 1
    assert "violates the predicate" in verdict.detail


def test_state_invariant_skips_a_missing_path():
    trace = make_trace(patch({"/other": 1}))
    verdict = state_invariant("/count", lambda value: True)(trace)
    assert verdict.passed is None
    assert "no value at /count" in verdict.detail


def test_state_invariant_turns_a_raising_predicate_into_a_failure():
    trace = make_trace(patch({"/count": 3}))

    def boom(value):
        raise ValueError("count must be positive")

    verdict = state_invariant("/count", boom)(trace)
    assert verdict.passed is False
    assert "ValueError" in verdict.detail
    assert "count must be positive" in verdict.detail


def test_state_invariant_supports_pointer_escapes_and_list_indexes():
    trace = make_trace(patch({"/a~1b": 7, "/items": ["x", "y"]}))
    assert state_invariant("/a~1b", lambda value: value == 7)(trace).passed is True
    assert state_invariant("/items/1", lambda value: value == "y")(trace).passed is True
    assert state_invariant("/items/9", lambda value: True)(trace).passed is None


def test_state_invariant_prefers_the_terminal_state_and_uses_its_seq():
    trace = make_trace(patch({"/count": 3}), done({"count": 9}))
    verdict = state_invariant("/count", lambda value: value == 3)(trace)
    assert verdict.passed is False
    assert verdict.first_offense_seq == 1  # the terminal event wins wholesale


def test_state_invariant_rejects_a_path_that_is_not_a_pointer():
    with pytest.raises(ValueError, match="JSON Pointer"):
        state_invariant("count", lambda value: True)


# --- output_schema ---------------------------------------------------------


def test_output_schema_accepts_a_matching_state():
    schema = {
        "type": "object",
        "required": ["status", "count"],
        "properties": {
            "status": {"type": "string", "enum": ["done", "failed"]},
            "count": {"type": "integer", "minimum": 0},
        },
    }
    verdict = output_schema(schema)(make_trace(done({"status": "done", "count": 2})))
    assert verdict.passed is True
    assert verdict.name == "output_schema(object)"


def test_output_schema_fails_on_an_enum_violation_and_reports_the_terminal_seq():
    schema = {"type": "object", "properties": {"status": {"type": "string", "enum": ["done", "failed"]}}}
    verdict = output_schema(schema)(make_trace(patch({"/x": 1}), done({"status": "bogus"})))
    assert verdict.passed is False
    assert verdict.first_offense_seq == 1
    assert "not one of" in verdict.detail


def test_output_schema_fails_on_a_missing_required_key_and_a_wrong_type():
    required = output_schema({"type": "object", "required": ["status"]})(make_trace(done({})))
    assert required.passed is False
    assert "required key 'status' is missing" in required.detail
    typed = output_schema({"type": "object", "properties": {"count": {"type": "integer"}}})(
        make_trace(done({"count": "2"}))
    )
    assert typed.passed is False
    assert "expected type integer, got string" in typed.detail


def test_output_schema_walks_items_and_patterns():
    nested = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "integer"}}},
    }
    bad_item = output_schema(nested)(make_trace(done({"items": [1, "x"]})))
    assert bad_item.passed is False
    assert "/items/1" in bad_item.detail
    patterned = output_schema({"type": "object", "properties": {"id": {"pattern": "^[a-z]+$"}}})(make_trace(done({"id": "AB"})))
    assert patterned.passed is False
    assert "does not match pattern" in patterned.detail


def test_output_schema_does_not_confuse_true_with_one():
    schema = {"type": "object", "properties": {"flag": {"enum": [True]}}}
    verdict = output_schema(schema)(make_trace(done({"flag": 1})))
    assert verdict.passed is False
    assert output_schema(schema)(make_trace(done({"flag": True}))).passed is True


def test_output_schema_rejects_an_unsupported_keyword_at_construction_time():
    with pytest.raises(ValueError, match="unsupported keyword 'additionalProperties'"):
        output_schema({"type": "object", "additionalProperties": False})
    with pytest.raises(ValueError, match="unsupported keyword 'oneOf'"):
        output_schema({"properties": {"a": {"oneOf": []}}})
    with pytest.raises(ValueError, match="invalid pattern"):
        output_schema({"pattern": "([a-z"})


def test_output_schema_skips_a_tape_with_no_output_state():
    verdict = output_schema({"type": "object"})(make_trace(("human_input", "user", {"input": "go"})))
    assert verdict.passed is None
    assert verdict.first_offense_seq is None
    assert "no terminal and no state_patch" in verdict.detail


# --- terminates_within -----------------------------------------------------


def test_terminates_within_fails_on_a_late_terminal():
    trace = make_trace(call("a"), call("b", "ah-2"), done())
    verdict = terminates_within(2)(trace)
    assert verdict.passed is False
    assert verdict.first_offense_seq == 2


def test_terminates_within_passes_before_the_bound():
    trace = make_trace(call("a"), done())
    verdict = terminates_within(5)(trace)
    assert verdict.passed is True
    assert verdict.first_offense_seq is None


def test_terminates_within_fails_on_an_overrun_without_a_terminal():
    trace = make_trace(*[call(f"t{i}", f"ah-{i}") for i in range(6)])
    verdict = terminates_within(5)(trace)
    assert verdict.passed is False
    assert verdict.first_offense_seq == 4
    assert "overran its bound" in verdict.detail


def test_terminates_within_skips_a_short_unterminated_tape():
    verdict = terminates_within(5)(make_trace(call("a")))
    assert verdict.passed is None
    assert verdict.first_offense_seq is None
    assert "not proven" in verdict.detail


def test_terminates_within_rejects_a_non_positive_bound():
    with pytest.raises(ValueError, match="integer >= 1"):
        terminates_within(0)


# --- load_yaml -------------------------------------------------------------


def test_load_yaml_builds_groups_with_their_modes(tape_dir):
    path = Path(tape_dir) / "asserts.yaml"
    path.write_text(
        "\n".join(
            [
                "- spec: max_repeats",
                "  args: {tool: shell, n: 2}",
                "- spec: never_calls",
                "  args: {tool: fs, args_contains: /etc}",
                "  warn: true",
                "- spec: terminates_within",
                "  args: {n: 9}",
                "  mode: any",
                "",
            ]
        ),
        encoding="utf-8",
    )
    groups = load_yaml(path)
    assert [group.mode for group in groups] == ["all", "warn", "any"]
    assert [group.name for group in groups] == [
        "max_repeats(shell, n=2)",
        "never_calls(fs, args_contains='/etc')",
        "terminates_within(n=9)",
    ]
    rows = groups[1].evaluate(make_trace(call("fs", "ah-1", "/etc/passwd")))
    assert rows[0].passed is False
    assert rows[0].detail.startswith("[warn] ")


def test_load_yaml_rejects_an_unknown_spec_key_or_entry_key(tape_dir):
    unknown_spec = Path(tape_dir) / "spec.yaml"
    unknown_spec.write_text("- spec: teleport\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown assertion 'teleport'"):
        load_yaml(unknown_spec)

    unknown_key = Path(tape_dir) / "key.yaml"
    unknown_key.write_text("- spec: no_deadlock\n  arg: {limit: 2}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key 'arg'"):
        load_yaml(unknown_key)


def test_load_yaml_reports_a_bad_argument_and_a_conflicting_mode(tape_dir):
    bad_args = Path(tape_dir) / "args.yaml"
    bad_args.write_text("- spec: max_repeats\n  args: {tool: shell, limit: 2}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_repeats rejected args"):
        load_yaml(bad_args)

    both = Path(tape_dir) / "both.yaml"
    both.write_text("- spec: no_deadlock\n  warn: true\n  mode: any\n", encoding="utf-8")
    with pytest.raises(ValueError, match="both 'mode' and 'warn'"):
        load_yaml(both)


def test_load_yaml_without_pyyaml_names_the_extra(tape_dir, monkeypatch):
    path = Path(tape_dir) / "plain.yaml"
    path.write_text("- spec: no_deadlock\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(RuntimeError, match=r"pip install agenttwin\[yaml\]"):
        load_yaml(path)


def test_load_yaml_accepts_an_empty_file(tape_dir):
    path = Path(tape_dir) / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_yaml(path) == []


# --- import discipline and report.py compatibility ------------------------


def test_yaml_is_not_imported_at_module_import_time():
    module_source = Path(asserts.__file__).read_text(encoding="utf-8")
    assert "\nimport yaml" not in module_source
    test_source = Path(__file__).read_text(encoding="utf-8")
    assert "\nimport yaml" not in test_source
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(asserts.__file__).resolve().parent.parent)
    probe = subprocess.run(
        [sys.executable, "-c", "import sys, agenttwin.asserts; print('yaml' in sys.modules)"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "False"


def test_report_helpers_read_both_group_modes():
    try:
        from agenttwin import report
    except ImportError:  # report.py imports the sibling diff.py, which lands separately
        report = None
    trace = death_loop()
    hard = assert_(max_repeats("shell", 3))
    soft = warn(max_repeats("shell", 3))
    for group in (hard, soft):
        results = group.evaluate(trace)
        rows = [report._result_dict(item) for item in results] if report is not None else [item.to_dict() for item in results]
        assert set(rows[0]) == ROW_KEYS
        assert rows[0]["status"] == "FAIL"
    hard_row = (report._result_dict if report is not None else AssertionResult.to_dict)(hard.evaluate(trace)[0])
    soft_row = (report._result_dict if report is not None else AssertionResult.to_dict)(soft.evaluate(trace)[0])
    assert hard_row["detail"] != soft_row["detail"]
    assert soft_row["detail"].startswith("[warn] ")
    assert hard.mode != soft.mode
    if report is not None:
        assert len(report._evaluate(hard, trace)) == 1
        assert len(report._evaluate(hard.specs[0], trace)) == 1


# --- the acceptance scenario and the failure table ------------------------


def test_death_loop_versus_repaired_trace():
    before = [max_repeats("shell", 3), never_calls("shell", args_contains="rm -rf")]
    after = [max_repeats("shell", 3), never_calls("shell", args_contains="rm -rf")]
    culprits = death_loop(6)
    repaired = make_trace(("human_input", "user", {"input": "tidy the tree"}), done({"status": "clean"}))
    assert not any(event.kind == "tool_call" for event in repaired.events)

    rows = []
    for label, trace, specs in (("death-loop", culprits, before), ("repaired", repaired, after)):
        for spec in specs:
            verdict = spec(trace)
            rows.append(verdict.to_dict())
            print(f"{label:>10}: {verdict.name:38s} {verdict.to_dict()['status']:4s} seq={verdict.first_offense_seq}")

    assert rows[0]["name"] == "max_repeats(shell, n=3)"
    assert rows[0]["status"] == "FAIL"
    assert rows[0]["first_offense_seq"] == 4
    assert rows[1]["status"] == "PASS"
    assert rows[2]["status"] == "PASS"
    assert rows[2]["first_offense_seq"] is None
    assert rows[3]["status"] == "PASS"


def test_every_factory_can_fail_and_names_its_first_offence():
    table = [
        ("max_repeats", max_repeats("shell", 3), death_loop(6), 4),
        ("no_deadlock", no_deadlock(3), make_trace(*[call("shell", "ah-same") for _ in range(4)]), 3),
        ("budget_within", budget_within(max_calls=2), make_trace(call("a"), call("b", "ah-2"), call("c", "ah-3")), 2),
        (
            "never_calls",
            never_calls("shell", args_contains="rm -rf"),
            make_trace(("human_input", "user", {"input": "go"}), call("shell", "ah-1", "rm -rf /")),
            1,
        ),
        ("call_order", call_order(("read", "edit")), make_trace(call("edit", "ah-e"), call("read", "ah-r")), 0),
        ("call_after", call_after("read", "edit"), make_trace(call("edit", "ah-e")), 0),
        ("state_invariant", state_invariant("/count", lambda value: value > 5), make_trace(patch({"/count": 3})), 0),
        (
            "output_schema",
            output_schema({"type": "object", "required": ["status"]}),
            make_trace(patch({"/x": 1}), done({"x": 1})),
            1,
        ),
        ("terminates_within", terminates_within(2), make_trace(call("a"), call("b", "ah-2"), done()), 2),
    ]
    for label, spec, trace, expected in table:
        verdict = spec(trace)
        assert verdict.passed is False, label
        assert verdict.first_offense_seq == expected, (label, verdict.detail)
        print(f"{label:16s} FAIL seq={verdict.first_offense_seq} :: {verdict.detail}")
    assert len(table) == 9
    assert MODES == ("all", "any", "warn")
