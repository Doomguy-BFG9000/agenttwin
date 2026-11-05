"""Contract tests for Report.json(), Report.markdown() and Report.exit_code.

CONTRACTS.md section 8 fixes the JSON key set and the result-row shape, section
9.18 fixes what divergent means, section 6 fixes the exit-code priority, and
section 5 fixes the AssertionResult/AssertionSpec/AssertionGroup shapes the tests
stand in for. Nothing here needs a real replay: Report is a pure view.
"""

from __future__ import annotations

import json

from agenttwin.asserts import AssertionResult
from agenttwin.report import Report
from agenttwin.trace import SCHEMA_VERSION, Header, Trace

JSON_KEYS = {
    "schema_version",
    "trace_id",
    "mode",
    "divergent",
    "exit_code",
    "results",
    "misses",
    "drifts",
    "diff",
    "header",
}


def make_header(trace_id: str = "checkout", redacted: int = 2) -> Header:
    return Header(
        schema_version=SCHEMA_VERSION,
        trace_id=trace_id,
        seed=0,
        env_fingerprint={
            "model": "gpt-4o-mini",
            "prompt_hash": "deadbeef",
            "python": "3.12",
            "agenttwin": "0.1.0",
            "platform": "win32",
            "deps": {"httpx": "0.27.0"},
        },
        redaction={"redacted": redacted, "policy": {}, "mode": "scrub"},
    )


def make_trace(trace_id: str = "checkout", *, response: str = "hello", patched: bool = False) -> Trace:
    trace = Trace(header=make_header(trace_id))
    trace.append("llm_call", "planner", {"prompt_hash": "p0", "response": response}, patched=patched)
    trace.append("tool_call", "shell", {"tool": "shell", "args_hash": "a1"})
    trace.append("tool_result", "shell", {"tool": "shell", "result": "ok"})
    trace.append("state_patch", "planner", {"patch": {"/step": "done"}})
    trace.append("terminal", "planner", {"state": {"step": "done"}, "state_hash": "z"})
    return trace


def make_report(*, response: str = "hello", patched: bool = False, current=None, **kwargs) -> Report:
    """An unpatched replay whose current tape is a copy of the baseline."""
    baseline = make_trace(response=response, patched=patched)
    return Report(
        baseline=baseline,
        trace=current if current is not None else baseline.copy(),
        **kwargs,
    )


class Spec:
    """Minimal AssertionSpec (CONTRACTS §5): __call__(trace) -> AssertionResult."""

    def __init__(self, name, passed, detail="detail", first_offense_seq=None):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.first_offense_seq = first_offense_seq
        self.seen = []

    def __call__(self, trace):
        self.seen.append(trace)
        return AssertionResult(
            name=self.name,
            passed=self.passed,
            detail=self.detail,
            first_offense_seq=self.first_offense_seq,
        )


class Group:
    """Minimal AssertionGroup (CONTRACTS §5): evaluate(trace) -> list[AssertionResult]."""

    def __init__(self, *specs):
        self.specs = specs

    def evaluate(self, trace):
        return [spec(trace) for spec in self.specs]


class Miss:
    """TapeMiss stand-in: Report only ever calls to_dict() (CONTRACTS §4)."""

    def __init__(self, seq=3, expected_kind="tool_call", actor="shell"):
        self.seq = seq
        self.expected_kind = expected_kind
        self.actor = actor

    def to_dict(self):
        return {
            "type": "TapeMiss",
            "seq": self.seq,
            "expected_kind": self.expected_kind,
            "actor": self.actor,
        }


class Drift:
    """DriftError stand-in, same reason (CONTRACTS §4)."""

    def __init__(self, type="PromptDrift", seq=1, message="prompt_hash mismatch"):
        self.type = type
        self.seq = seq
        self.message = message

    def to_dict(self):
        return {
            "type": self.type,
            "seq": self.seq,
            "expected": "aaa",
            "actual": "bbb",
            "message": self.message,
        }


# --- json() -----------------------------------------------------------------


def test_json_top_level_keys_are_exactly_the_frozen_set():
    payload = make_report().json()
    assert set(payload) == JSON_KEYS
    assert len(payload) == len(JSON_KEYS)


def test_json_round_trips_through_dumps_and_loads():
    report = make_report().assert_(Spec("max_repeats", True, detail="2 calls of shell"))
    payload = report.json()
    assert json.loads(json.dumps(payload)) == payload


def test_json_results_items_are_plain_dicts_with_the_frozen_four_keys():
    report = make_report().assert_(Spec("max_repeats", True, detail="2 calls"))
    item = report.json()["results"][0]
    assert item == {
        "name": "max_repeats",
        "status": "PASS",
        "detail": "2 calls",
        "first_offense_seq": None,
    }
    assert type(item) is dict


def test_json_failure_result_keeps_first_offense_seq():
    report = make_report().assert_(Spec("no_deadlock", False, detail="loop", first_offense_seq=2))
    item = report.json()["results"][0]
    assert item["status"] == "FAIL"
    assert item["first_offense_seq"] == 2


def test_json_skip_result_is_reported_as_skip():
    report = make_report().assert_(Spec("budget_within", None, detail="no token data"))
    assert report.json()["results"][0]["status"] == "SKIP"


def test_json_header_is_the_trace_header_and_schema_version():
    report = make_report()
    payload = report.json()
    assert payload["header"] == report.trace.header.to_json()
    assert payload["header"]["trace_id"] == "checkout"
    assert payload["trace_id"] == "checkout"
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["mode"] == "replay"


def test_json_diff_is_seq_aligned_and_lists_every_event():
    diff = make_report().json()["diff"]
    assert set(diff) == {"first_divergence_seq", "events", "state_diff", "assertion_matrix"}
    assert diff["first_divergence_seq"] is None
    assert [item["seq"] for item in diff["events"]] == [0, 1, 2, 3, 4]
    assert {item["change"] for item in diff["events"]} == {"same"}


def test_json_misses_and_drifts_delegate_to_to_dict():
    miss = Miss()
    drift = Drift()
    payload = make_report(misses=[miss], drifts=[drift]).json()
    assert payload["misses"] == [miss.to_dict()]
    assert payload["drifts"] == [drift.to_dict()]
    assert json.loads(json.dumps(payload)) == payload


# --- exit_code --------------------------------------------------------------


def test_exit_code_zero_when_all_hard_assertions_pass():
    report = make_report().assert_(Spec("max_repeats", True), Spec("call_order", True))
    assert report.exit_code == 0


def test_exit_code_one_when_a_hard_assertion_fails():
    report = make_report().assert_(Spec("max_repeats", True), Spec("no_deadlock", False))
    assert report.exit_code == 1


def test_exit_code_two_when_a_tape_miss_is_present():
    report = make_report(misses=[Miss()]).assert_(Spec("budget_within", False))
    assert report.exit_code == 2


def test_exit_code_two_when_a_drift_is_present():
    report = make_report(drifts=[Drift()])
    assert report.exit_code == 2


def test_exit_code_two_when_divergent_is_true():
    report = make_report(divergent=True)
    assert report.exit_code == 2


def test_divergence_is_not_fatal_in_a_counterfactual():
    # §9.36: the caller supplied the patch/policy that moved the trajectory, so the
    # assertions are the verdict; a faithful replay keeps the 2 (test above).
    report = make_report(divergent=True, mode="counterfactual")
    assert report.exit_code == 0
    text = report.markdown()
    assert "verdict: PASS (exit code 0)" in text
    assert "mode: counterfactual" in text
    assert "DIVERGENT" not in text
    assert "note: counterfactual replay" in text


def test_a_counterfactual_still_fails_on_a_failing_assertion():
    report = make_report(divergent=True, mode="counterfactual").assert_(Spec("budget", False))
    assert report.exit_code == 1
    assert "verdict: FAIL (exit code 1)" in report.markdown()


def test_divergent_is_false_for_a_patched_replay_that_differs_from_its_baseline():
    # §9.18: divergent tracks the R-TERM fork, not "the tapes differ", so a
    # patched replay (changed payload at seq 0) must still exit 0.
    current = make_trace(response="patched")
    report = make_report(current=current)
    assert report.diff.first_divergence_seq == 0
    assert report.divergent is False
    assert report.exit_code == 0


def test_divergent_defaults_to_false_and_is_json_serialisable():
    report = make_report()
    assert report.json()["divergent"] is False
    assert make_report(divergent=None).divergent is False


def test_strict_turns_skip_into_exit_code_one():
    report = make_report(strict=True).assert_(Spec("budget_within", None, detail="no data"))
    assert report.results[0].passed is None
    assert report.exit_code == 1


def test_skip_is_not_a_failure_without_strict():
    report = make_report().assert_(Spec("budget_within", None, detail="no data"))
    assert report.exit_code == 0


def test_warn_failure_never_changes_the_exit_code():
    report = make_report().warn(Spec("no_deadlock", False, detail="loop", first_offense_seq=4))
    assert report.results[0].passed is False
    assert report.json()["results"][0]["status"] == "FAIL"
    assert report.exit_code == 0


def test_warn_does_not_mask_a_hard_failure_next_to_it():
    report = make_report().warn(Spec("no_deadlock", False)).assert_(Spec("call_order", False))
    assert [result.name for result in report.results] == ["no_deadlock", "call_order"]
    assert report.exit_code == 1


def test_constructor_results_count_as_hard():
    skip = AssertionResult(name="budget_within", passed=None, detail="no data")
    report = Report(baseline=make_trace(), trace=make_trace(), results=[skip], strict=True)
    assert report.exit_code == 1


# --- assert_ / warn ---------------------------------------------------------


def test_assert_returns_self_extends_results_and_sees_the_replayed_trace():
    spec = Spec("max_repeats", True)
    report = make_report()
    assert report.assert_(spec) is report
    assert [result.name for result in report.results] == ["max_repeats"]
    assert spec.seen == [report.trace]
    assert spec.seen[0] is report.trace


def test_warn_returns_self_and_marks_only_its_own_results_soft():
    report = make_report()
    assert report.warn(Spec("soft_one", False)) is report
    report.assert_(Spec("hard_one", False))
    assert [result.name for result in report.results] == ["soft_one", "hard_one"]
    assert report.exit_code == 1


def test_assert_accepts_a_group_and_extends_one_result_per_member():
    report = make_report().assert_(Group(Spec("a", True), Spec("b", False)))
    assert [result.name for result in report.results] == ["a", "b"]
    assert report.exit_code == 1


def test_assert_with_no_specs_is_a_no_op():
    report = make_report()
    assert report.assert_() is report
    assert report.results == []
    assert report.exit_code == 0


# --- markdown() -------------------------------------------------------------


def test_markdown_reports_verdict_table_diff_fingerprint_and_redaction():
    report = make_report().assert_(Spec("max_repeats", True, detail="2 calls"))
    text = report.markdown()
    assert text.startswith("# AgentTwin report: checkout")
    assert "verdict: PASS (exit code 0)" in text
    assert "| name | status | detail | first_offense_seq |" in text
    assert "| max_repeats | PASS | 2 calls | - |" in text
    assert "1 passed, 0 failed, 0 skipped" in text
    assert "## Diff" in text
    assert "first divergence seq: none" in text
    assert "events: 5 total | same 5, changed 0, added 0, removed 0" in text
    assert "model: gpt-4o-mini" in text
    assert "python: 3.12" in text
    assert "deps: httpx=0.27.0" in text
    assert "this tape redacted 2 secrets" in text


def test_markdown_failure_row_and_verdict():
    report = make_report().assert_(Spec("no_deadlock", False, detail="loop", first_offense_seq=2))
    text = report.markdown()
    assert "verdict: FAIL (exit code 1)" in text
    assert "| no_deadlock | FAIL | loop | 2 |" in text


def test_markdown_escapes_pipes_and_newlines_in_detail():
    report = make_report().assert_(Spec("state_invariant", False, detail="a | b\nc", first_offense_seq=3))
    rows = [line for line in report.markdown().splitlines() if line.startswith("| state_invariant")]
    assert rows == ["| state_invariant | FAIL | a \\| b c | 3 |"]


def test_markdown_is_ascii_only():
    text = make_report().assert_(Spec("max_repeats", True)).warn(Spec("soft", False)).markdown()
    assert text.isascii()


def test_markdown_omits_the_failures_section_when_there_are_none():
    assert "## Failures" not in make_report().markdown()


def test_markdown_lists_misses_and_drifts_with_exit_code_two():
    report = make_report(misses=[Miss()], drifts=[Drift()])
    text = report.markdown()
    assert "verdict: DRIFT (exit code 2)" in text
    assert "## Failures" in text
    assert "- TapeMiss seq=3 expected_kind=tool_call actor=shell" in text
    assert "- PromptDrift seq=1: prompt_hash mismatch" in text


def test_markdown_marks_divergent_and_strict():
    text = make_report(divergent=True, strict=True).markdown()
    assert "divergent: yes" in text
    assert "strict: yes" in text
    assert "verdict: DIVERGENT (exit code 2)" in text

# --- §9.25: a warn group stays soft wherever it is evaluated ---------------


def test_a_warn_group_nested_inside_assert_never_moves_the_exit_code():
    from agenttwin.asserts import never_calls, warn

    report = make_report()
    assert report.exit_code == 0
    report.assert_(warn(never_calls("shell")))
    assert report.results[-1].passed is False
    assert report.results[-1].detail.startswith("[warn]")
    assert report.exit_code == 0


def test_the_same_spec_hard_wrapped_does_move_the_exit_code():
    from agenttwin.asserts import never_calls

    report = make_report().assert_(never_calls("shell"))
    assert report.results[-1].passed is False
    assert report.exit_code == 1

