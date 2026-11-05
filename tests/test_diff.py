"""Contract tests for TraceDiff — CONTRACTS §7 (seq alignment, state diff, matrix).

§7 fixes the row shape the HTML replayer already consumes, so the tests pin the
four change labels and the key names rather than merely checking that a diff
exists. Every test is named for the clause it guards.

The assertion fixtures stand in for asserts.AssertionResult/Spec/Group (§5)
instead of importing them: asserts.py is being written concurrently, and diff
only ever reads `name` and `passed` from a result.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import pytest

from agenttwin.diff import TraceDiff
from agenttwin.trace import SCHEMA_VERSION, Event, Header, Trace

CHANGE_LABELS = {"same", "added", "removed", "changed"}
TERMINAL = ("terminal", "planner", {"state": {}, "state_hash": "z"})


def make_header(trace_id: str = "checkout") -> Header:
    return Header(
        schema_version=SCHEMA_VERSION,
        trace_id=trace_id,
        seed=0,
        env_fingerprint={
            "model": "gpt-4o-mini",
            "prompt_hash": None,
            "python": "3.12",
            "agenttwin": "0.1.0",
            "platform": "win32",
            "deps": {},
        },
        redaction={"redacted": 0, "policy": {}, "mode": "scrub"},
    )


def make_custom(*events) -> Trace:
    """A tape built from (kind, actor, payload) triples, so a test varies exactly one field."""
    trace = Trace(header=make_header())
    for kind, actor, payload in events:
        trace.append(kind, actor, payload)
    return trace


def make_trace(count: int = 5, *, response: str = "hello") -> Trace:
    """A tape of `count` events ending in a terminal; payloads vary by position."""
    trace = Trace(header=make_header())
    for seq in range(max(count - 1, 0)):
        trace.append(*_body_event(seq, response))
    if count:
        trace.append("terminal", "planner", {"state": {"done": True}, "state_hash": "z"})
    return trace


def _body_event(seq: int, response: str) -> tuple:
    phase = seq % 4
    if phase == 0:
        return "llm_call", "planner", {"prompt_hash": f"p{seq}", "response": response}
    if phase == 1:
        return "tool_call", "shell", {"tool": "shell", "args_hash": f"a{seq}"}
    if phase == 2:
        return "tool_result", "shell", {"tool": "shell", "result": "ok"}
    return "state_patch", "planner", {"patch": {f"/step{seq}": seq}}


def make_state_trace(state: dict) -> Trace:
    """One event whose terminal payload IS the state: Trace.state() lets it win (§1.4)."""
    trace = Trace(header=make_header())
    trace.append("terminal", "planner", {"state": state, "state_hash": "z"})
    return trace


@dataclass(frozen=True)
class Result:
    """CONTRACTS §5 AssertionResult: diff reads name and passed and nothing else."""

    name: str
    passed: bool | None
    detail: str = ""
    first_offense_seq: int | None = None


class Spec:
    """CONTRACTS §5 AssertionSpec: __call__(trace) -> one result."""

    def __init__(self, name: str, passed: bool | None, detail: str = "") -> None:
        self.name = name
        self.passed = passed
        self.detail = detail
        self.seen: list[Trace] = []

    def __call__(self, trace: Trace) -> Result:
        self.seen.append(trace)
        return Result(self.name, self.passed, self.detail)


class Group:
    """CONTRACTS §5 AssertionGroup: evaluate(trace) -> a list of results."""

    def __init__(self, produce) -> None:
        self.produce = produce

    def evaluate(self, trace: Trace) -> list[Result]:
        return self.produce(trace)


def five_events(trace: Trace) -> Result:
    return Result("terminates_within", len(trace.events) == 5)


# --- alignment --------------------------------------------------------------


def test_identical_tapes_are_all_same_with_no_divergence():
    baseline = make_trace()
    diff = TraceDiff(baseline, baseline.copy())
    assert [row["change"] for row in diff.events] == ["same"] * 5
    assert diff.first_divergence_seq is None
    assert diff.state_diff == []
    assert diff.assertion_matrix == {}


def test_events_are_seq_aligned_and_carry_both_full_event_dicts():
    baseline = make_trace()
    diff = TraceDiff(baseline, baseline.copy())
    assert [row["seq"] for row in diff.events] == [0, 1, 2, 3, 4]
    for row in diff.events:
        assert set(row) == {"seq", "change", "baseline", "current"}
        assert row["baseline"] == baseline.events[row["seq"]].to_json()
        assert row["current"] == baseline.events[row["seq"]].to_json()


def test_payload_difference_marks_the_seq_changed_and_diverges_there():
    baseline = make_trace(3, response="hello")
    current = make_trace(3, response="patched")
    diff = TraceDiff(baseline, current)
    assert diff.events[0]["change"] == "changed"
    assert diff.events[1]["change"] == "same"
    assert diff.first_divergence_seq == 0


def test_kind_difference_marks_the_seq_changed():
    # decision allows extra keys (§1.4) and §9.9 accepts them, so both payloads
    # are identical and nothing but kind differs.
    baseline = make_custom(("decision", "planner", {"note": "x", "input": "x"}), TERMINAL)
    current = make_custom(("human_input", "planner", {"note": "x", "input": "x"}), TERMINAL)
    diff = TraceDiff(baseline, current)
    assert diff.events[0]["change"] == "changed"
    assert diff.first_divergence_seq == 0


def test_actor_difference_marks_the_seq_changed():
    baseline = make_custom(("decision", "planner", {"note": "x"}), TERMINAL)
    current = make_custom(("decision", "auditor", {"note": "x"}), TERMINAL)
    diff = TraceDiff(baseline, current)
    assert diff.events[0]["change"] == "changed"
    assert diff.events[1]["change"] == "same"


def test_extra_current_event_is_added_with_no_baseline_side():
    baseline = make_custom(("decision", "planner", {"note": "a"}), ("decision", "planner", {"note": "b"}))
    current = make_custom(
        ("decision", "planner", {"note": "a"}),
        ("decision", "planner", {"note": "b"}),
        ("decision", "planner", {"note": "c"}),
    )
    diff = TraceDiff(baseline, current)
    assert [row["change"] for row in diff.events] == ["same", "same", "added"]
    assert diff.events[2]["baseline"] is None
    assert diff.events[2]["current"]["payload"]["note"] == "c"
    assert diff.first_divergence_seq == 2


def test_extra_baseline_event_is_removed_with_no_current_side():
    baseline = make_custom(
        ("decision", "planner", {"note": "a"}),
        ("decision", "planner", {"note": "b"}),
        ("decision", "planner", {"note": "c"}),
    )
    current = make_custom(("decision", "planner", {"note": "a"}), ("decision", "planner", {"note": "b"}))
    diff = TraceDiff(baseline, current)
    assert [row["change"] for row in diff.events] == ["same", "same", "removed"]
    assert diff.events[2]["current"] is None
    assert diff.events[2]["baseline"]["payload"]["note"] == "c"
    assert diff.first_divergence_seq == 2


def test_first_divergence_seq_is_the_lowest_non_same_seq():
    baseline = make_trace(5, response="hello")
    current = make_trace(5, response="hello")
    current.events[3] = Event(seq=3, t=0.3, kind="decision", actor="planner", payload={"note": "instead"})
    diff = TraceDiff(baseline, current)
    assert diff.first_divergence_seq == 3
    assert [row["change"] for row in diff.events] == ["same", "same", "same", "changed", "same"]


def test_thirteen_event_tape_against_a_two_event_replay_diverges_at_zero():
    baseline = make_trace(13, response="hello")
    current = Trace(header=make_header())
    current.append("llm_call", "planner", {"prompt_hash": "p0", "response": "patched"})
    second = baseline.events[1]
    current.append(second.kind, second.actor, copy.deepcopy(second.payload))

    diff = TraceDiff(baseline, current)

    assert diff.first_divergence_seq == 0
    assert [row["seq"] for row in diff.events] == list(range(13))
    assert [row["change"] for row in diff.events] == ["changed", "same"] + ["removed"] * 11
    assert {row["change"] for row in diff.events} <= CHANGE_LABELS
    assert diff.events[0]["current"]["payload"]["response"] == "patched"
    assert diff.events[0]["baseline"]["payload"]["response"] == "hello"
    assert diff.events[2]["current"] is None


def test_patched_marker_is_carried_in_the_row_but_is_not_a_change():
    """§7 compares kind/actor/payload only; §9.10 writes patched solely when true."""
    baseline = make_custom(("decision", "planner", {"note": "x"}), TERMINAL)
    current = make_custom(("decision", "planner", {"note": "x"}), TERMINAL)
    current.events[0] = Event(
        seq=0, t=0.0, kind="decision", actor="planner", payload={"note": "x"}, patched=True
    )
    diff = TraceDiff(baseline, current)
    assert diff.events[0]["change"] == "same"
    assert diff.first_divergence_seq is None
    assert diff.events[0]["current"]["patched"] is True
    assert diff.events[0]["baseline"].get("patched") is None


def test_event_rows_never_alias_the_tape_payloads():
    baseline = make_trace(3, response="hello")
    current = make_trace(3, response="patched")
    diff = TraceDiff(baseline, current)

    assert diff.events[0]["baseline"]["payload"] is not baseline.events[0].payload
    diff.events[0]["baseline"]["payload"]["response"] = "tampered"
    assert baseline.events[0].payload["response"] == "hello"
    assert TraceDiff(baseline, current).events[0]["baseline"]["payload"]["response"] == "hello"


def test_each_row_owns_its_own_copy_of_the_same_event():
    baseline = make_trace(2)
    diff = TraceDiff(baseline, baseline.copy())
    assert diff.events[0]["baseline"] is not diff.events[0]["current"]
    diff.events[0]["baseline"]["payload"]["response"] = "tampered"
    assert diff.events[0]["current"]["payload"]["response"] == "hello"


def test_two_empty_tapes_are_identical():
    diff = TraceDiff(Trace(header=make_header()), Trace(header=make_header()))
    assert diff.events == []
    assert diff.first_divergence_seq is None
    assert diff.state_diff == []


# --- state diff -------------------------------------------------------------


def test_state_diff_reports_nested_additions_as_before_none():
    baseline = make_state_trace({"user": {"name": "bob"}})
    current = make_state_trace({"user": {"name": "bob", "age": 3}})
    assert TraceDiff(baseline, current).state_diff == [
        {"path": "/user/age", "before": None, "after": 3}
    ]


def test_state_diff_reports_nested_removals_as_after_none():
    baseline = make_state_trace({"user": {"name": "bob", "legacy": True}})
    current = make_state_trace({"user": {"name": "bob"}})
    assert TraceDiff(baseline, current).state_diff == [
        {"path": "/user/legacy", "before": True, "after": None}
    ]


def test_state_diff_reports_changed_leaf_and_whole_list_replacement():
    baseline = make_state_trace({"user": {"name": "bob"}, "roles": ["admin"]})
    current = make_state_trace({"user": {"name": "alice"}, "roles": ["admin", "dev"]})
    rows = TraceDiff(baseline, current).state_diff
    assert rows == [
        {"path": "/roles", "before": ["admin"], "after": ["admin", "dev"]},
        {"path": "/user/name", "before": "bob", "after": "alice"},
    ]


def test_state_diff_is_empty_when_the_folded_states_agree():
    baseline = make_state_trace({"user": {"name": "bob"}, "n": 1})
    current = make_state_trace({"user": {"name": "bob"}, "n": 1})
    diff = TraceDiff(baseline, current)
    assert diff.state_diff == []
    assert diff.first_divergence_seq is None


def test_state_diff_is_sorted_by_path_not_by_insertion_order():
    baseline = make_state_trace({"b": 1, "a": {"z": 1, "y": 2}})
    current = make_state_trace({"b": 2, "a": {"z": 3, "y": 4}})
    rows = TraceDiff(baseline, current).state_diff
    assert [row["path"] for row in rows] == ["/a/y", "/a/z", "/b"]


def test_state_diff_escapes_json_pointer_tokens():
    baseline = make_state_trace({"a/b": 1, "x~y": 2})
    current = make_state_trace({"a/b": 2, "x~y": 3})
    rows = TraceDiff(baseline, current).state_diff
    assert [row["path"] for row in rows] == ["/a~1b", "/x~0y"]


def test_state_diff_rows_are_copies_not_views_of_the_state():
    after = {"roles": ["admin"]}
    diff = TraceDiff(make_state_trace({"roles": []}), make_state_trace(after))
    diff.state_diff[0]["after"].append("dev")
    assert after == {"roles": ["admin"]}


# --- assertion matrix and summary -------------------------------------------


def test_assertion_matrix_is_empty_and_summary_is_zeroed_without_specs():
    diff = TraceDiff(make_trace(3), make_trace(3))
    assert diff.assertion_matrix == {}
    assert diff.summary == {"before": [0, 0, 0], "after": [0, 0, 0], "delta": 0}


def test_assertion_matrix_evaluates_each_spec_on_both_traces():
    baseline = make_trace(5)
    current = make_trace(3)
    diff = TraceDiff(baseline, current, specs=[five_events])
    assert diff.assertion_matrix == {"terminates_within": {"baseline": True, "current": False}}


def test_assertion_matrix_passes_the_real_traces_in_baseline_then_current_order():
    spec = Spec("max_repeats", True)
    baseline = make_trace(3)
    current = make_trace(3)
    TraceDiff(baseline, current, specs=[spec])
    assert spec.seen == [baseline, current]
    assert spec.seen[0] is baseline and spec.seen[1] is current


def test_assertion_matrix_takes_one_row_per_result_from_a_group():
    group = Group(lambda trace: [Result("call_order", True), Result("budget_within", None)])
    diff = TraceDiff(make_trace(2), make_trace(2), specs=[group])
    assert diff.assertion_matrix == {
        "call_order": {"baseline": True, "current": True},
        "budget_within": {"baseline": None, "current": None},
    }


def test_assertion_matrix_keeps_none_for_a_name_one_side_never_reports():
    group = Group(
        lambda trace: [Result("five_events", True)] if len(trace.events) == 5 else [Result("three_events", True)]
    )
    diff = TraceDiff(make_trace(5), make_trace(3), specs=[group])
    assert diff.assertion_matrix == {
        "five_events": {"baseline": True, "current": None},
        "three_events": {"baseline": None, "current": True},
    }


def test_summary_counts_passed_failed_and_skipped_per_side_with_a_delta():
    specs = [Spec("a", True), five_events, Spec("c", None)]
    diff = TraceDiff(make_trace(5), make_trace(3), specs=specs)
    assert diff.summary == {"before": [2, 0, 1], "after": [1, 1, 1], "delta": -1}


def test_a_spec_that_is_neither_callable_nor_evaluable_is_rejected():
    with pytest.raises(TypeError) as excinfo:
        TraceDiff(make_trace(2), make_trace(2), specs=[object()])
    assert "evaluate(trace)" in str(excinfo.value)


# --- to_dict and the constructor surface ------------------------------------


def test_trace_diff_takes_two_positional_traces_and_keyword_only_specs():
    with pytest.raises(TypeError):
        TraceDiff(make_trace(2), make_trace(2), [Spec("a", True)])


def test_to_dict_has_every_public_attribute_and_survives_a_json_round_trip():
    baseline = make_state_trace({"user": {"name": "bob", "roles": ["admin"]}, "count": 1})
    current = make_state_trace({"user": {"name": "bob", "roles": ["admin", "dev"], "age": 3}, "count": 1})
    diff = TraceDiff(baseline, current, specs=[Spec("max_repeats", True)])

    payload = diff.to_dict()

    assert set(payload) == {"first_divergence_seq", "events", "state_diff", "assertion_matrix", "summary"}
    assert payload["first_divergence_seq"] == 0
    assert payload["assertion_matrix"] == {"max_repeats": {"baseline": True, "current": True}}
    assert json.loads(json.dumps(payload)) == payload
    assert json.loads(json.dumps(diff.to_dict()))["state_diff"] == diff.state_diff


def test_to_dict_is_a_snapshot_not_a_live_view():
    diff = TraceDiff(make_trace(3, response="hello"), make_trace(3, response="patched"))
    payload = diff.to_dict()
    payload["events"][0]["baseline"]["payload"]["response"] = "tampered"
    assert diff.events[0]["baseline"]["payload"]["response"] == "hello"
