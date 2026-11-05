"""Drift taxonomy (CONTRACTS section 4): every error carries its own JSON row."""

from __future__ import annotations

import json

import pytest

from agenttwin.drift import ArgsDrift, DriftError, FingerprintDrift, PromptDrift, TapeMiss


def test_prompt_drift_keeps_its_type_name_and_fields():
    error = PromptDrift("prompt_hash mismatch", seq=3, expected="a" * 64, actual="b" * 64)
    assert isinstance(error, DriftError)
    assert error.type == "PromptDrift"
    assert (error.seq, error.expected, error.actual) == (3, "a" * 64, "b" * 64)
    assert str(error) == "prompt_hash mismatch"


def test_args_drift_keeps_its_seq():
    error = ArgsDrift("args_hash mismatch", seq=7, expected="x", actual="y")
    assert error.type == "ArgsDrift"
    assert error.seq == 7


def test_fingerprint_drift_keeps_both_sides():
    error = FingerprintDrift("model changed", expected="gpt-4o", actual=None)
    assert error.type == "FingerprintDrift"
    assert (error.expected, error.actual) == ("gpt-4o", None)
    assert error.seq is None


def test_drift_to_dict_has_the_five_contract_keys():
    row = ArgsDrift("args_hash mismatch", seq=1, expected="x", actual="y").to_dict()
    assert set(row) == {"type", "seq", "expected", "actual", "message"}
    assert row == {
        "type": "ArgsDrift",
        "seq": 1,
        "expected": "x",
        "actual": "y",
        "message": "args_hash mismatch",
    }


def test_base_drift_error_defaults_to_its_own_type():
    error = DriftError("boom")
    row = error.to_dict()
    assert row["type"] == "DriftError"
    assert row["seq"] is None
    assert row["message"] == "boom"


def test_drift_to_dict_is_json_serialisable():
    row = PromptDrift("m", seq=2, expected="a", actual="b").to_dict()
    assert json.loads(json.dumps(row))["type"] == "PromptDrift"


def test_tape_miss_to_dict_has_the_four_contract_keys():
    miss = TapeMiss(4, "tool_call", "shell")
    assert set(miss.to_dict()) == {"type", "seq", "expected_kind", "actor"}
    assert miss.to_dict() == {
        "type": "TapeMiss",
        "seq": 4,
        "expected_kind": "tool_call",
        "actor": "shell",
    }


def test_tape_miss_is_an_exception_but_not_a_drift_error():
    miss = TapeMiss(2, "llm_call", "llm")
    assert isinstance(miss, Exception)
    assert not isinstance(miss, DriftError)
    assert "tape miss at seq 2" in str(miss)
    assert "llm_call" in str(miss)


def test_every_drift_type_is_catchable_as_a_drift_error():
    for error in (
        PromptDrift("m"),
        ArgsDrift("m"),
        FingerprintDrift("m"),
    ):
        with pytest.raises(DriftError):
            raise error
