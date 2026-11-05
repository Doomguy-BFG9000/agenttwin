"""Tests for agenttwin.fingerprint (CONTRACTS §1.5 env_fingerprint)."""

from __future__ import annotations

import copy
import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenttwin import fingerprint  # noqa: E402

FINGERPRINT_KEYS = {"model", "prompt_hash", "python", "agenttwin", "platform", "deps"}


def test_fatal_fields_match_contract():
    assert fingerprint.FATAL_FIELDS == ("model", "prompt_hash", "python", "agenttwin")


def test_capture_returns_exactly_the_contract_keys():
    fp = fingerprint.capture(model="fake-1", prompt_hash="abc")
    assert set(fp) == FINGERPRINT_KEYS
    assert fp["model"] == "fake-1"
    assert fp["prompt_hash"] == "abc"


def test_capture_python_is_major_minor():
    fp = fingerprint.capture()
    assert re.fullmatch(r"\d+\.\d+", fp["python"]) is not None
    assert fp["python"] == f"{sys.version_info.major}.{sys.version_info.minor}"


def test_capture_platform_and_agenttwin_version():
    fp = fingerprint.capture()
    assert fp["platform"] == sys.platform
    assert isinstance(fingerprint.AGENTTWIN_VERSION, str)
    assert fingerprint.AGENTTWIN_VERSION
    assert fp["agenttwin"] == fingerprint.AGENTTWIN_VERSION


def test_capture_defaults_model_and_prompt_hash_to_none():
    fp = fingerprint.capture()
    assert fp["model"] is None
    assert fp["prompt_hash"] is None
    assert fp["deps"] == {}
    assert isinstance(fp["deps"], dict)


def test_capture_deps_are_stringified_and_copied():
    source = types.MappingProxyType({"pyyaml": "6.0.1", "port": 8080})
    fp = fingerprint.capture(deps=source)
    assert fp["deps"] == {"pyyaml": "6.0.1", "port": "8080"}
    fp["deps"]["extra"] = "1"
    assert dict(source) == {"pyyaml": "6.0.1", "port": 8080}


def test_compare_identical_fingerprints_is_empty():
    fp = fingerprint.capture(model="m", prompt_hash="h", deps={"x": "1"})
    assert fingerprint.compare(fp, dict(fp)) == []


def test_compare_python_ignores_patch_level():
    assert fingerprint.compare({"python": "3.12.7"}, {"python": "3.12.1"}) == []


def test_compare_python_major_minor_difference():
    assert fingerprint.compare({"python": "3.12"}, {"python": "3.11"}) == [
        {"field": "python", "expected": "3.12", "actual": "3.11"}
    ]


def test_compare_agenttwin_ignores_patch_level():
    assert fingerprint.compare({"agenttwin": "0.1.0"}, {"agenttwin": "0.1.9"}) == []
    assert fingerprint.compare({"agenttwin": "0.1.0"}, {"agenttwin": "0.2.0"}) == [
        {"field": "agenttwin", "expected": "0.1", "actual": "0.2"}
    ]


def test_compare_model_difference():
    assert fingerprint.compare({"model": "a"}, {"model": "b"}) == [
        {"field": "model", "expected": "a", "actual": "b"}
    ]


def test_compare_prompt_hash_difference():
    assert fingerprint.compare({"prompt_hash": "h1"}, {"prompt_hash": "h2"}) == [
        {"field": "prompt_hash", "expected": "h1", "actual": "h2"}
    ]


def test_compare_reports_each_field_once_in_stable_order():
    diff = fingerprint.compare(
        {"model": "a", "python": "3.12", "deps": {"x": "1"}},
        {"model": "b", "python": "3.11", "deps": {"x": "2"}},
    )
    assert [item["field"] for item in diff] == ["model", "python", "deps"]
    assert all(set(item) == {"field", "expected", "actual"} for item in diff)


def test_compare_skips_deps_and_platform_when_one_side_lacks_them():
    full = {"model": "m", "deps": {"x": "1"}, "platform": "win32"}
    assert fingerprint.compare({"model": "m"}, full) == []
    assert fingerprint.compare(full, {"model": "m"}) == []
    assert fingerprint.compare({"deps": None}, {"deps": {"x": "1"}}) == []


def test_compare_platform_when_both_sides_carry_it():
    assert fingerprint.compare({"platform": "win32"}, {"platform": "linux"}) == [
        {"field": "platform", "expected": "win32", "actual": "linux"}
    ]


def test_compare_deps_when_both_sides_carry_them():
    assert fingerprint.compare({"deps": {"x": "1"}}, {"deps": {"x": "2"}}) == [
        {"field": "deps", "expected": {"x": "1"}, "actual": {"x": "2"}}
    ]


def test_compare_does_not_mutate_its_inputs():
    expected = {"python": "3.12.7", "deps": {"x": "1"}}
    actual = {"python": "3.11.9", "deps": {"x": "2"}}
    before = copy.deepcopy((expected, actual))
    fingerprint.compare(expected, actual)
    assert (expected, actual) == before


def test_is_fatal_false_when_only_deps_and_platform_differ():
    expected = fingerprint.capture(model="fake-1", prompt_hash="abc", deps={"x": "1"})
    actual = fingerprint.capture(model="fake-1", prompt_hash="abc", deps={"x": "2"})
    actual["platform"] = "some-other-os"
    diff = fingerprint.compare(expected, actual)
    assert [item["field"] for item in diff] == ["platform", "deps"]
    assert fingerprint.is_fatal(diff) is False


def test_is_fatal_true_when_model_differs():
    expected = fingerprint.capture(model="fake-1", prompt_hash="abc")
    actual = fingerprint.capture(model="fake-2", prompt_hash="abc")
    diff = fingerprint.compare(expected, actual)
    assert [item["field"] for item in diff] == ["model"]
    assert fingerprint.is_fatal(diff) is True


@pytest.mark.parametrize("field", ["model", "prompt_hash", "python", "agenttwin"])
def test_is_fatal_for_every_fatal_field(field):
    assert fingerprint.is_fatal([{"field": field, "expected": "a", "actual": "b"}]) is True


def test_is_fatal_empty_or_non_fatal_diff():
    assert fingerprint.is_fatal([]) is False
    assert (
        fingerprint.is_fatal([{"field": "deps", "expected": {}, "actual": {"x": "1"}}])
        is False
    )
