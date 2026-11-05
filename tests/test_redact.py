"""Tests for agenttwin.redact (CONTRACTS §1.1 redaction policy, §1.4 payload keys)."""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenttwin.redact import (  # noqa: E402
    DEFAULT_DROP_KEYS,
    DEFAULT_PATTERNS,
    MANDATORY_PAYLOAD_KEYS,
    Redactor,
)

PATTERN_NAMES = [name for name, _ in DEFAULT_PATTERNS]
PATTERN_REGEX = dict(DEFAULT_PATTERNS)

# One positive and one negative sample per DEFAULT_PATTERNS entry.
POSITIVE_SAMPLES = {
    "openai_sk": "token=sk-abcdefghijklmnop1234",
    "bearer": "Authorization: Bearer abc123",
    "email": "contact jane.doe+tag@example.co.uk now",
    "long_hex": "digest 0123456789abcdef0123456789abcdef end",
    "win_user_path": "C:\\Users\\alice\\secret.txt",
    "posix_home_path": "/home/alice/.ssh/id_rsa",
    "aws_key": "AKIAIOSFODNN7EXAMPLE",
}

NEGATIVE_SAMPLES = {
    "openai_sk": "sk-short",
    "bearer": "Bearer",
    "email": "no-at-sign.example.com",
    "long_hex": "0123456789abcdef0123456789abcde",
    "win_user_path": "C:\\Windows\\System32",
    "posix_home_path": "/usr/local/bin/python",
    "aws_key": "akiaiosfodnn7example",
}


def test_mandatory_payload_keys_match_contract():
    assert set(MANDATORY_PAYLOAD_KEYS) == {
        "llm_call",
        "tool_call",
        "tool_result",
        "state_patch",
        "decision",
        "human_input",
        "error",
        "terminal",
    }
    assert MANDATORY_PAYLOAD_KEYS["llm_call"] == ("prompt_hash", "response")
    assert MANDATORY_PAYLOAD_KEYS["tool_call"] == ("tool", "args_hash")
    assert MANDATORY_PAYLOAD_KEYS["tool_result"] == ("tool", "result")
    assert MANDATORY_PAYLOAD_KEYS["state_patch"] == ("patch",)
    assert MANDATORY_PAYLOAD_KEYS["decision"] == ("note",)
    assert MANDATORY_PAYLOAD_KEYS["human_input"] == ("input",)
    assert MANDATORY_PAYLOAD_KEYS["error"] == ("type", "message")
    assert MANDATORY_PAYLOAD_KEYS["terminal"] == ("state", "state_hash")
    assert all(isinstance(keys, tuple) for keys in MANDATORY_PAYLOAD_KEYS.values())


def test_default_drop_keys_are_the_documented_set():
    assert isinstance(DEFAULT_DROP_KEYS, frozenset)
    assert DEFAULT_DROP_KEYS == frozenset(
        {
            "api_key",
            "apikey",
            "authorization",
            "password",
            "passwd",
            "secret",
            "token",
            "access_token",
            "refresh_token",
            "cookie",
            "set-cookie",
            "session_id",
            "private_key",
        }
    )


def test_default_patterns_declare_the_documented_names():
    assert isinstance(DEFAULT_PATTERNS, tuple)
    assert PATTERN_NAMES == [
        "openai_sk",
        "bearer",
        "email",
        "long_hex",
        "win_user_path",
        "posix_home_path",
        "aws_key",
    ]


def test_sample_tables_cover_every_pattern():
    assert set(POSITIVE_SAMPLES) == set(NEGATIVE_SAMPLES) == set(PATTERN_NAMES)


@pytest.mark.parametrize("name", PATTERN_NAMES)
def test_pattern_positive_sample_matches(name):
    assert re.search(PATTERN_REGEX[name], POSITIVE_SAMPLES[name]) is not None


@pytest.mark.parametrize("name", PATTERN_NAMES)
def test_pattern_negative_sample_does_not_match(name):
    assert re.search(PATTERN_REGEX[name], NEGATIVE_SAMPLES[name]) is None


def test_nested_authorization_is_dropped_and_secret_never_leaks():
    payload = {
        "tool": "http",
        "request": {
            "headers": {"Authorization": "Bearer abc123", "Accept": "application/json"}
        },
    }
    before = copy.deepcopy(payload)
    scrubbed, count = Redactor().payload("tool_call", payload)
    text = json.dumps(scrubbed, sort_keys=True)
    assert "abc123" not in text
    assert "Bearer" not in text
    assert scrubbed["request"]["headers"]["Authorization"] == "<redacted:Authorization>"
    assert scrubbed["request"]["headers"]["Accept"] == "application/json"
    assert count == 1
    assert payload == before


def test_email_and_windows_user_path_are_scrubbed():
    payload = {
        "prompt_hash": "abc",
        "response": {
            "contact": "jane@example.com",
            "path": "C:\\Users\\alice\\notes.txt",
        },
    }
    scrubbed, count = Redactor().payload("llm_call", payload)
    assert scrubbed == {
        "prompt_hash": "abc",
        "response": {
            "contact": "<redacted:email>",
            "path": "<redacted:win_user_path>",
        },
    }
    assert count == 2


def test_bearer_value_is_scrubbed_without_a_dropped_key():
    scrubbed, count = Redactor().payload(
        "tool_call", {"tool": "http", "args_preview": "Bearer abc123"}
    )
    assert scrubbed["args_preview"] == "<redacted:bearer>"
    assert count == 1


def test_pattern_match_replaces_the_whole_string_value():
    scrubbed, count = Redactor().payload(
        "human_input", {"input": "ping me at bob@example.com before noon"}
    )
    assert scrubbed["input"] == "<redacted:email>"
    assert count == 1


def test_redaction_count_matches_number_of_substitutions():
    payload = {
        "note": "all good",
        "token": "sk-abcdefghijklmnop1234",
        "nested": {"password": "hunter2", "keep": "plain"},
        "blob": ["hi@example.com", 7, {"secret": "shh"}],
    }
    scrubbed, count = Redactor().payload("decision", payload)
    assert scrubbed == {
        "note": "all good",
        "token": "<redacted:token>",
        "nested": {"password": "<redacted:password>", "keep": "plain"},
        "blob": ["<redacted:email>", 7, {"secret": "<redacted:secret>"}],
    }
    assert count == 4
    assert json.dumps(scrubbed).count("<redacted:") == count


def test_input_payload_is_not_mutated():
    payload = {"note": "n", "token": "t", "nested": {"list": ["a@b.co", {"secret": "s"}]}}
    before = copy.deepcopy(payload)
    scrubbed, _ = Redactor().payload("decision", payload)
    assert payload == before
    assert scrubbed is not payload
    assert scrubbed["nested"] is not payload["nested"]
    assert scrubbed["nested"]["list"] is not payload["nested"]["list"]


def test_drop_key_matching_is_case_insensitive():
    scrubbed, count = Redactor().payload(
        "decision", {"PASSWORD": "x", "Set-Cookie": "sid=1", "note": "n"}
    )
    assert scrubbed["PASSWORD"] == "<redacted:PASSWORD>"
    assert scrubbed["Set-Cookie"] == "<redacted:Set-Cookie>"
    assert count == 2


def test_lists_are_recursed():
    scrubbed, count = Redactor().payload(
        "tool_result", {"tool": "t", "result": [{"token": "a"}, ["b@c.io"]]}
    )
    assert scrubbed["result"] == [{"token": "<redacted:token>"}, ["<redacted:email>"]]
    assert count == 2


def test_non_string_scalars_pass_through():
    payload = {"note": "n", "count": 3, "flag": True, "nothing": None, "ratio": 0.5}
    scrubbed, count = Redactor().payload("decision", payload)
    assert scrubbed == payload
    assert scrubbed is not payload
    assert count == 0


def test_whitelist_mode_drops_keys_not_mandatory_or_whitelisted():
    redactor = Redactor(mode="whitelist", whitelist={"llm_call": ("model",)})
    payload = {
        "prompt_hash": "abc",
        "response": "hello",
        "model": "m",
        "traceback": "boom",
    }
    scrubbed, count = redactor.payload("llm_call", payload)
    assert set(scrubbed) == {"prompt_hash", "response", "model"}
    assert count == 1


def test_whitelist_mode_still_scrubs_kept_values():
    scrubbed, count = Redactor(mode="whitelist").payload(
        "error", {"type": "E", "message": "mail x@y.io", "debug": 1}
    )
    assert scrubbed == {"type": "E", "message": "<redacted:email>"}
    assert count == 2


def test_scrub_mode_keeps_extra_keys():
    redactor = Redactor(whitelist={"llm_call": ("model",)})
    scrubbed, count = redactor.payload(
        "llm_call", {"prompt_hash": "a", "response": "r", "extra": 1}
    )
    assert set(scrubbed) == {"prompt_hash", "response", "extra"}
    assert scrubbed["extra"] == 1
    assert count == 0


def test_dropped_key_wins_over_whitelist():
    redactor = Redactor(mode="whitelist", whitelist={"llm_call": ("authorization",)})
    scrubbed, count = redactor.payload(
        "llm_call", {"prompt_hash": "a", "response": "r", "authorization": "Bearer zzz"}
    )
    assert scrubbed["authorization"] == "<redacted:authorization>"
    assert "zzz" not in json.dumps(scrubbed)
    assert count == 1


def test_custom_drop_keys_replace_the_default_set():
    scrubbed, count = Redactor(drop_keys=["APIKEY"]).payload(
        "decision", {"apikey": "k", "password": "p", "note": "n"}
    )
    assert scrubbed["apikey"] == "<redacted:apikey>"
    assert scrubbed["password"] == "p"
    assert count == 1


def test_patterns_can_be_overridden():
    redactor = Redactor(patterns=[("ticket", r"T-\d{4}")])
    scrubbed, count = redactor.payload("decision", {"note": "see T-1234 please"})
    assert scrubbed["note"] == "<redacted:ticket>"
    assert count == 1
    assert redactor.policy()["patterns"] == ["ticket"]


def test_policy_reports_the_active_policy():
    redactor = Redactor(
        mode="whitelist",
        drop_keys=["Token", "api_key"],
        whitelist={"llm_call": ("model", "tokens")},
    )
    assert redactor.policy() == {
        "mode": "whitelist",
        "drop_keys": ["api_key", "token"],
        "patterns": PATTERN_NAMES,
        "whitelist": {"llm_call": ["model", "tokens"]},
    }
    assert Redactor().policy()["mode"] == "scrub"
    assert redactor.mode == "whitelist"
    assert Redactor().mode == "scrub"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        Redactor(mode="hash")


def test_payload_rejects_non_mapping():
    with pytest.raises(TypeError):
        Redactor().payload("decision", ["note"])
