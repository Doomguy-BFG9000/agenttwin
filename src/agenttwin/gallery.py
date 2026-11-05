"""The shipped fixture gallery (design doc section 8): two real tapes, one controller.

Gallery is the first flagship artefact: the demo is not a screenshot but a replayed
recording.  Each fixture pairs a committed tape with the very controller that
produced it, so Twin.load(fixture.tape).replay(fixture.agent) runs the same
callable that ran during recording -- Record and Replay inject their own
llm/tools (CONTRACTS section 3) and the controller cannot tell them apart.  That
identity is what makes "same assertion set, FAIL before / PASS after" a fact about
the tapes instead of a staged narrative.

CONTRACTS (sections 1-9) has no clause for this surface: tape, replay, api, drift,
asserts, exit codes, diff and report are frozen there, the gallery is frozen here,
against the interface the CLI is written against -- Fixture / list_fixtures / get.

The two controllers differ only in the policy dict handed to
_checkout_controller: same body, same goal, same offline fakes.  deathloop ignores
the failed observation and reruns one shell command six times; repaired stops on
that first failed observation and never reaches the shell tool at all.

The fake llm and fake tools below are what recorded the tapes (drive them with
tools/gen_fixtures.py); each is a pure function of its input, so regenerating a
tape reproduces every event line exactly -- only env_fingerprint in the header
still remembers the machine that ran the generator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import asserts

__all__ = [
    "FIXTURE_DIR",
    "Fixture",
    "deathloop_agent",
    "get",
    "list_fixtures",
    "repaired_agent",
]

# Absolute by construction: the tapes are read from next to this module, whatever
# the cwd is, so a replay never depends on where it was launched from.
FIXTURE_DIR: Path = Path(__file__).resolve().with_name("fixtures")

GOAL: str = "get the checkout green"

# The whole difference between the two fixtures.  Six is the runaway guard that
# makes the deathloop tape finite (and therefore replayable); it is not a stopping
# condition the controller ever reaches on purpose.
DEATHLOOP_POLICY: dict[str, Any] = {"max_attempts": 6, "stop_on_failure": False}
REPAIRED_POLICY: dict[str, Any] = {"max_attempts": 6, "stop_on_failure": True}


@dataclass(frozen=True)
class Fixture:
    """One gallery entry: a committed tape, its controller, and the claims on it.

    agent is the module-level controller that produced tape (CONTRACTS section 3:
    agent(llm, tools), policy accepted as a third argument).  asserts holds
    CONTRACTS section 5 AssertionSpec values, evaluated with Report.assert_.
    commands are copy-pasteable reproductions of this fixture's verdict; they use
    pytest rather than inventing CLI syntax for a cli.py this module does not own.
    """

    id: str
    title: str
    description: str
    tape: Path
    agent: Callable[..., Any]
    asserts: tuple
    commands: tuple[str, ...] = ()


# --- the offline fakes every tape was recorded with ------------------------


def _fake_checkout_llm(request: dict) -> dict:
    """A model stand-in that is a pure function of the request (same bytes in, same bytes out)."""
    return {
        "tool": "shell",
        "args": ["python", "-m", "pytest", "tests", "-q"],
        "note": f"attempt {request['attempt']}: rerun the suite",
    }


def _fake_repo_status() -> dict:
    """The observation both runs share: the checkout is dirty, so a green suite is not reachable."""
    return {"ok": False, "detail": "working tree has uncommitted changes"}


def _fake_shell(*args: str) -> dict:
    """A shell that never runs anything: no subprocess, no network, one canned failure."""
    return {"ok": False, "exit_code": 1, "detail": "3 failed, 12 passed"}


# --- the one controller, under two policies --------------------------------


def _checkout_controller(llm: Callable[[dict], Any], tools: Any, policy: dict) -> dict:
    """Observe once, then retry the planned command until the policy or the guard stops it."""
    max_attempts = int(policy["max_attempts"])
    stop_on_failure = bool(policy["stop_on_failure"])

    status = tools["repo_status"]()
    observation = {"tool": "repo_status", "ok": bool(status["ok"]), "detail": status["detail"]}
    shell_calls = 0
    for attempt in range(1, max_attempts + 1):
        plan = llm({"goal": GOAL, "attempt": attempt, "observation": observation})
        if stop_on_failure and not observation["ok"]:
            return {
                "status": "stopped",
                "reason": "first observation failed",
                "attempt": attempt,
                "shell_calls": shell_calls,
            }
        result = tools[plan["tool"]](*plan["args"])
        shell_calls += 1
        observation = {"tool": plan["tool"], "ok": bool(result["ok"]), "detail": result["detail"]}
    return {"status": "gave_up", "attempts": max_attempts, "shell_calls": shell_calls}


def deathloop_agent(llm: Any, tools: Any, policy: dict | None = None) -> dict:
    """The controller that ignores a failed observation and reruns the same command."""
    return _checkout_controller(llm, tools, dict(policy or DEATHLOOP_POLICY))


def repaired_agent(llm: Any, tools: Any, policy: dict | None = None) -> dict:
    """The same controller under the policy that stops after the first failed observation."""
    return _checkout_controller(llm, tools, dict(policy or REPAIRED_POLICY))


# --- the gallery -----------------------------------------------------------

# One shared tuple, on purpose: the two fixtures must be judged by the identical
# assertion set, not by two tuples that happen to hold equal values today.
_ASSERTIONS: tuple = (
    asserts.max_repeats("shell", 3),
    asserts.budget_within(max_calls=8),
    asserts.never_calls("shell", args_contains="rm -rf"),
)

_GALLERY: tuple[Fixture, ...] = (
    Fixture(
        id="deathloop",
        title="Checkout retry loop",
        description=(
            "Reads one failed observation, ignores it, and reruns python -m pytest six times "
            "with identical arguments: max_repeats and budget_within fail."
        ),
        tape=FIXTURE_DIR / "deathloop.jsonl",
        agent=deathloop_agent,
        asserts=_ASSERTIONS,
        commands=("python -m pytest tests/test_gallery.py -q -k deathloop",),
    ),
    Fixture(
        id="repaired",
        title="Checkout that stops on the failed observation",
        description=(
            "The same controller under a policy that stops after the first failed observation: "
            "zero shell calls, the same assertion set, every row passes."
        ),
        tape=FIXTURE_DIR / "repaired.jsonl",
        agent=repaired_agent,
        asserts=_ASSERTIONS,
        commands=("python -m pytest tests/test_gallery.py -q -k repaired",),
    ),
)


def list_fixtures() -> list[Fixture]:
    """Every shipped fixture, in gallery order (tape order is presentation order)."""
    return list(_GALLERY)


def get(id: str) -> Fixture:
    """The fixture named id; KeyError when the gallery holds no such id."""
    for fixture in _GALLERY:
        if fixture.id == id:
            return fixture
    known = ", ".join(fixture.id for fixture in _GALLERY)
    raise KeyError(f"unknown fixture id {id!r} (known: {known})")
