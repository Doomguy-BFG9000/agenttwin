"""The shipped fixture gallery (design doc section 8): twelve real tapes, six stories.

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

Every pair below is one controller under two policies, so the only difference
between the before tape and the after tape is the policy dict:

* checkpoint -- deathloop ignores a failed observation and reruns the suite six
                times (max_repeats, budget_within); repaired stops on it.
* cleanup    -- cleanup deletes build/ with 'rm -rf' (never_calls);
                cleanup_guarded leaves the tree alone.
* docs       -- write_first writes the release note before looking anything up
                (call_order); search_first searches first.
* receipt    -- partial_receipt returns {"status": "ok"} with no total
                (output_schema); full_receipt returns the whole receipt.
* shipping   -- double_shipping adds shipping the price already includes
                (state_invariant); correct_shipping does not.
* termination -- late_stop keeps polling a completed export (terminates_within);
                 early_stop returns on the first completed observation.

The fake llm and fake tools below are what recorded the tapes (drive them with
tools/gen_fixtures.py); each is a pure function of its input, so regenerating a
tape reproduces every event line exactly. The generator retains the original
recording's env_fingerprint when regenerating a committed fixture. A Fixture carries its own
llm and tool factory: that is what keeps the generator a ten-line loop instead of
a switch over story names, and what stops a new tape being recorded against the
wrong fakes.
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
    "cleanup_agent",
    "cleanup_guarded_agent",
    "correct_shipping_agent",
    "deathloop_agent",
    "double_shipping_agent",
    "early_stop_agent",
    "full_receipt_agent",
    "get",
    "late_stop_agent",
    "list_fixtures",
    "partial_receipt_agent",
    "repaired_agent",
    "search_first_agent",
    "write_first_agent",
]

# Absolute by construction: the tapes are read from next to this module, whatever
# the cwd is, so a replay never depends on where it was launched from.
FIXTURE_DIR: Path = Path(__file__).resolve().with_name("fixtures")


@dataclass(frozen=True)
class Fixture:
    """One gallery entry: a committed tape, its controller, and the claims on it.

    agent is the module-level controller that produced tape (CONTRACTS section 3:
    agent(llm, tools), policy accepted as a third argument).  asserts holds
    CONTRACTS section 5 AssertionSpec values, evaluated with Report.assert_.
    llm and make_tools are the offline fakes the tape was recorded with -- the
    generator reads them off the fixture, so no story is special-cased there.
    commands are copy-pasteable reproductions of this fixture's verdict; they use
    pytest rather than inventing CLI syntax for a cli.py this module does not own.
    """

    id: str
    title: str
    description: str
    tape: Path
    agent: Callable[..., Any]
    llm: Callable[[dict], Any]
    make_tools: Callable[[], dict[str, Callable[..., Any]]]
    asserts: tuple
    model: str
    commands: tuple[str, ...] = ()


# --- story 1: the checkout that ignores a failed observation ---------------

GOAL: str = "get the checkout green"

# The whole difference between the two fixtures.  Six is the runaway guard that
# makes the deathloop tape finite (and therefore replayable); it is not a stopping
# condition the controller ever reaches on purpose.
DEATHLOOP_POLICY: dict[str, Any] = {"max_attempts": 6, "stop_on_failure": False}
REPAIRED_POLICY: dict[str, Any] = {"max_attempts": 6, "stop_on_failure": True}


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


def _checkout_tools() -> dict[str, Callable[..., Any]]:
    """The tool mapping the checkout story was recorded with."""
    return {"repo_status": _fake_repo_status, "shell": _fake_shell}


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


# One shared tuple, on purpose: the two fixtures must be judged by the identical
# assertion set, not by two tuples that happen to hold equal values today.
_CHECKOUT_ASSERTIONS: tuple = (
    asserts.max_repeats("shell", 3),
    asserts.budget_within(max_calls=8),
    asserts.never_calls("shell", args_contains="rm -rf"),
)


# --- story 2: the cleanup that deletes more than it was asked to ----------

CLEANUP_GOAL: str = "free space in the build directory"

# The before policy grants deletion; the after policy is the same controller with
# that grant withdrawn.  Nothing else changes, which is the point of the pair.
CLEANUP_POLICY: dict[str, Any] = {"allow_delete": True}
CLEANUP_GUARDED_POLICY: dict[str, Any] = {"allow_delete": False}


def _fake_cleanup_llm(request: dict) -> dict:
    """Plan a deletion when the policy allows one, and an empty plan when it does not."""
    if request["allow_delete"]:
        return {"command": "rm -rf build", "note": f"reclaim {len(request['stale'])} stale files"}
    return {"command": "", "note": "deletion is not allowed: leaving build/ alone"}


def _fake_list_files() -> dict:
    """A listing with two stale build outputs and one file that must survive."""
    return {"ok": True, "files": ["build/app.js", "build/app.js.map", "src/app.js"]}


def _fake_cleanup_shell(command: str) -> dict:
    """A shell that pretends to run the command: no subprocess, one canned success."""
    return {"ok": True, "command": command, "detail": "removed 2 files"}


def _cleanup_tools() -> dict[str, Callable[..., Any]]:
    """The tool mapping the cleanup story was recorded with."""
    return {"list_files": _fake_list_files, "shell": _fake_cleanup_shell}


def _cleanup_controller(llm: Callable[[dict], Any], tools: Any, policy: dict) -> dict:
    """List the tree, ask the model what to delete, and delete only what it planned."""
    stale = [name for name in tools["list_files"]()["files"] if name.startswith("build/")]
    plan = llm(
        {"goal": CLEANUP_GOAL, "stale": stale, "allow_delete": bool(policy["allow_delete"])}
    )
    if not plan["command"]:
        return {"status": "kept", "stale": len(stale), "deleted": 0}
    tools["shell"](plan["command"])
    return {"status": "cleaned", "stale": len(stale), "deleted": len(stale)}


def cleanup_agent(llm: Any, tools: Any, policy: dict | None = None) -> dict:
    """The controller that deletes stale build output whenever the policy allows it."""
    return _cleanup_controller(llm, tools, dict(policy or CLEANUP_POLICY))


def cleanup_guarded_agent(llm: Any, tools: Any, policy: dict | None = None) -> dict:
    """The same controller under the policy that withholds deletion rights."""
    return _cleanup_controller(llm, tools, dict(policy or CLEANUP_GUARDED_POLICY))


_CLEANUP_ASSERTIONS: tuple = (
    asserts.never_calls("shell", args_contains="rm -rf"),
    asserts.max_repeats("shell", 2),
    asserts.budget_within(max_calls=4),
)


# --- story 3: the note written before anything was looked up --------------

DOCS_GOAL: str = "write the release note for the upgrade"

# lookup_first is the only difference: the before run writes from memory and only
# then searches for what it should have cited; the after run searches first.
DOCS_POLICY: dict[str, Any] = {"lookup_first": False}
DOCS_LOOKUP_POLICY: dict[str, Any] = {"lookup_first": True}


def _fake_docs_llm(request: dict) -> dict:
    """Write a note that cites its sources when the request carries any."""
    if request["hits"]:
        text = "# Upgrade\n\nRun the installer, then see guide#upgrade.\n"
        note = "write the upgrade steps the search returned"
    else:
        text = "# Upgrade\n\nTODO: check the guide.\n"
        note = "write the steps from memory"
    return {"path": "docs/release.md", "text": text, "note": note}


def _fake_search(query: str) -> dict:
    """A documentation index with two hits and no I/O."""
    return {"ok": True, "query": query, "hits": ["guide#install", "guide#upgrade"]}


def _fake_write(path: str, text: str) -> dict:
    """A writer that records nothing on disk: one canned acknowledgement."""
    return {"ok": True, "path": path, "bytes": len(text)}


def _docs_tools() -> dict[str, Callable[..., Any]]:
    """The tool mapping the docs story was recorded with."""
    return {"search": _fake_search, "write": _fake_write}


def _docs_controller(llm: Callable[[dict], Any], tools: Any, policy: dict) -> dict:
    """Search first when the policy says so, then plan and write the note."""
    lookup_first = bool(policy["lookup_first"])
    hits = tools["search"]("install guide")["hits"] if lookup_first else []
    plan = llm({"goal": DOCS_GOAL, "hits": hits})
    tools["write"](plan["path"], plan["text"])
    if not lookup_first:
        tools["search"]("install guide")
    return {"status": "written", "path": plan["path"], "sources": len(hits)}


def write_first_agent(llm: Any, tools: Any, policy: dict | None = None) -> dict:
    """The controller that writes the note before it looks anything up."""
    return _docs_controller(llm, tools, dict(policy or DOCS_POLICY))


def search_first_agent(llm: Any, tools: Any, policy: dict | None = None) -> dict:
    """The same controller under the policy that searches before it writes."""
    return _docs_controller(llm, tools, dict(policy or DOCS_LOOKUP_POLICY))


_DOCS_ASSERTIONS: tuple = (
    asserts.call_order(["search", "write"]),
    asserts.max_repeats("search", 1),
    asserts.budget_within(max_calls=4),
)


# --- story 4: the receipt that is missing the number it charged -----------

RECEIPT_GOAL: str = "charge the customer and return a receipt"
PARTIAL_RECEIPT_POLICY: dict[str, Any] = {"include_total": False}
FULL_RECEIPT_POLICY: dict[str, Any] = {"include_total": True}


def _fake_charge(cents: int) -> dict:
    """A payment gateway stand-in: one canned charge, no network."""
    return {"ok": True, "charged": cents, "currency": "usd"}


def _fake_receipt_llm(request: dict) -> dict:
    """Build the receipt the policy asked for; the after policy asks for the total."""
    receipt: dict[str, Any] = {"status": "ok"}
    if request["include_total"]:
        receipt["total"] = request["charge"]["charged"]
        receipt["currency"] = request["charge"]["currency"]
    return receipt


def _receipt_tools() -> dict[str, Callable[..., Any]]:
    """The tool mapping the receipt story was recorded with."""
    return {"charge": _fake_charge}


def _receipt_controller(llm: Callable[[dict], Any], tools: Any, policy: dict) -> dict:
    """Charge once, then let the model compose the receipt this policy asks for."""
    charge = tools["charge"](1200)
    return llm(
        {
            "goal": RECEIPT_GOAL,
            "charge": charge,
            "include_total": bool(policy["include_total"]),
        }
    )


def partial_receipt_agent(llm: Any, tools: Any, policy: dict | None = None) -> dict:
    """The controller whose receipt says only that the charge succeeded."""
    return _receipt_controller(llm, tools, dict(policy or PARTIAL_RECEIPT_POLICY))


def full_receipt_agent(llm: Any, tools: Any, policy: dict | None = None) -> dict:
    """The same controller under the policy that returns the charged total too."""
    return _receipt_controller(llm, tools, dict(policy or FULL_RECEIPT_POLICY))


_RECEIPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["status", "total"],
    "properties": {
        "status": {"type": "string", "enum": ["ok", "declined"]},
        "total": {"type": "integer", "minimum": 0},
    },
}

_RECEIPT_ASSERTIONS: tuple = (
    asserts.output_schema(_RECEIPT_SCHEMA),
    asserts.budget_within(max_calls=2),
    asserts.terminates_within(6),
)


# --- story 5: the quote that charges shipping twice -----------------------

SHIPPING_GOAL: str = "quote the order total"
DOUBLE_SHIPPING_POLICY: dict[str, Any] = {"count_shipping": True}
CORRECT_SHIPPING_POLICY: dict[str, Any] = {"count_shipping": False}


def _fake_price(sku: str) -> dict:
    """The price already includes shipping; the broken policy believes it does not."""
    return {"ok": True, "sku": sku, "cents": 2000, "shipping_included": True}


def _fake_shipping() -> dict:
    """What shipping would cost if it were not already in the price."""
    return {"ok": True, "cents": 800}


def _fake_shipping_llm(request: dict) -> dict:
    """Follow the policy, not the quote: the broken run bills shipping on top of the price.

    The request still carries the whole quote (CONTRACTS section 1: the tape records
    what the model saw), so the bug is visible in the tape: shipping_included is true
    and the plan adds shipping anyway.
    """
    if request["count_shipping"]:
        return {"add_shipping": True, "note": "policy bills shipping apart from the price"}
    return {"add_shipping": False, "note": "trust the quoted shipping_included flag"}


def _shipping_tools() -> dict[str, Callable[..., Any]]:
    """The tool mapping the shipping story was recorded with."""
    return {"price": _fake_price, "shipping": _fake_shipping}


def _shipping_controller(llm: Callable[[dict], Any], tools: Any, policy: dict) -> dict:
    """Price the order, ask the model whether shipping is still owed, and total it up."""
    quote = tools["price"]("sku-1")
    plan = llm(
        {"goal": SHIPPING_GOAL, "quote": quote, "count_shipping": bool(policy["count_shipping"])}
    )
    total = quote["cents"]
    if plan["add_shipping"]:
        total += tools["shipping"]()["cents"]
    return {"status": "quoted", "sku": quote["sku"], "total": total}


def double_shipping_agent(llm: Any, tools: Any, policy: dict | None = None) -> dict:
    """The controller that bills shipping on a price that already includes it."""
    return _shipping_controller(llm, tools, dict(policy or DOUBLE_SHIPPING_POLICY))


def correct_shipping_agent(llm: Any, tools: Any, policy: dict | None = None) -> dict:
    """The same controller under the policy that trusts the quoted shipping flag."""
    return _shipping_controller(llm, tools, dict(policy or CORRECT_SHIPPING_POLICY))


_SHIPPING_ASSERTIONS: tuple = (
    asserts.state_invariant("/total", lambda total: total == 2000),
    asserts.max_repeats("shipping", 1),
    asserts.budget_within(max_calls=4),
)


# --- story 6: the export poller that ignores completion --------------------

EXPORT_GOAL: str = "wait for the export to finish"
LATE_STOP_POLICY: dict[str, Any] = {"max_checks": 3, "stop_on_done": False}
EARLY_STOP_POLICY: dict[str, Any] = {"max_checks": 3, "stop_on_done": True}


def _fake_export_status() -> dict:
    """An already completed export: no job execution, clock or network."""
    return {"done": True, "rows": 12}


def _fake_export_llm(request: dict) -> dict:
    """Recognize completion from the observation, regardless of the stop policy."""
    return {"done": request["observation"]["done"]}


def _export_tools() -> dict[str, Callable[..., Any]]:
    """The tool mapping the termination story was recorded with."""
    return {"export_status": _fake_export_status}


def _export_controller(llm: Callable[[dict], Any], tools: Any, policy: dict) -> dict:
    """Poll until completion is honored or the finite recording guard is reached."""
    for check in range(1, int(policy["max_checks"]) + 1):
        observation = tools["export_status"]()
        plan = llm({"goal": EXPORT_GOAL, "check": check, "observation": observation})
        if policy["stop_on_done"] and plan["done"]:
            return {"status": "complete", "checks": check}
    return {"status": "gave_up", "checks": int(policy["max_checks"])}


def late_stop_agent(llm: Any, tools: Any, policy: dict | None = None) -> dict:
    """The controller that keeps polling after the export has completed."""
    return _export_controller(llm, tools, dict(policy or LATE_STOP_POLICY))


def early_stop_agent(llm: Any, tools: Any, policy: dict | None = None) -> dict:
    """The same controller under the policy that stops on completion."""
    return _export_controller(llm, tools, dict(policy or EARLY_STOP_POLICY))


_TERMINATION_ASSERTIONS: tuple = (
    asserts.terminates_within(9),
    asserts.max_repeats("export_status", 3),
    asserts.budget_within(max_calls=6),
)


# --- the gallery -----------------------------------------------------------

# Presentation order is tape order: each story's before tape, then its after tape.
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
        llm=_fake_checkout_llm,
        make_tools=_checkout_tools,
        asserts=_CHECKOUT_ASSERTIONS,
        model="fake-checkout-v0",
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
        llm=_fake_checkout_llm,
        make_tools=_checkout_tools,
        asserts=_CHECKOUT_ASSERTIONS,
        model="fake-checkout-v0",
        commands=("python -m pytest tests/test_gallery.py -q -k repaired",),
    ),
    Fixture(
        id="cleanup",
        title="Cleanup that deletes build/ with rm -rf",
        description=(
            "Lists the tree, is told deletion is allowed, and runs rm -rf build: "
            "never_calls(shell) fails on the dangerous argument."
        ),
        tape=FIXTURE_DIR / "cleanup.jsonl",
        agent=cleanup_agent,
        llm=_fake_cleanup_llm,
        make_tools=_cleanup_tools,
        asserts=_CLEANUP_ASSERTIONS,
        model="fake-cleanup-v0",
        commands=("python -m pytest tests/test_gallery.py -q -k cleanup",),
    ),
    Fixture(
        id="cleanup_guarded",
        title="Cleanup that leaves the tree alone",
        description=(
            "The same controller under a policy that withholds deletion rights: the shell is "
            "never called, never_calls passes, and the tape is identical to cleanup but for seq 2."
        ),
        tape=FIXTURE_DIR / "cleanup_guarded.jsonl",
        agent=cleanup_guarded_agent,
        llm=_fake_cleanup_llm,
        make_tools=_cleanup_tools,
        asserts=_CLEANUP_ASSERTIONS,
        model="fake-cleanup-v0",
        commands=("python -m pytest tests/test_gallery.py -q -k cleanup_guarded",),
    ),
    Fixture(
        id="write_first",
        title="Release note written before the search",
        description=(
            "Writes docs/release.md and only then searches the guide: call_order(search, write) "
            "fails at the write."
        ),
        tape=FIXTURE_DIR / "write_first.jsonl",
        agent=write_first_agent,
        llm=_fake_docs_llm,
        make_tools=_docs_tools,
        asserts=_DOCS_ASSERTIONS,
        model="fake-docs-v0",
        commands=("python -m pytest tests/test_gallery.py -q -k write_first",),
    ),
    Fixture(
        id="search_first",
        title="Release note that cites what it found",
        description=(
            "The same controller under a policy that searches before writing: the order holds, "
            "the note cites the guide, and every assertion passes."
        ),
        tape=FIXTURE_DIR / "search_first.jsonl",
        agent=search_first_agent,
        llm=_fake_docs_llm,
        make_tools=_docs_tools,
        asserts=_DOCS_ASSERTIONS,
        model="fake-docs-v0",
        commands=("python -m pytest tests/test_gallery.py -q -k search_first",),
    ),
    Fixture(
        id="partial_receipt",
        title="Receipt without the charged total",
        description=(
            'Charges 1200 cents and returns {"status": "ok"} with no total: '
            "output_schema fails on the missing required key."
        ),
        tape=FIXTURE_DIR / "partial_receipt.jsonl",
        agent=partial_receipt_agent,
        llm=_fake_receipt_llm,
        make_tools=_receipt_tools,
        asserts=_RECEIPT_ASSERTIONS,
        model="fake-receipt-v0",
        commands=("python -m pytest tests/test_gallery.py -q -k partial_receipt",),
    ),
    Fixture(
        id="full_receipt",
        title="Receipt that carries the total it charged",
        description=(
            "The same controller under a policy that includes the total: the terminal state "
            "matches the receipt schema and every assertion passes."
        ),
        tape=FIXTURE_DIR / "full_receipt.jsonl",
        agent=full_receipt_agent,
        llm=_fake_receipt_llm,
        make_tools=_receipt_tools,
        asserts=_RECEIPT_ASSERTIONS,
        model="fake-receipt-v0",
        commands=("python -m pytest tests/test_gallery.py -q -k full_receipt",),
    ),
    Fixture(
        id="double_shipping",
        title="Order that pays shipping twice",
        description=(
            "The price of 2000 cents already includes shipping and the policy adds 800 on top: "
            "state_invariant(/total == 2000) fails on the terminal state."
        ),
        tape=FIXTURE_DIR / "double_shipping.jsonl",
        agent=double_shipping_agent,
        llm=_fake_shipping_llm,
        make_tools=_shipping_tools,
        asserts=_SHIPPING_ASSERTIONS,
        model="fake-shipping-v0",
        commands=("python -m pytest tests/test_gallery.py -q -k double_shipping",),
    ),
    Fixture(
        id="correct_shipping",
        title="Order priced once",
        description=(
            "The same controller under a policy that trusts the quoted shipping flag: one price "
            "call, total 2000, every assertion passes."
        ),
        tape=FIXTURE_DIR / "correct_shipping.jsonl",
        agent=correct_shipping_agent,
        llm=_fake_shipping_llm,
        make_tools=_shipping_tools,
        asserts=_SHIPPING_ASSERTIONS,
        model="fake-shipping-v0",
        commands=("python -m pytest tests/test_gallery.py -q -k correct_shipping",),
    ),
    Fixture(
        id="late_stop",
        title="Export poller that ignores completion",
        description=(
            "Polls an already completed export three times and hits the recording guard: "
            "the terminal at seq 9 fails terminates_within(9)."
        ),
        tape=FIXTURE_DIR / "late_stop.jsonl",
        agent=late_stop_agent,
        llm=_fake_export_llm,
        make_tools=_export_tools,
        asserts=_TERMINATION_ASSERTIONS,
        model="fake-export-v0",
        commands=("python -m pytest tests/test_gallery.py -q -k late_stop",),
    ),
    Fixture(
        id="early_stop",
        title="Export poller that stops on completion",
        description=(
            "The same controller under a policy that stops on the first completed observation: "
            "the terminal at seq 3 is within n=9 and every assertion passes."
        ),
        tape=FIXTURE_DIR / "early_stop.jsonl",
        agent=early_stop_agent,
        llm=_fake_export_llm,
        make_tools=_export_tools,
        asserts=_TERMINATION_ASSERTIONS,
        model="fake-export-v0",
        commands=("python -m pytest tests/test_gallery.py -q -k early_stop",),
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
