[English](README.md) · [中文](README.zh-CN.md)

# AgentTwin

**The VCR for agents.** Record one agent run to a tape, replay it offline afterwards, and you have a unit test you can commit to CI.

Not a trace dashboard, not a monitoring platform: no service, no database, no dashboard, just one JSONL file and one ordinary assertion.

The 30-second version:

1. Recording calls the model and writes the request hash and the response into the tape;
2. Replay calls no model and executes no real tool; it walks the tape event by event;
3. A mismatch is a failure (not "run it again and see"); the assertion result ends up as a process exit code.

Every code block below was actually run against this repository: blocks 1 to 4 in order, block 5 only against the fixtures shipped with the repo. The output under a block is the output of that run.

## 1. Record one run

`record_demo.py`

```python
"""Record one agent run to a JSONL tape.  Run: python record_demo.py"""
import json

from agenttwin import Twin


def fake_llm(request):                    # stands in for your model client
    return {"text": f"order {request['prompt']} is on its way"}


def lookup_order(order_id):               # stands in for a real, side-effecting tool
    return {"order_id": order_id, "status": "paid"}


def agent(llm, tools):                    # your agent: identical when replaying
    reply = llm({"prompt": "A-1"})
    order = tools["lookup_order"]("A-1")
    return {"answer": reply["text"], "status": order["status"]}


with Twin.record("checkout", out="checkout.jsonl",
                 llm=fake_llm, tools={"lookup_order": lookup_order}) as rec:
    rec.returned(agent(rec.llm, rec.tools()))     # leaving the block writes the tape

lines = open("checkout.jsonl", encoding="utf-8").read().splitlines()
header = json.loads(lines[0])
print("header:", header["schema_version"], header["trace_id"], "seed=" + str(header["seed"]))
print("events:", " -> ".join(json.loads(line)["kind"] for line in lines[1:]))
print("line 2:", lines[1][:96] + " ...")
```

Output:

```text
header: 0.1 checkout seed=0
events: llm_call -> tool_call -> tool_result -> terminal
line 2: {"actor":"llm","kind":"llm_call","payload":{"prompt_hash":"1d783fcd9d8b565fc5b19532a4538139c2ab6 ...
```

Only the last three lines do the recording: run the agent once inside `with Twin.record(...) as rec:`, keep the return value with `rec.returned(...)`, and the tape is written when the `with` block exits.

`checkout.jsonl` is plain JSONL: the first line is the header (`schema_version` / `trace_id` / `seed` / `env_fingerprint` / `redaction`), then one event per line; there are exactly 8 `kind`s: `llm_call`, `tool_call`, `tool_result`, `state_patch`, `decision`, `human_input`, `error`, `terminal`. Any text tool can read it.

## 2. Replay offline, then assert

Keep going in the same directory:

`replay_demo.py`

```python
"""Replay the tape offline and assert on it.  Run after record_demo.py, same dir."""
import sys

from agenttwin import Twin, asserts


def agent(llm, tools):                    # the same agent, character for character
    reply = llm({"prompt": "A-1"})
    order = tools["lookup_order"]("A-1")
    return {"answer": reply["text"], "status": order["status"]}


report = Twin.load("checkout.jsonl").replay(agent)   # note: no llm=, no tools=
report.assert_(
    asserts.max_repeats("lookup_order", n=1),
    asserts.never_calls("refund"),
    asserts.terminates_within(10),
)

for result in report.results:
    status = "SKIP" if result.passed is None else ("PASS" if result.passed else "FAIL")
    print(f"{status}  {result.name}")
page = report.html("report.html")         # one self-contained file, opens from file://
print(f"html: {page.name} ({page.stat().st_size} bytes)")
print(f"divergent: {report.divergent}")
print(f"exit code: {report.exit_code}")
sys.exit(report.exit_code)
```

Output:

```text
PASS  max_repeats(lookup_order, n=1)
PASS  never_calls(refund)
PASS  terminates_within(n=10)
html: report.html (56546 bytes)
divergent: False
exit code: 0
```

Two things that are easy to miss:

- `Twin.load("checkout.jsonl").replay(agent)` takes no `llm=` and no `tools=`: both the model and the tools come from the tape. `llm(...)` returns the response as recorded, `tools["lookup_order"](...)` returns the result as recorded, and the real functions are never called.
- `sys.exit(report.exit_code)` turns the assertion result into an ordinary exit code that `pytest` or any CI consumes directly.

The same result in three shapes: `report.markdown()` (a table for humans), `report.json()` (for machines, with fixed top-level keys `schema_version` / `trace_id` / `mode` / `divergent` / `exit_code` / `results` / `misses` / `drifts` / `diff` / `header`), and `report.html("report.html")` (a single-file report, no external references, opens straight from `file://`; the byte count above varies a little with the tape and the environment).

## 3. A changed request is a hard failure

```python
"""A changed request is a hard failure, not a flaky test.  Run after record_demo.py."""
from agenttwin import DriftError, Twin


def changed_agent(llm, tools):
    llm({"prompt": "A-2"})                # one value changed in the request
    tools["lookup_order"]("A-1")


try:
    Twin.load("checkout.jsonl").replay(changed_agent)
except DriftError as exc:                 # PromptDrift / ArgsDrift / FingerprintDrift
    detail = exc.message.replace(exc.expected, exc.expected[:8] + "...").replace(exc.actual, exc.actual[:8] + "...")
    print(f"{type(exc).__name__}: {detail}")
    raise SystemExit(2)
```

Output:

```text
PromptDrift: prompt_hash mismatch at seq 0: tape expects 1d783fcd..., controller sent 4250ccd5...
```

Note that it raises instead of noting the mismatch and carrying on: the replay stops right there, and `SystemExit(2)` turns it into a red job in CI.

## 4. Assertions can be data too (optional `[yaml]`)

`asserts.yaml`:

```yaml
# The same three checks as replay_demo.py, kept as data.
- {spec: max_repeats, args: {tool: lookup_order, n: 1}}
- {spec: never_calls, args: {tool: refund}}
- {spec: terminates_within, args: {n: 10}}
```

`yaml_demo.py`:

```python
"""The same three checks as data.  Run after record_demo.py."""
from agenttwin import Twin, asserts


def agent(llm, tools):
    reply = llm({"prompt": "A-1"})
    order = tools["lookup_order"]("A-1")
    return {"answer": reply["text"], "status": order["status"]}


report = Twin.load("checkout.jsonl").replay(agent)
report.assert_(*asserts.load_yaml("asserts.yaml"))   # needs the [yaml] extra

for result in report.results:
    print(f"{'PASS' if result.passed else 'FAIL'}  {result.name}")
print("exit code:", report.exit_code)
```

Output:

```text
PASS  max_repeats(lookup_order, n=1)
PASS  never_calls(refund)
PASS  terminates_within(n=10)
exit code: 0
```

## What this tape does not promise

This section sits on the second screen because the most common question is "why did nothing change when I changed the temperature".

- **What is frozen is the request and the response as recorded, not an inference pass you can run again.** The model is not called a second time during replay, so sampling parameters (temperature / top_p / seed) do not participate and do not affect the result. Conversely, if your agent puts temperature into the request dict, it is part of the request, it changes the `prompt_hash`, and it raises `PromptDrift`: a changed request is supposed to fail.
- **No real side effects.** A tool call during replay returns the `result` from the tape and the real function does not execute; there is no "call it for real while we are here" either, `Twin.load(..., allow_real=True)` raises `ValueError` outright.
- **No second model call.** No network requests, no token bill; as many `llm_call`s as the tape holds is as many replays as you get, and not one more appears out of nowhere.
- **All four drift kinds are hard failures.** `DriftError` and its three subclasses `PromptDrift` / `ArgsDrift` / `FingerprintDrift` are raised and stop the replay; `TapeMiss` (nothing recorded for that call) is not a drift and is not raised, but it also turns `exit_code` into 2.
- **Replay is not "run the agent again".** A tape hit does not re-execute the model or the tool, and a tape miss does not degrade into a real call.

## What it does not do

| Not doing | Why |
| --- | --- |
| Calling the LLM again | Replay is worth exactly its determinism and its zero cost; for new results, record a new tape |
| Producing real side effects | Replay has to run in CI, on a machine with no network |
| Promising zero-config support for every framework | You write the agent as `agent(llm, tools)`, or as an object with `run(llm, tools)`; the tape knows nothing about your framework |
| SaaS / dashboard / database | The tape is a file and the report is a file; there is no backend to host |
| LLM-as-judge for pass or fail | The verdict has to be a deterministic assertion; asking a model to grade brings the uncertainty right back |

## Install

```bash
python -m pip install -e .
```

An extra dependency is needed only to read and write YAML assertion files (`pyyaml` is imported lazily, inside `load_yaml()` only):

```bash
python -m pip install -e ".[yaml]"
```

Python 3.10+, zero third-party dependencies at runtime: `pyproject.toml` declares `dependencies = []`, and `import agenttwin` does not pull in `pyyaml`.

### Exit codes

| code | meaning |
| --- | --- |
| 0 | everything PASSes (SKIP is not a failure) |
| 1 | an assertion FAILed |
| 2 | Drift / TapeMiss / SchemaError / incompatible schema |
| 3 | configuration or install error (missing tape, broken assertion file, missing dependency) |

`--strict` turns SKIP into 1, with precedence 3 > 2 > 1 > 0. `Report.exit_code` only ever produces 0 / 1 / 2; 3 comes from the CLI's usage and configuration layer (`EXIT_USAGE = 3` in `agenttwin.cli`: the tape does not exist, `--agent` is malformed, the assertion file cannot be read).

### Command line

Once installed, `agenttwin` and `python -m agenttwin` are the same entry point (`agenttwin.cli:main`), with five subcommands: `record` / `replay` / `diff` / `doctor` / `gallery`. To see the two fixtures shipped with the repo, without recording a tape first:

```bash
python -m agenttwin gallery list
python -m agenttwin gallery run deathloop
```

Output (the second command prints the head of a report and its assertion table; the Diff / Fingerprint / Redaction sections in between are elided, `...` marks the gap):

```text
deathloop  Checkout retry loop
repaired  Checkout that stops on the failed observation

# AgentTwin report: gallery-deathloop

verdict: FAIL (exit code 1)
mode: replay
divergent: no
strict: no
...
| name | status | detail | first_offense_seq |
| --- | --- | --- | --- |
| max_repeats(shell, n=3) | FAIL | 6 calls of shell exceeds the limit of 3; first excess at seq 12 | 12 |
| budget_within(calls<=8) | FAIL | calls 13 > 8 (first excess at seq 12) | 12 |
| never_calls(shell, args_contains='rm -rf') | PASS | no call of shell matches | - |
```

`gallery run deathloop` exits 1 and `gallery run repaired` exits 0: same assertion set, and the only difference is whether that agent repeats the same tool call a thirteenth time. `python -m agenttwin <command> --help` shows the arguments of each command.

## Status

0.1.0 is the version in the repository, not a release number: nothing has been published yet.

- Done and tested: the record/replay core (`Twin.record` / `Twin.load`), drift detection, 9 assertion factories plus the combinators and YAML loading, `TraceDiff`, `Report` (JSON / Markdown / single-file HTML), the five `agenttwin` subcommands, the openai-compatible adapter (duck-typed, no SDK import), and the two bundled fixtures.
- Not verified: `.github/workflows/ci.yml` has only been parsed as YAML locally and has never actually run on GitHub; this machine's sandbox cannot install packages, so every example runs with `PYTHONPATH=src`, and the `agenttwin` console script is declared in `pyproject.toml` but has never actually been executed (`python -m agenttwin` has).
- Not done yet: the PyPI release, a coverage gate, any hosted service or dashboard.
