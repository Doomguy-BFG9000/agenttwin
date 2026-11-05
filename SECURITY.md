# Security policy

## What this project is, and is not

AgentTwin is a **development-time testing tool**. It freezes one agent run into
a tape and replays it without touching the outside world. It provides no
runtime guardrails, no sandbox and no policy enforcement for a live agent, and
it must not be deployed as one. That is a design boundary, not a missing
feature.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private reporting channel:

<https://github.com/ravenkernel/agenttwin/security/advisories/new>

Include the version (`agenttwin --version`), the platform, a minimal
reproduction and what an attacker gains. You can expect an answer within a
week and a fix or a written explanation in the next patch release.

## In scope

- **Secret leakage through a tape**: a value that the redaction policy promises
  to drop or replace, still present in `*.jsonl`, in `prompt_preview` /
  `args_preview`, in the header, or in a generated `report.html`.
- **`doctor` false negatives**: a tape with an unredacted secret that
  `agenttwin doctor` reports as clean.
- **Code execution during replay**: anything that makes a replay execute a real
  tool, make a network request, or import attacker-controlled code. Replay is
  supposed to be a lookup table.
- **Injection in the HTML viewer**: content taken from a tape (trace id, actor,
  payload, assertion detail) escaping its context and executing script in
  `report.html`, which is a single file people open with `file://`.

## Out of scope

- Anything that requires running the agent itself; that is the user's code.
- Denial of service through a deliberately huge tape; there are no size limits
  by design, and a tape is a local file you already trust.
- The PyPI token, the CI configuration or the repository itself — report those
  privately to the maintainer instead.

## Supported versions

The latest patch release of the current minor version receives fixes. Tape
format `schema 0.1` is stable across them, so upgrading never invalidates a
tape you already committed.
