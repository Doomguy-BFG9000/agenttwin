# Contributing to AgentTwin

Thanks for looking. This project is small on purpose: one narrow protocol
(`agent(llm, tools)`), one tape format, one adapter. The most useful
contributions are usually **a new assertion** or **a new gallery fixture**, not
a new integration.

## Set up

```bash
git clone https://github.com/ravenkernel/agenttwin
cd agenttwin
python -m pip install -e ".[yaml,dev]"
python -m pytest tests -q
```

Python 3.10+ is required. The test suite must pass on 3.10 through 3.13 and on
Windows, macOS and Linux — CI runs all of them. No test may write to the OS
temp directory: use the `tape_dir` fixture from `tests/conftest.py`.

## The four rules that are not negotiable

1. **The core has zero third-party dependencies.** `dependencies = []` in
   `pyproject.toml` and a stdlib-only CI job that uninstalls PyYAML and imports
   the package. PyYAML is optional and lazily imported; nothing else may enter
   the core path.
2. **The tape format is frozen at `schema 0.1`.** Adding a field or a new
   `kind` is a schema version bump plus a migration note. Padding a payload is
   fine; changing the meaning of `seq`, `t` or the eight kinds is not.
3. **Drift is a hard failure.** Never downgrade a `PromptDrift`, `ArgsDrift` or
   `FingerprintDrift` to a warning to make a test green. "The prompt changed
   but the test still passes" is the failure mode this project exists to
   prevent.
4. **Replay never touches the world.** No model call, no real tool execution,
   no network, no wall-clock reads. If a change needs any of those, it belongs
   in record mode.

## Adding an assertion

Assertions are pure functions `Trace -> AssertionResult`; they cannot do IO,
call a model or look at the clock.

- Add the factory to `src/agenttwin/asserts.py` next to its siblings.
- Return `passed=None` (SKIP) only when the tape genuinely cannot be judged;
  "nothing went wrong" is a PASS, not a SKIP.
- Set `first_offense_seq` when there is an offending event — it is what makes
  the failure clickable in the HTML viewer.
- Cover it in `tests/test_asserts.py`, then use it in a gallery pair so the
  documented demo and the code can never disagree.

## Adding a gallery fixture

The gallery is the front door of the project: five before/after pairs, one
assertion family each, all replayable offline. To add a sixth pair, register
both halves in `src/agenttwin/gallery.py` (`llm`, `make_tools`, `model`,
`agent`, `asserts`), then:

```bash
python tools/gen_fixtures.py     # writes src/agenttwin/fixtures/*.jsonl
python -m pytest tests/test_gallery.py -q
```

Constraints enforced by tests: each tape stays under 8 KiB, the before half
exits 1 and the after half exits 0, every assertion in the pair really
evaluates (no SKIP rows), and both READMEs quote `agenttwin gallery list`
verbatim.

## Regenerating the demo pages

`docs/` is published by GitHub Pages and is generated, not hand-edited:

```bash
python tools/gen_demo_pages.py   # replays the tape into docs/before.html, docs/after.html
```

The README hero transcript is the real stdout of `examples/hero.py`; a test
runs it and compares, so changing its output means changing both READMEs.

## Pull requests

- One concern per pull request; keep the diff readable.
- Run `python -m pytest tests -q` and `python -m coverage report` before
  pushing. The CI gate is 82% of statements.
- Say *why* in the commit message. The repository history is part of the
  documentation.
- New behaviour needs a test that fails without it. Fixes for the viewer or the
  CLI need the same treatment as fixes for the core.

## Reporting a bug

Open an issue with `agenttwin --version`, the output of
`agenttwin doctor <your-tape.jsonl>`, and — when you can — a tape made from a
bundled fixture. Never paste real API keys or prompt text you cannot share.
