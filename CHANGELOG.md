# Changelog

Every release is tagged in git and published on PyPI.  The tape format is
`schema 0.1` for all of them, so a tape recorded by an older release still
replays on a newer one: the package version in the fingerprint is compared
major.minor, so a patch release never invalidates a tape.

## 0.1.1 — 2026-11-05

Documentation, one viewer fix and one change of meaning; the tape format is
untouched.

### Added

- `examples/hero.py` and `assets/hero.gif`: one committed tape replayed twice —
  faithful first (exit 1), then with a single policy key changed (exit 0).  Both
  README editions embed the animation and quote the same transcript, and
  `tests/test_hero.py` keeps that transcript equal to a real run.
- A `Changelog` link in the package metadata.

### Fixed

- The offline viewer's state-diff panel drew eighteen rows as ten-pixel slivers:
  the list is a flex column with `overflow: auto`, so every row was shrinking to
  fit instead of scrolling.  Rows keep their height now.
- `README` installation instructions now lead with `pip install agenttwin`;
  the `-e .` path is mentioned only for working on a checkout.

### Changed

- A replay that carries a patch or an explicit `policy=` now reports
  `mode: counterfactual`, and its divergence no longer forces exit code 2 — the
  assertions decide.  A faithful replay (neither supplied) still treats any
  divergence as exit 2, which is the check that catches a controller that stopped
  reproducing the recorded run.
- `fingerprint._DEFAULT_VERSION` no longer repeats the package version; the
  single source of truth is `agenttwin.__version__`.

## 0.1.0 — 2026-11-05

First public release: record / replay / counterfactual, the four drift classes,
nine assertion factories plus combinators and YAML loading, seq-aligned
`TraceDiff`, the single-file HTML viewer and report, the five `agenttwin`
commands, the openai-compatible adapter, and ten bundled fixtures in five
before/after pairs.
