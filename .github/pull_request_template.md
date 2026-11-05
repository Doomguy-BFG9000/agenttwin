## What this changes

<!-- One paragraph. Say why, not only what. -->

## Checks

- [ ] `python -m pytest tests -q` passes locally
- [ ] `python -m coverage report` stays above the 82% gate
- [ ] New behaviour has a test that fails without this change
- [ ] No new third-party dependency in the core path (`dependencies = []`)
- [ ] The tape format is untouched, or the schema version and migration note are updated
- [ ] If a fixture changed: `python tools/gen_fixtures.py` was run and the tapes were committed
- [ ] If the demo changed: `python tools/gen_demo_pages.py` was run and `docs/` was committed

## Notes for the reviewer

<!-- Anything you are unsure about, and what you would like looked at hardest. -->
