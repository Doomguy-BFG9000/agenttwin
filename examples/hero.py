"""One failing tape, one policy change, the assertions flip -- offline.

Run it from the repository root:

    PYTHONPATH=src python examples/hero.py

No model call, no network, no API key: the committed tape is the whole
environment.  The two reports it writes (build/hero/before.html and
build/hero/after.html) are single files you can open by double-clicking.
"""
from __future__ import annotations

import os
from pathlib import Path

from agenttwin import Twin, gallery

# tools/gen_demo_pages.py points this at docs/ so the published GitHub Pages
# demo is the same replay, not a copy of it.
OUT = Path(os.environ.get("AGENTTWIN_HERO_OUT", "build/hero"))

FIXTURE = gallery.get("deathloop")

# "before" passes no policy at all: the agent falls back to the one it was
# recorded under, so the replay is faithful and the run fails as recorded.
# "after" changes one key; that single difference is the whole demo.
FIXED_POLICY = {**gallery.DEATHLOOP_POLICY, "stop_on_failure": True}


def replay(name: str, policy: dict | None = None):
    """Replay the one tape under policy, judge it with the fixture's assertions."""
    session = Twin.load(FIXTURE.tape)
    if policy is None:
        report = session.replay(FIXTURE.agent)
    else:
        report = session.replay(FIXTURE.agent, policy=policy)
    report = report.assert_(*FIXTURE.asserts)
    OUT.mkdir(parents=True, exist_ok=True)
    report.html(OUT / name)
    return report


def rows(report) -> list[str]:
    """The assertion table as the report states it, one line per assertion."""
    out = []
    for result in report.results:
        verdict = "SKIP" if result.passed is None else ("PASS" if result.passed else "FAIL")
        where = "" if result.first_offense_seq is None else f"  (seq {result.first_offense_seq})"
        out.append(f"  {verdict}  {result.name}  -- {result.detail}{where}")
    return out


def banner(title: str, policy: dict | None, report) -> list[str]:
    label = "recorded (no policy passed)" if policy is None else f"policy={policy}"
    head = f"{title}  mode={report.mode}  {label}"
    return [head, f"  exit code {report.exit_code}"] + rows(report)


def main() -> int:
    before = replay("before.html")
    after = replay("after.html", FIXED_POLICY)

    print(f"tape: {Path(FIXTURE.tape).name}  ({len(Twin.load(FIXTURE.tape).tape.events)} events)")
    print("no model call, no network, no API key: the tape is the whole environment")
    print()
    for line in banner("before", None, before):
        print(line)
    print()
    for line in banner("after ", FIXED_POLICY, after):
        print(line)
    print()

    before_json, after_json = before.json(), after.json()
    first = after_json["diff"]["first_divergence_seq"]
    print(f"first divergence: seq {first}  (same tape, different trajectory)")
    print(f"reports: {(OUT / 'before.html').as_posix()}  {(OUT / 'after.html').as_posix()}")

    if before.exit_code != 1 or after.exit_code != 0:
        print(f"unexpected: before={before.exit_code} (want 1), after={after.exit_code} (want 0)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
