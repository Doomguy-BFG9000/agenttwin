"""Regenerate the two committed gallery tapes (design doc section 8).

Every byte under src/agenttwin/fixtures/ is produced by this script: the tapes are
real recordings made through Twin.record, not hand-written JSONL.  The llm and the
tools come from agenttwin.gallery, so the controller that runs here is byte-for-byte
the one that replays later -- Record and Replay inject their own objects (CONTRACTS
section 3) and it cannot tell them apart.

Nothing here touches the network, the clock or the filesystem: the fake llm is a
pure function of the request and the fake tools are constants, so a rerun of this
script reproduces both tapes byte for byte (fixtures carry no timestamps, CONTRACTS
section 1).

    python tools/gen_fixtures.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agenttwin import Twin  # noqa: E402
from agenttwin import gallery  # noqa: E402

FIXTURES = REPO_ROOT / "src" / "agenttwin" / "fixtures"
MODEL = "fake-checkout-v0"


def record(fixture: "gallery.Fixture", *, out_dir: Path = FIXTURES) -> Path:
    """Record one fixture by running its own controller against the offline fakes."""
    path = Path(out_dir) / f"{fixture.id}.jsonl"
    tools = {"repo_status": gallery._fake_repo_status, "shell": gallery._fake_shell}
    with Twin.record(
        f"gallery-{fixture.id}",
        out=path,
        seed=0,
        llm=gallery._fake_checkout_llm,
        tools=tools,
        model=MODEL,
    ) as recorder:
        result = fixture.agent(recorder.llm, recorder.tools())
        recorder.returned(result)
    return path


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for fixture in gallery.list_fixtures():
        path = record(fixture)
        data = path.read_bytes()
        print(f"{fixture.id:10s} {len(data):5d} bytes  sha256={hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
