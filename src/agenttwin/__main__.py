"""Module entry point so "python -m agenttwin" reaches the CLI (CONTRACTS section 6).

The console script installed by pyproject.toml points at agenttwin.cli:main; this
module exists so the same surface works from a source checkout without an
install.  Keep it to the delegation and the exit status -- everything else,
including status mapping, belongs to agenttwin.cli.main.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
