"""Static assets for the standalone AgentTwin HTML report (CONTRACTS.md section 8).

The report writer reads TEMPLATE_PATH (or calls load_template) and substitutes
the single __AGENTTWIN_PAYLOAD__ placeholder with the JSON payload, producing
one self-contained file that opens over file://.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["TEMPLATE_PATH", "load_template"]

TEMPLATE_PATH = Path(__file__).resolve().parent / "replayer.html"


def load_template() -> str:
    """Return the replayer template, payload placeholder still in place."""
    return TEMPLATE_PATH.read_text(encoding="utf-8")
