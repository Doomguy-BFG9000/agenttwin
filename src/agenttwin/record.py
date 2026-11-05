"""The L1 instrumentation layer (CONTRACTS section 3): wrap one callable, record it.

twin_llm and twin_tool carry no policy of their own: they hand back the
recorder's own proxies, so a project can adopt AgentTwin without restructuring.
The env-var pair lets a CLI subprocess attach to a recorder its parent opened.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

__all__ = ["bind_record_env", "record_from_env", "twin_llm", "twin_tool"]

TAPE_OUT_ENV = "AGENTTWIN_TAPE_OUT"
TAPE_ID_ENV = "AGENTTWIN_TAPE_ID"


def twin_llm(recorder: Any) -> Callable[[dict], Any]:
    """The recording llm callable of recorder: one call in, one llm_call out."""
    return recorder.llm


def twin_tool(recorder: Any, name: str | None = None) -> Callable[[Callable], Callable]:
    """Register fn on recorder as name and return its recorded wrapper."""

    def decorate(fn: Callable) -> Callable:
        tool = name or getattr(fn, "__name__", "tool")
        recorder.register(tool, fn)
        return recorder.tools()[tool]

    return decorate


def bind_record_env(recorder: Any, *, environ: Any = None) -> dict[str, str]:
    """Publish recorder as the parent tape for later subprocesses; return the pair."""
    env = os.environ if environ is None else environ
    values = {TAPE_OUT_ENV: str(recorder.out), TAPE_ID_ENV: recorder.trace.header.trace_id}
    env.update(values)
    return values


def record_from_env(
    *,
    seed: int = 0,
    llm: Callable[[dict], Any] | None = None,
    tools: Mapping[str, Callable[..., Any]] | None = None,
    model: str | None = None,
    redactor: Any = None,
    environ: Any = None,
) -> Any:
    """Attach to the tape named by AGENTTWIN_TAPE_OUT/AGENTTWIN_TAPE_ID, else None."""
    env = os.environ if environ is None else environ
    out = env.get(TAPE_OUT_ENV)
    trace_id = env.get(TAPE_ID_ENV)
    if not out or not trace_id:
        return None
    from .env import Twin

    return Twin.record(
        trace_id,
        out=out,
        seed=seed,
        llm=llm,
        tools=tools,
        model=model,
        redactor=redactor,
    )
