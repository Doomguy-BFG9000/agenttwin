"""The L1 instrumentation layer (CONTRACTS section 3): wrap one callable, record it.

twin_llm and twin_tool carry no policy of their own: they hand back the
recorder's own proxies, so a project can adopt AgentTwin without restructuring.
The env-var pairs let a CLI subprocess attach to a recorder its parent opened.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

__all__ = [
    "RECORD_OUT_ENV",
    "RECORD_TRACE_ID_ENV",
    "TAPE_ID_ENV",
    "TAPE_OUT_ENV",
    "bind_record_env",
    "record_from_env",
    "twin_llm",
    "twin_tool",
]

TAPE_OUT_ENV = "AGENTTWIN_TAPE_OUT"
TAPE_ID_ENV = "AGENTTWIN_TAPE_ID"

# CONTRACTS section 3: the second spelling of the same handshake.  The CLI
# (agenttwin record) and adapters/openai_compatible publish this pair, so a
# subprocess the CLI starts is readable only if record.py knows it too.
RECORD_OUT_ENV = "AGENTTWIN_RECORD_OUT"
RECORD_TRACE_ID_ENV = "AGENTTWIN_TRACE_ID"


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
    """Publish recorder as the parent tape for later subprocesses; return what was set.

    CONTRACTS section 3: both spellings of the handshake are written with the same
    values -- TAPE_OUT/TAPE_ID and the CLI's RECORD_OUT/TRACE_ID -- so a child
    attaches whichever consumer it uses.
    """
    env = os.environ if environ is None else environ
    out = str(recorder.out)
    trace_id = recorder.trace.header.trace_id
    values = {
        TAPE_OUT_ENV: out,
        TAPE_ID_ENV: trace_id,
        RECORD_OUT_ENV: out,
        RECORD_TRACE_ID_ENV: trace_id,
    }
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
    """Attach to the tape named by the environment, else None (CONTRACTS section 3).

    Precedence: AGENTTWIN_TAPE_OUT / AGENTTWIN_TAPE_ID first, then the CLI's
    AGENTTWIN_RECORD_OUT / AGENTTWIN_TRACE_ID.  A pair counts only when both of
    its variables are set, so an inherited half pair cannot open the wrong tape.
    """
    env = os.environ if environ is None else environ
    for out_name, id_name in ((TAPE_OUT_ENV, TAPE_ID_ENV), (RECORD_OUT_ENV, RECORD_TRACE_ID_ENV)):
        out = env.get(out_name)
        trace_id = env.get(id_name)
        if out and trace_id:
            break
    else:
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
